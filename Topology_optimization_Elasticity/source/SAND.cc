/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2021 by Justin O'Connor.
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Justin O'Connor, Colorado State University, 2021.
 */


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <iostream>
#include <fstream>
#include <algorithm>

// Above are fairly normal files to include.  I also use the sparse
// direct package, which requires BLAS/LAPACK to perform a direct
// solve while I work on a fast iterative solver for this problem.
//
// Below is the main class for this problem. The majority of functions
// follow the usual naming schemes of tutorial programs, though there
// are a couple that have been broken out of what is usually called
// the `setup_system()` function because of their length, and there
// are also a number that deal with various aspects of the
// optimization algorithm.
//
// As an added bonus, the program writes the computed design as an STL
// file that one can, for example, send to a 3d printer.
namespace SAND {
    using namespace dealii;

    template<int dim>
    class SANDTopOpt
    {
    public:
        SANDTopOpt();

        void
        run();

    private:
        void
        create_triangulation();

        void
        setup_boundary_values();

        void
        setup_block_system();

        void
        setup_filter_matrix();

        void
        assemble_system(const double barrier_size);

        void
        solve();
        std::pair<double,double>
        calculate_max_step_size(const BlockVector<double> &state, const BlockVector<double> &step, const double barrier_size) const;

        BlockVector<double>
        calculate_test_rhs(const BlockVector<double> &test_solution, const double barrier_size, const double penalty_parameter) const;

        double
        calculate_exact_merit(const BlockVector<double> &test_solution, const double barrier_size, const double penalty_parameter);

        BlockVector<double>
        find_max_step(const BlockVector<double> &state, const double barrier_size);

        BlockVector<double>
        take_scaled_step(const BlockVector<double> &state, const BlockVector<double> &step, const double descent_requirement,const double barrier_size);

        bool
        check_convergence(const BlockVector<double> &state, const double barrier_size);

        void
        output_results(const unsigned int j) const;

        void
        write_as_stl();


      // Most of the member variables are also standard. There are,
      // however, a number of variables that are specifically related
      // to the optimization algorithm (such the various scalar
      // factors below) as well as the filter matrix to ensure that
      // the design remains smooth.
        Triangulation<dim> triangulation;
        FESystem<dim> fe;
        DoFHandler<dim> dof_handler;
        AffineConstraints<double> constraints;

        std::map<types::global_dof_index, double> boundary_values;
      
        BlockSparsityPattern sparsity_pattern;
        BlockSparseMatrix<double> system_matrix;
      
        SparsityPattern filter_sparsity_pattern;
        SparseMatrix<double> filter_matrix;
      
        BlockVector<double> linear_solution;
        BlockVector<double> system_rhs;
        BlockVector<double> nonlinear_solution;
      
        const double density_ratio;
        const double density_penalty_exponent;
        const double filter_r;
        double penalty_multiplier;


        TimerOutput timer;
    };

    // This problem has quite a lot of variables. We initialize a
    // FESystem composed of 2×dim `FE_Q(1)` elements for the
    // displacement variable and its Lagrange multiplier, and 7
    // `FE_DGQ(0)` elements.  These piecewise constant functions are
    // for density-related variables: the density itself, the
    // unfiltered density, the slack variables for the lower and upper
    // bounds on the unfiltered density, and then Lagrange multipliers
    // for the connection between filtered and unfiltered densities as
    // well as for the inequality constraints.
    //
    // The order in which these elements appear is documented above.
    template<int dim>
    SANDTopOpt<dim>::SANDTopOpt()
            :
            fe(FE_DGQ<dim>(0) ^ 1,
               (FESystem<dim>(FE_Q<dim>(1) ^ dim)) ^ 1,
               FE_DGQ<dim>(0) ^ 1,
               (FESystem<dim>(FE_Q<dim>(1) ^ dim)) ^ 1,
               FE_DGQ<dim>(0) ^ 5),
            dof_handler(triangulation),
            density_ratio (.5),
            density_penalty_exponent (3),
            filter_r (.25),
            penalty_multiplier (1),
            timer(std::cout,
                  TimerOutput::summary,
                  TimerOutput::wall_times)
    {}


  // The first step then is to create the triangulation that matches the problem description in the introduction --
    // a 6-by-1 rectangle where a point force will be applied in the top center.
    template<int dim>
    void
    SANDTopOpt<dim>::create_triangulation() {
        GridGenerator::subdivided_hyper_rectangle (triangulation, 
                                                   {6,1},
                                                   Point<dim>(0,0),
                                                   Point<dim>(6,1));
        triangulation.refine_global(3);

        /*Set BCIDs   */
        for (const auto &cell : triangulation.active_cell_iterators()) {
            for (unsigned int face_number = 0;
                 face_number < GeometryInfo<dim>::faces_per_cell;
                 ++face_number) {
                if (cell->face(face_number)->at_boundary()) {
                    const auto center = cell->face(face_number)->center();
                    if (std::fabs(center(1) - 0) < 1e-12) {
                        /*Boundary ID of 2 is the 0 neumann, so no external force*/
                        cell->face(face_number)->set_boundary_id(2);
                    }
                    if (std::fabs(center(1) - 1) < 1e-12) {
                        /*Find top middle*/
                        if ((std::fabs(center(0) - 3) < .1)) {
                            /*downward force is boundary id of 1*/
                            cell->face(face_number)->set_boundary_id(1);
                        } else {
                            cell->face(face_number)->set_boundary_id(2);
                        }
                    }
                    if (std::fabs(center(0) - 0) < 1e-12) {
                        cell->face(face_number)->set_boundary_id(2);
                    }
                    if (std::fabs(center(0) - 6) < 1e-12) {
                        cell->face(face_number)->set_boundary_id(2);
                    }
                }
            }
        }

        dof_handler.distribute_dofs(fe);

        DoFRenumbering::component_wise(dof_handler);

    }

// The  bottom  corners  are  kept  in  place  in  the  y  direction  -  the  bottom  left  also  in  the  x direction.
// Because deal.ii is formulated to enforce boundary conditions along regions of the boundary,
// we do this to ensure these BCs are only enforced at points.
    template<int dim>
    void
    SANDTopOpt<dim>::setup_boundary_values() {
        for (const auto &cell : dof_handler.active_cell_iterators()) {
            for (unsigned int face_number = 0;
                 face_number < GeometryInfo<dim>::faces_per_cell;
                 ++face_number) {
                if (cell->face(face_number)->at_boundary()) {
                    const auto center = cell->face(face_number)->center();
                    if (std::fabs(center(1) - 0) < 1e-12) {

                        for (unsigned int vertex_number = 0;
                             vertex_number < GeometryInfo<dim>::vertices_per_cell;
                             ++vertex_number) {
                            const auto vert = cell->vertex(vertex_number);
                            /*Find bottom left corner*/
                            if (std::fabs(vert(0) - 0) < 1e-12 && std::fabs(
                                    vert(1) - 0) < 1e-12) {

                                const unsigned int x_displacement =
                                        cell->vertex_dof_index(vertex_number, 0);
                                const unsigned int y_displacement =
                                        cell->vertex_dof_index(vertex_number, 1);
                                const unsigned int x_displacement_multiplier =
                                        cell->vertex_dof_index(vertex_number, 2);
                                const unsigned int y_displacement_multiplier =
                                        cell->vertex_dof_index(vertex_number, 3);
                                /*set bottom left BC*/
                                boundary_values[x_displacement] = 0;
                                boundary_values[y_displacement] = 0;
                                boundary_values[x_displacement_multiplier] = 0;
                                boundary_values[y_displacement_multiplier] = 0;
                            }
                            /*Find bottom right corner*/
                            if (std::fabs(vert(0) - 6) < 1e-12 && std::fabs(
                                    vert(
                                            1)
                                    - 0)
                                                                  < 1e-12) {
                                const unsigned int y_displacement =
                                        cell->vertex_dof_index(vertex_number, 1);
                                const unsigned int y_displacement_multiplier =
                                        cell->vertex_dof_index(vertex_number, 3);
                                // const unsigned int x_displacement =
                                //         cell->vertex_dof_index(vertex_number, 0);
                                // const unsigned int x_displacement_multiplier =
                                //         cell->vertex_dof_index(vertex_number, 2);
                                /*set bottom left BC*/
                                boundary_values[y_displacement] = 0;
                                boundary_values[y_displacement_multiplier] = 0;
                            }
                        }
                    }
                }
            }
        }
    }


    // This makes a giant 9-by-9 block matrix, and also sets up the necessary block vectors.  The
    // sparsity pattern for this matrix includes the sparsity pattern for the filter matrix. It also initializes
    // any block vectors we will use.
    template<int dim>
    void
    SANDTopOpt<dim>::setup_block_system() {
        const FEValuesExtractors::Scalar densities(0);

        //MAKE n_u and n_P*****************************************************************

        /*Setup 9 by 9 block matrix*/

        std::vector<unsigned int> block_component(9, 2);
        block_component[0] = 0;
        block_component[1] = 1;
        const std::vector<types::global_dof_index> dofs_per_block =
                DoFTools::count_dofs_per_fe_block(dof_handler, block_component);

        const unsigned int n_p = dofs_per_block[0];
        const unsigned int n_u = dofs_per_block[1];
        std::cout << "n_p:  " << n_p << "   n_u   " << n_u << std::endl;
        const std::vector<unsigned int> block_sizes = {n_p, n_u, n_p, n_u, n_p, n_p, n_p, n_p, n_p};

        BlockDynamicSparsityPattern dsp(9, 9);

        for (unsigned int k = 0; k < 9; k++) {
            for (unsigned int j = 0; j < 9; j++) {
                dsp.block(j, k).reinit(block_sizes[j], block_sizes[k]);
            }
        }

        dsp.collect_sizes();

        Table<2, DoFTools::Coupling> coupling(2 * dim + 7, 2 * dim + 7);

        coupling[0][0] = DoFTools::always;

        for (unsigned int i = 0; i < dim; i++) {
            coupling[0][1 + i] = DoFTools::always;
            coupling[1 + i][0] = DoFTools::always;
        }

        coupling[0][1 + dim] = DoFTools::none;
        coupling[1 + dim][0] = DoFTools::none;

        for (unsigned int i = 0; i < dim; i++) {
            coupling[0][2 + dim + i] = DoFTools::always;
            coupling[2 + dim + i][0] = DoFTools::always;
        }

        coupling[0][2 + 2 * dim] = DoFTools::always;
        coupling[2 + 2 * dim][0] = DoFTools::always;


        coupling[0][2 + 2 * dim + 1] = DoFTools::none;
        coupling[0][2 + 2 * dim + 2] = DoFTools::none;
        coupling[0][2 + 2 * dim + 3] = DoFTools::none;
        coupling[0][2 + 2 * dim + 4] = DoFTools::none;
        coupling[2 + 2 * dim + 1][0] = DoFTools::none;
        coupling[2 + 2 * dim + 2][0] = DoFTools::none;
        coupling[2 + 2 * dim + 3][0] = DoFTools::none;
        coupling[2 + 2 * dim + 4][0] = DoFTools::none;




//Coupling for displacement

        for (unsigned int i = 0; i < dim; i++) {
            for (unsigned int k = 0; k < dim; k++) {
                coupling[1 + i][1 + k] = DoFTools::none;
            }
            coupling[1 + i][1 + dim ] = DoFTools::none;
            coupling[1 + dim ][1 + i] = DoFTools::none;

            for (unsigned int k = 0; k < dim; k++) {
                coupling[1 + i][2 + dim + k] = DoFTools::always;
                coupling[2 + dim + k][1 + i] = DoFTools::always;
            }
            for (unsigned int k = 0; k < 5; k++) {
                coupling[1 + i][2 + 2 * dim + k] = DoFTools::none;
                coupling[2 + 2 * dim + k][1 + i] = DoFTools::none;
            }
        }

// coupling for unfiltered density
        coupling[1+dim][1+dim]= DoFTools::none;
        for (unsigned int i = 0; i < dim; i++) {
            coupling[1 + dim][2 + dim + i] = DoFTools::none;
            coupling[2 + dim + i][1 + dim] = DoFTools::none;
        }

        coupling[1 + dim][3 + 2 * dim] = DoFTools::none;
        coupling[3 + 2 * dim][1 + dim] = DoFTools::none;
        coupling[1 + dim][4 + 2 * dim] = DoFTools::none;
        coupling[4 + 2 * dim][1 + dim] = DoFTools::none;
        coupling[1 + dim][5 + 2 * dim] = DoFTools::always;
        coupling[5 + 2 * dim][1 + dim] = DoFTools::always;
        coupling[1 + dim][6 + 2 * dim] = DoFTools::always;
        coupling[6 + 2 * dim][1 + dim] = DoFTools::always;

//Coupling for equality multipliers
        for (unsigned int i = 0; i < dim + 1; i++) {
            for (unsigned int k = 0; k < dim + 5; k++) {
                coupling[2 + dim + i][2 + dim + k] = DoFTools::none;
                coupling[2 + dim + k][2 + dim + i] = DoFTools::none;
            }
        }

//        Coupling for lower slack
        coupling[3 + 2 * dim][3 + 2 * dim] = DoFTools::always;
        coupling[3 + 2 * dim][4 + 2 * dim] = DoFTools::none;
        coupling[4 + 2 * dim][3 + 2 * dim] = DoFTools::none;
        coupling[3 + 2 * dim][5 + 2 * dim] = DoFTools::always;
        coupling[5 + 2 * dim][3 + 2 * dim] = DoFTools::always;
        coupling[3 + 2 * dim][6 + 2 * dim] = DoFTools::none;
        coupling[6 + 2 * dim][3 + 2 * dim] = DoFTools::none;

//
        coupling[4 + 2 * dim][4 + 2 * dim] = DoFTools::always;
        coupling[4 + 2 * dim][5 + 2 * dim] = DoFTools::none;
        coupling[5 + 2 * dim][4 + 2 * dim] = DoFTools::none;
        coupling[4 + 2 * dim][6 + 2 * dim] = DoFTools::always;
        coupling[6 + 2 * dim][4 + 2 * dim] = DoFTools::always;

//
        coupling[5 + 2 * dim][5 + 2 * dim] = DoFTools::none;
        coupling[5 + 2 * dim][6 + 2 * dim] = DoFTools::none;
        coupling[6 + 2 * dim][5 + 2 * dim] = DoFTools::none;
        coupling[6 + 2 * dim][6 + 2 * dim] = DoFTools::none;

        constraints.clear();

        const ComponentMask density_mask = fe.component_mask(densities);

        const IndexSet density_dofs = DoFTools::extract_dofs(dof_handler,
                                                       density_mask);


        const unsigned int last_density_dof = density_dofs.nth_index_in_set(density_dofs.n_elements() - 1);
        constraints.add_line(last_density_dof);
        for (unsigned int i = 1;
             i < density_dofs.n_elements(); ++i) {
            constraints.add_entry(last_density_dof,
                                  density_dofs.nth_index_in_set(i - 1), -1);
        }


      constraints.set_inhomogeneity (last_density_dof, 0);

        constraints.close();

//      DoFTools::make_sparsity_pattern (dof_handler, coupling, dsp, constraints,
//          false);
//changed it to below - works now?

        DoFTools::make_sparsity_pattern(dof_handler, coupling,dsp, constraints);

        std::set<unsigned int> neighbor_ids;
        std::set<typename Triangulation<dim>::cell_iterator> cells_to_check;
        std::set<typename Triangulation<dim>::cell_iterator> cells_to_check_temp;
        unsigned int n_neighbors, i;
        double distance;

        for (const auto &cell : dof_handler.active_cell_iterators()) {
            i = cell->active_cell_index();
            neighbor_ids.clear();
            neighbor_ids.insert(i);
            cells_to_check.clear();
            cells_to_check.insert(cell);
            n_neighbors = 1;
            while (true) {
                cells_to_check_temp.clear();
                for (auto check_cell : cells_to_check) {
                    for (unsigned int n = 0;
                         n < GeometryInfo<dim>::faces_per_cell; ++n) {
                        if (!(check_cell->face(n)->at_boundary())) {
                            distance = cell->center().distance(
                                    check_cell->neighbor(n)->center());
                            if ((distance < filter_r) &&
                                !(neighbor_ids.count(check_cell->neighbor(n)->active_cell_index()))) {
                                cells_to_check_temp.insert(check_cell->neighbor(n));
                                neighbor_ids.insert(check_cell->neighbor(n)->active_cell_index());
                            }
                        }
                    }
                }

                if (neighbor_ids.size() == n_neighbors) {
                    break;
                } else {
                    cells_to_check = cells_to_check_temp;
                    n_neighbors = neighbor_ids.size();
                }
            }
/*add all of these to the sparsity pattern*/
            for (auto j : neighbor_ids) {
                dsp.block(2, 4).add(i, j);
                dsp.block(4, 2).add(i, j);
            }
        }
//        constraints.condense(dsp);
        sparsity_pattern.copy_from(dsp);

//        This also breaks everything
//        sparsity_pattern.block(4,2).copy_from( filter_sparsity_pattern);
//        sparsity_pattern.block(2,4).copy_from( filter_sparsity_pattern);

        std::ofstream out("sparsity.plt");
        sparsity_pattern.print_gnuplot(out);

        system_matrix.reinit(sparsity_pattern);


        linear_solution.reinit(9);
        nonlinear_solution.reinit(9);
        system_rhs.reinit(9);

        for (unsigned int j = 0; j < 9; j++) {
            linear_solution.block(j).reinit(block_sizes[j]);
            nonlinear_solution.block(j).reinit(block_sizes[j]);
            system_rhs.block(j).reinit(block_sizes[j]);
        }

        linear_solution.collect_sizes();
        nonlinear_solution.collect_sizes();
        system_rhs.collect_sizes();

        for (unsigned int k = 0; k < n_u; k++) {
            nonlinear_solution.block(1)[k] = 0;
            nonlinear_solution.block(3)[k] = 0;
        }
        for (unsigned int k = 0; k < n_p; k++) {
            nonlinear_solution.block(0)[k] = density_ratio;
            nonlinear_solution.block(2)[k] = density_ratio;
            nonlinear_solution.block(4)[k] = density_ratio;
            nonlinear_solution.block(5)[k] = density_ratio;
            nonlinear_solution.block(6)[k] = 50;
            nonlinear_solution.block(7)[k] = 1 - density_ratio;
            nonlinear_solution.block(8)[k] = 50;
        }

    }

  
    // A  function  used  once  at  the  beginning  of  the  program,  this  creates  a  matrix  H  so  that H* unfiltered density = filtered density
    template<int dim>
    void
    SANDTopOpt<dim>::setup_filter_matrix() {
        DynamicSparsityPattern filter_dsp(dof_handler.get_triangulation().n_active_cells(),
                                          dof_handler.get_triangulation().n_active_cells());
        std::set<unsigned int> neighbor_ids;
        std::set<typename Triangulation<dim>::cell_iterator> cells_to_check;
        std::set<typename Triangulation<dim>::cell_iterator> cells_to_check_temp;
        double distance;

        /*finds neighbors-of-neighbors until it is out to specified radius*/
        for (const auto &cell : dof_handler.active_cell_iterators()) {
            const unsigned int i = cell->active_cell_index();
            neighbor_ids = {i};
            cells_to_check = {cell};
            
            unsigned int n_neighbors = 1;
            while (true) {
                cells_to_check_temp.clear();
                for (auto check_cell : cells_to_check) {
                    for (unsigned int n = 0;
                         n < GeometryInfo<dim>::faces_per_cell; ++n) {
                        if (!(check_cell->face(n)->at_boundary())) {
                            distance = cell->center().distance(
                                    check_cell->neighbor(n)->center());
                            if ((distance < filter_r) &&
                                !(neighbor_ids.count(check_cell->neighbor(n)->active_cell_index()))) {
                                cells_to_check_temp.insert(check_cell->neighbor(n));
                                neighbor_ids.insert(check_cell->neighbor(n)->active_cell_index());
                            }
                        }
                    }
                }

                if (neighbor_ids.size() == n_neighbors) {
                    break;
                } else {
                    cells_to_check = cells_to_check_temp;
                    n_neighbors = neighbor_ids.size();
                }
            }
/*add all of these to the sparsity pattern*/
            for (auto j : neighbor_ids) {
                filter_dsp.add(i, j);
            }
        }

        filter_sparsity_pattern.copy_from(filter_dsp);
        filter_matrix.reinit(filter_sparsity_pattern);

/*find these cells again to add values to matrix*/
        for (const auto &cell : dof_handler.active_cell_iterators()) {
            const unsigned int i = cell->active_cell_index();
            neighbor_ids = {i};
            cells_to_check = {cell};
            cells_to_check_temp = {};
            
            unsigned int n_neighbors = 1;
            filter_matrix.add(i, i, filter_r);
            while (true) {
                cells_to_check_temp.clear();
                for (auto check_cell : cells_to_check) {
                    for (unsigned int n = 0; n < GeometryInfo<dim>::faces_per_cell; ++n) {
                        if (!(check_cell->face(n)->at_boundary())) {
                            distance = cell->center().distance(
                                    check_cell->neighbor(n)->center());
                            if ((distance < filter_r) && !(neighbor_ids.count(
                                    check_cell->neighbor(n)->active_cell_index()))) {
                                cells_to_check_temp.insert(
                                        check_cell->neighbor(n));
                                neighbor_ids.insert(
                                        check_cell->neighbor(n)->active_cell_index());
/*value should be max radius - distance between cells*/
                                filter_matrix.add(i, check_cell->neighbor(n)->active_cell_index(),
                                                  filter_r - distance);
                            }
                        }
                    }
                }

                if (neighbor_ids.size() == n_neighbors) {
                    break;
                } else {
                    cells_to_check = cells_to_check_temp;
                    n_neighbors = neighbor_ids.size();
                }
            }
        }

        for (const auto &cell : dof_handler.active_cell_iterators()) {
            const unsigned int i = cell->active_cell_index();
            double denominator = 0;
            typename SparseMatrix<double>::iterator iter = filter_matrix.begin(
                    i);
            for (; iter != filter_matrix.end(i); iter++) {
                denominator = denominator + iter->value();
            }
            iter = filter_matrix.begin(i);
            for (; iter != filter_matrix.end(i); iter++) {
                iter->value() = iter->value() / denominator;
            }
        }
        std::cout << "filled in filter matrix" << std::endl;
    }


    // This  is  where  the  magic  happens.   The  equations  describing  the newtons method for finding 0s in the KKT conditions are implemented here.


    template<int dim>
    void
    SANDTopOpt<dim>::assemble_system(double barrier_size) {
        TimerOutput::Scope t(timer, "assembly");
        const FEValuesExtractors::Scalar densities(0);
        const FEValuesExtractors::Vector displacements(1);
        const FEValuesExtractors::Scalar unfiltered_densities(1 + dim);
        const FEValuesExtractors::Vector displacement_multipliers(2 + dim);
        const FEValuesExtractors::Scalar unfiltered_density_multipliers(2 + 2 * dim);
        const FEValuesExtractors::Scalar density_lower_slacks(3 + 2 * dim);
        const FEValuesExtractors::Scalar density_lower_slack_multipliers(
                4 + 2 * dim);
        const FEValuesExtractors::Scalar density_upper_slacks(5 + 2 * dim);
        const FEValuesExtractors::Scalar density_upper_slack_multipliers(
                6 + 2 * dim);

        /*Remove any values from old iterations*/
        system_matrix.reinit(sparsity_pattern);
        linear_solution = 0;
        system_rhs = 0;

        QGauss<dim> quadrature_formula(fe.degree + 1);
        QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);
        FEValues<dim> fe_values(fe, quadrature_formula,
                                update_values | update_gradients | update_quadrature_points
                                | update_JxW_values);
        FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                         update_values | update_quadrature_points | update_normal_vectors
                                         | update_JxW_values);

        const unsigned int dofs_per_cell = fe.dofs_per_cell;
        const unsigned int n_q_points = quadrature_formula.size();
        const unsigned int n_face_q_points = face_quadrature_formula.size();

        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> cell_rhs(dofs_per_cell);
        FullMatrix<double> full_density_cell_matrix(dofs_per_cell,
                                                    dofs_per_cell);
        FullMatrix<double> full_density_cell_matrix_for_Au(dofs_per_cell,
                                                           dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        std::vector<double> lambda_values(n_q_points);
        std::vector<double> mu_values(n_q_points);

        const Functions::ConstantFunction<dim> lambda(1.), mu(1.);
        std::vector<Tensor<1, dim>> rhs_values(n_q_points);

        BlockVector<double> filtered_unfiltered_density_solution = nonlinear_solution;
        BlockVector<double> filter_adjoint_unfiltered_density_multiplier_solution = nonlinear_solution;
        filtered_unfiltered_density_solution.block(2) = 0;
        filter_adjoint_unfiltered_density_multiplier_solution.block(4) = 0;

        filter_matrix.vmult(filtered_unfiltered_density_solution.block(2), nonlinear_solution.block(2));
        filter_matrix.Tvmult(filter_adjoint_unfiltered_density_multiplier_solution.block(4),
                             nonlinear_solution.block(4));


        std::vector<double> old_density_values(n_q_points);
        std::vector<Tensor<1, dim>> old_displacement_values(n_q_points);
        std::vector<double> old_displacement_divs(n_q_points);
        std::vector<SymmetricTensor<2, dim>> old_displacement_symmgrads(
          n_q_points);
        std::vector<Tensor<1, dim>> old_displacement_multiplier_values(
          n_q_points);
        std::vector<double> old_displacement_multiplier_divs(n_q_points);
        std::vector<SymmetricTensor<2, dim>> old_displacement_multiplier_symmgrads(
          n_q_points);
        std::vector<double> old_lower_slack_multiplier_values(n_q_points);
        std::vector<double> old_upper_slack_multiplier_values(n_q_points);
        std::vector<double> old_lower_slack_values(n_q_points);
        std::vector<double> old_upper_slack_values(n_q_points);
        std::vector<double> old_unfiltered_density_values(n_q_points);
        std::vector<double> old_unfiltered_density_multiplier_values(n_q_points);
        std::vector<double> filtered_unfiltered_density_values(n_q_points);
        std::vector<double> filter_adjoint_unfiltered_density_multiplier_values(n_q_points);


        for (const auto &cell : dof_handler.active_cell_iterators()) {
            cell_matrix = 0;
            full_density_cell_matrix = 0;
            full_density_cell_matrix_for_Au = 0;
            cell_rhs = 0;

            cell->get_dof_indices(local_dof_indices);

            fe_values.reinit(cell);

            lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
            mu.value_list(fe_values.get_quadrature_points(), mu_values);

            fe_values[densities].get_function_values(nonlinear_solution,
                                                     old_density_values);
            fe_values[displacements].get_function_values(nonlinear_solution,
                                                         old_displacement_values);
            fe_values[displacements].get_function_divergences(nonlinear_solution,
                                                              old_displacement_divs);
            fe_values[displacements].get_function_symmetric_gradients(
                    nonlinear_solution, old_displacement_symmgrads);
            fe_values[displacement_multipliers].get_function_values(
                    nonlinear_solution, old_displacement_multiplier_values);
            fe_values[displacement_multipliers].get_function_divergences(
                    nonlinear_solution, old_displacement_multiplier_divs);
            fe_values[displacement_multipliers].get_function_symmetric_gradients(
                    nonlinear_solution, old_displacement_multiplier_symmgrads);
            fe_values[density_lower_slacks].get_function_values(
                    nonlinear_solution, old_lower_slack_values);
            fe_values[density_lower_slack_multipliers].get_function_values(
                    nonlinear_solution, old_lower_slack_multiplier_values);
            fe_values[density_upper_slacks].get_function_values(
                    nonlinear_solution, old_upper_slack_values);
            fe_values[density_upper_slack_multipliers].get_function_values(
                    nonlinear_solution, old_upper_slack_multiplier_values);
            fe_values[unfiltered_densities].get_function_values(
                    nonlinear_solution, old_unfiltered_density_values);
            fe_values[unfiltered_density_multipliers].get_function_values(
                    nonlinear_solution, old_unfiltered_density_multiplier_values);
            fe_values[unfiltered_densities].get_function_values(
                    filtered_unfiltered_density_solution, filtered_unfiltered_density_values);
            fe_values[unfiltered_density_multipliers].get_function_values(
                    filter_adjoint_unfiltered_density_multiplier_solution,
                    filter_adjoint_unfiltered_density_multiplier_values);

            Tensor<1, dim> traction;
            traction[1] = -1;

            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {

                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    const SymmetricTensor<2, dim> displacement_phi_i_symmgrad =
                            fe_values[displacements].symmetric_gradient(i, q_point);
                    const double displacement_phi_i_div =
                            fe_values[displacements].divergence(i, q_point);

                    const SymmetricTensor<2, dim> displacement_multiplier_phi_i_symmgrad =
                            fe_values[displacement_multipliers].symmetric_gradient(i,
                                                                                   q_point);
                    const double displacement_multiplier_phi_i_div =
                            fe_values[displacement_multipliers].divergence(i,
                                                                           q_point);


                    const double density_phi_i = fe_values[densities].value(i,
                                                                            q_point);
                    const double unfiltered_density_phi_i = fe_values[unfiltered_densities].value(i,
                                                                                                  q_point);
                    const double unfiltered_density_multiplier_phi_i = fe_values[unfiltered_density_multipliers].value(
                            i, q_point);

                    const double lower_slack_multiplier_phi_i =
                            fe_values[density_lower_slack_multipliers].value(i,
                                                                             q_point);

                    const double lower_slack_phi_i =
                            fe_values[density_lower_slacks].value(i, q_point);

                    const double upper_slack_phi_i =
                            fe_values[density_upper_slacks].value(i, q_point);

                    const double upper_slack_multiplier_phi_i =
                            fe_values[density_upper_slack_multipliers].value(i,
                                                                             q_point);


                    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                        const SymmetricTensor<2, dim> displacement_phi_j_symmgrad =
                                fe_values[displacements].symmetric_gradient(j,
                                                                            q_point);
                        const double displacement_phi_j_div =
                                fe_values[displacements].divergence(j, q_point);

                        const SymmetricTensor<2, dim> displacement_multiplier_phi_j_symmgrad =
                                fe_values[displacement_multipliers].symmetric_gradient(
                                        j, q_point);
                        const double displacement_multiplier_phi_j_div =
                                fe_values[displacement_multipliers].divergence(j,
                                                                               q_point);

                        const double density_phi_j = fe_values[densities].value(
                                j, q_point);

                        const double unfiltered_density_phi_j = fe_values[unfiltered_densities].value(j,
                                                                                                      q_point);
                        const double unfiltered_density_multiplier_phi_j = fe_values[unfiltered_density_multipliers].value(
                                j, q_point);


                        const double lower_slack_phi_j =
                                fe_values[density_lower_slacks].value(j, q_point);

                        const double upper_slack_phi_j =
                                fe_values[density_upper_slacks].value(j, q_point);

                        const double lower_slack_multiplier_phi_j =
                                fe_values[density_lower_slack_multipliers].value(j,
                                                                                 q_point);

                        const double upper_slack_multiplier_phi_j =
                                fe_values[density_upper_slack_multipliers].value(j,
                                                                                 q_point);

                        //Equation 0
                        cell_matrix(i, j) +=
                                fe_values.JxW(q_point) *
                                (

                                        -density_phi_i * unfiltered_density_multiplier_phi_j

                                        + density_penalty_exponent * (density_penalty_exponent
                                                                      - 1)
                                          * std::pow(
                                                old_density_values[q_point],
                                                density_penalty_exponent - 2)
                                          * density_phi_i
                                          * density_phi_j
                                          * (old_displacement_multiplier_divs[q_point] * old_displacement_divs[q_point]
                                             * lambda_values[q_point]
                                             + 2 * mu_values[q_point]
                                               * (old_displacement_symmgrads[q_point] *
                                                  old_displacement_multiplier_symmgrads[q_point]))

                                        + density_penalty_exponent * std::pow(
                                                old_density_values[q_point],
                                                density_penalty_exponent - 1)
                                          * density_phi_i
                                          * (displacement_multiplier_phi_j_div * old_displacement_divs[q_point]
                                             * lambda_values[q_point]
                                             + 2 * mu_values[q_point]
                                               *
                                               (old_displacement_symmgrads[q_point] *
                                                displacement_multiplier_phi_j_symmgrad))

                                        + density_penalty_exponent * std::pow(
                                                old_density_values[q_point],
                                                density_penalty_exponent - 1)
                                          * density_phi_i
                                          * (displacement_phi_j_div * old_displacement_multiplier_divs[q_point]
                                             * lambda_values[q_point]
                                             + 2 * mu_values[q_point]
                                               * (old_displacement_multiplier_symmgrads[q_point] *
                                                  displacement_phi_j_symmgrad)));

                        //Equation 1

                        cell_matrix(i, j) +=
                                fe_values.JxW(q_point) * (
                                        density_penalty_exponent * std::pow(
                                                old_density_values[q_point],
                                                density_penalty_exponent - 1)
                                        * density_phi_j
                                        * (old_displacement_multiplier_divs[q_point] * displacement_phi_i_div
                                           * lambda_values[q_point]
                                           + 2 * mu_values[q_point]
                                             * (old_displacement_multiplier_symmgrads[q_point] *
                                                displacement_phi_i_symmgrad))

                                        + std::pow(old_density_values[q_point],
                                                   density_penalty_exponent)
                                          * (displacement_multiplier_phi_j_div * displacement_phi_i_div
                                             * lambda_values[q_point]
                                             + 2 * mu_values[q_point]
                                               * (displacement_multiplier_phi_j_symmgrad * displacement_phi_i_symmgrad))

                                );

                        //Equation 2 has to do with the filter, which is calculated elsewhere.
                        cell_matrix(i, j) +=
                                fe_values.JxW(q_point) * (
                                        -1 * unfiltered_density_phi_i * lower_slack_multiplier_phi_j
                                        + unfiltered_density_phi_i * upper_slack_multiplier_phi_j);


                        //Equation 3 - Primal Feasibility

                        cell_matrix(i, j) +=
                                fe_values.JxW(q_point) * (

                                        density_penalty_exponent * std::pow(
                                                old_density_values[q_point],
                                                density_penalty_exponent - 1)
                                        * density_phi_j
                                        * (old_displacement_divs[q_point] * displacement_multiplier_phi_i_div
                                           * lambda_values[q_point]
                                           + 2 * mu_values[q_point]
                                             * (old_displacement_symmgrads[q_point] *
                                                displacement_multiplier_phi_i_symmgrad))

                                        + std::pow(old_density_values[q_point],
                                                   density_penalty_exponent)
                                          * (displacement_phi_j_div * displacement_multiplier_phi_i_div
                                             * lambda_values[q_point]
                                             + 2 * mu_values[q_point]
                                               *
                                               (displacement_phi_j_symmgrad * displacement_multiplier_phi_i_symmgrad)));

                        //Equation 4 - more primal feasibility
                        cell_matrix(i, j) +=
                                -1 * fe_values.JxW(q_point) * lower_slack_multiplier_phi_i *
                                (unfiltered_density_phi_j - lower_slack_phi_j);

                        //Equation 5 - more primal feasibility
                        cell_matrix(i, j) +=
                                -1 * fe_values.JxW(q_point) * upper_slack_multiplier_phi_i * (
                                        -1 * unfiltered_density_phi_j - upper_slack_phi_j);

                        //Equation 6 - more primal feasibility - part with filter added later
                        cell_matrix(i, j) +=
                                -1 * fe_values.JxW(q_point) * unfiltered_density_multiplier_phi_i * (
                                        density_phi_j);

                        //Equation 7 - complementary slackness
                        cell_matrix(i, j) += fe_values.JxW(q_point)
                                             * (lower_slack_phi_i * lower_slack_multiplier_phi_j

                                                + lower_slack_phi_i * lower_slack_phi_j *
                                                  old_lower_slack_multiplier_values[q_point] /
                                                  old_lower_slack_values[q_point]);

                        //Equation 8 - complementary slackness
                        cell_matrix(i, j) += fe_values.JxW(q_point)
                                             * (upper_slack_phi_i * upper_slack_multiplier_phi_j


                                                + upper_slack_phi_i * upper_slack_phi_j
                                                  * old_upper_slack_multiplier_values[q_point] /
                                                  old_upper_slack_values[q_point]);

                    }

                    //rhs eqn 0
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (
                                    density_penalty_exponent *
                                    std::pow(old_density_values[q_point], density_penalty_exponent - 1) * density_phi_i
                                    * (old_displacement_multiplier_divs[q_point] * old_displacement_divs[q_point]
                                       * lambda_values[q_point]
                                       + 2 * mu_values[q_point] * (old_displacement_symmgrads[q_point]
                                                                   * old_displacement_multiplier_symmgrads[q_point]))
                                    - density_phi_i * old_unfiltered_density_multiplier_values[q_point]);

                    //rhs eqn 1 - boundary terms counted later
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (
                                    std::pow(old_density_values[q_point], density_penalty_exponent)
                                    * (old_displacement_multiplier_divs[q_point] * displacement_phi_i_div
                                       * lambda_values[q_point]
                                       + 2 * mu_values[q_point] * (old_displacement_multiplier_symmgrads[q_point]
                                                                   * displacement_phi_i_symmgrad))
                            );

                    //rhs eqn 2
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (
                                    unfiltered_density_phi_i *
                                    filter_adjoint_unfiltered_density_multiplier_values[q_point]
                                    + unfiltered_density_phi_i * old_upper_slack_multiplier_values[q_point]
                                    + -1 * unfiltered_density_phi_i * old_lower_slack_multiplier_values[q_point]
                            );




                    //rhs eqn 3 - boundary terms counted later
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (
                                    std::pow(old_density_values[q_point], density_penalty_exponent)
                                    * (old_displacement_divs[q_point] * displacement_multiplier_phi_i_div
                                       * lambda_values[q_point]
                                       + 2 * mu_values[q_point] * (displacement_multiplier_phi_i_symmgrad
                                                                   * old_displacement_symmgrads[q_point]))
                            );

                    //rhs eqn 4
                    cell_rhs(i) +=
                            fe_values.JxW(q_point) *
                            (lower_slack_multiplier_phi_i
                             * (old_unfiltered_density_values[q_point] - old_lower_slack_values[q_point])
                            );

                    //rhs eqn 5
                    cell_rhs(i) +=
                            fe_values.JxW(q_point) * (
                                    upper_slack_multiplier_phi_i
                                    * (1 - old_unfiltered_density_values[q_point]
                                       - old_upper_slack_values[q_point]));

                    //rhs eqn 6
                    cell_rhs(i) +=
                            fe_values.JxW(q_point) * (
                                    unfiltered_density_multiplier_phi_i
                                    * (old_density_values[q_point] - filtered_unfiltered_density_values[q_point])
                            );

                    //rhs eqn 7
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (lower_slack_phi_i
                                                           * (old_lower_slack_multiplier_values[q_point] -
                                                              barrier_size / old_lower_slack_values[q_point])
                            );

                    //rhs eqn 8
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (upper_slack_phi_i
                                                           * (old_upper_slack_multiplier_values[q_point] -
                                                              barrier_size / old_upper_slack_values[q_point]));

                }

            }
            for (unsigned int face_number = 0;
                 face_number < GeometryInfo<dim>::faces_per_cell;
                 ++face_number) {
                if (cell->face(face_number)->at_boundary() && cell->face(
                        face_number)->boundary_id()
                                                              == 1) {
                    fe_face_values.reinit(cell, face_number);

                    for (unsigned int face_q_point = 0;
                         face_q_point < n_face_q_points; ++face_q_point) {
                        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                            cell_rhs(i) += -1
                                           * traction
                                           * fe_face_values[displacements].value(i,
                                                                                 face_q_point)
                                           * fe_face_values.JxW(face_q_point);

                            cell_rhs(i) += traction
                                           * fe_face_values[displacement_multipliers].value(
                                    i, face_q_point)
                                           * fe_face_values.JxW(face_q_point);
                        }
                    }
                }
            }


            MatrixTools::local_apply_boundary_values(boundary_values, local_dof_indices,
                                                     cell_matrix, cell_rhs, true);

            constraints.distribute_local_to_global(
                    cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);


        }


        for (const auto &cell : dof_handler.active_cell_iterators()) {
            unsigned int i = cell->active_cell_index();
            typename SparseMatrix<double>::iterator iter = filter_matrix.begin(
                    i);
            for (; iter != filter_matrix.end(i); iter++) {
                unsigned int j = iter->column();
                double value = iter->value() * cell->measure();

                system_matrix.block(4, 2).add(i, j, value);
                system_matrix.block(2, 4).add(j, i, value);
            }
        }
    }


  
    // A direct solver, for now. The complexity of the system means that an iterative solver algorithm will take some more work in the future.
    template<int dim>
    void
    SANDTopOpt<dim>::solve() {
        TimerOutput::Scope t(timer, "solver");
        SparseDirectUMFPACK A_direct;
        A_direct.initialize(system_matrix);
        A_direct.vmult(linear_solution, system_rhs);

        constraints.distribute(linear_solution);
    }

    // A binary search figures out the maximum step that meets the dual feasibility - that s>0 and z>0. The fraction to boundary increases as the barrier size decreases.

    template<int dim>
    std::pair<double,double>
    SANDTopOpt<dim>::calculate_max_step_size(const BlockVector<double> &state, const BlockVector<double> &step, const double barrier_size) const {

        double fraction_to_boundary;
        const double min_fraction_to_boundary = .8;
        const double max_fraction_to_boundary = .99999;

        if (min_fraction_to_boundary < 1 - barrier_size)
        {
            if (1 - barrier_size < max_fraction_to_boundary)
            {
                fraction_to_boundary = 1-barrier_size;
            }
            else
            {
                fraction_to_boundary = max_fraction_to_boundary;
            }

        }
        else
        {
            fraction_to_boundary = min_fraction_to_boundary;
        }

        double step_size_s_low = 0;
        double step_size_z_low = 0;
        double step_size_s_high = 1;
        double step_size_z_high = 1;
        double step_size_s, step_size_z;

        for (unsigned int k = 0; k < 50; k++) {
            step_size_s = (step_size_s_low + step_size_s_high) / 2;
            step_size_z = (step_size_z_low + step_size_z_high) / 2;

            const BlockVector<double> state_test_s =
                    (fraction_to_boundary * state) + (step_size_s * step);

            const BlockVector<double> state_test_z =
                    (fraction_to_boundary * state) + (step_size_z * step);

            const bool accept_s = (state_test_s.block(5).is_non_negative())
                                  && (state_test_s.block(7).is_non_negative());
            const bool accept_z = (state_test_z.block(6).is_non_negative())
                                  && (state_test_z.block(8).is_non_negative());

            if (accept_s) {
                step_size_s_low = step_size_s;
            } else {
                step_size_s_high = step_size_s;
            }
            if (accept_z) {
                step_size_z_low = step_size_z;
            } else {
                step_size_z_high = step_size_z;
            }
        }
//        std::cout << step_size_s_low << "    " << step_size_z_low << std::endl;
        return {step_size_s_low, step_size_z_low};
    }

// Creates a rhs vector that we can use to look at the magnitude of the KKT conditions.  This is then used for testing the convergence before shrinking barrier size, as well as in the calculation of the l1 merit.

    template<int dim>
    BlockVector<double>
    SANDTopOpt<dim>::calculate_test_rhs(const BlockVector<double> &test_solution, const double barrier_size, const double /*penalty_parameter*/) const {
        const FEValuesExtractors::Scalar densities(0);
        const FEValuesExtractors::Vector displacements(1);
        const FEValuesExtractors::Scalar unfiltered_densities(1 + dim);
        const FEValuesExtractors::Vector displacement_multipliers(2 + dim);
        const FEValuesExtractors::Scalar unfiltered_density_multipliers(2 + 2 * dim);
        const FEValuesExtractors::Scalar density_lower_slacks(3 + 2 * dim);
        const FEValuesExtractors::Scalar density_lower_slack_multipliers(
                4 + 2 * dim);
        const FEValuesExtractors::Scalar density_upper_slacks(5 + 2 * dim);
        const FEValuesExtractors::Scalar density_upper_slack_multipliers(
                6 + 2 * dim);

        /*Remove any values from old iterations*/

        BlockVector<double> test_rhs;
        test_rhs = system_rhs;
        test_rhs = 0;

        const QGauss<dim> quadrature_formula(fe.degree + 1);
        const QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);
        FEValues<dim> fe_values(fe, quadrature_formula,
                                update_values | update_gradients | update_quadrature_points
                                | update_JxW_values);
        FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                         update_values | update_quadrature_points | update_normal_vectors
                                         | update_JxW_values);

        const unsigned int dofs_per_cell = fe.dofs_per_cell;
        const unsigned int n_q_points = quadrature_formula.size();
        const unsigned int n_face_q_points = face_quadrature_formula.size();

        Vector<double> cell_rhs(dofs_per_cell);
        FullMatrix<double> dummy_cell_matrix(dofs_per_cell,dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        std::vector<double> lambda_values(n_q_points);
        std::vector<double> mu_values(n_q_points);

        const Functions::ConstantFunction<dim> lambda(1.), mu(1.);
        std::vector<Tensor<1, dim>> rhs_values(n_q_points);

        BlockVector<double> filtered_unfiltered_density_solution = nonlinear_solution;
        BlockVector<double> filter_adjoint_unfiltered_density_multiplier_solution = nonlinear_solution;
        filtered_unfiltered_density_solution.block(2) = 0;
        filter_adjoint_unfiltered_density_multiplier_solution.block(4) = 0;

        filter_matrix.vmult(filtered_unfiltered_density_solution.block(2), nonlinear_solution.block(2));
        filter_matrix.Tvmult(filter_adjoint_unfiltered_density_multiplier_solution.block(4),
                             nonlinear_solution.block(4));


        std::vector<double> old_density_values(n_q_points);
        std::vector<Tensor<1, dim>> old_displacement_values(n_q_points);
        std::vector<double> old_displacement_divs(n_q_points);
        std::vector<SymmetricTensor<2, dim>> old_displacement_symmgrads(
          n_q_points);
        std::vector<Tensor<1, dim>> old_displacement_multiplier_values(
          n_q_points);
        std::vector<double> old_displacement_multiplier_divs(n_q_points);
        std::vector<SymmetricTensor<2, dim>> old_displacement_multiplier_symmgrads(
          n_q_points);
        std::vector<double> old_lower_slack_multiplier_values(n_q_points);
        std::vector<double> old_upper_slack_multiplier_values(n_q_points);
        std::vector<double> old_lower_slack_values(n_q_points);
        std::vector<double> old_upper_slack_values(n_q_points);
        std::vector<double> old_unfiltered_density_values(n_q_points);
        std::vector<double> old_unfiltered_density_multiplier_values(n_q_points);
        std::vector<double> filtered_unfiltered_density_values(n_q_points);
        std::vector<double> filter_adjoint_unfiltered_density_multiplier_values(n_q_points);

        for (const auto &cell : dof_handler.active_cell_iterators()) {
            cell_rhs = 0;

            cell->get_dof_indices(local_dof_indices);

            fe_values.reinit(cell);

            lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
            mu.value_list(fe_values.get_quadrature_points(), mu_values);

            fe_values[densities].get_function_values(test_solution,
                                                     old_density_values);
            fe_values[displacements].get_function_values(test_solution,
                                                         old_displacement_values);
            fe_values[displacements].get_function_divergences(test_solution,
                                                              old_displacement_divs);
            fe_values[displacements].get_function_symmetric_gradients(
                    test_solution, old_displacement_symmgrads);
            fe_values[displacement_multipliers].get_function_values(
                    test_solution, old_displacement_multiplier_values);
            fe_values[displacement_multipliers].get_function_divergences(
                    test_solution, old_displacement_multiplier_divs);
            fe_values[displacement_multipliers].get_function_symmetric_gradients(
                    test_solution, old_displacement_multiplier_symmgrads);
            fe_values[density_lower_slacks].get_function_values(
                    test_solution, old_lower_slack_values);
            fe_values[density_lower_slack_multipliers].get_function_values(
                    test_solution, old_lower_slack_multiplier_values);
            fe_values[density_upper_slacks].get_function_values(
                    test_solution, old_upper_slack_values);
            fe_values[density_upper_slack_multipliers].get_function_values(
                    test_solution, old_upper_slack_multiplier_values);
            fe_values[unfiltered_densities].get_function_values(
                    test_solution, old_unfiltered_density_values);
            fe_values[unfiltered_density_multipliers].get_function_values(
                    test_solution, old_unfiltered_density_multiplier_values);
            fe_values[unfiltered_densities].get_function_values(
                    filtered_unfiltered_density_solution, filtered_unfiltered_density_values);
            fe_values[unfiltered_density_multipliers].get_function_values(
                    filter_adjoint_unfiltered_density_multiplier_solution,
                    filter_adjoint_unfiltered_density_multiplier_values);

            Tensor<1, dim> traction;
            traction[1] = -1;

            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {

                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    const SymmetricTensor<2, dim> displacement_phi_i_symmgrad =
                            fe_values[displacements].symmetric_gradient(i, q_point);
                    const double displacement_phi_i_div =
                            fe_values[displacements].divergence(i, q_point);

                    const SymmetricTensor<2, dim> displacement_multiplier_phi_i_symmgrad =
                            fe_values[displacement_multipliers].symmetric_gradient(i,
                                                                                   q_point);
                    const double displacement_multiplier_phi_i_div =
                            fe_values[displacement_multipliers].divergence(i,
                                                                           q_point);


                    const double density_phi_i = fe_values[densities].value(i,
                                                                            q_point);
                    const double unfiltered_density_phi_i = fe_values[unfiltered_densities].value(i,
                                                                                                  q_point);
                    const double unfiltered_density_multiplier_phi_i = fe_values[unfiltered_density_multipliers].value(
                            i, q_point);

                    const double lower_slack_multiplier_phi_i =
                            fe_values[density_lower_slack_multipliers].value(i,
                                                                             q_point);

                    const double lower_slack_phi_i =
                            fe_values[density_lower_slacks].value(i, q_point);

                    const double upper_slack_phi_i =
                            fe_values[density_upper_slacks].value(i, q_point);

                    const double upper_slack_multiplier_phi_i =
                            fe_values[density_upper_slack_multipliers].value(i,
                                                                             q_point);

                    //rhs eqn 0
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (
                                    density_penalty_exponent *
                                    std::pow(old_density_values[q_point], density_penalty_exponent - 1) * density_phi_i
                                    * (old_displacement_multiplier_divs[q_point] * old_displacement_divs[q_point]
                                       * lambda_values[q_point]
                                       + 2 * mu_values[q_point] * (old_displacement_symmgrads[q_point]
                                                                   * old_displacement_multiplier_symmgrads[q_point]))
                                    - density_phi_i * old_unfiltered_density_multiplier_values[q_point]);

                    //rhs eqn 1 - boundary terms counted later
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (
                                    std::pow(old_density_values[q_point], density_penalty_exponent)
                                    * (old_displacement_multiplier_divs[q_point] * displacement_phi_i_div
                                       * lambda_values[q_point]
                                       + 2 * mu_values[q_point] * (old_displacement_multiplier_symmgrads[q_point]
                                                                   * displacement_phi_i_symmgrad))
                            );

                    //rhs eqn 2
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (
                                    unfiltered_density_phi_i *
                                    filter_adjoint_unfiltered_density_multiplier_values[q_point]
                                    + unfiltered_density_phi_i * old_upper_slack_multiplier_values[q_point]
                                    + -1 * unfiltered_density_phi_i * old_lower_slack_multiplier_values[q_point]
                            );




                    //rhs eqn 3 - boundary terms counted later
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (
                                    std::pow(old_density_values[q_point], density_penalty_exponent)
                                    * (old_displacement_divs[q_point] * displacement_multiplier_phi_i_div
                                       * lambda_values[q_point]
                                       + 2 * mu_values[q_point] * (displacement_multiplier_phi_i_symmgrad
                                                                   * old_displacement_symmgrads[q_point]))
                            );

                    //rhs eqn 4
                    cell_rhs(i) +=
                            fe_values.JxW(q_point) *
                            (lower_slack_multiplier_phi_i
                             * (old_unfiltered_density_values[q_point] - old_lower_slack_values[q_point])
                            );

                    //rhs eqn 5
                    cell_rhs(i) +=
                            fe_values.JxW(q_point) * (
                                    upper_slack_multiplier_phi_i * (1 - old_unfiltered_density_values[q_point]
                                       - old_upper_slack_values[q_point]));

                    //rhs eqn 6
                    cell_rhs(i) +=
                            fe_values.JxW(q_point) * (
                                    unfiltered_density_multiplier_phi_i
                                    * (old_density_values[q_point] - filtered_unfiltered_density_values[q_point])
                            );

                    //rhs eqn 7
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (lower_slack_phi_i
                                                           * (old_lower_slack_multiplier_values[q_point] -
                                                              barrier_size / old_lower_slack_values[q_point])
                            );

                    //rhs eqn 8
                    cell_rhs(i) +=
                            -1 * fe_values.JxW(q_point) * (upper_slack_phi_i
                                                           * (old_upper_slack_multiplier_values[q_point] -
                                                              barrier_size / old_upper_slack_values[q_point]));

                }


            }

            for (unsigned int face_number = 0;
                 face_number < GeometryInfo<dim>::faces_per_cell;
                 ++face_number) {
                if (cell->face(face_number)->at_boundary() && cell->face(
                        face_number)->boundary_id()
                                                              == 1) {
                    fe_face_values.reinit(cell, face_number);

                    for (unsigned int face_q_point = 0;
                         face_q_point < n_face_q_points; ++face_q_point) {
                        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                            cell_rhs(i) += -1
                                           * traction
                                           * fe_face_values[displacements].value(i,
                                                                                 face_q_point)
                                           * fe_face_values.JxW(face_q_point);

                            cell_rhs(i) += traction
                                           * fe_face_values[displacement_multipliers].value(
                                    i, face_q_point)
                                           * fe_face_values.JxW(face_q_point);
                        }
                    }
                }
            }

            MatrixTools::local_apply_boundary_values(boundary_values, local_dof_indices,
                                                     dummy_cell_matrix, cell_rhs, true);

            constraints.distribute_local_to_global(
                    cell_rhs, local_dof_indices, test_rhs);


        }
        return test_rhs;

    }


// I use an exact l1 merit function in my watchdog algorithm to determine steps. This calculates the exact l1 merit
    template<int dim>
    double
    SANDTopOpt<dim>::calculate_exact_merit(const BlockVector<double> &test_solution, const double barrier_size, const double /*penalty_parameter*/)
    {
       TimerOutput::Scope t(timer, "merit function");

       double objective_function_merit = 0;
       double elasticity_constraint_merit = 0;
       double filter_constraint_merit = 0;
       double lower_slack_merit = 0;
       double upper_slack_merit = 0;

       //Calculate objective function
       //Loop over cells, integrate along boundary because I only have external force
       {
            const FEValuesExtractors::Vector displacements(1);
            const QGauss<dim> quadrature_formula(fe.degree + 1);
            const QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);
            FEValues<dim> fe_values(fe, quadrature_formula,
                                    update_values | update_gradients | update_quadrature_points
                                    | update_JxW_values);
            FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                             update_values | update_quadrature_points | update_normal_vectors
                                             | update_JxW_values);

            const unsigned int n_face_q_points = face_quadrature_formula.size();


            std::vector<Tensor<1, dim>> old_displacement_face_values(n_face_q_points);

            for (const auto &cell : dof_handler.active_cell_iterators()) {

                Tensor<1, dim> traction;
                traction[1] = -1;

                for (unsigned int face_number = 0;
                     face_number < GeometryInfo<dim>::faces_per_cell;
                     ++face_number) {
                    if (cell->face(face_number)->at_boundary() && cell->face(
                            face_number)->boundary_id()== 1)
                    {
                        fe_face_values.reinit(cell, face_number);
                        fe_face_values[displacements].get_function_values(test_solution,
                                                                          old_displacement_face_values);
                        for (unsigned int face_q_point = 0;
                             face_q_point < n_face_q_points; ++face_q_point) {
                            objective_function_merit +=
                                    traction
                                    * old_displacement_face_values[face_q_point]
                                    * fe_face_values.JxW(face_q_point);
                        }
                    }
                }
            }
        }
        //
        const BlockVector<double> test_rhs = calculate_test_rhs(test_solution, barrier_size, 1);


        //calculate elasticity constraint merit
        {

            elasticity_constraint_merit = penalty_multiplier * test_rhs.block(3).l1_norm();
        }

        //calculate filter constraint merit
        {
            filter_constraint_merit = penalty_multiplier * test_rhs.block(4).l1_norm();
        }

        //calculate lower slack merit
        {

            lower_slack_merit = penalty_multiplier * test_rhs.block(6).l1_norm();
        }

        //calculate upper slack merit
        {

            upper_slack_merit = penalty_multiplier * test_rhs.block(8).l1_norm();
        }



        double total_merit;

        total_merit = objective_function_merit + elasticity_constraint_merit + filter_constraint_merit + lower_slack_merit + upper_slack_merit;

        return total_merit;
    }

    // This updates the penalty multiplier in the merit function, and then returns the largest scaled feasible step

    template<int dim>
    BlockVector<double>
    SANDTopOpt<dim>::find_max_step(const BlockVector<double> &state,const double barrier_size)
    {
        nonlinear_solution = state;
        assemble_system(barrier_size);
        solve();
        const BlockVector<double> step = linear_solution;

        //Going to update penalty_multiplier in here too. Taken from 18.36 in Nocedal Wright

        double test_penalty_multiplier;
        double hess_part = 0;
        double grad_part = 0;
        double constraint_norm = 0;
        const std::vector<unsigned int> decision_variable_locations = {0, 1, 2};

        const std::vector<unsigned int> equality_constraint_locations = {3, 4, 6, 8};

        for(unsigned int i = 0; i<3; i++)
        {
            for(unsigned int j = 0; j<3; j++)
            {
                Vector<double> temp_vector;
                temp_vector.reinit(step.block(decision_variable_locations[i]).size());
                system_matrix.block(decision_variable_locations[i],decision_variable_locations[j]).vmult(temp_vector, step.block(decision_variable_locations[j]));
                hess_part = hess_part + step.block(decision_variable_locations[i]) * temp_vector;
            }
            grad_part = grad_part - system_rhs.block(decision_variable_locations[i])*step.block(decision_variable_locations[i]);
        }

        for(unsigned int i = 0; i<4; i++)
        {
            constraint_norm =   constraint_norm + system_rhs.block(equality_constraint_locations[i]).linfty_norm();
        }

        if (hess_part > 0)
        {
            test_penalty_multiplier = (grad_part + .5 * hess_part)/(.05 * constraint_norm);
        }
        else
        {
            test_penalty_multiplier = (grad_part)/(.05 * constraint_norm);
        }
        if (test_penalty_multiplier > penalty_multiplier)
        {
            penalty_multiplier = test_penalty_multiplier;
            std::cout << "penalty multiplier updated to " << penalty_multiplier << std::endl;
        }
        else
        {
            std::cout << "penalty multiplier kept at " << penalty_multiplier << std::endl;
        }

        const auto max_step_sizes= calculate_max_step_size(state,step,barrier_size);
        const double step_size_s = max_step_sizes.first;
        const double step_size_z = max_step_sizes.second;
        BlockVector<double> max_step(9);

        max_step.block(0) = step_size_s * step.block(0);
        max_step.block(1) = step_size_s * step.block(1);
        max_step.block(2) = step_size_s * step.block(2);
        max_step.block(3) = step_size_z * step.block(3);
        max_step.block(4) = step_size_z * step.block(4);
        max_step.block(5) = step_size_s * step.block(5);
        max_step.block(6) = step_size_z * step.block(6);
        max_step.block(7) = step_size_s * step.block(7);
        max_step.block(8) = step_size_z * step.block(8);

        return max_step;
    }

    // This is my back-stepping algorithm for a line search - keeps shrinking step size until it finds a step where the merit is decreased.

    template<int dim>
    BlockVector<double>
    SANDTopOpt<dim>::take_scaled_step(const BlockVector<double> &state,const BlockVector<double> &max_step,const double descent_requirement, const double barrier_size)
    {
        double step_size = 1;
            for(unsigned int k = 0; k<10; k++)
            {
                const double merit_derivative = (calculate_exact_merit(state + .0001 * max_step,barrier_size, 1) - calculate_exact_merit(state,barrier_size, 1))/.0001;
                if(calculate_exact_merit(state + step_size * max_step,barrier_size, 1) <calculate_exact_merit(state,barrier_size, 1) + step_size * descent_requirement * merit_derivative )
                {
                    break;
                }
                else
                {
                    step_size = step_size/2;
                }
            }
        return state + (step_size * max_step);

    }



    // Checks to see if the KKT conditions are sufficiently met to lower barrier size.
    template<int dim>
    bool
    SANDTopOpt<dim>::check_convergence(const BlockVector<double> &state,  const double barrier_size)
    {
               const double convergence_condition = 1e-2;
               const BlockVector<double> test_rhs = calculate_test_rhs(state,barrier_size,1);
               std::cout << "current rhs norm is " << test_rhs.linfty_norm() << std::endl;
               if (test_rhs.l1_norm()<convergence_condition * barrier_size)
               {
                   return true;
               }
               else
               {
                   return false;
               }
    }


    // Outputs information in a VTK file
    template<int dim>
    void
    SANDTopOpt<dim>::output_results(const unsigned int j) const {
        std::vector<std::string> solution_names(1, "density");
        std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(
                1, DataComponentInterpretation::component_is_scalar);
        for (unsigned int i = 0; i < dim; i++) {
            solution_names.emplace_back("displacement");
            data_component_interpretation.push_back(
                    DataComponentInterpretation::component_is_part_of_vector);
        }
        solution_names.emplace_back("unfiltered_density");
        data_component_interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);
        for (unsigned int i = 0; i < dim; i++) {
            solution_names.emplace_back("displacement_multiplier");
            data_component_interpretation.push_back(
                    DataComponentInterpretation::component_is_part_of_vector);
        }
        solution_names.emplace_back("unfiltered_density_multiplier");
        data_component_interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);
        solution_names.emplace_back("low_slack");
        data_component_interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);
        solution_names.emplace_back("low_slack_multiplier");
        data_component_interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);
        solution_names.emplace_back("high_slack");
        data_component_interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);
        solution_names.emplace_back("high_slack_multiplier");
        data_component_interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);
        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(nonlinear_solution, solution_names,
                                 DataOut<dim>::type_dof_data, data_component_interpretation);
//      data_out.add_data_vector (linear_solution, solution_names,
//          DataOut<dim>::type_dof_data, data_component_interpretation);
        data_out.build_patches();
        std::ofstream output("solution" + std::to_string(j) + ".vtk");
        data_out.write_vtk(output);
    }


    // This outputs an .stl file for 3d printing the result! .stl files  made up of normal vectors and triangles.
    // The triangle nodes must go counter-clockwise when looking from the outside, which requires a few checks.
    template<int dim>
    void
    SANDTopOpt<dim>::write_as_stl()
    {
    std::ofstream stlfile;
    stlfile.open ("bridge.stl");
    stlfile << "solid bridge\n" << std::scientific;
    double height = .25;
         for (const auto cell : dof_handler.active_cell_iterators())
         {

             if (nonlinear_solution.block(0)[cell->active_cell_index()]>0.5)
             {
                 if ((cell->vertex(1)[0]-cell->vertex(0)[0])*(cell->vertex(2)[1]-cell->vertex(0)[1]) - (cell->vertex(2)[0]-cell->vertex(0)[0])*(cell->vertex(1)[1]-cell->vertex(0)[1]) >0)
                 {
                     //Write one side at z = 0

                     stlfile << "   facet normal " <<0.000000e+00 <<" " << 0.000000e+00 << " " << -1.000000e+00 << "\n";
                     stlfile << "      outer loop\n";
                     stlfile << "         vertex " << cell->vertex(0)[0] << " " << cell->vertex(0)[1] << " " << 0.000000e+00 << "\n";
                     stlfile << "         vertex " << cell->vertex(2)[0] << " " << cell->vertex(2)[1] << " " << 0.000000e+00 << "\n";
                     stlfile << "         vertex " << cell->vertex(1)[0] << " " << cell->vertex(1)[1] << " " << 0.000000e+00 << "\n";
                     stlfile << "      endloop\n";
                     stlfile << "   endfacet\n";
                     stlfile << "   facet normal " <<0.000000e+00 <<" " << 0.000000e+00 << " " << -1.000000e+00 << "\n";
                     stlfile << "      outer loop\n";
                     stlfile << "         vertex " << cell->vertex(1)[0] << " " << cell->vertex(1)[1] << " " << 0.000000e+00 << "\n";
                     stlfile << "         vertex " << cell->vertex(2)[0] << " " << cell->vertex(2)[1] << " " << 0.000000e+00 << "\n";
                     stlfile << "         vertex " << cell->vertex(3)[0] << " " << cell->vertex(3)[1] << " " << 0.000000e+00 << "\n";
                     stlfile << "      endloop\n";
                     stlfile << "   endfacet\n";



                     //Write one side at z = height

                     stlfile << "   facet normal " <<0.000000e+00 <<" " << 0.000000e+00 << " " << 1.000000e+00 << "\n";
                     stlfile << "      outer loop\n";
                     stlfile << "         vertex " << cell->vertex(0)[0] << " " << cell->vertex(0)[1] << " " << height << "\n";
                     stlfile << "         vertex " << cell->vertex(1)[0] << " " << cell->vertex(1)[1] << " " << height << "\n";
                     stlfile << "         vertex " << cell->vertex(2)[0] << " " << cell->vertex(2)[1] << " " << height << "\n";
                     stlfile << "      endloop\n";
                     stlfile << "   endfacet\n";
                     stlfile << "   facet normal " <<0.000000e+00 <<" " << 0.000000e+00 << " " << 1.000000e+00 << "\n";
                     stlfile << "      outer loop\n";
                     stlfile << "         vertex " << cell->vertex(1)[0] << " " << cell->vertex(1)[1] << " " << height << "\n";
                     stlfile << "         vertex " << cell->vertex(3)[0] << " " << cell->vertex(3)[1] << " " << height << "\n";
                     stlfile << "         vertex " << cell->vertex(2)[0] << " " << cell->vertex(2)[1] << " " << height << "\n";
                     stlfile << "      endloop\n";
                     stlfile << "   endfacet\n";
                 }
                 else
                 {
                     //Write one side at z = 0

                     stlfile << "   facet normal " <<0.000000e+00 <<" " << 0.000000e+00 << " " << -1.000000e+00 << "\n";
                     stlfile << "      outer loop\n";
                     stlfile << "         vertex " << cell->vertex(0)[0] << " " << cell->vertex(0)[1] << " " << 0.000000e+00 << "\n";
                     stlfile << "         vertex " << cell->vertex(1)[0] << " " << cell->vertex(1)[1] << " " << 0.000000e+00 << "\n";
                     stlfile << "         vertex " << cell->vertex(2)[0] << " " << cell->vertex(2)[1] << " " << 0.000000e+00 << "\n";
                     stlfile << "      endloop\n";
                     stlfile << "   endfacet\n";
                     stlfile << "   facet normal " <<0.000000e+00 <<" " << 0.000000e+00 << " " << -1.000000e+00 << "\n";
                     stlfile << "      outer loop\n";
                     stlfile << "         vertex " << cell->vertex(1)[0] << " " << cell->vertex(1)[1] << " " << 0.000000e+00 << "\n";
                     stlfile << "         vertex " << cell->vertex(3)[0] << " " << cell->vertex(3)[1] << " " << 0.000000e+00 << "\n";
                     stlfile << "         vertex " << cell->vertex(2)[0] << " " << cell->vertex(2)[1] << " " << 0.000000e+00 << "\n";
                     stlfile << "      endloop\n";
                     stlfile << "   endfacet\n";



                     //Write one side at z = height

                     stlfile << "   facet normal " <<0.000000e+00 <<" " << 0.000000e+00 << " " << 1.000000e+00 << "\n";
                     stlfile << "      outer loop\n";
                     stlfile << "         vertex " << cell->vertex(0)[0] << " " << cell->vertex(0)[1] << " " << height << "\n";
                     stlfile << "         vertex " << cell->vertex(2)[0] << " " << cell->vertex(2)[1] << " " << height << "\n";
                     stlfile << "         vertex " << cell->vertex(1)[0] << " " << cell->vertex(1)[1] << " " << height << "\n";
                     stlfile << "      endloop\n";
                     stlfile << "   endfacet\n";
                     stlfile << "   facet normal " <<0.000000e+00 <<" " << 0.000000e+00 << " " << 1.000000e+00 << "\n";
                     stlfile << "      outer loop\n";
                     stlfile << "         vertex " << cell->vertex(1)[0] << " " << cell->vertex(1)[1] << " " << height << "\n";
                     stlfile << "         vertex " << cell->vertex(2)[0] << " " << cell->vertex(2)[1] << " " << height << "\n";
                     stlfile << "         vertex " << cell->vertex(3)[0] << " " << cell->vertex(3)[1] << " " << height << "\n";
                     stlfile << "      endloop\n";
                     stlfile << "   endfacet\n";
                 }



                     for (unsigned int face_number = 0;
                          face_number < GeometryInfo<dim>::faces_per_cell;
                          ++face_number)
                     {
                         if ((cell->face(face_number)->at_boundary())
                             ||
                             (!cell->face(face_number)->at_boundary()
                              &&
                              (nonlinear_solution.block(0)[cell->neighbor(face_number)->active_cell_index()]<0.5)))
                         {
                             const Tensor<1,dim> normal_vector
                                     = (cell->face(face_number)->center() - cell->center()); // maybe something better
                             double normal_norm = std::pow(normal_vector[0]*normal_vector[0] + normal_vector[1]*normal_vector[1],.5);

                             // also need to normalize

                             //                 write face into STL as two triangles, using normal_vector;
                             if ((cell->face(face_number)->vertex(0)[0] - cell->face(face_number)->vertex(0)[0])  *  (cell->face(face_number)->vertex(1)[1] - cell->face(face_number)->vertex(0)[1])  *  0.000000e+00
                                +(cell->face(face_number)->vertex(0)[1] - cell->face(face_number)->vertex(0)[1])  *  (0 - 0)                                                                                *  normal_vector[0]
                                +(height - 0)                                                                          *  (cell->face(face_number)->vertex(1)[0] - cell->face(face_number)->vertex(0)[0])        *  normal_vector[1]
                                -(cell->face(face_number)->vertex(0)[0] - cell->face(face_number)->vertex(0)[0])  *  (0 - 0)                                                                                *  normal_vector[1]
                                -(cell->face(face_number)->vertex(0)[1] - cell->face(face_number)->vertex(0)[1])  *  (cell->face(face_number)->vertex(1)[0] - cell->face(face_number)->vertex(0)[0]) *  normal_vector[0]
                                -(height - 0)                                                                     *  (cell->face(face_number)->vertex(1)[1] - cell->face(face_number)->vertex(0)[1])        *  0 >0)
                             {
                                 stlfile << "   facet normal " <<normal_vector[0]/normal_norm <<" " << normal_vector[1]/normal_norm << " " << 0.000000e+00 << "\n";
                                 stlfile << "      outer loop\n";
                                 stlfile << "         vertex " << cell->face(face_number)->vertex(0)[0] << " " << cell->face(face_number)->vertex(0)[1] << " " << 0.000000e+00 << "\n";
                                 stlfile << "         vertex " << cell->face(face_number)->vertex(0)[0] << " " << cell->face(face_number)->vertex(0)[1] << " " << height << "\n";
                                 stlfile << "         vertex " << cell->face(face_number)->vertex(1)[0] << " " << cell->face(face_number)->vertex(1)[1] << " " << 0.000000e+00 << "\n";
                                 stlfile << "      endloop\n";
                                 stlfile << "   endfacet\n";
                                 stlfile << "   facet normal " <<normal_vector[0]/normal_norm <<" " << normal_vector[1]/normal_norm << " " << 0.000000e+00 << "\n";
                                 stlfile << "      outer loop\n";
                                 stlfile << "         vertex " << cell->face(face_number)->vertex(0)[0] << " " << cell->face(face_number)->vertex(0)[1] << " " << height << "\n";
                                 stlfile << "         vertex " << cell->face(face_number)->vertex(1)[0] << " " << cell->face(face_number)->vertex(1)[1] << " " << height << "\n";
                                 stlfile << "         vertex " << cell->face(face_number)->vertex(1)[0] << " " << cell->face(face_number)->vertex(1)[1] << " " << 0.000000e+00 << "\n";
                                 stlfile << "      endloop\n";
                                 stlfile << "   endfacet\n";
                             }
                             else
                             {
                                 stlfile << "   facet normal " <<normal_vector[0]/normal_norm <<" " << normal_vector[1]/normal_norm << " " << 0.000000e+00 << "\n";
                                 stlfile << "      outer loop\n";
                                 stlfile << "         vertex " << cell->face(face_number)->vertex(0)[0] << " " << cell->face(face_number)->vertex(0)[1] << " " << 0.000000e+00 << "\n";
                                 stlfile << "         vertex " << cell->face(face_number)->vertex(1)[0] << " " << cell->face(face_number)->vertex(1)[1] << " " << 0.000000e+00 << "\n";
                                 stlfile << "         vertex " << cell->face(face_number)->vertex(0)[0] << " " << cell->face(face_number)->vertex(0)[1] << " " << height << "\n";
                                 stlfile << "      endloop\n";
                                 stlfile << "   endfacet\n";
                                 stlfile << "   facet normal " <<normal_vector[0]/normal_norm <<" " << normal_vector[1]/normal_norm << " " << 0.000000e+00 << "\n";
                                 stlfile << "      outer loop\n";
                                 stlfile << "         vertex " << cell->face(face_number)->vertex(0)[0] << " " << cell->face(face_number)->vertex(0)[1] << " " << height << "\n";
                                 stlfile << "         vertex " << cell->face(face_number)->vertex(1)[0] << " " << cell->face(face_number)->vertex(1)[1] << " " << 0.000000e+00 << "\n";
                                 stlfile << "         vertex " << cell->face(face_number)->vertex(1)[0] << " " << cell->face(face_number)->vertex(1)[1] << " " << height << "\n";
                                 stlfile << "      endloop\n";
                                 stlfile << "   endfacet\n";

                             }





                     }
                 }
             }
         }
        stlfile << "endsolid bridge";
        stlfile.close();
    }


    // Contains watchdog algorithm
    template<int dim>
    void
    SANDTopOpt<dim>::run() {
        {
          TimerOutput::Scope t(timer, "setup");
          
          create_triangulation();
          setup_block_system();
          setup_boundary_values();
          setup_filter_matrix();
        }
        
        double barrier_size = 25;
        const double min_barrier_size = .0005;
        
        const unsigned int max_uphill_steps = 8;
        unsigned int iteration_number = 0;
        const double descent_requirement = .0001;
        //while barrier value above minimal value and total iterations under some value
        BlockVector<double> current_state = nonlinear_solution;
        BlockVector<double> current_step;
        

        while((barrier_size > .0005 || !check_convergence(current_state, barrier_size)) && iteration_number < 10000)
        {
            bool converged = false;
            //while not converged
            while(!converged && iteration_number < 10000)
            {
                bool found_step = false;
                //save current state as watchdog state

                const BlockVector<double> watchdog_state = current_state;
                BlockVector<double> watchdog_step;
                double goal_merit;
                //for 1-8 steps - this is the number of steps away we will let it go uphill before demanding downhill
                for(unsigned int k = 0; k<max_uphill_steps; k++)
                {
                    //compute step from current state  - function from kktSystem
                    current_step = find_max_step(current_state, barrier_size);
                    // save the first of these as the watchdog step
                    if(k==0)
                    {
                        watchdog_step = current_step;
                    }
                    //apply full step to current state
                    current_state=current_state+current_step;
                    //if merit of current state is less than goal
                    double current_merit = calculate_exact_merit(current_state, barrier_size, 1);
                    std::cout << "current merit is: " <<current_merit << "  and  ";
                    double merit_derivative = ((calculate_exact_merit(watchdog_state+.0001*watchdog_step,barrier_size,1) - calculate_exact_merit(watchdog_state,barrier_size,1 ))/.0001);
                    goal_merit = calculate_exact_merit(watchdog_state,barrier_size,1) + descent_requirement * merit_derivative;
                    std::cout << "goal merit is "<<goal_merit <<std::endl;
                    if(current_merit < goal_merit)
                    {
                        //Accept current state
                        // iterate number of steps by number of steps taken in this process
                        iteration_number = iteration_number + k + 1;
                        //found step = true
                        found_step = true;
                        std::cout << "found workable step after " << k+1 << " iterations"<<std::endl;
                        //break for loop
                        break;
                        //end if
                    }
                    //end for
                }
                //if found step = false
                if (!found_step)
                {
                    //Compute step from current state
                    current_step = find_max_step(current_state,barrier_size);
                    //find step length so that merit of stretch state - sized step from current length - is less than merit of (current state + descent requirement * linear derivative of merit of current state in direction of current step)
                    //update stretch state with found step length
                    const BlockVector<double> stretch_state = take_scaled_step(current_state, current_step, descent_requirement, barrier_size);
                    //if current merit is less than watchdog merit, or if stretch merit is less than earlier goal merit
                    if(calculate_exact_merit(current_state,barrier_size,1) < calculate_exact_merit(watchdog_state,barrier_size,1) || calculate_exact_merit(stretch_state,barrier_size,1) < goal_merit)
                    {
                        std::cout << "in then" << std::endl;
                        current_state = stretch_state;
                        iteration_number = iteration_number + max_uphill_steps + 1;
                    }
                    else
                    {
                        std::cout << "in else" << std::endl;
                        //if merit of stretch state is bigger than watchdog merit
                        if (calculate_exact_merit(stretch_state,barrier_size,1) > calculate_exact_merit(watchdog_state,barrier_size,1))
                        {
                            //find step length from watchdog state that meets descent requirement
                            current_state = take_scaled_step(watchdog_state, watchdog_step, descent_requirement, barrier_size);
                            //update iteration count
                            iteration_number = iteration_number +  max_uphill_steps + 1;
                        }
                        else
                        {
                            //calculate direction from stretch state
                            const BlockVector<double> stretch_step = find_max_step(stretch_state,barrier_size);
                            //find step length from stretch state that meets descent requirement
                            current_state = take_scaled_step(stretch_state, stretch_step, descent_requirement,barrier_size);
                            //update iteration count
                            iteration_number = iteration_number + max_uphill_steps + 2;
                        }
                    }
                }
                //output current state
                output_results(iteration_number);
                //check convergence
                converged = check_convergence(current_state, barrier_size);
                //end while
            }
            const double barrier_size_multiplier = .5;
            const double barrier_size_exponent = 1.2;

            if (barrier_size * barrier_size_multiplier < std::pow(barrier_size, barrier_size_exponent))
            {
                if (barrier_size * barrier_size_multiplier < min_barrier_size)
                {
                    barrier_size = min_barrier_size;
                }
                else
                {
                    barrier_size = barrier_size * barrier_size_multiplier;
                }
            }
            else
            {
                if (std::pow(barrier_size, barrier_size_exponent) < min_barrier_size)
                {
                    barrier_size = min_barrier_size;
                }
                else
                {
                    barrier_size = std::pow(barrier_size, barrier_size_exponent);
                }
            }



//            barrier_size = barrier_size * barrier_size_multiplier;
            std::cout << "barrier size reduced to " << barrier_size << " on iteration number " << iteration_number << std::endl;
//
//            penalty_multiplier = 1;
            //end while
        }

        write_as_stl();
        timer.print_summary ();
    }

} // namespace SAND



// The remainder of the code, the `main()` function, is as usual:
int
main() {
    try {
        SAND::SANDTopOpt<2> elastic_problem_2d;
        elastic_problem_2d.run();
    }
    catch (std::exception &exc) {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------" << std::endl;
        std::cerr << "Exception on processing: " << std::endl << exc.what()
                  << std::endl << "Aborting!" << std::endl
                  << "----------------------------------------------------" << std::endl;

        return 1;
    }
    catch (...) {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------" << std::endl;
        std::cerr << "Unknown exception!" << std::endl << "Aborting!" << std::endl
                  << "----------------------------------------------------" << std::endl;
        return 1;
    }

    return 0;
}
