// Copyright (C) 2023 Igor A. Baratta and Chris Richardson
// SPDX-License-Identifier:    MIT

#include <dolfinx.h>
#include <dolfinx/common/log.h>
#include <dolfinx/fem/dolfinx_fem.h>
#include <dolfinx/la/MatrixCSR.h>
#include <rocalution/rocalution.hpp>

template <typename T>
class RocalutionOperator
{
public:
  RocalutionOperator(std::shared_ptr<fem::Form<T, T>> a,
                     const std::vector<std::shared_ptr<const fem::DirichletBC<T, T>>>& bcs)
  {
    dolfinx::common::Timer t0("~setup phase PETScOperator");
    static_assert(std::is_same<T, PetscScalar>(), "Type mismatch");

    if (a->rank() != 2)
      throw std::runtime_error("Form should have rank be 2.");

    la::SparsityPattern pattern = fem::create_sparsity_pattern(*a);
    pattern.assemble();

    LOG(INFO) << "Create matrix...";
    dolfinx::la::MatrixCSR<T> mat(pattern);

    LOG(INFO) << "Assemble matrix...";
    fem::assemble_matrix(mat.mat_add_values(), *a, bcs);

    LOG(INFO) << "Assemble matrix 2...";
    auto V = a->function_spaces()[0];
    fem::set_diagonal<T>(mat.mat_set_values(), *V, bcs);
    mat.finalize();

    LOG(INFO) << "Convert matrix...";
    // Create HIP matrix
    dolfinx::common::Timer t1("~Convert matrix to ROCALUTION");

    // Start to create Rocalution data structures
    auto pm = std::make_shared<rocalution::ParallelManager>();

    // Get communicator from mesh
    MPI_Comm _comm = V->mesh()->comm();

    auto _map = std::make_shared<const common::IndexMap>(pattern.column_index_map());
    const std::int32_t local_size = _map->size_local();
    const std::int64_t global_size = _map->size_global();

    // Initialize manager
    pm->SetMPICommunicator(_comm);
    pm->SetGlobalNrow(global_size);
    pm->SetGlobalNcol(global_size);
    pm->SetLocalNrow(local_size);
    pm->SetLocalNcol(local_size);

    int mpi_size = dolfinx::MPI::size(_comm);

    if (mpi_size > 1)
    {
      pm->SetBoundaryIndex(boundary_nnz, boundary.data());
      pm->SetReceivers(nrecv, recvs.data(), recv_index_offset.data());
      pm->SetSenders(nsend, sends.data(), send_index_offset.data());
    }

    auto roc_mat = std::make_shared<rocalution::GlobalMatrix<T>>(*pm);

    t1.stop();

    // LOG(INFO) << "Create vecs...";
    // VecCreateMPIHIPWithArray(_comm, PetscInt(1), local_size, global_size, NULL, &_x_petsc);
    // VecCreateMPIHIPWithArray(_comm, PetscInt(1), local_size, global_size, NULL, &_y_petsc);
  }
};
