
#include "operators.hpp"
#include "vector.hpp"
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Form.h>

#pragma once

template <typename T>
class CoarseSolverType
{
public:
  CoarseSolverType(std::shared_ptr<dolfinx::fem::Form<T, T>> a,
                   std::shared_ptr<const dolfinx::fem::DirichletBC<T, T>> bcs)
  {
    auto V = a->function_spaces()[0];
    MPI_Comm comm = a->mesh()->comm();

    // Create Coarse Operator using PETSc and Hypre
    spdlog::info("Create PETScOperator");
    coarse_op = std::make_unique<PETScOperator<T>>(a, std::vector{bcs});
    err_check(hipDeviceSynchronize());

    auto im_op = coarse_op->index_map();
    spdlog::info("OP:{}/{}/{}", im_op->size_global(), im_op->size_local(), im_op->num_ghosts());
    auto im_V = V->dofmap()->index_map;
    spdlog::info("V:{}/{}/{}", im_V->size_global(), im_V->size_local(), im_V->num_ghosts());

    spdlog::info("Get device matrix");
    Mat A = coarse_op->device_matrix();
    spdlog::info("Create Petsc KSP");
    KSPCreate(comm, &_solver);
    spdlog::info("Set KSP Type");
    KSPSetType(_solver, KSPCG);
    spdlog::info("Set Operators");
    KSPSetOperators(_solver, A, A);
    spdlog::info("Set iteration count");
    KSPSetTolerances(_solver, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 10);
    spdlog::info("Set PC Type");
    PC prec;
    KSPGetPC(_solver, &prec);
    PCSetType(prec, PCHYPRE);
    KSPSetFromOptions(_solver);
    spdlog::info("KSP Setup");
    KSPSetUp(_solver);

    const PetscInt local_size = V->dofmap()->index_map->size_local();
    const PetscInt global_size = V->dofmap()->index_map->size_global();
    VecCreateMPIHIPWithArray(comm, PetscInt(1), local_size, global_size, NULL, &_x);
    VecCreateMPIHIPWithArray(comm, PetscInt(1), local_size, global_size, NULL, &_b);
  }

  ~CoarseSolverType()
  {
    VecDestroy(&_x);
    VecDestroy(&_b);
    KSPDestroy(&_solver);
  }

  void solve(dolfinx::acc::Vector<T, acc::Device::HIP>& x,
             dolfinx::acc::Vector<T, acc::Device::HIP>& y)
  {
    VecHIPPlaceArray(_b, y.array().data());
    VecHIPPlaceArray(_x, x.array().data());

    KSPSolve(_solver, _b, _x);
    KSPView(_solver, PETSC_VIEWER_STDOUT_WORLD);

    KSPConvergedReason reason;
    KSPGetConvergedReason(_solver, &reason);

    PetscInt num_iterations = 0;
    int ierr = KSPGetIterationNumber(_solver, &num_iterations);
    if (ierr != 0)
      spdlog::error("KSPGetIterationNumber Error:{}", ierr);

    spdlog::info("Converged reason: {}", (int)reason);
    spdlog::info("Num iterations: {}", num_iterations);

    VecHIPResetArray(_b);
    VecHIPResetArray(_x);
  }

private:
  Vec _b, _x;
  KSP _solver;
  std::unique_ptr<PETScOperator<T>> coarse_op;
};
