#include "operators.hpp"
#include "vector.hpp"
#include <dolfinx/fem/dolfinx_fem.h>
#include <dolfinx/fem/petsc.h>
#include <petscmat.h>

#pragma once

template <typename Vector>
class CoarseSolverType
{

  using T = typename Vector::value_type;

public:
  CoarseSolverType(std::shared_ptr<fem::Form<T, T>> a,
                   std::shared_ptr<const fem::DirichletBC<T, T>> bcs)
  {
    auto V = a->function_spaces()[0];
    MPI_Comm comm = a->mesh()->comm();

    // Create Coarse Operator using PETSc and Hypre
    LOG(INFO) << "Create PETScOperator";
    coarse_op = std::make_unique<PETScOperator<T>>(a, std::vector{bcs});
    err_check(hipDeviceSynchronize());

    auto im_op = coarse_op->index_map();
    LOG(INFO) << "OP:" << im_op->size_global() << "/" << im_op->size_local() << "/"
              << im_op->num_ghosts();
    auto im_V = V->dofmap()->index_map;
    LOG(INFO) << "V:" << im_V->size_global() << "/" << im_V->size_local() << "/"
              << im_V->num_ghosts();

    LOG(INFO) << "Get device matrix";
    Mat A = coarse_op->device_matrix();
    LOG(INFO) << "Create Petsc KSP";
    KSPCreate(comm, &_solver);
    LOG(INFO) << "Set KSP Type";
    KSPSetType(_solver, KSPCG);
    LOG(INFO) << "Set Operators";
    KSPSetOperators(_solver, A, A);
    LOG(INFO) << "Set iteration count";
    KSPSetTolerances(_solver, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 10);
    LOG(INFO) << "Set PC Type";
    PC prec;
    KSPGetPC(_solver, &prec);
    PCSetType(prec, PCHYPRE);
    KSPSetFromOptions(_solver);
    LOG(INFO) << "KSP Setup";
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

  void solve(Vector& x, Vector& y)
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
      LOG(ERROR) << "KSPGetIterationNumber Error:" << ierr;

    LOG(INFO) << "Converged reason: " << reason;
    LOG(INFO) << "Num iterations: " << num_iterations;

    VecHIPResetArray(_b);
    VecHIPResetArray(_x);
  }

private:
  Vec _b, _x;
  KSP _solver;
  std::unique_ptr<PETScOperator<T>> coarse_op;
};
