// Copyright (C) 2022 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#pragma once

#include <algorithm>
#include <basix/finite-element.h>
#include <basix/mdspan.hpp>
#include <basix/quadrature.h>
#include <dolfinx.h>
#include <vector>

using namespace dolfinx;

/// ---------------------------------------------------------------------------
/// Transpose matrix A and store in B
/// @param[in] A Input matrix
/// @param[out] B Output matrix
template <typename U, typename V>
void transpose(const U& A, V& B)
{
  for (std::size_t i = 0; i < A.extent(0); ++i)
  {
    for (std::size_t j = 0; j < A.extent(1); ++j)
    {
      B(i, j) = A(j, i);
    }
  }
}

/// ---------------------------------------------------------------------------
/// Compute the scaled determinant of the Jacobian ([cell][point])
/// @param[in] mesh The mesh object (which contains the coordinate map)
/// @param[in] points The quadrature points to compute Jacobian of the map
/// @param[in] weights The weights evaluated at the quadrature points
template <typename T>
std::vector<T> compute_scaled_jacobian_determinant(std::shared_ptr<const mesh::Mesh<T>> mesh,
                                                   std::vector<T> points, std::vector<T> weights)
{
  // Number of points
  std::size_t nq = weights.size();

  // Get geometry data
  const fem::CoordinateElement<T>& cmap = mesh->geometry().cmap();
  auto x_dofmap = mesh->geometry().dofmap();
  const std::size_t num_dofs_g = cmap.dim();
  std::span<const T> x_g = mesh->geometry().x();

  // Get dimensions
  const std::size_t tdim = mesh->topology()->dim();
  const std::size_t gdim = mesh->geometry().dim();
  const std::size_t nc = mesh->topology()->index_map(tdim)->size_local()
                         + mesh->topology()->index_map(tdim)->num_ghosts();

  // Tabulate basis functions at quadrature points
  std::array<std::size_t, 4> phi_shape = cmap.tabulate_shape(1, nq);
  std::vector<T> phi_b(std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
  std::mdspan<const T, std::dextents<std::size_t, 4>> phi(phi_b.data(), phi_shape);
  cmap.tabulate(1, points, {nq, gdim}, phi_b);

  // Create working arrays
  std::vector<T> coord_dofs_b(num_dofs_g * gdim);
  std::mdspan<T, std::dextents<std::size_t, 2>> coord_dofs(coord_dofs_b.data(), num_dofs_g, gdim);

  std::vector<T> J_b(tdim * gdim);
  std::mdspan<T, std::dextents<std::size_t, 2>> J(J_b.data(), tdim, gdim);
  std::vector<T> detJ_b(nc * nq);
  std::mdspan<T, std::dextents<std::size_t, 2>> detJ(detJ_b.data(), nc, nq);
  std::vector<T> det_scratch(2 * tdim * gdim);

  for (std::size_t c = 0; c < nc; ++c)
  {
    // Get cell geometry (coordinates dofs)
    for (std::size_t i = 0; i < x_dofmap.extent(1); ++i)
    {
      for (std::size_t j = 0; j < gdim; ++j)
        coord_dofs(i, j) = x_g[3 * x_dofmap(c, i) + j];
    }

    // Compute the scaled Jacobian determinant
    for (std::size_t q = 0; q < nq; ++q)
    {
      std::fill(J_b.begin(), J_b.end(), 0.0);

      // Get the derivatives at each quadrature points
      auto dphi = std::submdspan(phi, std::pair(1, tdim + 1), q, std::full_extent, 0);

      // Compute Jacobian matrix
      auto _J = std::submdspan(J, std::full_extent, std::full_extent);
      cmap.compute_jacobian(dphi, coord_dofs, _J);

      // Compute the determinant of the Jacobian
      detJ(c, q) = cmap.compute_jacobian_determinant(_J, det_scratch);

      // Scaled the determinant of the Jacobian
      detJ(c, q) = std::fabs(detJ(c, q)) * weights[q];
    }
  }

  return detJ_b;
}

/// ---------------------------------------------------------------------------
/// Compute the scaled of the geometrical factor ([cell][points][tdim][gdim])
/// @param[in] mesh The mesh object (which contains the coordinate map)
/// @param[in] points The quadrature points to compute Jacobian of the map
/// @param[in] weights The weights evaluated at the quadrature points
template <typename T>
std::vector<T> compute_scaled_geometrical_factor(std::shared_ptr<const mesh::Mesh<T>> mesh,
                                                 std::vector<T> points, std::vector<T> weights)
{
  // The number of element of the upper triangular matrix
  std::map<int, int> gdim2dim;
  gdim2dim[2] = 3;
  gdim2dim[3] = 6;

  // Number of points
  std::size_t nq = weights.size();

  // Get geometry data
  const fem::CoordinateElement<T>& cmap = mesh->geometry().cmap();
  auto x_dofmap = mesh->geometry().dofmap();
  const std::size_t num_dofs_g = cmap.dim();
  std::span<const T> x_g = mesh->geometry().x();

  // Get dimensions
  const std::size_t tdim = mesh->topology()->dim();
  const std::size_t gdim = mesh->geometry().dim();
  const std::size_t nc = mesh->topology()->index_map(tdim)->size_local()
                         + mesh->topology()->index_map(tdim)->num_ghosts();
  int dim = gdim2dim[gdim];

  // Tabulate basis functions at quadrature points
  std::array<std::size_t, 4> phi_shape = cmap.tabulate_shape(1, nq);
  std::vector<T> phi_b(std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
  std::mdspan<const T, std::dextents<std::size_t, 4>> phi(phi_b.data(), phi_shape);
  cmap.tabulate(1, points, {nq, gdim}, phi_b);

  // Create working arrays
  std::vector<T> coord_dofs_b(num_dofs_g * gdim);
  std::mdspan<T, std::dextents<std::size_t, 2>> coord_dofs(coord_dofs_b.data(), num_dofs_g, gdim);

  // Jacobian
  std::vector<T> J_b(gdim * tdim);
  std::mdspan<T, std::dextents<std::size_t, 2>> J(J_b.data(), gdim, tdim);

  // Jacobian inverse J^{-1}
  std::vector<T> K_b(tdim * gdim);
  std::mdspan<T, std::dextents<std::size_t, 2>> K(K_b.data(), tdim, gdim);

  // Jacobian inverse transpose J^{-T}
  std::vector<T> KT_b(gdim * tdim);
  std::mdspan<T, std::dextents<std::size_t, 2>> KT(KT_b.data(), gdim, tdim);

  // G = J^{-1} * J^{-T}
  std::vector<T> G_b(gdim * tdim);
  std::mdspan<T, std::dextents<std::size_t, 2>> G(G_b.data(), gdim, tdim);

  // G small
  std::vector<T> Gs_b(nc * nq * dim);
  std::mdspan<T, std::dextents<std::size_t, 3>> Gs(Gs_b.data(), nc, nq, dim);

  // Jacobian determinants
  //  std::vector<T> detJ_b(nc * nq);
  //  std::mdspan<T, std::dextents<std::size_t, 2>> detJ(detJ_b.data(), nc, nq);
  //  std::vector<T> det_scratch(2 * gdim * tdim);

  for (std::size_t c = 0; c < nc; ++c)
  {
    // Get cell geometry (coordinates dofs)
    for (std::size_t i = 0; i < x_dofmap.extent(1); ++i)
    {
      for (std::size_t j = 0; j < gdim; ++j)
        coord_dofs(i, j) = x_g[3 * x_dofmap(c, i) + j];
    }

    // Compute the scaled geometrical factor
    for (std::size_t q = 0; q < nq; ++q)
    {
      std::fill(J_b.begin(), J_b.end(), 0.0);
      std::fill(K_b.begin(), K_b.end(), 0.0);
      std::fill(KT_b.begin(), KT_b.end(), 0.0);
      std::fill(G_b.begin(), G_b.end(), 0.0);

      // Get the derivatives at each quadrature points
      auto dphi = std::submdspan(phi, std::pair(1, tdim + 1), q, std::full_extent, 0);

      // Compute Jacobian matrix
      //      auto _J = std::submdspan(J, std::full_extent, std::full_extent);
      //      cmap.compute_jacobian(dphi, coord_dofs, _J);

      for (std::size_t i = 0; i < coord_dofs.extent(1); i++)
        for (std::size_t j = 0; j < dphi.extent(0); j++)
          for (std::size_t k = 0; k < coord_dofs.extent(0); k++)
            J(i, j) += coord_dofs(k, i) * dphi(j, k);

      T Ja = J(0, 1) * J(1, 2) - J(0, 2) * J(1, 1);
      T Jb = J(0, 1) * J(2, 2) - J(0, 2) * J(2, 1);
      T Jc = J(1, 1) * J(2, 2) - J(1, 2) * J(2, 1);
      T Jd = J(0, 0) * J(1, 2) - J(0, 2) * J(1, 0);
      T Je = J(0, 0) * J(2, 2) - J(0, 2) * J(2, 0);
      T Jf = J(1, 0) * J(2, 2) - J(1, 2) * J(2, 0);
      T Jg = J(0, 0) * J(1, 1) - J(0, 1) * J(1, 0);
      T Jh = J(0, 0) * J(2, 1) - J(0, 1) * J(2, 0);
      T Ji = J(1, 0) * J(2, 1) - J(1, 1) * J(2, 0);

      T detJ = J(0, 0) * Jc - J(1, 0) * Jf + J(0, 2) * Ji;

      // // Compute the inverse Jacobian matrix
      // auto _K = std::submdspan(K, std::full_extent, std::full_extent);
      // cmap.compute_jacobian_inverse(_J, _K);

      // // Transpose K -> K^{T}
      // auto _KT = std::submdspan(KT, std::full_extent, std::full_extent);
      // transpose(_K, _KT);

      // // Compute the scaled geometrical factor (K * K^{T})
      // auto _G = std::submdspan(G, std::full_extent, std::full_extent);
      // math::dot(_K, _KT, _G);

      G(0, 0) = (Ja * Ja + Jb * Jb + Jc * Jc);
      G(0, 1) = -(Jd * Ja + Je * Jb + Jf * Jc);
      G(0, 2) = (Jg * Ja + Jh * Jb + Ji * Jc);
      G(1, 1) = (Jd * Jd + Je * Je + Jf * Jf);
      G(1, 2) = -(Jg * Jd + Jh * Je + Ji * Jf);
      G(2, 2) = (Jg * Jg + Jh * Jh + Ji * Ji);

      // Compute the scaled Jacobian determinant
      //      detJ(c, q) = cmap.compute_jacobian_determinant(_J, det_scratch);
      detJ = std::fabs(detJ);

      // Only store the upper triangular values since G is symmetric
      if (gdim == 2)
      {
        Gs(c, q, 0) = detJ * G(0, 0);
        Gs(c, q, 1) = detJ * G(0, 1);
        Gs(c, q, 2) = detJ * G(1, 1);
      }
      else if (gdim == 3)
      {
        Gs(c, q, 0) = weights[q] / detJ * G(0, 0);
        Gs(c, q, 1) = weights[q] / detJ * G(0, 1);
        Gs(c, q, 2) = weights[q] / detJ * G(0, 2);
        Gs(c, q, 3) = weights[q] / detJ * G(1, 1);
        Gs(c, q, 4) = weights[q] / detJ * G(1, 2);
        Gs(c, q, 5) = weights[q] / detJ * G(2, 2);
      }
    }
  }
  return Gs_b;
}

/// ---------------------------------------------------------------------------
/// Tabulate degree P basis functions on an interval
template <typename T>
std::vector<T> tabulate_1d(int P, int Q, int derivative)
{
  // Create element
  auto element = basix::create_element<T>(basix::element::family::P, basix::cell::type::interval, P,
                                          basix::element::lagrange_variant::gll_warped,
                                          basix::element::dpc_variant::unset, false);

  // Create quadrature
  auto [points, weights] = basix::quadrature::make_quadrature<T>(
      basix::quadrature::type::gll, basix::cell::type::interval, basix::polyset::type::standard, Q);

  // Tabulate
  auto [table, shape] = element.tabulate(1, points, {weights.size(), 1});

  return table;
}
