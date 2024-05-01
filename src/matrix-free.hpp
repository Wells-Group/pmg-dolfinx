#include "hip/hip_runtime.h"
#include "small-csr.hpp"
#include <cstdint>
#include <thrust/device_vector.h>

namespace
{

template <typename T>
__device__ void tabulate_local_Q3(T* A, const T* w, const T* c, const T* coordinate_dofs)
{
  // Quadrature rules
  static const double weights_34a[125]
      = {0.001662467052579079, 0.003358438595671477, 0.003991775919106033, 0.003358438595671477,
         0.001662467052579079, 0.003358438595671477, 0.006784561404328516, 0.008063999999999995,
         0.006784561404328516, 0.003358438595671477, 0.003991775919106033, 0.008063999999999995,
         0.009584716258668146, 0.008063999999999995, 0.003991775919106033, 0.003358438595671477,
         0.006784561404328516, 0.008063999999999995, 0.006784561404328516, 0.003358438595671477,
         0.001662467052579079, 0.003358438595671477, 0.003991775919106033, 0.003358438595671477,
         0.001662467052579079, 0.003358438595671477, 0.006784561404328516, 0.008063999999999995,
         0.006784561404328516, 0.003358438595671477, 0.006784561404328516, 0.01370585530681735,
         0.01629051763370603,  0.01370585530681735,  0.006784561404328516, 0.008063999999999995,
         0.01629051763370603,  0.0193625978702756,   0.01629051763370603,  0.008063999999999995,
         0.006784561404328516, 0.01370585530681735,  0.01629051763370603,  0.01370585530681735,
         0.006784561404328516, 0.003358438595671477, 0.006784561404328516, 0.008063999999999995,
         0.006784561404328516, 0.003358438595671477, 0.003991775919106033, 0.008063999999999995,
         0.009584716258668146, 0.008063999999999995, 0.003991775919106033, 0.008063999999999995,
         0.01629051763370603,  0.0193625978702756,   0.01629051763370603,  0.008063999999999995,
         0.009584716258668146, 0.0193625978702756,   0.02301401371742111,  0.0193625978702756,
         0.009584716258668146, 0.008063999999999995, 0.01629051763370603,  0.0193625978702756,
         0.01629051763370603,  0.008063999999999995, 0.003991775919106033, 0.008063999999999995,
         0.009584716258668146, 0.008063999999999995, 0.003991775919106033, 0.003358438595671477,
         0.006784561404328516, 0.008063999999999995, 0.006784561404328516, 0.003358438595671477,
         0.006784561404328516, 0.01370585530681735,  0.01629051763370603,  0.01370585530681735,
         0.006784561404328516, 0.008063999999999995, 0.01629051763370603,  0.0193625978702756,
         0.01629051763370603,  0.008063999999999995, 0.006784561404328516, 0.01370585530681735,
         0.01629051763370603,  0.01370585530681735,  0.006784561404328516, 0.003358438595671477,
         0.006784561404328516, 0.008063999999999995, 0.006784561404328516, 0.003358438595671477,
         0.001662467052579079, 0.003358438595671477, 0.003991775919106033, 0.003358438595671477,
         0.001662467052579079, 0.003358438595671477, 0.006784561404328516, 0.008063999999999995,
         0.006784561404328516, 0.003358438595671477, 0.003991775919106033, 0.008063999999999995,
         0.009584716258668146, 0.008063999999999995, 0.003991775919106033, 0.003358438595671477,
         0.006784561404328516, 0.008063999999999995, 0.006784561404328516, 0.003358438595671477,
         0.001662467052579079, 0.003358438595671477, 0.003991775919106033, 0.003358438595671477,
         0.001662467052579079};
  // Precomputed values of basis functions and precomputations
  // FE* dimensions: [permutation][entities][points][dofs]
  static const double FE_TF0[1][1][5][4]
      = {{{{-5.094806789291987, 0.5639075595986678, 6.356016612033354, -1.825117382340034},
           {-2.183482767485546, -0.5088637830428693, 0.9823862405771633, 1.709960309951253},
           {0.2499999999999998, -0.2500000000000002, -2.795084971874736, 2.795084971874737},
           {0.5088637830428694, 2.183482767485546, -1.709960309951253, -0.9823862405771616},
           {-0.5639075595986669, 5.094806789291987, 1.825117382340032, -6.356016612033352}}}};
  static const double FE_TF1[1][1][5][4]
      = {{{{0.7400289499867195, 0.03642344149505652, 0.3382587987733497, -0.1147111902551258},
           {0.08649005029831021, 0.02594644710880267, 0.9781189357260486, -0.09055543313316147},
           {-0.125, -0.1250000000000001, 0.6250000000000001, 0.6250000000000001},
           {0.02594644710880263, 0.08649005029831, -0.09055543313316122, 0.9781189357260488},
           {0.03642344149505661, 0.740028949986719, -0.1147111902551258, 0.3382587987733504}}}};
  static const double FE_TF2[1][1][5][2] = {{{{0.953089922969332, 0.04691007703066796},
                                              {0.7692346550528416, 0.2307653449471584},
                                              {0.5, 0.5},
                                              {0.2307653449471585, 0.7692346550528415},
                                              {0.04691007703066802, 0.9530899229693319}}}};
  static const double FE_TF3[1][1][5][2]
      = {{{{-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}}}};

  for (int iq0 = 0; iq0 < 5; ++iq0)
  {
    for (int iq1 = 0; iq1 < 5; ++iq1)
    {
      for (int iq2 = 0; iq2 < 5; ++iq2)
      {
        // ------------------------
        // Section: Jacobian
        // Inputs: FE_TF2, FE_TF3, coordinate_dofs
        // Outputs: J_c6, J_c7, J_c5, J_c0, J_c4, J_c2, J_c8, J_c3, J_c1
        double J_c4 = 0.0;
        double J_c8 = 0.0;
        double J_c5 = 0.0;
        double J_c7 = 0.0;
        double J_c0 = 0.0;
        double J_c3 = 0.0;
        double J_c6 = 0.0;
        double J_c1 = 0.0;
        double J_c2 = 0.0;
        {
          for (int ic0 = 0; ic0 < 2; ++ic0)
          {
            for (int ic1 = 0; ic1 < 2; ++ic1)
            {
              for (int ic2 = 0; ic2 < 2; ++ic2)
              {
                J_c4
                    += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3 + 1]
                       * (FE_TF2[0][0][iq0][ic0] * FE_TF3[0][0][iq1][ic1] * FE_TF2[0][0][iq2][ic2]);
              }
            }
            for (int ic1 = 0; ic1 < 2; ++ic1)
            {
              for (int ic2 = 0; ic2 < 2; ++ic2)
              {
                J_c8
                    += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3 + 2]
                       * (FE_TF2[0][0][iq0][ic0] * FE_TF2[0][0][iq1][ic1] * FE_TF3[0][0][iq2][ic2]);
              }
            }
            for (int ic1 = 0; ic1 < 2; ++ic1)
            {
              for (int ic2 = 0; ic2 < 2; ++ic2)
              {
                J_c5
                    += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3 + 1]
                       * (FE_TF2[0][0][iq0][ic0] * FE_TF2[0][0][iq1][ic1] * FE_TF3[0][0][iq2][ic2]);
              }
            }
            for (int ic1 = 0; ic1 < 2; ++ic1)
            {
              for (int ic2 = 0; ic2 < 2; ++ic2)
              {
                J_c7
                    += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3 + 2]
                       * (FE_TF2[0][0][iq0][ic0] * FE_TF3[0][0][iq1][ic1] * FE_TF2[0][0][iq2][ic2]);
              }
            }
            for (int ic1 = 0; ic1 < 2; ++ic1)
            {
              for (int ic2 = 0; ic2 < 2; ++ic2)
              {
                J_c0
                    += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3]
                       * (FE_TF3[0][0][iq0][ic0] * FE_TF2[0][0][iq1][ic1] * FE_TF2[0][0][iq2][ic2]);
              }
            }
            for (int ic1 = 0; ic1 < 2; ++ic1)
            {
              for (int ic2 = 0; ic2 < 2; ++ic2)
              {
                J_c3
                    += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3 + 1]
                       * (FE_TF3[0][0][iq0][ic0] * FE_TF2[0][0][iq1][ic1] * FE_TF2[0][0][iq2][ic2]);
              }
            }
            for (int ic1 = 0; ic1 < 2; ++ic1)
            {
              for (int ic2 = 0; ic2 < 2; ++ic2)
              {
                J_c6
                    += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3 + 2]
                       * (FE_TF3[0][0][iq0][ic0] * FE_TF2[0][0][iq1][ic1] * FE_TF2[0][0][iq2][ic2]);
              }
            }
            for (int ic1 = 0; ic1 < 2; ++ic1)
            {
              for (int ic2 = 0; ic2 < 2; ++ic2)
              {
                J_c1
                    += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3]
                       * (FE_TF2[0][0][iq0][ic0] * FE_TF3[0][0][iq1][ic1] * FE_TF2[0][0][iq2][ic2]);
              }
            }
            for (int ic1 = 0; ic1 < 2; ++ic1)
            {
              for (int ic2 = 0; ic2 < 2; ++ic2)
              {
                J_c2
                    += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3]
                       * (FE_TF2[0][0][iq0][ic0] * FE_TF2[0][0][iq1][ic1] * FE_TF3[0][0][iq2][ic2]);
              }
            }
          }
        }
        // ------------------------
        // ------------------------
        // Section: Coefficient
        // Inputs: w, FE_TF1, FE_TF0
        // Outputs: w0_d010, w0_d100, w0_d001
        double w0_d100 = 0.0;
        double w0_d010 = 0.0;
        double w0_d001 = 0.0;
        {
          for (int ic0 = 0; ic0 < 4; ++ic0)
          {
            for (int ic1 = 0; ic1 < 4; ++ic1)
            {
              for (int ic2 = 0; ic2 < 4; ++ic2)
              {
                w0_d100
                    += w[16 * ic0 + 4 * ic1 + ic2]
                       * (FE_TF0[0][0][iq0][ic0] * FE_TF1[0][0][iq1][ic1] * FE_TF1[0][0][iq2][ic2]);
              }
            }
            for (int ic1 = 0; ic1 < 4; ++ic1)
            {
              for (int ic2 = 0; ic2 < 4; ++ic2)
              {
                w0_d010
                    += w[16 * ic0 + 4 * ic1 + ic2]
                       * (FE_TF1[0][0][iq0][ic0] * FE_TF0[0][0][iq1][ic1] * FE_TF1[0][0][iq2][ic2]);
              }
            }
            for (int ic1 = 0; ic1 < 4; ++ic1)
            {
              for (int ic2 = 0; ic2 < 4; ++ic2)
              {
                w0_d001
                    += w[16 * ic0 + 4 * ic1 + ic2]
                       * (FE_TF1[0][0][iq0][ic0] * FE_TF1[0][0][iq1][ic1] * FE_TF0[0][0][iq2][ic2]);
              }
            }
          }
        }
        // ------------------------
        // ------------------------
        // Section: Intermediates
        // Inputs: J_c6, J_c7, J_c5, J_c1, J_c4, J_c2, J_c8, J_c3, J_c0, w0_d001, w0_d100, w0_d010
        // Outputs: fw0, fw1, fw2
        double fw0 = 0;
        double fw1 = 0;
        double fw2 = 0;
        {
          double sv_34a_0 = J_c4 * J_c8;
          double sv_34a_1 = J_c5 * J_c7;
          double sv_34a_2 = -sv_34a_1;
          double sv_34a_3 = sv_34a_0 + sv_34a_2;
          double sv_34a_4 = J_c0 * sv_34a_3;
          double sv_34a_5 = J_c3 * J_c8;
          double sv_34a_6 = J_c5 * J_c6;
          double sv_34a_7 = -sv_34a_6;
          double sv_34a_8 = sv_34a_5 + sv_34a_7;
          double sv_34a_9 = -J_c1;
          double sv_34a_10 = sv_34a_8 * sv_34a_9;
          double sv_34a_11 = sv_34a_4 + sv_34a_10;
          double sv_34a_12 = J_c3 * J_c7;
          double sv_34a_13 = J_c4 * J_c6;
          double sv_34a_14 = -sv_34a_13;
          double sv_34a_15 = sv_34a_12 + sv_34a_14;
          double sv_34a_16 = J_c2 * sv_34a_15;
          double sv_34a_17 = sv_34a_11 + sv_34a_16;
          double sv_34a_18 = sv_34a_3 / sv_34a_17;
          double sv_34a_19 = -J_c8;
          double sv_34a_20 = J_c3 * sv_34a_19;
          double sv_34a_21 = sv_34a_6 + sv_34a_20;
          double sv_34a_22 = sv_34a_21 / sv_34a_17;
          double sv_34a_23 = sv_34a_15 / sv_34a_17;
          double sv_34a_24 = w0_d100 * sv_34a_18;
          double sv_34a_25 = w0_d010 * sv_34a_22;
          double sv_34a_26 = sv_34a_24 + sv_34a_25;
          double sv_34a_27 = w0_d001 * sv_34a_23;
          double sv_34a_28 = sv_34a_26 + sv_34a_27;
          double sv_34a_29 = sv_34a_28 * sv_34a_18;
          double sv_34a_30 = sv_34a_28 * sv_34a_22;
          double sv_34a_31 = sv_34a_28 * sv_34a_23;
          double sv_34a_32 = J_c2 * J_c7;
          double sv_34a_33 = J_c8 * sv_34a_9;
          double sv_34a_34 = sv_34a_32 + sv_34a_33;
          double sv_34a_35 = sv_34a_34 / sv_34a_17;
          double sv_34a_36 = J_c0 * J_c8;
          double sv_34a_37 = -J_c2;
          double sv_34a_38 = J_c6 * sv_34a_37;
          double sv_34a_39 = sv_34a_36 + sv_34a_38;
          double sv_34a_40 = sv_34a_39 / sv_34a_17;
          double sv_34a_41 = J_c1 * J_c6;
          double sv_34a_42 = J_c0 * J_c7;
          double sv_34a_43 = -sv_34a_42;
          double sv_34a_44 = sv_34a_41 + sv_34a_43;
          double sv_34a_45 = sv_34a_44 / sv_34a_17;
          double sv_34a_46 = w0_d100 * sv_34a_35;
          double sv_34a_47 = w0_d010 * sv_34a_40;
          double sv_34a_48 = sv_34a_46 + sv_34a_47;
          double sv_34a_49 = w0_d001 * sv_34a_45;
          double sv_34a_50 = sv_34a_48 + sv_34a_49;
          double sv_34a_51 = sv_34a_50 * sv_34a_35;
          double sv_34a_52 = sv_34a_50 * sv_34a_40;
          double sv_34a_53 = sv_34a_50 * sv_34a_45;
          double sv_34a_54 = sv_34a_51 + sv_34a_29;
          double sv_34a_55 = sv_34a_52 + sv_34a_30;
          double sv_34a_56 = sv_34a_31 + sv_34a_53;
          double sv_34a_57 = J_c1 * J_c5;
          double sv_34a_58 = J_c2 * J_c4;
          double sv_34a_59 = -sv_34a_58;
          double sv_34a_60 = sv_34a_57 + sv_34a_59;
          double sv_34a_61 = sv_34a_60 / sv_34a_17;
          double sv_34a_62 = J_c2 * J_c3;
          double sv_34a_63 = J_c0 * J_c5;
          double sv_34a_64 = -sv_34a_63;
          double sv_34a_65 = sv_34a_62 + sv_34a_64;
          double sv_34a_66 = sv_34a_65 / sv_34a_17;
          double sv_34a_67 = J_c0 * J_c4;
          double sv_34a_68 = J_c1 * J_c3;
          double sv_34a_69 = -sv_34a_68;
          double sv_34a_70 = sv_34a_67 + sv_34a_69;
          double sv_34a_71 = sv_34a_70 / sv_34a_17;
          double sv_34a_72 = w0_d100 * sv_34a_61;
          double sv_34a_73 = w0_d010 * sv_34a_66;
          double sv_34a_74 = sv_34a_72 + sv_34a_73;
          double sv_34a_75 = w0_d001 * sv_34a_71;
          double sv_34a_76 = sv_34a_74 + sv_34a_75;
          double sv_34a_77 = sv_34a_76 * sv_34a_61;
          double sv_34a_78 = sv_34a_76 * sv_34a_66;
          double sv_34a_79 = sv_34a_76 * sv_34a_71;
          double sv_34a_80 = sv_34a_54 + sv_34a_77;
          double sv_34a_81 = sv_34a_55 + sv_34a_78;
          double sv_34a_82 = sv_34a_56 + sv_34a_79;
          double sv_34a_83 = c[0] * sv_34a_80;
          double sv_34a_84 = c[0] * sv_34a_81;
          double sv_34a_85 = c[0] * sv_34a_82;
          double sv_34a_86 = fabs(sv_34a_17);
          double sv_34a_87 = sv_34a_83 * sv_34a_86;
          double sv_34a_88 = sv_34a_84 * sv_34a_86;
          double sv_34a_89 = sv_34a_85 * sv_34a_86;
          fw0 = sv_34a_87 * weights_34a[25 * iq0 + 5 * iq1 + iq2];
          fw1 = sv_34a_88 * weights_34a[25 * iq0 + 5 * iq1 + iq2];
          fw2 = sv_34a_89 * weights_34a[25 * iq0 + 5 * iq1 + iq2];
        }
        // ------------------------
        // ------------------------
        // Section: Tensor Computation
        // Inputs: FE_TF1, fw1, fw0, fw2, FE_TF0
        // Outputs: A
        {
          for (int i0 = 0; i0 < 4; ++i0)
          {
            for (int i1 = 0; i1 < 4; ++i1)
            {
              for (int i2 = 0; i2 < 4; ++i2)
              {
                A[(16 * i0 + 4 * i1 + i2)]
                    += fw0
                       * (FE_TF0[0][0][iq0][i0] * FE_TF1[0][0][iq1][i1] * FE_TF1[0][0][iq2][i2]);
                A[(16 * i0 + 4 * i1 + i2)]
                    += fw1
                       * (FE_TF1[0][0][iq0][i0] * FE_TF0[0][0][iq1][i1] * FE_TF1[0][0][iq2][i2]);
                A[(16 * i0 + 4 * i1 + i2)]
                    += fw2
                       * (FE_TF1[0][0][iq0][i0] * FE_TF1[0][0][iq1][i1] * FE_TF0[0][0][iq2][i2]);
              }
            }
          }
        }
        // ------------------------
      }
    }
  }
}

template <typename T>
__global__ void tabulate_tensor_Q1(int N, const std::int32_t* cell_list, T* Aglobal,
                                   const T* wglobal, const T* c, const T* coordinate_dofs_global,
                                   const std::int32_t* geom_dofmap, const std::int32_t* dofmap,
                                   const std::int8_t* bc_dofs, const SmallCSRDevice<T>** mat)
{
  // Calculate the row index for this thread.
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  // Check if the row index is out of bounds.
  if (id < N)
  {
    int cell = cell_list[id];
    // Extract w from wglobal
    const int space_dim = 8;
    double w1[space_dim];
    for (int i = 0; i < space_dim; ++i)
    {
      int dof = dofmap[cell * space_dim + i];
      w1[i] = wglobal[dof] * bc_dofs[dof];
    }
    double coordinate_dofs[24];
    for (int i = 0; i < 8; ++i)
    {
      coordinate_dofs[i * 3] = coordinate_dofs_global[geom_dofmap[cell * 8 + i] * 3];
      coordinate_dofs[i * 3 + 1] = coordinate_dofs_global[geom_dofmap[cell * 8 + i] * 3 + 1];
      coordinate_dofs[i * 3 + 2] = coordinate_dofs_global[geom_dofmap[cell * 8 + i] * 3 + 2];
    }

    T w2[27];
    mat[2]->apply(w1, w2);
    T w3[64];
    mat[3]->apply(w2, w3);
    T A3[64] = {0};
    tabulate_local_Q3(A3, w3, c, coordinate_dofs);
    T A2[27];
    mat[0]->apply(A3, A2);
    T A1[space_dim];
    mat[1]->apply(A2, A1);

    for (int i = 0; i < space_dim; ++i)
    {
      int dof = dofmap[cell * space_dim + i];
      atomicAdd(&Aglobal[dof], A1[i] * bc_dofs[dof]);
    }
  }
}

template <typename T>
__global__ void tabulate_tensor_Q2(int N, const std::int32_t* cell_list, T* Aglobal,
                                   const T* wglobal, const T* c, const T* coordinate_dofs_global,
                                   const std::int32_t* geom_dofmap, const std::int32_t* dofmap,
                                   const std::int8_t* bc_dofs, const SmallCSRDevice<T>** mat)
{
  // Calculate the row index for this thread.
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  // Check if the row index is out of bounds.
  if (id < N)
  {
    int cell = cell_list[id];
    // Extract w from wglobal
    const int space_dim = 27;
    T w2[space_dim];
    for (int i = 0; i < space_dim; ++i)
    {
      int dof = dofmap[cell * space_dim + i];
      w2[i] = wglobal[dof] * bc_dofs[dof];
    }
    T w3[64];
    mat[3]->apply(w2, w3);

    double coordinate_dofs[24];
    for (int i = 0; i < 8; ++i)
    {
      coordinate_dofs[i * 3] = coordinate_dofs_global[geom_dofmap[cell * 8 + i] * 3];
      coordinate_dofs[i * 3 + 1] = coordinate_dofs_global[geom_dofmap[cell * 8 + i] * 3 + 1];
      coordinate_dofs[i * 3 + 2] = coordinate_dofs_global[geom_dofmap[cell * 8 + i] * 3 + 2];
    }

    T A3[64] = {0};
    T A2[space_dim];
    tabulate_local_Q3(A3, w3, c, coordinate_dofs);
    mat[0]->apply(A3, A2);

    for (int i = 0; i < space_dim; ++i)
    {
      int dof = dofmap[cell * space_dim + i];
      atomicAdd(&Aglobal[dof], A2[i] * bc_dofs[dof]);
    }
  }
}

template <typename T>
__global__ void tabulate_tensor_Q3(int N, const std::int32_t* cell_list, T* Aglobal,
                                   const T* wglobal, const T* c, const T* coordinate_dofs_global,
                                   const std::int32_t* geom_dofmap, const std::int32_t* dofmap,
                                   const std::int8_t* bc_dofs, const SmallCSRDevice<T>** mat)
{
  // Calculate the row index for this thread.
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  // Check if the row index is out of bounds.
  if (id < N)
  {
    int cell = cell_list[id];
    // Extract w from wglobal
    const int space_dim = 64;
    double w[space_dim];
    for (int i = 0; i < space_dim; ++i)
    {
      int dof = dofmap[cell * space_dim + i];
      w[i] = wglobal[dof] * bc_dofs[dof];
    }

    double coordinate_dofs[24];
    for (int i = 0; i < 8; ++i)
    {
      coordinate_dofs[i * 3] = coordinate_dofs_global[geom_dofmap[cell * 8 + i] * 3];
      coordinate_dofs[i * 3 + 1] = coordinate_dofs_global[geom_dofmap[cell * 8 + i] * 3 + 1];
      coordinate_dofs[i * 3 + 2] = coordinate_dofs_global[geom_dofmap[cell * 8 + i] * 3 + 2];
    }

    T A[space_dim] = {0};
    tabulate_local_Q3(A, w, c, coordinate_dofs);
    for (int i = 0; i < space_dim; ++i)
    {
      int dof = dofmap[cell * space_dim + i];
      atomicAdd(&Aglobal[dof], A[i] * bc_dofs[dof]);
    }
  }
}
} // namespace

namespace dolfinx::acc
{
template <typename T>
class MatFreeLaplace
{
public:
  MatFreeLaplace(int degree, std::span<const std::int32_t> local_cells,
                 std::span<const std::int32_t> boundary_cells, std::span<const T> constants,
                 std::span<const T> x, std::span<const std::int32_t> x_dofmap,
                 std::span<const std::int32_t> dofmap, std::span<const std::int8_t> bc_dofs)
      : local_cells(local_cells), boundary_cells(boundary_cells), constants(constants), x(x),
        x_dofmap(x_dofmap), dofmap(dofmap), bc_dofs(bc_dofs)
  {
    switch (degree)
    {
    case 1:
      tabulate_tensor = tabulate_tensor_Q1;
      break;
    case 2:
      tabulate_tensor = tabulate_tensor_Q2;
      break;
    case 3:
      tabulate_tensor = tabulate_tensor_Q3;
      break;
    default:
      throw std::runtime_error("Invalid degree");
      break;
    }
    // TODO Other degrees
  }

  ~MatFreeLaplace() {}

  // Set interpolation matrix pointers on device
  void set_mats(std::vector<const SmallCSRDevice<T>*>& mat_vec)
  {
    mats.resize(mat_vec.size());
    thrust::copy(mat_vec.begin(), mat_vec.end(), mats.begin());
  }

  template <typename Vector>
  void operator()(Vector& in, Vector& out)
  {
    dolfinx::common::Timer tmf("% MatFree operator " + std::to_string(dofmap.size()));

    // Start vector update of input_vector
    in.scatter_fwd_begin();

    T* wglobal = in.mutable_array().data();
    T* Aglobal = out.mutable_array().data();

    // Zero result vector
    err_check(hipMemset(Aglobal, 0, out.mutable_array().size() * sizeof(T)));

    int num_cells = local_cells.size();
    dim3 block_size(256);
    dim3 grid_size((num_cells + block_size.x - 1) / block_size.x);
    hipLaunchKernelGGL(tabulate_tensor, grid_size, block_size, 0, 0, num_cells, local_cells.data(),
                       Aglobal, wglobal, constants.data(), x.data(), x_dofmap.data(), dofmap.data(),
                       bc_dofs.data(), thrust::raw_pointer_cast(mats.data()));
    err_check(hipGetLastError());

    // Wait for vector update of input_vector to complete
    in.scatter_fwd_end();

    num_cells = boundary_cells.size();
    if (num_cells > 0)
    {
      grid_size.x = ((num_cells + block_size.x - 1) / block_size.x);
      hipLaunchKernelGGL(tabulate_tensor, grid_size, block_size, 0, 0, num_cells,
                         boundary_cells.data(), Aglobal, wglobal, constants.data(), x.data(),
                         x_dofmap.data(), dofmap.data(), bc_dofs.data(),
                         thrust::raw_pointer_cast(mats.data()));
      err_check(hipGetLastError());
    }
  }

private:
  // List of cells which are local, and which are on boundary (requiring update before matvec)
  std::span<const std::int32_t> local_cells;
  std::span<const std::int32_t> boundary_cells;

  std::span<const T> constants;
  std::span<const T> x;
  std::span<const std::int32_t> x_dofmap;
  std::span<const std::int32_t> dofmap;
  std::span<const std::int8_t> bc_dofs;

  // list of per-cell interpolation matrices (on device)
  thrust::device_vector<const SmallCSRDevice<T>*> mats;

  void (*tabulate_tensor)(int, const std::int32_t*, T*, const T*, const T*, const T*,
                          const std::int32_t*, const std::int32_t*, const std::int8_t*,
                          const SmallCSRDevice<T>**);
};
} // namespace dolfinx::acc
