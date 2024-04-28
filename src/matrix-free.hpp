#include "hip/hip_runtime.h"
#include <cstdint>

namespace
{

template <typename T>
__global__ void tabulate_tensor_Q1(int N, T* Aglobal, const T* wglobal, const T* c,
                                   const T* coordinate_dofs_global, const std::int32_t* geom_dofmap,
                                   const std::int32_t* dofmap)
{
  // Calculate the row index for this thread.
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  // Check if the row index is out of bounds.
  if (id < N)
  {
    // Extract w from wglobal
    const int space_dim = 8;
    double w[space_dim];
    for (int i = 0; i < space_dim; ++i)
      w[i] = wglobal[dofmap[id * space_dim + i]];

    double coordinate_dofs[24];
    for (int i = 0; i < 8; ++i)
    {
      coordinate_dofs[i * 3] = coordinate_dofs_global[geom_dofmap[id * 8 + i] * 3];
      coordinate_dofs[i * 3 + 1] = coordinate_dofs_global[geom_dofmap[id * 8 + i] * 3 + 1];
      coordinate_dofs[i * 3 + 2] = coordinate_dofs_global[geom_dofmap[id * 8 + i] * 3 + 2];
    }
    // Quadrature rules
    static const double weights_461[27]
        = {0.02143347050754454, 0.03429355281207129, 0.02143347050754456, 0.03429355281207129,
           0.05486968449931409, 0.03429355281207132, 0.02143347050754457, 0.03429355281207132,
           0.02143347050754458, 0.03429355281207129, 0.05486968449931409, 0.03429355281207132,
           0.05486968449931409, 0.0877914951989026,  0.05486968449931414, 0.03429355281207132,
           0.05486968449931413, 0.03429355281207135, 0.02143347050754457, 0.03429355281207132,
           0.02143347050754458, 0.03429355281207132, 0.05486968449931413, 0.03429355281207135,
           0.02143347050754458, 0.03429355281207135, 0.0214334705075446};
    // Precomputed values of basis functions and precomputations
    // FE* dimensions: [permutation][entities][points][dofs]
    static const double FE_TF0[1][1][3][2] = {{{{-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}}}};
    static const double FE_TF1[1][1][3][2] = {{{{0.8872983346207417, 0.1127016653792582},
                                                {0.5, 0.5},
                                                {0.1127016653792582, 0.8872983346207417}}}};
    double A[space_dim];
    for (int i = 0; i < space_dim; ++i)
      A[i] = 0.0;
    for (int iq0 = 0; iq0 < 3; ++iq0)
    {
      for (int iq1 = 0; iq1 < 3; ++iq1)
      {
        for (int iq2 = 0; iq2 < 3; ++iq2)
        {
          // ------------------------
          // Section: Jacobian
          // Inputs: coordinate_dofs, FE_TF0, FE_TF1
          // Outputs: J_c8, J_c2, J_c6, J_c3, J_c0, J_c4, J_c7, J_c1, J_c5
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
                  J_c4 += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3 + 1]
                          * (FE_TF1[0][0][iq0][ic0] * FE_TF0[0][0][iq1][ic1]
                             * FE_TF1[0][0][iq2][ic2]);
                }
              }
              for (int ic1 = 0; ic1 < 2; ++ic1)
              {
                for (int ic2 = 0; ic2 < 2; ++ic2)
                {
                  J_c8 += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3 + 2]
                          * (FE_TF1[0][0][iq0][ic0] * FE_TF1[0][0][iq1][ic1]
                             * FE_TF0[0][0][iq2][ic2]);
                }
              }
              for (int ic1 = 0; ic1 < 2; ++ic1)
              {
                for (int ic2 = 0; ic2 < 2; ++ic2)
                {
                  J_c5 += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3 + 1]
                          * (FE_TF1[0][0][iq0][ic0] * FE_TF1[0][0][iq1][ic1]
                             * FE_TF0[0][0][iq2][ic2]);
                }
              }
              for (int ic1 = 0; ic1 < 2; ++ic1)
              {
                for (int ic2 = 0; ic2 < 2; ++ic2)
                {
                  J_c7 += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3 + 2]
                          * (FE_TF1[0][0][iq0][ic0] * FE_TF0[0][0][iq1][ic1]
                             * FE_TF1[0][0][iq2][ic2]);
                }
              }
              for (int ic1 = 0; ic1 < 2; ++ic1)
              {
                for (int ic2 = 0; ic2 < 2; ++ic2)
                {
                  J_c0 += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3]
                          * (FE_TF0[0][0][iq0][ic0] * FE_TF1[0][0][iq1][ic1]
                             * FE_TF1[0][0][iq2][ic2]);
                }
              }
              for (int ic1 = 0; ic1 < 2; ++ic1)
              {
                for (int ic2 = 0; ic2 < 2; ++ic2)
                {
                  J_c3 += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3 + 1]
                          * (FE_TF0[0][0][iq0][ic0] * FE_TF1[0][0][iq1][ic1]
                             * FE_TF1[0][0][iq2][ic2]);
                }
              }
              for (int ic1 = 0; ic1 < 2; ++ic1)
              {
                for (int ic2 = 0; ic2 < 2; ++ic2)
                {
                  J_c6 += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3 + 2]
                          * (FE_TF0[0][0][iq0][ic0] * FE_TF1[0][0][iq1][ic1]
                             * FE_TF1[0][0][iq2][ic2]);
                }
              }
              for (int ic1 = 0; ic1 < 2; ++ic1)
              {
                for (int ic2 = 0; ic2 < 2; ++ic2)
                {
                  J_c1 += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3]
                          * (FE_TF1[0][0][iq0][ic0] * FE_TF0[0][0][iq1][ic1]
                             * FE_TF1[0][0][iq2][ic2]);
                }
              }
              for (int ic1 = 0; ic1 < 2; ++ic1)
              {
                for (int ic2 = 0; ic2 < 2; ++ic2)
                {
                  J_c2 += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3]
                          * (FE_TF1[0][0][iq0][ic0] * FE_TF1[0][0][iq1][ic1]
                             * FE_TF0[0][0][iq2][ic2]);
                }
              }
            }
          }
          // ------------------------
          // ------------------------
          // Section: Coefficient
          // Inputs: w, FE_TF0, FE_TF1
          // Outputs: w0_d100, w0_d010, w0_d001
          double w0_d100 = 0.0;
          double w0_d010 = 0.0;
          double w0_d001 = 0.0;
          {
            for (int ic0 = 0; ic0 < 2; ++ic0)
            {
              for (int ic1 = 0; ic1 < 2; ++ic1)
              {
                for (int ic2 = 0; ic2 < 2; ++ic2)
                {
                  w0_d100 += w[4 * ic0 + 2 * ic1 + ic2]
                             * (FE_TF0[0][0][iq0][ic0] * FE_TF1[0][0][iq1][ic1]
                                * FE_TF1[0][0][iq2][ic2]);
                }
              }
              for (int ic1 = 0; ic1 < 2; ++ic1)
              {
                for (int ic2 = 0; ic2 < 2; ++ic2)
                {
                  w0_d010 += w[4 * ic0 + 2 * ic1 + ic2]
                             * (FE_TF1[0][0][iq0][ic0] * FE_TF0[0][0][iq1][ic1]
                                * FE_TF1[0][0][iq2][ic2]);
                }
              }
              for (int ic1 = 0; ic1 < 2; ++ic1)
              {
                for (int ic2 = 0; ic2 < 2; ++ic2)
                {
                  w0_d001 += w[4 * ic0 + 2 * ic1 + ic2]
                             * (FE_TF1[0][0][iq0][ic0] * FE_TF1[0][0][iq1][ic1]
                                * FE_TF0[0][0][iq2][ic2]);
                }
              }
            }
          }
          // ------------------------
          // ------------------------
          // Section: Intermediates
          // Inputs: J_c8, J_c2, J_c6, J_c3, J_c0, J_c4, J_c7, J_c1, J_c5, w0_d100, w0_d010, w0_d001
          // Outputs: fw0, fw1, fw2
          double fw0 = 0;
          double fw1 = 0;
          double fw2 = 0;
          {
            double sv_461_0 = J_c4 * J_c8;
            double sv_461_1 = J_c5 * J_c7;
            double sv_461_2 = -sv_461_1;
            double sv_461_3 = sv_461_0 + sv_461_2;
            double sv_461_4 = J_c0 * sv_461_3;
            double sv_461_5 = J_c3 * J_c8;
            double sv_461_6 = J_c5 * J_c6;
            double sv_461_7 = -sv_461_6;
            double sv_461_8 = sv_461_5 + sv_461_7;
            double sv_461_9 = -J_c1;
            double sv_461_10 = sv_461_8 * sv_461_9;
            double sv_461_11 = sv_461_4 + sv_461_10;
            double sv_461_12 = J_c3 * J_c7;
            double sv_461_13 = J_c4 * J_c6;
            double sv_461_14 = -sv_461_13;
            double sv_461_15 = sv_461_12 + sv_461_14;
            double sv_461_16 = J_c2 * sv_461_15;
            double sv_461_17 = sv_461_11 + sv_461_16;
            double sv_461_18 = sv_461_3 / sv_461_17;
            double sv_461_19 = -J_c8;
            double sv_461_20 = J_c3 * sv_461_19;
            double sv_461_21 = sv_461_6 + sv_461_20;
            double sv_461_22 = sv_461_21 / sv_461_17;
            double sv_461_23 = sv_461_15 / sv_461_17;
            double sv_461_24 = w0_d100 * sv_461_18;
            double sv_461_25 = w0_d010 * sv_461_22;
            double sv_461_26 = sv_461_24 + sv_461_25;
            double sv_461_27 = w0_d001 * sv_461_23;
            double sv_461_28 = sv_461_26 + sv_461_27;
            double sv_461_29 = sv_461_28 * sv_461_18;
            double sv_461_30 = sv_461_28 * sv_461_22;
            double sv_461_31 = sv_461_28 * sv_461_23;
            double sv_461_32 = J_c2 * J_c7;
            double sv_461_33 = J_c8 * sv_461_9;
            double sv_461_34 = sv_461_32 + sv_461_33;
            double sv_461_35 = sv_461_34 / sv_461_17;
            double sv_461_36 = J_c0 * J_c8;
            double sv_461_37 = -J_c2;
            double sv_461_38 = J_c6 * sv_461_37;
            double sv_461_39 = sv_461_36 + sv_461_38;
            double sv_461_40 = sv_461_39 / sv_461_17;
            double sv_461_41 = J_c1 * J_c6;
            double sv_461_42 = J_c0 * J_c7;
            double sv_461_43 = -sv_461_42;
            double sv_461_44 = sv_461_41 + sv_461_43;
            double sv_461_45 = sv_461_44 / sv_461_17;
            double sv_461_46 = w0_d100 * sv_461_35;
            double sv_461_47 = w0_d010 * sv_461_40;
            double sv_461_48 = sv_461_46 + sv_461_47;
            double sv_461_49 = w0_d001 * sv_461_45;
            double sv_461_50 = sv_461_48 + sv_461_49;
            double sv_461_51 = sv_461_50 * sv_461_35;
            double sv_461_52 = sv_461_50 * sv_461_40;
            double sv_461_53 = sv_461_50 * sv_461_45;
            double sv_461_54 = sv_461_51 + sv_461_29;
            double sv_461_55 = sv_461_52 + sv_461_30;
            double sv_461_56 = sv_461_31 + sv_461_53;
            double sv_461_57 = J_c1 * J_c5;
            double sv_461_58 = J_c2 * J_c4;
            double sv_461_59 = -sv_461_58;
            double sv_461_60 = sv_461_57 + sv_461_59;
            double sv_461_61 = sv_461_60 / sv_461_17;
            double sv_461_62 = J_c2 * J_c3;
            double sv_461_63 = J_c0 * J_c5;
            double sv_461_64 = -sv_461_63;
            double sv_461_65 = sv_461_62 + sv_461_64;
            double sv_461_66 = sv_461_65 / sv_461_17;
            double sv_461_67 = J_c0 * J_c4;
            double sv_461_68 = J_c1 * J_c3;
            double sv_461_69 = -sv_461_68;
            double sv_461_70 = sv_461_67 + sv_461_69;
            double sv_461_71 = sv_461_70 / sv_461_17;
            double sv_461_72 = w0_d100 * sv_461_61;
            double sv_461_73 = w0_d010 * sv_461_66;
            double sv_461_74 = sv_461_72 + sv_461_73;
            double sv_461_75 = w0_d001 * sv_461_71;
            double sv_461_76 = sv_461_74 + sv_461_75;
            double sv_461_77 = sv_461_76 * sv_461_61;
            double sv_461_78 = sv_461_76 * sv_461_66;
            double sv_461_79 = sv_461_76 * sv_461_71;
            double sv_461_80 = sv_461_54 + sv_461_77;
            double sv_461_81 = sv_461_55 + sv_461_78;
            double sv_461_82 = sv_461_56 + sv_461_79;
            double sv_461_83 = c[0] * sv_461_80;
            double sv_461_84 = c[0] * sv_461_81;
            double sv_461_85 = c[0] * sv_461_82;
            double sv_461_86 = fabs(sv_461_17);
            double sv_461_87 = sv_461_83 * sv_461_86;
            double sv_461_88 = sv_461_84 * sv_461_86;
            double sv_461_89 = sv_461_85 * sv_461_86;
            fw0 = sv_461_87 * weights_461[9 * iq0 + 3 * iq1 + iq2];
            fw1 = sv_461_88 * weights_461[9 * iq0 + 3 * iq1 + iq2];
            fw2 = sv_461_89 * weights_461[9 * iq0 + 3 * iq1 + iq2];
          }
          // ------------------------
          // ------------------------
          // Section: Tensor Computation
          // Inputs: fw0, fw1, fw2, FE_TF0, FE_TF1
          // Outputs: A
          {
            for (int i0 = 0; i0 < 2; ++i0)
            {
              for (int i1 = 0; i1 < 2; ++i1)
              {
                for (int i2 = 0; i2 < 2; ++i2)
                {
                  A[(4 * i0 + 2 * i1 + i2)]
                      += fw0
                         * (FE_TF0[0][0][iq0][i0] * FE_TF1[0][0][iq1][i1] * FE_TF1[0][0][iq2][i2]);
                  A[(4 * i0 + 2 * i1 + i2)]
                      += fw1
                         * (FE_TF1[0][0][iq0][i0] * FE_TF0[0][0][iq1][i1] * FE_TF1[0][0][iq2][i2]);
                  A[(4 * i0 + 2 * i1 + i2)]
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
    for (int i = 0; i < space_dim; ++i)
      atomicAdd(&Aglobal[dofmap[id * space_dim + i]], A[i]);
  }
}

template <typename T>
__global__ void tabulate_tensor_Q2(int N, T* Aglobal, const T* wglobal, const T* c,
                                   const T* coordinate_dofs_global, const std::int32_t* geom_dofmap,
                                   const std::int32_t* dofmap)
{
  // Calculate the row index for this thread.
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  // Check if the row index is out of bounds.
  if (id < N)
  {
    // Extract w from wglobal
    const int space_dim = 27;
    double w[space_dim];
    for (int i = 0; i < space_dim; ++i)
      w[i] = wglobal[dofmap[id * space_dim + i]];

    double coordinate_dofs[24];
    for (int i = 0; i < 8; ++i)
    {
      coordinate_dofs[i * 3] = coordinate_dofs_global[geom_dofmap[id * 8 + i] * 3];
      coordinate_dofs[i * 3 + 1] = coordinate_dofs_global[geom_dofmap[id * 8 + i] * 3 + 1];
      coordinate_dofs[i * 3 + 2] = coordinate_dofs_global[geom_dofmap[id * 8 + i] * 3 + 2];
    }

    // Quadrature rules
    static const double weights_be1[64]
        = {0.005261434686316431, 0.009863939474383817, 0.009863939474383819, 0.00526143468631643,
           0.009863939474383817, 0.01849254200709766,  0.01849254200709766,  0.009863939474383814,
           0.009863939474383819, 0.01849254200709766,  0.01849254200709766,  0.009863939474383816,
           0.00526143468631643,  0.009863939474383814, 0.009863939474383816, 0.005261434686316428,
           0.009863939474383817, 0.01849254200709766,  0.01849254200709766,  0.009863939474383814,
           0.01849254200709766,  0.03466912086923912,  0.03466912086923912,  0.01849254200709765,
           0.01849254200709766,  0.03466912086923912,  0.03466912086923913,  0.01849254200709766,
           0.009863939474383814, 0.01849254200709765,  0.01849254200709766,  0.00986393947438381,
           0.009863939474383819, 0.01849254200709766,  0.01849254200709766,  0.009863939474383816,
           0.01849254200709766,  0.03466912086923912,  0.03466912086923913,  0.01849254200709766,
           0.01849254200709766,  0.03466912086923912,  0.03466912086923913,  0.01849254200709766,
           0.009863939474383817, 0.01849254200709766,  0.01849254200709766,  0.009863939474383814,
           0.00526143468631643,  0.009863939474383814, 0.009863939474383816, 0.005261434686316428,
           0.009863939474383814, 0.01849254200709765,  0.01849254200709766,  0.00986393947438381,
           0.009863939474383817, 0.01849254200709766,  0.01849254200709766,  0.009863939474383814,
           0.005261434686316428, 0.00986393947438381,  0.009863939474383812, 0.005261434686316426};
    // Precomputed values of basis functions and precomputations
    // FE* dimensions: [permutation][entities][points][dofs]
    static const double FE_TF0[1][1][4][3]
        = {{{{-2.722272623188105, -0.7222726231881051, 3.44454524637621},
             {-1.679962087169713, 0.3200379128302875, 1.359924174339425},
             {-0.3200379128302875, 1.679962087169713, -1.359924174339425},
             {0.7222726231881049, 2.722272623188105, -3.444545246376209}}}};
    static const double FE_TF1[1][1][4][3]
        = {{{{0.8013460293699309, -0.05979028222412167, 0.2584442528541908},
             {0.227784076790952, -0.1121969667939042, 0.884412890002952},
             {-0.1121969667939043, 0.2277840767909521, 0.884412890002952},
             {-0.05979028222412186, 0.8013460293699308, 0.258444252854191}}}};
    static const double FE_TF2[1][1][4][2] = {{{{0.9305681557970263, 0.06943184420297366},
                                                {0.6699905217924281, 0.3300094782075719},
                                                {0.3300094782075719, 0.6699905217924281},
                                                {0.06943184420297371, 0.9305681557970262}}}};
    static const double FE_TF3[1][1][4][2]
        = {{{{-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}}}};

    double A[space_dim];
    for (int i = 0; i < space_dim; ++i)
      A[i] = 0.0;
    for (int iq0 = 0; iq0 < 4; ++iq0)
    {
      for (int iq1 = 0; iq1 < 4; ++iq1)
      {
        for (int iq2 = 0; iq2 < 4; ++iq2)
        {
          // ------------------------
          // Section: Jacobian
          // Inputs: FE_TF2, coordinate_dofs, FE_TF3
          // Outputs: J_c5, J_c3, J_c7, J_c0, J_c6, J_c4, J_c8, J_c1, J_c2
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
                  J_c4 += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3 + 1]
                          * (FE_TF2[0][0][iq0][ic0] * FE_TF3[0][0][iq1][ic1]
                             * FE_TF2[0][0][iq2][ic2]);
                }
              }
              for (int ic1 = 0; ic1 < 2; ++ic1)
              {
                for (int ic2 = 0; ic2 < 2; ++ic2)
                {
                  J_c8 += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3 + 2]
                          * (FE_TF2[0][0][iq0][ic0] * FE_TF2[0][0][iq1][ic1]
                             * FE_TF3[0][0][iq2][ic2]);
                }
              }
              for (int ic1 = 0; ic1 < 2; ++ic1)
              {
                for (int ic2 = 0; ic2 < 2; ++ic2)
                {
                  J_c5 += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3 + 1]
                          * (FE_TF2[0][0][iq0][ic0] * FE_TF2[0][0][iq1][ic1]
                             * FE_TF3[0][0][iq2][ic2]);
                }
              }
              for (int ic1 = 0; ic1 < 2; ++ic1)
              {
                for (int ic2 = 0; ic2 < 2; ++ic2)
                {
                  J_c7 += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3 + 2]
                          * (FE_TF2[0][0][iq0][ic0] * FE_TF3[0][0][iq1][ic1]
                             * FE_TF2[0][0][iq2][ic2]);
                }
              }
              for (int ic1 = 0; ic1 < 2; ++ic1)
              {
                for (int ic2 = 0; ic2 < 2; ++ic2)
                {
                  J_c0 += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3]
                          * (FE_TF3[0][0][iq0][ic0] * FE_TF2[0][0][iq1][ic1]
                             * FE_TF2[0][0][iq2][ic2]);
                }
              }
              for (int ic1 = 0; ic1 < 2; ++ic1)
              {
                for (int ic2 = 0; ic2 < 2; ++ic2)
                {
                  J_c3 += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3 + 1]
                          * (FE_TF3[0][0][iq0][ic0] * FE_TF2[0][0][iq1][ic1]
                             * FE_TF2[0][0][iq2][ic2]);
                }
              }
              for (int ic1 = 0; ic1 < 2; ++ic1)
              {
                for (int ic2 = 0; ic2 < 2; ++ic2)
                {
                  J_c6 += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3 + 2]
                          * (FE_TF3[0][0][iq0][ic0] * FE_TF2[0][0][iq1][ic1]
                             * FE_TF2[0][0][iq2][ic2]);
                }
              }
              for (int ic1 = 0; ic1 < 2; ++ic1)
              {
                for (int ic2 = 0; ic2 < 2; ++ic2)
                {
                  J_c1 += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3]
                          * (FE_TF2[0][0][iq0][ic0] * FE_TF3[0][0][iq1][ic1]
                             * FE_TF2[0][0][iq2][ic2]);
                }
              }
              for (int ic1 = 0; ic1 < 2; ++ic1)
              {
                for (int ic2 = 0; ic2 < 2; ++ic2)
                {
                  J_c2 += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3]
                          * (FE_TF2[0][0][iq0][ic0] * FE_TF2[0][0][iq1][ic1]
                             * FE_TF3[0][0][iq2][ic2]);
                }
              }
            }
          }
          // ------------------------
          // ------------------------
          // Section: Coefficient
          // Inputs: FE_TF1, FE_TF0, w
          // Outputs: w0_d001, w0_d100, w0_d010
          double w0_d100 = 0.0;
          double w0_d010 = 0.0;
          double w0_d001 = 0.0;
          {
            for (int ic0 = 0; ic0 < 3; ++ic0)
            {
              for (int ic1 = 0; ic1 < 3; ++ic1)
              {
                for (int ic2 = 0; ic2 < 3; ++ic2)
                {
                  w0_d100 += w[9 * ic0 + 3 * ic1 + ic2]
                             * (FE_TF0[0][0][iq0][ic0] * FE_TF1[0][0][iq1][ic1]
                                * FE_TF1[0][0][iq2][ic2]);
                }
              }
              for (int ic1 = 0; ic1 < 3; ++ic1)
              {
                for (int ic2 = 0; ic2 < 3; ++ic2)
                {
                  w0_d010 += w[9 * ic0 + 3 * ic1 + ic2]
                             * (FE_TF1[0][0][iq0][ic0] * FE_TF0[0][0][iq1][ic1]
                                * FE_TF1[0][0][iq2][ic2]);
                }
              }
              for (int ic1 = 0; ic1 < 3; ++ic1)
              {
                for (int ic2 = 0; ic2 < 3; ++ic2)
                {
                  w0_d001 += w[9 * ic0 + 3 * ic1 + ic2]
                             * (FE_TF1[0][0][iq0][ic0] * FE_TF1[0][0][iq1][ic1]
                                * FE_TF0[0][0][iq2][ic2]);
                }
              }
            }
          }
          // ------------------------
          // ------------------------
          // Section: Intermediates
          // Inputs: J_c5, J_c7, J_c0, J_c3, J_c6, J_c4, J_c8, J_c1, J_c2, w0_d001, w0_d100, w0_d010
          // Outputs: fw0, fw1, fw2
          double fw0 = 0;
          double fw1 = 0;
          double fw2 = 0;
          {
            double sv_be1_0 = J_c4 * J_c8;
            double sv_be1_1 = J_c5 * J_c7;
            double sv_be1_2 = -sv_be1_1;
            double sv_be1_3 = sv_be1_0 + sv_be1_2;
            double sv_be1_4 = J_c0 * sv_be1_3;
            double sv_be1_5 = J_c3 * J_c8;
            double sv_be1_6 = J_c5 * J_c6;
            double sv_be1_7 = -sv_be1_6;
            double sv_be1_8 = sv_be1_5 + sv_be1_7;
            double sv_be1_9 = -J_c1;
            double sv_be1_10 = sv_be1_8 * sv_be1_9;
            double sv_be1_11 = sv_be1_4 + sv_be1_10;
            double sv_be1_12 = J_c3 * J_c7;
            double sv_be1_13 = J_c4 * J_c6;
            double sv_be1_14 = -sv_be1_13;
            double sv_be1_15 = sv_be1_12 + sv_be1_14;
            double sv_be1_16 = J_c2 * sv_be1_15;
            double sv_be1_17 = sv_be1_11 + sv_be1_16;
            double sv_be1_18 = sv_be1_3 / sv_be1_17;
            double sv_be1_19 = -J_c8;
            double sv_be1_20 = J_c3 * sv_be1_19;
            double sv_be1_21 = sv_be1_6 + sv_be1_20;
            double sv_be1_22 = sv_be1_21 / sv_be1_17;
            double sv_be1_23 = sv_be1_15 / sv_be1_17;
            double sv_be1_24 = w0_d100 * sv_be1_18;
            double sv_be1_25 = w0_d010 * sv_be1_22;
            double sv_be1_26 = sv_be1_24 + sv_be1_25;
            double sv_be1_27 = w0_d001 * sv_be1_23;
            double sv_be1_28 = sv_be1_26 + sv_be1_27;
            double sv_be1_29 = sv_be1_28 * sv_be1_18;
            double sv_be1_30 = sv_be1_28 * sv_be1_22;
            double sv_be1_31 = sv_be1_28 * sv_be1_23;
            double sv_be1_32 = J_c2 * J_c7;
            double sv_be1_33 = J_c8 * sv_be1_9;
            double sv_be1_34 = sv_be1_32 + sv_be1_33;
            double sv_be1_35 = sv_be1_34 / sv_be1_17;
            double sv_be1_36 = J_c0 * J_c8;
            double sv_be1_37 = -J_c2;
            double sv_be1_38 = J_c6 * sv_be1_37;
            double sv_be1_39 = sv_be1_36 + sv_be1_38;
            double sv_be1_40 = sv_be1_39 / sv_be1_17;
            double sv_be1_41 = J_c1 * J_c6;
            double sv_be1_42 = J_c0 * J_c7;
            double sv_be1_43 = -sv_be1_42;
            double sv_be1_44 = sv_be1_41 + sv_be1_43;
            double sv_be1_45 = sv_be1_44 / sv_be1_17;
            double sv_be1_46 = w0_d100 * sv_be1_35;
            double sv_be1_47 = w0_d010 * sv_be1_40;
            double sv_be1_48 = sv_be1_46 + sv_be1_47;
            double sv_be1_49 = w0_d001 * sv_be1_45;
            double sv_be1_50 = sv_be1_48 + sv_be1_49;
            double sv_be1_51 = sv_be1_50 * sv_be1_35;
            double sv_be1_52 = sv_be1_50 * sv_be1_40;
            double sv_be1_53 = sv_be1_50 * sv_be1_45;
            double sv_be1_54 = sv_be1_51 + sv_be1_29;
            double sv_be1_55 = sv_be1_52 + sv_be1_30;
            double sv_be1_56 = sv_be1_31 + sv_be1_53;
            double sv_be1_57 = J_c1 * J_c5;
            double sv_be1_58 = J_c2 * J_c4;
            double sv_be1_59 = -sv_be1_58;
            double sv_be1_60 = sv_be1_57 + sv_be1_59;
            double sv_be1_61 = sv_be1_60 / sv_be1_17;
            double sv_be1_62 = J_c2 * J_c3;
            double sv_be1_63 = J_c0 * J_c5;
            double sv_be1_64 = -sv_be1_63;
            double sv_be1_65 = sv_be1_62 + sv_be1_64;
            double sv_be1_66 = sv_be1_65 / sv_be1_17;
            double sv_be1_67 = J_c0 * J_c4;
            double sv_be1_68 = J_c1 * J_c3;
            double sv_be1_69 = -sv_be1_68;
            double sv_be1_70 = sv_be1_67 + sv_be1_69;
            double sv_be1_71 = sv_be1_70 / sv_be1_17;
            double sv_be1_72 = w0_d100 * sv_be1_61;
            double sv_be1_73 = w0_d010 * sv_be1_66;
            double sv_be1_74 = sv_be1_72 + sv_be1_73;
            double sv_be1_75 = w0_d001 * sv_be1_71;
            double sv_be1_76 = sv_be1_74 + sv_be1_75;
            double sv_be1_77 = sv_be1_76 * sv_be1_61;
            double sv_be1_78 = sv_be1_76 * sv_be1_66;
            double sv_be1_79 = sv_be1_76 * sv_be1_71;
            double sv_be1_80 = sv_be1_54 + sv_be1_77;
            double sv_be1_81 = sv_be1_55 + sv_be1_78;
            double sv_be1_82 = sv_be1_56 + sv_be1_79;
            double sv_be1_83 = c[0] * sv_be1_80;
            double sv_be1_84 = c[0] * sv_be1_81;
            double sv_be1_85 = c[0] * sv_be1_82;
            double sv_be1_86 = fabs(sv_be1_17);
            double sv_be1_87 = sv_be1_83 * sv_be1_86;
            double sv_be1_88 = sv_be1_84 * sv_be1_86;
            double sv_be1_89 = sv_be1_85 * sv_be1_86;
            fw0 = sv_be1_87 * weights_be1[16 * iq0 + 4 * iq1 + iq2];
            fw1 = sv_be1_88 * weights_be1[16 * iq0 + 4 * iq1 + iq2];
            fw2 = sv_be1_89 * weights_be1[16 * iq0 + 4 * iq1 + iq2];
          }
          // ------------------------
          // ------------------------
          // Section: Tensor Computation
          // Inputs: FE_TF1, fw1, FE_TF0, fw0, fw2
          // Outputs: A
          {
            for (int i0 = 0; i0 < 3; ++i0)
            {
              for (int i1 = 0; i1 < 3; ++i1)
              {
                for (int i2 = 0; i2 < 3; ++i2)
                {
                  A[(9 * i0 + 3 * i1 + i2)]
                      += fw0
                         * (FE_TF0[0][0][iq0][i0] * FE_TF1[0][0][iq1][i1] * FE_TF1[0][0][iq2][i2]);
                  A[(9 * i0 + 3 * i1 + i2)]
                      += fw1
                         * (FE_TF1[0][0][iq0][i0] * FE_TF0[0][0][iq1][i1] * FE_TF1[0][0][iq2][i2]);
                  A[(9 * i0 + 3 * i1 + i2)]
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
    for (int i = 0; i < space_dim; ++i)
      atomicAdd(&Aglobal[dofmap[id * space_dim + i]], A[i]);
  }
}

template <typename T>
__global__ void tabulate_tensor_Q3(int N, T* Aglobal, const T* wglobal, const T* c,
                                   const T* coordinate_dofs_global, const std::int32_t* geom_dofmap,
                                   const std::int32_t* dofmap)
{
  // Calculate the row index for this thread.
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  // Check if the row index is out of bounds.
  if (id < N)
  {
    // Extract w from wglobal
    const int space_dim = 64;
    double w[space_dim];
    for (int i = 0; i < space_dim; ++i)
      w[i] = wglobal[dofmap[id * space_dim + i]];

    double coordinate_dofs[24];
    for (int i = 0; i < 8; ++i)
    {
      coordinate_dofs[i * 3] = coordinate_dofs_global[geom_dofmap[id * 8 + i] * 3];
      coordinate_dofs[i * 3 + 1] = coordinate_dofs_global[geom_dofmap[id * 8 + i] * 3 + 1];
      coordinate_dofs[i * 3 + 2] = coordinate_dofs_global[geom_dofmap[id * 8 + i] * 3 + 2];
    }

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
    double A[space_dim];
    for (int i = 0; i < space_dim; ++i)
      A[i] = 0.0;
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
                  J_c4 += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3 + 1]
                          * (FE_TF2[0][0][iq0][ic0] * FE_TF3[0][0][iq1][ic1]
                             * FE_TF2[0][0][iq2][ic2]);
                }
              }
              for (int ic1 = 0; ic1 < 2; ++ic1)
              {
                for (int ic2 = 0; ic2 < 2; ++ic2)
                {
                  J_c8 += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3 + 2]
                          * (FE_TF2[0][0][iq0][ic0] * FE_TF2[0][0][iq1][ic1]
                             * FE_TF3[0][0][iq2][ic2]);
                }
              }
              for (int ic1 = 0; ic1 < 2; ++ic1)
              {
                for (int ic2 = 0; ic2 < 2; ++ic2)
                {
                  J_c5 += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3 + 1]
                          * (FE_TF2[0][0][iq0][ic0] * FE_TF2[0][0][iq1][ic1]
                             * FE_TF3[0][0][iq2][ic2]);
                }
              }
              for (int ic1 = 0; ic1 < 2; ++ic1)
              {
                for (int ic2 = 0; ic2 < 2; ++ic2)
                {
                  J_c7 += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3 + 2]
                          * (FE_TF2[0][0][iq0][ic0] * FE_TF3[0][0][iq1][ic1]
                             * FE_TF2[0][0][iq2][ic2]);
                }
              }
              for (int ic1 = 0; ic1 < 2; ++ic1)
              {
                for (int ic2 = 0; ic2 < 2; ++ic2)
                {
                  J_c0 += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3]
                          * (FE_TF3[0][0][iq0][ic0] * FE_TF2[0][0][iq1][ic1]
                             * FE_TF2[0][0][iq2][ic2]);
                }
              }
              for (int ic1 = 0; ic1 < 2; ++ic1)
              {
                for (int ic2 = 0; ic2 < 2; ++ic2)
                {
                  J_c3 += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3 + 1]
                          * (FE_TF3[0][0][iq0][ic0] * FE_TF2[0][0][iq1][ic1]
                             * FE_TF2[0][0][iq2][ic2]);
                }
              }
              for (int ic1 = 0; ic1 < 2; ++ic1)
              {
                for (int ic2 = 0; ic2 < 2; ++ic2)
                {
                  J_c6 += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3 + 2]
                          * (FE_TF3[0][0][iq0][ic0] * FE_TF2[0][0][iq1][ic1]
                             * FE_TF2[0][0][iq2][ic2]);
                }
              }
              for (int ic1 = 0; ic1 < 2; ++ic1)
              {
                for (int ic2 = 0; ic2 < 2; ++ic2)
                {
                  J_c1 += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3]
                          * (FE_TF2[0][0][iq0][ic0] * FE_TF3[0][0][iq1][ic1]
                             * FE_TF2[0][0][iq2][ic2]);
                }
              }
              for (int ic1 = 0; ic1 < 2; ++ic1)
              {
                for (int ic2 = 0; ic2 < 2; ++ic2)
                {
                  J_c2 += coordinate_dofs[(4 * ic0 + 2 * ic1 + ic2) * 3]
                          * (FE_TF2[0][0][iq0][ic0] * FE_TF2[0][0][iq1][ic1]
                             * FE_TF3[0][0][iq2][ic2]);
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
                  w0_d100 += w[16 * ic0 + 4 * ic1 + ic2]
                             * (FE_TF0[0][0][iq0][ic0] * FE_TF1[0][0][iq1][ic1]
                                * FE_TF1[0][0][iq2][ic2]);
                }
              }
              for (int ic1 = 0; ic1 < 4; ++ic1)
              {
                for (int ic2 = 0; ic2 < 4; ++ic2)
                {
                  w0_d010 += w[16 * ic0 + 4 * ic1 + ic2]
                             * (FE_TF1[0][0][iq0][ic0] * FE_TF0[0][0][iq1][ic1]
                                * FE_TF1[0][0][iq2][ic2]);
                }
              }
              for (int ic1 = 0; ic1 < 4; ++ic1)
              {
                for (int ic2 = 0; ic2 < 4; ++ic2)
                {
                  w0_d001 += w[16 * ic0 + 4 * ic1 + ic2]
                             * (FE_TF1[0][0][iq0][ic0] * FE_TF1[0][0][iq1][ic1]
                                * FE_TF0[0][0][iq2][ic2]);
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
    for (int i = 0; i < space_dim; ++i)
      atomicAdd(&Aglobal[dofmap[id * space_dim + i]], A[i]);
  }
}
} // namespace

namespace dolfinx::acc
{
template <typename T>
class MatFreeLaplace
{
public:
  MatFreeLaplace(int degree, int num_cells, std::span<const T> constants, std::span<const T> x,
                 std::span<const std::int32_t> x_dofmap, std::span<const std::int32_t> dofmap)
      : num_cells(num_cells), constants(constants), x(x), x_dofmap(x_dofmap), dofmap(dofmap)
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
      // TODO throw error
      break;
    }
    // TODO Other degrees
  }

  ~MatFreeLaplace() {}

  template <typename Vector>
  void operator()(Vector& in, Vector& out)
  {
    T* wglobal = in.mutable_array().data();
    T* Aglobal = out.mutable_array().data();

    dim3 block_size(256);
    dim3 grid_size((num_cells + block_size.x - 1) / block_size.x);
    hipLaunchKernelGGL(tabulate_tensor, grid_size, block_size, 0, 0, num_cells, Aglobal, wglobal,
                       constants.data(), x.data(), x_dofmap.data(), dofmap.data());
    err_check(hipGetLastError());
  }

private:
  int num_cells;
  std::span<const T> constants;
  std::span<const T> x;
  std::span<const std::int32_t> x_dofmap;
  std::span<const std::int32_t> dofmap;
  void (*tabulate_tensor)(int, T*, const T*, const T*, const T*, const std::int32_t*,
                          const std::int32_t*);
};
} // namespace dolfinx::acc
