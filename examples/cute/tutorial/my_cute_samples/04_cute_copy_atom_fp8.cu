#include <cute/tensor.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <stdio.h>

using namespace cute;

template <typename TC, typename TA, typename TB,
          int kTileM, int kTileN, int kTileK,
          typename TiledMMA,
          typename TiledCopyA, typename TiledCopyB, typename TiledCopyC>
__global__ void gemm_kernel(TC *Cptr, TA *Aptr, TB *Bptr, int m, int n, int k) {
    auto mA = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
    auto mB = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
    auto mC = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

    auto tiler = make_tile(Int<kTileM>{}, Int<kTileN>{}, Int<kTileK>{});
    auto coord = make_coord(0, 0, 0);

    Tensor gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{}); // (kTileM, kTileK)
    Tensor gB = local_tile(mB, tiler, coord, Step<X, _1, _1>{}); // (kTileN, kTileK)
    Tensor gC = local_tile(mC, tiler, coord, Step<_1, _1, X>{}); // (kTileM, kTileN)

    __shared__ TA smem_A[kTileM * kTileK];
    __shared__ TB smem_B[kTileN * kTileK];

    int tid = threadIdx.x;

    auto sA = make_tensor(make_smem_ptr(smem_A), make_shape(m, k), make_stride(k, Int<1>{}));
    auto sB = make_tensor(make_gmem_ptr(smem_B), make_shape(n, k), make_stride(k, Int<1>{}));

#if 1
    // before G2S
    if (thread0()) {
        print(" sA:"); print_tensor(sA); print("\n");
        print(" sB:"); print_tensor(sB); print("\n");
    }
#endif

    // G2S
    copy(gA, sA);
    copy(gB, sB);

    // end G2S
    if (thread0()) {
        print(" sA:"); print_tensor(sA); print("\n");
        print(" sB:"); print_tensor(sB); print("\n");
    }

#if 0
    if (thread0()) {
        print(" mA:"); print(mA); print("\n");
        print(" mB:"); print(mB); print("\n");
        print(" mC:"); print(mC); print("\n");
        print(mA(9,0)); print("\n");
        print(mB(9,0)); print("\n");
    }
#endif

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);
    auto tAgA = thr_mma.partition_A(gA);
    auto tBgB = thr_mma.partition_B(gB);
    auto tCgC = thr_mma.partition_C(gC);

#if 0
    if (thread0()) {
        print(" tAgA:"); print(tAgA); print("\n");
        print(" tBgB:"); print(tBgB); print("\n");
        print(" tCgC:"); print(tCgC); print("\n");
    }
#endif

    auto tArA = thr_mma.partition_fragment_A(gA);
    auto tBrB = thr_mma.partition_fragment_B(gB);
    auto tCrC = thr_mma.partition_fragment_C(gC);
    clear(tCrC);

    TiledCopyA s2r_tiled_copy_a;
    ThrCopy s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(tid);
    Tensor tAsA_s2r = s2r_thr_copy_a.partition_S(sA); // (CPY, CPY_M, CPY_K)
    Tensor tArA_s2r = s2r_thr_copy_a.retile_D(tArA);  // (CPY, CPY_M, CPY_K)

    TiledCopyB s2r_tiled_copy_b;
    ThrCopy s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(tid);
    Tensor tBsB_s2r = s2r_thr_copy_b.partition_S(sB); // (CPY, CPY_M, CPY_K)
    Tensor tBrB_s2r = s2r_thr_copy_b.retile_D(tBrB);  // (CPY, CPY_M, CPY_K)

#if 0
    if (thread0()) {
        print(" tArA:"); print(tArA); print("\n");
        print(" tBrB:"); print(tBrB); print("\n");
        print(" tCrC:"); print(tCrC); print("\n");
    }
#endif

    int num_tile_k = size<1>(gA) / kTileK;

    if (thread0()) {
        printf(" num_tile_k: %d\n", num_tile_k);
    }

#pragma unroll 1
    for (int itile = 0; itile < num_tile_k; ++itile) {
        copy(s2r_tiled_copy_a, tAsA_s2r, tArA_s2r);
        copy(s2r_tiled_copy_b, tBsB_s2r, tBrB_s2r);

#if 0
        if (thread0()) {
            for (int i = 0; i < size(tArA); ++i) {
                auto element = tArA(i);
                print(element); print("\n");
            }
        }
#endif

        gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
    }

    copy(tCrC, tCgC);
}

/*
 * Just use one tensor core(SM89_16x8x32_F32E4M3E4M3F32_TN) calculate mma,
 * and use ldmatrix move data from shared memory to registers.
 */

int main(int argc, char** argv) {
    int m = 16;
    if (argc >= 2)
        sscanf(argv[1], "%d", &m);

    int n = 8;
    if (argc >= 3)
        sscanf(argv[2], "%d", &n);

    int k = 32;
    if (argc >= 4)
        sscanf(argv[3], "%d", &k);

    if (m != 16 || n != 8 || k != 32) {
        printf("This case only show the example of the M=16, N=8, K=16 shape, please use the correct inputs\n");
        return -1;
    }

    using TA = cute::float_e4m3_t;
    using TB = cute::float_e4m3_t;
    using TC = float;

    printf("m: %d, n: %d, k: %d\n", m, n, k);

    thrust::host_vector<TA> h_A(m*k);
    thrust::host_vector<TB> h_B(n*k);
    thrust::host_vector<TC> h_C(m*n);

    srand(42);
    for (int j = 0; j < m*k; ++j) h_A[j] = __nv_fp8_e4m3( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < n*k; ++j) h_B[j] = __nv_fp8_e4m3( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<TC>( 2*(rand() / double(RAND_MAX)) - 1 );

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;

    using mma_op      = SM89_16x8x32_F32E4M3E4M3F32_TN;
    using mma_traits  = MMA_Traits<mma_op>;
    using mma_atom    = MMA_Atom<mma_traits>;
    using TiledMMA    = decltype(make_tiled_mma(mma_atom{}));

    using copy_g2s_op      = UniversalCopy<cute::uint64_t>;
    using copy_g2s_traits  = Copy_Traits<copy_g2s_op>;
    using CopyA_g2s_atom   = Copy_Atom<copy_g2s_traits, TA>;
    using CopyB_g2s_atom   = Copy_Atom<copy_g2s_traits, TB>;

    using copy_s2r_op_m8n8_1 = SM75_U32x1_LDSM_N;
    using copy_s2r_op_m8n8_2 = SM75_U32x2_LDSM_N;
    using copy_s2r_op_m8n8_4 = SM75_U32x4_LDSM_N;
    // using CopyA_s2r_atom   = Copy_Atom<copy_s2r_op_m8n8_1, TA>;
    // using CopyA_s2r_atom   = Copy_Atom<copy_s2r_op_m8n8_2, TA>;
    using CopyA_s2r_atom   = Copy_Atom<copy_s2r_op_m8n8_4, TA>;
    using CopyB_s2r_atom   = Copy_Atom<copy_s2r_op_m8n8_1, TB>;
    using CopyC_s2r_atom   = Copy_Atom<copy_s2r_op_m8n8_1, TC>;

    using TiledCopyA = decltype(make_tiled_copy_A(CopyA_s2r_atom{}, TiledMMA{}));
    using TiledCopyB = decltype(make_tiled_copy_B(CopyB_s2r_atom{}, TiledMMA{}));
    using TiledCopyC = decltype(make_tiled_copy_C(CopyC_s2r_atom{}, TiledMMA{}));

    dim3 block(32);
    dim3 grid(1);

    printf("========================grid: (%d, %d, %d), block: (%d, %d, %d)========================\n",
           grid.x, grid.y, grid.z,
           block.x, block.y, block.z);

    gemm_kernel<TC, TA, TB, 16, 8, 32, TiledMMA, TiledCopyA, TiledCopyB, TiledCopyC>
        <<<grid, block>>>(d_C.data().get(), d_A.data().get(), d_B.data().get(), m, n, k);

    CUTE_CHECK_LAST();

    thrust::host_vector<TC> cute_result = d_C;

    return 0;
}