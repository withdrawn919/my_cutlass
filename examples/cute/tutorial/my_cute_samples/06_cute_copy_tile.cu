#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>
#include <cstdint>

using namespace cute;

template <typename TC, typename TA, typename TB,
          int kTileM, int kTileN, int kTileK,
          typename TiledMMA,
          typename TiledCopyA_S2R, typename TiledCopyB_S2R,
          typename TiledCopyC_R2S,
          typename TiledCopyA_G2S, typename TiledCopyB_G2S>
__global__ void gemm_kernel(TC *Cptr, TA *Aptr, TB *Bptr, int m, int n, int k) {
#if 0
    __shared__ TA smem_A[kTileM * kTileK];
    __shared__ TB smem_B[kTileN * kTileK];
    __shared__ TC smem_C[kTileM * kTileN];
#else
    extern __shared__ char smem[];

    size_t offset_A = 0;
    TA* smem_A = reinterpret_cast<TA*>(&smem[offset_A]);

    size_t offset_B = offset_A + sizeof(TA) * kTileM * kTileK;
    TB* smem_B = reinterpret_cast<TB*>(&smem[offset_B]);

    size_t offset_C = offset_B + sizeof(TB) * kTileN * kTileK;
    TC* smem_C = reinterpret_cast<TC*>(&smem[offset_C]);
#endif

    int tid = threadIdx.x;

    Tensor mA = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor mB = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor mC = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

    Tensor sA = make_tensor(make_smem_ptr(smem_A), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor sB = make_tensor(make_smem_ptr(smem_B), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor sC = make_tensor(make_smem_ptr(smem_C), make_shape(m, n), make_stride(n, Int<1>{}));

    auto tiler = make_tile(Int<kTileM>{}, Int<kTileN>{}, Int<kTileK>{});
    auto coord = make_coord(0, 0, 0);

    Tensor gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{}); // (kTileM, kTileK)
    Tensor gB = local_tile(mB, tiler, coord, Step<X, _1, _1>{}); // (kTileN, kTileK)
    Tensor gC = local_tile(mC, tiler, coord, Step<_1, _1, X>{}); // (kTileM, kTileN)

#if 0
    copy(gA, sA);
    copy(gB, sB);
    __syncthreads();
#else
    TiledCopyA_G2S g2s_tiled_copy_a;
    ThrCopy g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(tid);
    Tensor tAgA_g2s = g2s_thr_copy_a.partition_S(gA); // (CPY, CPY_M, CPY_K)
    Tensor tAsA_g2s = g2s_thr_copy_a.partition_D(sA); // (CPY, CPY_M, CPY_K)

    TiledCopyB_G2S g2s_tiled_copy_b;
    ThrCopy g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(tid);
    Tensor tBgB_g2s = g2s_thr_copy_b.partition_S(gB); // (CPY, CPY_N, CPY_K)
    Tensor tBsB_g2s = g2s_thr_copy_b.partition_D(sB); // (CPY, CPY_N, CPY_K)

    copy(g2s_tiled_copy_a, tAgA_g2s, tAsA_g2s);
    copy(g2s_tiled_copy_b, tBgB_g2s, tBsB_g2s);

    cp_async_fence();              // Label the end of (potential) cp.async instructions
    cp_async_wait<0>();             // Sync on all (potential) cp.async instructions
    __syncthreads();                // Wait for all threads to write to smem
#endif

#if 0
    if (thread0()) {
        print("mA:"); print_tensor(mA);
        print("sA:"); print_tensor(sA);
        print("mB:"); print_tensor(mB);
        print("sB:"); print_tensor(sB);
    }
#endif

    TiledMMA tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_slice(tid);

    // thread-globalMem layout
    Tensor tCgA = thr_mma.partition_A(gA);
    Tensor tCgB = thr_mma.partition_B(gB);
    Tensor tCgC = thr_mma.partition_C(gC);

    // thread-regMem layout
    Tensor tCrA = thr_mma.partition_fragment_A(gA);
    Tensor tCrB = thr_mma.partition_fragment_B(gB);
    Tensor tCrC = thr_mma.partition_fragment_C(gC);

    // copy from sharedMem to regMem
    TiledCopyA_S2R s2r_tiled_copy_a;
    ThrCopy s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(tid);
    Tensor tAsA_s2r = s2r_thr_copy_a.partition_S(sA); // (CPY, CPY_M, CPY_K)
    Tensor tArA_s2r = s2r_thr_copy_a.retile_D(tCrA);  // (CPY, CPY_M, CPY_K)

    TiledCopyB_S2R s2r_tiled_copy_b;
    ThrCopy s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(tid);
    Tensor tBsB_s2r = s2r_thr_copy_b.partition_S(sB); // (CPY, CPY_M, CPY_K)
    Tensor tBrB_s2r = s2r_thr_copy_b.retile_D(tCrB);  // (CPY, CPY_M, CPY_K)

    copy(s2r_tiled_copy_a, tAsA_s2r, tArA_s2r);
    copy(s2r_tiled_copy_b, tBsB_s2r, tBrB_s2r);

    // mma compute
    gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

    // copy from regMem to sharedMem
    TiledCopyC_R2S r2s_tiled_copy_c;
    ThrCopy r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(tid);
    Tensor tCrC_r2s = r2s_thr_copy_c.retile_S(tCrC); // (CPY, CPY_M, CPY_N)
    Tensor tCsC_r2s = r2s_thr_copy_c.partition_D(sC); // (CPY, CPY_M, CPY_N)
    copy(r2s_tiled_copy_c, tCrC_r2s, tCsC_r2s);

    __syncthreads();

    // copy from sharedMem to globalMem
    copy(sC, mC);

    if (thread0()) {
        print(" mC: "); print_tensor(mC);
    }
}

/*
 * Expand tile shape, use more tensor core(SM80_16x8x16_F32F16F16F32_TN) calculate mma,
 * and use cp.async to copy date from G2S.
 */

int main(int argc, char** argv) {
    int m = 32;
    if (argc >= 2)
        sscanf(argv[1], "%d", &m);

    int n = 32;
    if (argc >= 3)
        sscanf(argv[2], "%d", &n);

    int k = 16;
    if (argc >= 4)
        sscanf(argv[3], "%d", &k);

    printf("m: %d, n: %d, k: %d\n", m, n, k);

    using TA = cute::half_t;
    using TB = cute::half_t;
    using TC = float;

    thrust::host_vector<TA> h_A(m * k);
    thrust::host_vector<TB> h_B(n * k);
    thrust::host_vector<TC> h_C(m * n);

    srand(42);
    for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<TC>(-1);

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;

    using TiledMMA = decltype(make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                                             Layout<Shape<_2, _4, _1>>{},
                                             Tile<_32, _32, _16>{}));

    dim3 grid(1);
    dim3 block(size(TiledMMA{}));

    static constexpr int kShmSize = sizeof(TA) * 32 * 16
                                 + sizeof(TB) * 32 * 16
                                 + sizeof(TC) * 32 * 32;

    // get device max shared memory size
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("=== 设备 0 信息 ===\n");
    printf("设备名称: %s\n", deviceProp.name);
    printf("每个线程块(block)的静态共享内存大小: %zu 字节 ( %f KB)\n", deviceProp.sharedMemPerBlock, deviceProp.sharedMemPerBlock / 1024.0);
    printf("每个SM的最大共享内存配置大小 %zu 字节 ( %f KB)\n", deviceProp.sharedMemPerMultiprocessor, deviceProp.sharedMemPerMultiprocessor / 1024.0);

    if (deviceProp.sharedMemPerBlock < kShmSize) {
        printf("申请的共享内存(%d 字节)超过设备上限(%zu 字节)\n", kShmSize, deviceProp.sharedMemPerBlock);
        return -1;
    }

    using TiledCopyA_G2S = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, cute::half_t>{},
                                                   Layout<Shape<_32, _2>, Stride<_2, _1>>{},
                                                   Layout<Shape<_1, _8>>{}));
    using TiledCopyB_G2S = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, cute::half_t>{},
                                                   Layout<Shape<_32, _2>, Stride<_2, _1>>{},
                                                   Layout<Shape<_1, _8>>{}));

    using copy_op_ldmatirx_x1 = SM75_U32x1_LDSM_N;
    using copy_op_ldmatirx_x2 = SM75_U32x2_LDSM_N;
    using copy_op_ldmatirx_x4 = SM75_U32x4_LDSM_N;
    using copy_atom_A = Copy_Atom<copy_op_ldmatirx_x4, TA>;
    using copy_atom_B = Copy_Atom<copy_op_ldmatirx_x1, TB>;

    using TiledCopyA_S2R = decltype(make_tiled_copy_A(copy_atom_A{}, TiledMMA{}));
    using TiledCopyB_S2R = decltype(make_tiled_copy_B(copy_atom_B{}, TiledMMA{}));

    using Copy_R2S_op = AutoVectorizingCopy;
    using CopyC_R2S_atom = Copy_Atom<Copy_R2S_op, TC>;
    using TiledCopyC_R2S = decltype(make_tiled_copy_C(CopyC_R2S_atom{}, TiledMMA{}));

    printf("========================grid: (%d, %d, %d), block: (%d, %d, %d)========================\n",
           grid.x, grid.y, grid.z,
           block.x, block.y, block.z);

    gemm_kernel<TC, TA, TB, 32, 32, 16, TiledMMA, TiledCopyA_S2R, TiledCopyB_S2R, TiledCopyC_R2S, TiledCopyA_G2S, TiledCopyB_G2S>
        <<<grid, block, kShmSize>>>(d_C.data().get(), d_A.data().get(), d_B.data().get(), m, n, k);

    CUTE_CHECK_LAST();

    thrust::host_vector<TC> cute_result = d_C;

    return 0;
}