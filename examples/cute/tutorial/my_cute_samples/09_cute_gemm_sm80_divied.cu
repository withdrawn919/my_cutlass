#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include <cstdint>
#include <cmath>
#include <iostream>
#include <vector>

using namespace cute;

// Kernel: each block computes one (kTileM x kTileN) output tile.
// Loops over K-tiles and accumulates into register fragment tCrC.
template <typename TC, typename TA, typename TB,
          int kTileM, int kTileN, int kTileK,
          typename TiledMMA,
          typename TiledCopyA_G2S, typename TiledCopyB_G2S,
          typename TiledCopyA_S2R, typename TiledCopyB_S2R,
          typename TiledCopyC_R2G>
__global__ void gemm_kernel(TC *Cptr, TA *Aptr, TB *Bptr, int m, int n, int k) {

    extern __shared__ char smem[];

    TA* smem_A = (TA*)smem;
    TB* smem_B = (TB*)(smem + sizeof(TA) * kTileM * kTileK);

    int tid   = threadIdx.x;
    int tile_m = blockIdx.x;   // M-tile index for this block
    int tile_n = blockIdx.y;   // N-tile index for this block

    // Global memory tensors (full matrices)
    Tensor mA = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor mB = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor mC = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

    // Swizzled shared memory layouts (same as swizzle_copy.cu)
    using SmemLayoutAtomA = decltype(
        composition(Swizzle<3, 3, 3>{},
                    Layout<Shape<_128, _64>, Stride<_64, _1>>{}));
    using SmemLayoutAtomB = decltype(
        composition(Swizzle<3, 3, 3>{},
                    Layout<Shape<_128, _64>, Stride<_64, _1>>{}));

    Tensor sA = make_tensor(make_smem_ptr(smem_A), SmemLayoutAtomA{});
    Tensor sB = make_tensor(make_smem_ptr(smem_B), SmemLayoutAtomB{});

    auto tiler = make_tile(Int<kTileM>{}, Int<kTileN>{}, Int<kTileK>{});

    // Output tile: fixed for the lifetime of this block
    Tensor gC = local_tile(mC, tiler, make_coord(tile_m, tile_n, 0), Step<_1, _1, X>{});

    TiledMMA tiled_mma;
    ThrMMA   thr_mma = tiled_mma.get_slice(tid);

    Tensor tCgC = thr_mma.partition_C(gC);            // (MMA, MMA_M, MMA_N)
    Tensor tCrC = thr_mma.partition_fragment_C(gC);   // (MMA, MMA_M, MMA_N)
    clear(tCrC);  // zero-initialize accumulator

    // Copy objects (constructed once, reused across K-loop)
    TiledCopyA_G2S g2s_tiled_copy_a;
    TiledCopyB_G2S g2s_tiled_copy_b;
    TiledCopyA_S2R s2r_tiled_copy_a;
    TiledCopyB_S2R s2r_tiled_copy_b;

    ThrCopy g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(tid);
    ThrCopy g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(tid);
    ThrCopy s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(tid);
    ThrCopy s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(tid);

    // -------------------------------------------------------------------------
    // Main K-loop: iterate over K-tiles and accumulate
    // -------------------------------------------------------------------------
    int num_k_tiles = k / kTileK;

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        auto coord = make_coord(tile_m, tile_n, k_tile);

        // Global tiles for this K-slice
        Tensor gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{});  // (kTileM, kTileK)
        Tensor gB = local_tile(mB, tiler, coord, Step<X, _1, _1>{});  // (kTileN, kTileK)

        // ----- Global → Shared (async) -----
        Tensor tAgA_g2s = g2s_thr_copy_a.partition_S(gA);
        Tensor tAsA_g2s = g2s_thr_copy_a.partition_D(sA);
        Tensor tBgB_g2s = g2s_thr_copy_b.partition_S(gB);
        Tensor tBsB_g2s = g2s_thr_copy_b.partition_D(sB);

        copy(g2s_tiled_copy_a, tAgA_g2s, tAsA_g2s);
        copy(g2s_tiled_copy_b, tBgB_g2s, tBsB_g2s);

        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();   // all threads see completed smem writes

        // ----- Shared → Register -----
        Tensor tCrA = thr_mma.partition_fragment_A(gA);  // (MMA, MMA_M, MMA_K)
        Tensor tCrB = thr_mma.partition_fragment_B(gB);  // (MMA, MMA_N, MMA_K)

        Tensor tAgA_s2r = s2r_thr_copy_a.partition_S(sA);
        Tensor tArA_s2r = s2r_thr_copy_a.retile_D(tCrA);
        Tensor tBgB_s2r = s2r_thr_copy_b.partition_S(sB);
        Tensor tBrB_s2r = s2r_thr_copy_b.retile_D(tCrB);

        copy(s2r_tiled_copy_a, tAgA_s2r, tArA_s2r);
        copy(s2r_tiled_copy_b, tBgB_s2r, tBrB_s2r);

        // ----- MMA accumulate -----
        gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

        __syncthreads();   // ensure all threads done reading smem before next write
    }

    // -------------------------------------------------------------------------
    // Write accumulator back to global memory
    // -------------------------------------------------------------------------
    TiledCopyC_R2G r2g_tiled_copy_c;
    ThrCopy r2g_thr_copy_c = r2g_tiled_copy_c.get_slice(tid);
    Tensor tCrC_r2g = r2g_thr_copy_c.retile_S(tCrC);   // (CPY, CPY_M, CPY_N)
    Tensor tCgC_r2g = r2g_thr_copy_c.retile_D(tCgC);   // (CPY, CPY_M, CPY_N)
    copy(r2g_tiled_copy_c, tCrC_r2g, tCgC_r2g);
}

// -------------------------------------------------------------------------
// CPU reference: C[m,n] = sum_k A[m,k] * B[n,k]   (B is row-major, k-minor)
// -------------------------------------------------------------------------
void cpu_gemm_ref(const std::vector<float>& A, const std::vector<float>& B,
                  std::vector<float>& C, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.f;
            for (int l = 0; l < k; ++l)
                sum += A[i * k + l] * B[j * k + l];
            C[i * n + j] = sum;
        }
    }
}

int main(int argc, char** argv) {

    // Print device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Shared mem per block: " << prop.sharedMemPerBlock / 1024.0 << " KB\n\n";

    // Tile sizes (must divide M, N, K respectively)
    constexpr int kTileM = 128;
    constexpr int kTileN = 128;
    constexpr int kTileK = 64;

    // Matrix dimensions (must be multiples of tile sizes)
    int m = 1024;
    if (argc >= 2) sscanf(argv[1], "%d", &m);
    int n = 1024;
    if (argc >= 3) sscanf(argv[2], "%d", &n);
    int k = 256;
    if (argc >= 4) sscanf(argv[3], "%d", &k);

    // Validate divisibility
    if (m % kTileM != 0 || n % kTileN != 0 || k % kTileK != 0) {
        fprintf(stderr,
                "Error: m=%d n=%d k=%d must be divisible by tile sizes %d/%d/%d\n",
                m, n, k, kTileM, kTileN, kTileK);
        return 1;
    }
    printf("m=%d, n=%d, k=%d  (grid: %d x %d blocks)\n",
           m, n, k, m / kTileM, n / kTileN);

    using TA = cute::bfloat16_t;
    using TB = cute::bfloat16_t;
    using TC = float;

    // Host data
    thrust::host_vector<TA> h_A(m * k);
    thrust::host_vector<TB> h_B(n * k);
    thrust::host_vector<TC> h_C(m * n, TC(0));

    srand(42);
    for (int j = 0; j < m * k; ++j)
        h_A[j] = static_cast<TA>(2.0 * (rand() / double(RAND_MAX)) - 1.0);
    for (int j = 0; j < n * k; ++j)
        h_B[j] = static_cast<TB>(2.0 * (rand() / double(RAND_MAX)) - 1.0);

    // Device data
    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;

    // MMA / copy type aliases (identical to swizzle_copy.cu)
    using TiledMMA = decltype(make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{},
                                             Layout<Shape<_2, _4, _1>>{},
                                             Tile<_32, _32, _32>{}));

    using TiledCopyA_G2S = decltype(
        make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, TA>{},
                        Layout<Shape<_32, _8>, Stride<_8, _1>>{},
                        Layout<Shape<_1, _8>>{}));
    using TiledCopyB_G2S = decltype(
        make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, TB>{},
                        Layout<Shape<_32, _8>, Stride<_8, _1>>{},
                        Layout<Shape<_1, _8>>{}));

    using CopyA_atom = Copy_Atom<SM75_U32x4_LDSM_N, TA>;
    using CopyB_atom = Copy_Atom<SM75_U32x4_LDSM_N, TB>;
    using CopyC_atom = Copy_Atom<AutoVectorizingCopy, TC>;
    using TiledCopyA_S2R = decltype(make_tiled_copy_A(CopyA_atom{}, TiledMMA{}));
    using TiledCopyB_S2R = decltype(make_tiled_copy_B(CopyB_atom{}, TiledMMA{}));
    using TiledCopyC_R2G = decltype(make_tiled_copy_C(CopyC_atom{}, TiledMMA{}));

    // Launch config
    dim3 grid(m / kTileM, n / kTileN);
    dim3 block(size(TiledMMA{}));

    static constexpr int kShmSize = kTileM * kTileK * sizeof(TA)
                                  + kTileN * kTileK * sizeof(TB);

    printf("block threads: %d, smem: %d bytes (%.1f KB)\n",
           block.x, kShmSize, kShmSize / 1024.0f);
    printf("grid: (%d, %d, %d)\n\n", grid.x, grid.y, grid.z);

    using KernelT = decltype(&gemm_kernel<TC, TA, TB, kTileM, kTileN, kTileK,
                                          TiledMMA,
                                          TiledCopyA_G2S, TiledCopyB_G2S,
                                          TiledCopyA_S2R, TiledCopyB_S2R,
                                          TiledCopyC_R2G>);

    KernelT kernel_ptr = gemm_kernel<TC, TA, TB, kTileM, kTileN, kTileK,
                                     TiledMMA,
                                     TiledCopyA_G2S, TiledCopyB_G2S,
                                     TiledCopyA_S2R, TiledCopyB_S2R,
                                     TiledCopyC_R2G>;

    cudaFuncSetAttribute(kernel_ptr,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kShmSize);
    CUTE_CHECK_LAST();

    kernel_ptr<<<grid, block, kShmSize>>>(
        d_C.data().get(), d_A.data().get(), d_B.data().get(), m, n, k);
    CUTE_CHECK_LAST();
    cudaDeviceSynchronize();
    
    printf("Kernel finished\n");
    // -------------------------------------------------------------------------
    // Verification against CPU reference
    // -------------------------------------------------------------------------
    thrust::host_vector<TC> h_C_gpu = d_C;

    // Convert bf16 inputs to float for CPU reference
    std::vector<float> A_fp32(m * k), B_fp32(n * k);
    for (int i = 0; i < m * k; ++i) A_fp32[i] = static_cast<float>(h_A[i]);
    for (int i = 0; i < n * k; ++i) B_fp32[i] = static_cast<float>(h_B[i]);

    std::vector<float> C_ref(m * n, 0.f);
    cpu_gemm_ref(A_fp32, B_fp32, C_ref, m, n, k);
    
    printf("cpu calc finished\n");
    printf("compare gpu&cpu result m: %d, n: %d\n", m, n);

    float max_err = 0.f, sum_err = 0.f;
    for (int i = 0; i < m * n; ++i) {
        float err = std::abs(h_C_gpu[i] - C_ref[i]);
        max_err = std::max(max_err, err);
        sum_err += err;
    }
    printf("Verification: max_abs_err = %.6f,  avg_abs_err = %.6f\n",
           max_err, sum_err / (m * n));

    if (max_err < 0.5f)
        printf("PASS\n");
    else
        printf("FAIL\n");

    return 0;
}
