#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include <cstdint>
#include <cmath>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <chrono>

using namespace cute;

// Kernel: each block computes one (kTileM x kTileN) output tile.
// Loops over K-tiles and accumulates into register fragment tCrC.
template <typename TC, typename TA, typename TB,
          int kTileM, int kTileN, int kTileK,
          typename SmemLayoutA, typename SmemLayoutB,
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

    Tensor sA = make_tensor(make_smem_ptr(smem_A), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(smem_B), SmemLayoutB{});

    auto tiler = make_tile(Int<kTileM>{}, Int<kTileN>{}, Int<kTileK>{});

    // one block task
    Tensor gA = local_tile(mA, tiler, make_coord(tile_m, tile_n, _), Step<_1,  X, _1>{}); //(kTileM, kTileK, num_tile_k)
    Tensor gB = local_tile(mB, tiler, make_coord(tile_m, tile_n, _), Step< X, _1, _1>{}); //(kTileN, kTileK, num_tile_k)
    Tensor gC = local_tile(mC, tiler, make_coord(tile_m, tile_n, _), Step<_1, _1,  X>{}); //(kTileM, kTileN)
    if(thread0()){
        print("\n");
        print("gA: ");print(gA);print("\n");
        print("gB: ");print(gB);print("\n");
        print("gC: ");print(gC);print("\n");
    }

    // 创建TiledCopyA_G2S的对象，每一次从gA拷贝一个(kTileM, kTileK)大小的块到sA
    TiledCopyA_G2S g2s_tiled_copy_a;
    ThrCopy g2s_thr_copy_a  = g2s_tiled_copy_a.get_slice(tid);
    Tensor  tAgA_g2s        = g2s_thr_copy_a.partition_S(gA);
    Tensor  tAsA_g2s        = g2s_thr_copy_a.partition_D(sA);
    // 创建TiledCopyB_G2S的对象，每一次从gB拷贝一个(kTileN, kTileK)大小的块到sB
    TiledCopyB_G2S g2s_tiled_copy_b;
    ThrCopy g2s_thr_copy_b  = g2s_tiled_copy_b.get_slice(tid);
    Tensor  tBgB_g2s        = g2s_thr_copy_b.partition_S(gB);
    Tensor  tBsB_g2s        = g2s_thr_copy_b.partition_D(sB);
    if(thread0()){
        print("\n");
        print("tAgA_g2s: ");print(tAgA_g2s);print("\n");
        print("tAsA_g2s: ");print(tAsA_g2s);print("\n");
        print("tBgB_g2s: ");print(tBgB_g2s);print("\n");
        print("tBsB_g2s: ");print(tBsB_g2s);print("\n");
    }

    // TiledMMA compute  (kTileM, kTileN, kTileK)
    TiledMMA tiled_mma;
    ThrMMA   thr_mma = tiled_mma.get_slice(tid);
    // 创建每个线程上的寄存器中的tensor
    Tensor tArA = thr_mma.partition_fragment_A(gA(_, _, 0));        // reg  (MMA, MMA_M, MMA_K)
    Tensor tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));        // reg  (MMA, MMA_M, MMA_K)
    Tensor tCrC = thr_mma.partition_fragment_C(gC);                 // reg  (MMA, MMA_M, MMA_N)
    if(thread0()){
        print("\n");
        print("tArA: ");print(tArA);print("\n");
        print("tBrB: ");print(tBrB);print("\n");
        print("tCrC: ");print(tCrC);print("\n");
    }
    // 创建TiledCopyA_S2R的对象,每一次按照tArA的layout，从sA拷贝一个(kTileM, kTileK)大小的块到tArA
    TiledCopyA_S2R  s2r_tiled_copy_a;
    ThrCopy         s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(tid);
    Tensor          tAsA_s2r = s2r_thr_copy_a.partition_S(sA);
    Tensor          tArA_s2r = s2r_thr_copy_a.retile_D(tArA);
    if(thread0()){
        print("\n");
        print("tAsA_s2r: ");print(tAsA_s2r);print("\n");
        print("tArA_s2r: ");print(tArA_s2r);print("\n");
    }
    // 创建TiledCopyB_S2R的对象,每一次按照tBrB的layout，从sB拷贝一个(kTileN, kTileK)大小的块到tBrB
    TiledCopyB_S2R  s2r_tiled_copy_b;
    ThrCopy         s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(tid);
    Tensor          tBsB_s2r = s2r_thr_copy_b.partition_S(sB);
    Tensor          tBrB_s2r = s2r_thr_copy_b.retile_D(tBrB);
    if(thread0()){
        print("\n");
        print("tBsB_s2r: ");print(tBsB_s2r);print("\n");
        print("tBrB_s2r: ");print(tBrB_s2r);print("\n");
    }
    // 创建 TiledCopyC_R2G 的对象，把每个线程上的tCrC拷贝到全局内存gC上
    TiledCopyC_R2G r2g_tiled_copy_c;
    ThrCopy r2g_thr_copy_c = r2g_tiled_copy_c.get_slice(tid);
    Tensor  tCrC_r2g = r2g_thr_copy_c.retile_S(tCrC);    // (CPY, CPY_M, CPY_N)
    Tensor  tCgC_r2g = r2g_thr_copy_c.partition_D(gC);   // (CPY, CPY_M, CPY_N)

     if(thread0()){
        print("\n");
        print("tCrC_r2g: ");print(tCrC_r2g);print("\n");
        print("tCgC_r2g: ");print(tCgC_r2g);print("\n");
    }
    // -------------------------------------------------------------------------
    // Main K-loop: iterate over K-tiles and accumulate
    // -------------------------------------------------------------------------
    int num_k_tiles = k / kTileK;

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        copy(g2s_tiled_copy_a, tAgA_g2s(_, _, _, k_tile), tAsA_g2s);
        copy(g2s_tiled_copy_b, tBgB_g2s(_, _, _, k_tile), tBsB_g2s);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();   // all threads see completed smem writes

        copy(s2r_tiled_copy_a, tAsA_s2r, tArA_s2r);
        copy(s2r_tiled_copy_b, tBsB_s2r, tBrB_s2r);

        gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);

        __syncthreads();   // ensure all threads done reading smem before next write
    }
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

    using SmemLayoutAtomA = decltype(
        composition(Swizzle<3, 3, 3>{},
                    Layout<Shape<_8, _64>,
                           Stride<_64, _1>>{}));
    using SmemLayoutAtomB = decltype(
        composition(Swizzle<3, 3, 3>{},
                    Layout<Shape<_8, _64>, 
                           Stride<_64, _1>>{}));
    using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{}, make_shape(Int<128>{}, Int<64>{})));
    using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{}, make_shape(Int<128>{}, Int<64>{})));

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
                                          SmemLayoutA, SmemLayoutB,
                                          TiledMMA,
                                          TiledCopyA_G2S, TiledCopyB_G2S,
                                          TiledCopyA_S2R, TiledCopyB_S2R,
                                          TiledCopyC_R2G>);

    KernelT kernel_ptr = gemm_kernel<TC, TA, TB, kTileM, kTileN, kTileK,
                                     SmemLayoutA, SmemLayoutB,    
                                     TiledMMA,
                                     TiledCopyA_G2S, TiledCopyB_G2S,
                                     TiledCopyA_S2R, TiledCopyB_S2R,
                                     TiledCopyC_R2G>;

    cudaFuncSetAttribute(kernel_ptr,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kShmSize);
    CUTE_CHECK_LAST();

    // ========================================
    // GPU核函数时间统计 - 开始
    // ========================================
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float gpu_elapsed_time_ms;

    // 记录开始时间
    cudaEventRecord(start, 0);

    // 启动核函数
    kernel_ptr<<<grid, block, kShmSize>>>(
        d_C.data().get(), d_A.data().get(), d_B.data().get(), m, n, k);
    
    // 记录结束时间
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    
    // 释放事件资源
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // ========================================
    // GPU核函数时间统计 - 结束
    // ========================================

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

    // ========================================
    // CPU参考实现时间统计 - 开始
    // ========================================
    auto cpu_start = std::chrono::high_resolution_clock::now();
    
    // 执行CPU参考实现
    cpu_gemm_ref(A_fp32, B_fp32, C_ref, m, n, k);
    
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_elapsed_time_ms = cpu_end - cpu_start;
    // ========================================
    // CPU参考实现时间统计 - 结束
    // ========================================
    
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

    // ========================================
    // 输出时间统计结果
    // ========================================
    printf("\n========================================");
    printf("\nPerformance Statistics:");
    printf("\n----------------------------------------");
    printf("\nGPU Kernel Time:   %.3f ms", gpu_elapsed_time_ms);
    printf("\nCPU Reference Time: %.3f ms", cpu_elapsed_time_ms.count());
    printf("\nSpeedup (CPU/GPU):  %.2f x", cpu_elapsed_time_ms.count() / gpu_elapsed_time_ms);
    
    // 计算GEMM的FLOPs（浮点运算数）：2*m*n*k（每个元素需要k次乘法+ k-1次加法，近似2*m*n*k）
    double flops = 2.0 * m * n * k;
    double gpu_flops = flops / (gpu_elapsed_time_ms * 1e-3) / 1e12; // TFLOPS
    double cpu_flops = flops / (cpu_elapsed_time_ms.count() * 1e-3) / 1e9; // GFLOPS
    printf("\n----------------------------------------");
    printf("\nTotal FLOPs:        %.2f GFLOPs", flops / 1e9);
    printf("\nGPU Performance:    %.3f TFLOPS", gpu_flops);
    printf("\nCPU Performance:    %.3f GFLOPS", cpu_flops);
    printf("\n========================================\n");

    return 0;
}
