#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include <cstdint>

using namespace cute;

template <typename TC,typename TA,typename TB,
          int kTileM, int kTileN, int kTileK,
          typename SmemLayoutA, typename SmemLayoutB,
          typename TiledMMA,
          typename TiledCopyA_G2S, typename TiledCopyB_G2S,
          typename TiledCopyA_S2R, typename TiledCopyB_S2R,
          typename TiledCopyC_R2G>
__global__ void gemm_kernel(TC *Cptr, TA *Aptr, TB *Bptr, int m, int n, int k){

    extern __shared__ char smem[]; 

    TA* smem_A = (TA*)smem;
    TB* smem_B = (TB*)(smem + sizeof(TA) * kTileM * kTileK);

    int tid = threadIdx.x;

    Tensor mA = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor mB = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor mC = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));
#if 1   
    using SmemLayoutAtomA = decltype(
        composition(Swizzle<3, 3, 3>{}, 
                   Layout<Shape<_128,_64>,
                          Stride<_64,_1>>{}));

    using SmemLayoutAtomB = decltype(
        composition(Swizzle<3, 3, 3>{}, 
                   Layout<Shape<_128,_64>,
                          Stride<_64,_1>>{}));

    Tensor sA = make_tensor(make_smem_ptr(smem_A), SmemLayoutAtomA{});                          
    Tensor sB = make_tensor(make_smem_ptr(smem_B), SmemLayoutAtomB{});
#else
    Tensor sA = make_tensor(make_smem_ptr(smem_A), make_shape(m , k), make_stride(k, Int<1>{}));
    Tensor sB = make_tensor(make_smem_ptr(smem_B), make_shape(n , k), make_stride(k, Int<1>{}));
#endif
    auto tiler = make_tile(Int<kTileM>{}, Int<kTileN>{}, Int<kTileK>{});
    auto coord = make_coord(0, 0, 0);

    Tensor gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{}); // (kTileM, kTileK)
    Tensor gB = local_tile(mB, tiler, coord, Step<X, _1, _1>{}); // (kTileN, kTileK)
    Tensor gC = local_tile(mC, tiler, coord, Step<_1, _1, X>{}); // (kTileM, kTileN)

    // if(thread0()){
    //     // print("mA:");print_tensor(mA);
    //     // print("mB:");print_tensor(mB);
    //     // print("mC:");print_tensor(mC);
    //     print("gA:");print(gA);print("\n");
    //     print("gB:");print(gB);print("\n");
    //     print("gC:");print(gC);print("\n");
    // }

    TiledMMA tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_slice(tid);

    // thread-globalMem layout
    Tensor tCgA = thr_mma.partition_A(gA); // (MMA, MMA_M, MMA_K)
    Tensor tCgB = thr_mma.partition_B(gB); // (MMA, MMA_N, MMA_K)
    Tensor tCgC = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N)
    
    // thread-sharedMem layout
    Tensor tCsA = thr_mma.partition_A(sA); // (MMA, MMA_M, MMA_K)
    Tensor tCsB = thr_mma.partition_B(sB); // (MMA, MMA_N, MMA_K)
 

    // thread-regMem layout
    Tensor tCrA = thr_mma.partition_fragment_A(gA); // (MMA, MMA_M, MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(gB); // (MMA, MMA_N, MMA_K)
    Tensor tCrC = thr_mma.partition_fragment_C(gC); // (MMA, MMA_M, MMA_N)



    // if(thread0()){
    //     print("tCgA:");print(tCgA);print("\n");
    //     print("tCgB:");print(tCgB);print("\n");
    //     print("tCgC:");print(tCgC);print("\n");
    //     print("tCrA:");print(tCrA);print("\n");
    //     print("tCrB:");print(tCrB);print("\n");
    //     print("tCrC:");print(tCrC);print("\n");
    // }

    // copy from globalMem to sharedMem
    TiledCopyA_G2S g2s_tiled_copy_a;
    ThrCopy g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(tid);
    Tensor tAgA_g2s = g2s_thr_copy_a.partition_S(gA);
    Tensor tAsA_g2s = g2s_thr_copy_a.partition_D(sA);
    
    TiledCopyB_G2S g2s_tiled_copy_b;
    ThrCopy g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(tid);
    Tensor tBgB_g2s = g2s_thr_copy_b.partition_S(gB);
    Tensor tBsB_g2s = g2s_thr_copy_b.partition_D(sB);

    copy(g2s_tiled_copy_a, tAgA_g2s, tAsA_g2s);
    copy(g2s_tiled_copy_b, tBgB_g2s, tBsB_g2s);

    cp_async_fence();        // Label the end of (potential) cp.async instructions
    cp_async_wait<0>();      // Sync on all (potential) cp.async instructions
    __syncthreads();         // Wait for all threads to write to smem

    // if(thread0()){
    //     print("sA:");print_tensor(sA);print("\n");
    //     print("sB:");print_tensor(sB);print("\n");
    // }
    
    // copy from sharedMem to regMem    
    TiledCopyA_S2R s2r_tiled_copy_a;
    ThrCopy s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(tid);
    Tensor tAgA_s2r = s2r_thr_copy_a.partition_S(sA); // (CPY, CPY_M, CPY_K)
    Tensor tArA_s2r = s2r_thr_copy_a.retile_D(tCrA); // (CPY, CPY_M, CPY_K)

    TiledCopyB_S2R s2r_tiled_copy_b;
    ThrCopy s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(tid);
    Tensor tBgB_s2r = s2r_thr_copy_b.partition_S(sB); // (CPY, CPY_M, CPY_K)
    Tensor tBrB_s2r = s2r_thr_copy_b.retile_D(tCrB); // (CPY, CPY_M, CPY_K)


    // if(thread0()){
    //     print("tAgA_s2r:");print(tAgA_s2r);print("\n");
    //     print("tArA_s2r:");print(tArA_s2r);print("\n");
    //     print("tBgB_s2r:");print(tBgB_s2r);print("\n");
    //     print("tBrB_s2r:");print(tBrB_s2r);print("\n");
    // }

#if 1
    copy(s2r_tiled_copy_a, tAgA_s2r, tArA_s2r);
    copy(s2r_tiled_copy_b, tBgB_s2r, tBrB_s2r);
    
    // mma compute 
    gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);
#elif 0
    constexpr int Kmma_per_copy = tiled_mma.template tile_size_mnk<2>() / 
                                  get<2>(typename TiledMMA::AtomShape_MNK{});
    CUTE_UNROLL
    for(int ik = 0; ik < (kTileK / tiled_mma.template tile_size_mnk<2>()); ik++){
        copy(s2r_tiled_copy_a, tAgA_s2r(_,_,ik), tArA_s2r(_,_,ik));
        copy(s2r_tiled_copy_b, tBgB_s2r(_,_,ik), tBrB_s2r(_,_,ik));
        CUTE_UNROLL
        for(int gk = ik * Kmma_per_copy; gk < (ik + 1) * Kmma_per_copy; gk++){
            gemm(tiled_mma, tCrC, tCrA(_,_,gk), tCrB(_,_,gk), tCrC);
        }
    }
#elif 0
    CUTE_UNROLL
    for(int ik = 0; ik < (kTileK / get<2>(typename TiledMMA::AtomShape_MNK{})); ik++){
        copy(tCgA(_,_,ik), tCrA(_,_,ik));
        copy(tCgB(_,_,ik), tCrB(_,_,ik));
        gemm(tiled_mma, tCrC, tCrA(_,_,ik), tCrB(_,_,ik), tCrC);
    }
#else
    constexpr int Kmma_per_copy_M = 1;
    constexpr int Kmma_per_copy_N = 1;
    constexpr int Kmma_per_copy_K = tiled_mma.template tile_size_mnk<2>() / 
                                  get<2>(typename TiledMMA::AtomShape_MNK{});

    CUTE_UNROLL                              
    for(int m_tile = 0; m_tile < (kTileM / tiled_mma.template tile_size_mnk<0>()); m_tile++){
        CUTE_UNROLL
        for(int n_tile = 0; n_tile < (kTileN / tiled_mma.template tile_size_mnk<1>()); n_tile++){
            CUTE_UNROLL
            for(int k_tile = 0; k_tile < (kTileK / tiled_mma.template tile_size_mnk<2>()); k_tile++){
                copy(s2r_tiled_copy_a, tAgA_s2r(_,m_tile,k_tile), tArA_s2r(_,m_tile,k_tile));
                copy(s2r_tiled_copy_b, tBgB_s2r(_,n_tile,k_tile), tBrB_s2r(_,n_tile,k_tile));
                CUTE_UNROLL
                for(int im = m_tile * Kmma_per_copy_M; im < (m_tile + 1) * Kmma_per_copy_M; im++){
                    CUTE_UNROLL
                    for(int in = n_tile * Kmma_per_copy_N; in < (n_tile + 1) * Kmma_per_copy_N; in++){
                        CUTE_UNROLL
                        for(int ik = k_tile * Kmma_per_copy_K; ik < (k_tile + 1) * Kmma_per_copy_K; ik++){
                            gemm(tiled_mma, tCrC(_, im, in), tCrA(_, im, ik), tCrB(_, in, ik), tCrC(_, im, in));
                        }
                    }
                }
            }
        }
    }
                             
#endif
    // copy from regMem to globalMem
    TiledCopyC_R2G r2g_tiled_copy_c;
    ThrCopy r2g_thr_copy_c = r2g_tiled_copy_c.get_slice(tid);
    Tensor tCrC_r2G = r2g_thr_copy_c.retile_S(tCrC);  // (CPY, CPY_M, CPY_N)
    Tensor tCgC_r2g = r2g_thr_copy_c.retile_D(tCgC); //  (CPY, CPY_M, CPY_N)
    copy(r2g_tiled_copy_c, tCrC_r2G, tCgC_r2g);
    __syncthreads();

    // if(thread0()){
    //     print(" mC: ");print_tensor(mC);
    // }
}



int main(int argc, char** argv){


    // 2. 获取第0个设备的属性
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    // 3. 打印设备基本信息
    std::cout << "\n=== 设备 " << 0 << " 信息 ===" << std::endl;
    std::cout << "设备名称: " << deviceProp.name << std::endl;

    // 4. 打印共享内存相关信息
    // 每个block的静态共享内存大小（字节）
    std::cout << "每个线程块(block)的静态共享内存大小: " 
              << deviceProp.sharedMemPerBlock << " 字节 (" 
              << deviceProp.sharedMemPerBlock / 1024.0 << " KB)" << std::endl;
    
    // 每个SM的最大共享内存大小（字节）
    std::cout << "每个SM的最大共享内存配置大小: "
              << deviceProp.sharedMemPerMultiprocessor << " 字节 ("
              << deviceProp.sharedMemPerMultiprocessor / 1024.0 << " KB)" << std::endl;

    // 通过 cudaFuncAttributeMaxDynamicSharedMemorySize 可申请到的最大动态共享内存
    std::cout << "每个Block可申请的最大动态共享内存(sharedMemPerBlockOptin): "
              << deviceProp.sharedMemPerBlockOptin << " 字节 ("
              << deviceProp.sharedMemPerBlockOptin / 1024.0 << " KB)" << std::endl;

    int m = 128;
    if (argc >= 2)
        sscanf(argv[1], "%d", &m);

    int n = 128;
    if (argc >= 3)
        sscanf(argv[2], "%d", &n);

    int k = 64;
    if (argc >= 4)
        sscanf(argv[3], "%d", &k);

    printf("m: %d, n: %d, k: %d\n", m, n, k);

    using TA = cute::bfloat16_t;
    using TB = cute::bfloat16_t;
    using TC = float;

    thrust::host_vector<TA> h_A(m*k);
    thrust::host_vector<TB> h_B(n*k);
    thrust::host_vector<TC> h_C(m*n);

    srand(42);
    for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<TC>(-1);    

    for(int i = 0;i < m; i++){
        h_A[i * k] = static_cast<TA>( i * 10);
        for(int j = 1;j < k; j++){
            h_A[i * k + j] = static_cast<TA>( 1);
        }
    }

    for(int i = 0;i < n; i++){
        for(int j = 0;j < k; j++){
            if(j == (i % k)){
                h_B[i * k + j] = static_cast<TB>( 1);
            }else{
                h_B[i * k + j] = static_cast<TB>( 0);
            }
        }
    }

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;

    using MMA_op = SM80_16x8x16_F32BF16BF16F32_TN;
    using MMA_traits = MMA_Traits<MMA_op>;
    using MMA_atom = MMA_Atom<MMA_traits>;
    using MMA_shape = MMA_traits::Shape_MNK;

    using TiledMMA = decltype(make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{},
                                             Layout<Shape<_2, _4, _1>>{},
                                             Tile<_32, _32, _32>{}));

    using TiledCopyA_G2S = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, cute::bfloat16_t>{},
                                                    Layout<Shape<_32, _8>, Stride<_8, _1>>{},
                                                    Layout<Shape<_1, _8>>{}));
    using TiledCopyB_G2S = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, cute::bfloat16_t>{},
                                                    Layout<Shape<_32, _8>, Stride<_8, _1>>{},
                                                    Layout<Shape<_1, _8>>{}));
    using SmemLayoutAtomA = decltype(
        composition(Swizzle<3, 3, 3>{}, 
                   Layout<Shape<_8,_64>,
                          Stride<_64,_1>>{}));
    using SmemLayoutAtomB = decltype(
        composition(Swizzle<3, 3, 3>{}, 
                   Layout<Shape<_8,_64>,
                          Stride<_64,_1>>{}));
    using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{}, make_shape(Int<128>{}, Int<64>{}), Step<_1, _2>{}));
    using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{}, make_shape(Int<128>{}, Int<64>{}), Step<_1, _2>{}));

    using Copy_op = AutoVectorizingCopy;
    using Copy_op_ldmatrix_x1 = SM75_U32x1_LDSM_N;
    using Copy_op_ldmatrix_x2 = SM75_U32x2_LDSM_N;
    using Copy_op_ldmatrix_x4 = SM75_U32x4_LDSM_N;
    using CopyA_atom = Copy_Atom<Copy_op_ldmatrix_x4, TA>;
    using CopyB_atom = Copy_Atom<Copy_op_ldmatrix_x4, TB>;
    using CopyC_atom = Copy_Atom<Copy_op, TC>;
    using TiledCopyA_S2R = decltype(make_tiled_copy_A(CopyA_atom{}, TiledMMA{}));
    using TiledCopyB_S2R = decltype(make_tiled_copy_B(CopyB_atom{}, TiledMMA{}));
    using TiledCopyC_R2G = decltype(make_tiled_copy_C(CopyC_atom{}, TiledMMA{}));

    dim3 grid(1);
    dim3 block(size(TiledMMA{}));
    static constexpr int kShmSize = 128 * 64 * sizeof(TA)  // size smem_A
                                  + 128 * 64 * sizeof(TB); // size smem_B
    printf("%d\n",kShmSize);
    printf("====================grid: (%d, %d, %d), block: (%d, %d, %d)====================\n",
       grid.x, grid.y, grid.z,
       block.x, block.y, block.z);

    // 申请量超过设备上限时直接退出，避免后续 kernel launch 崩溃
    if (kShmSize > deviceProp.sharedMemPerBlockOptin) {
        fprintf(stderr,
                "Error: 申请的动态共享内存 %d 字节 (%.1f KB) 超过设备上限 %zu 字节 (%.1f KB)，程序退出。\n",
                kShmSize,      kShmSize / 1024.0,
                deviceProp.sharedMemPerBlockOptin,
                deviceProp.sharedMemPerBlockOptin / 1024.0);
        return 1;
    }

    cudaFuncSetAttribute(gemm_kernel<TC, TA, TB, 128, 128, 64,
                                    SmemLayoutA, SmemLayoutB,
                                    TiledMMA,
                                    TiledCopyA_G2S, TiledCopyB_G2S,
                                    TiledCopyA_S2R, TiledCopyB_S2R,
                                    TiledCopyC_R2G>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kShmSize);
    CUTE_CHECK_LAST();

    gemm_kernel<TC, TA, TB, 128, 128, 64, 
                SmemLayoutA, SmemLayoutB,
                TiledMMA, 
                TiledCopyA_G2S, TiledCopyB_G2S, 
                TiledCopyA_S2R, TiledCopyB_S2R, 
                TiledCopyC_R2G>
                <<<grid,block,kShmSize>>>(d_C.data().get(), d_A.data().get(), d_B.data().get(), m, n, k);

    CUTE_CHECK_LAST();

    return 0;
}