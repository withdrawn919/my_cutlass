#include <cute/tensor.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <stdio.h>


using namespace cute;

template <typename TC, typename TA, typename TB, int kTileM, int kTileN, int kTileK, typename TiledMMA>
__global__ void gemm_kernel(TC *Cptr, TA *Aptr, TB *Bptr, int m, int n, int k) {
    auto mA = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
    auto mB = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
    auto mC = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

#if 0
    if (thread0()) {
        print(" mA: "); print(mA); print("\n");
        print(" mB: "); print(mB); print("\n");
        print(" mC: "); print(mC); print("\n");
        print(mA(9,0)); print("\n");
        print(mB(9,0)); print("\n");
    }
#endif

    int tid = threadIdx.x;
    auto tiler = make_tile(Int<kTileM>{}, Int<kTileN>{}, Int<kTileK>{});
    auto coord = make_coord(0, 0, 0);

    auto gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{});
    auto gB = local_tile(mB, tiler, coord, Step<X, _1, _1>{});
    auto gC = local_tile(mC, tiler, coord, Step<_1, _1, X>{});

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);
    auto tAgA = thr_mma.partition_A(gA);
    auto tBgB = thr_mma.partition_B(gB);
    auto tCgC = thr_mma.partition_C(gC);

#if 0
    if (thread0()) {
        print(" tAgA: "); print(tAgA); print("\n");
        print(" tBgB: "); print(tBgB); print("\n");
        print(" tCgC: "); print(tCgC); print("\n");
    }
#endif

    auto tArA = thr_mma.partition_fragment_A(gA);
    auto tBrB = thr_mma.partition_fragment_B(gB);
    auto tCrC = thr_mma.partition_fragment_C(gC);
    clear(tCrC);

#if 0
    if (thread0()) {
        print(" tArA: "); print(tArA); print("\n");
        print(" tBrB: "); print(tBrB); print("\n");
        print(" tCrC: "); print(tCrC); print("\n");
    }
#endif

    int num_tile_k = size<1>(gA) / kTileK;

    auto copy_atom = AutoVectorizingCopy{};
    #pragma unroll 1
    for (int itile = 0; itile < num_tile_k; ++itile) {
        copy(copy_atom, tAgA, tArA);
        copy(copy_atom, tBgB, tBrB);

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
 * Just use one tensor core(SM80_16x8x16_F16F16F16F16_TN) calculate mma
 */

int main(int argc, char** argv) {
    int m = 16;
    if (argc >= 2)
        sscanf(argv[1], "%d", &m);

    int n = 8;
    if (argc >= 3)
        sscanf(argv[2], "%d", &n);

    int k = 16;
    if (argc >= 4)
        sscanf(argv[3], "%d", &k);

    if (m != 16 || n != 8 || k != 16) {
        printf("This case only show the example of the M=16, N=8, K=16 shape, please use the correct inputs\n");
        return -1;
    }

    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    using TiledMMA = decltype(make_tiled_mma(mma_atom{}));

    using TA = cute::half_t;
    using TB = cute::half_t;
    using TC = cute::half_t;
    using TI = cute::half_t;

    printf("m: %d, n: %d, k: %d\n", m, n, k);

    thrust::host_vector<TA> h_A(m*k);
    thrust::host_vector<TB> h_B(n*k);
    thrust::host_vector<TC> h_C(m*n);

    srand(42);
    for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<TC>( 2*(rand() / double(RAND_MAX)) - 1 );


    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;

    dim3 block(32);
    dim3 grid(1);

    printf("========================grid: (%d, %d, %d), block: (%d, %d, %d)========================\n",
           grid.x, grid.y, grid.z,
           block.x, block.y, block.z);

    gemm_kernel<TC, TA, TB, 16, 8, 16, TiledMMA><<<grid, block>>>(d_C.data().get(), d_A.data().get(), d_B.data().get(), m, n, k);

    CUTE_CHECK_LAST();

    printf("Run once end ! \n");
    thrust::host_vector<TC> cute_result = d_C;

    return 0;
}