#include <cstdint>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_fp8.h>
#include <cute/tensor.hpp>


using namespace cute;

template <typename TypeA, typename TypeB, typename TypeC,
          int kTileM, int kTileN, int kTileK, typename TiledMMA>
__global__ void gemm_kernel(TypeC *Cptr, TypeA *Aptr, TypeB *Bptr, int m, int n, int k) {
    auto mA = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
    auto mB = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
    auto mC = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

#if 0
    if (thread0()) {
        print(" mA:"); print_tensor(mA); print("\n");
        print(" mB:"); print_tensor(mB); print("\n");
        print(" mC:"); print_tensor(mC); print("\n");
        print("B(1,1):"); print(mB(1,1)); print("\n");
    }
#endif

    int tid = threadIdx.x;
    auto tiler = make_tile(Int<kTileM>{}, Int<kTileN>{}, Int<kTileK>{});
    auto coord = make_coord(0, 0, 0);

    auto gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{});
    auto gB = local_tile(mB, tiler, coord, Step<X, _1, _1>{});
    auto gC = local_tile(mC, tiler, coord, Step<_1, _1, X>{});

    TiledMMA tiled_mma;
    auto thr_mma  = tiled_mma.get_slice(tid);
    auto tAgA     = thr_mma.partition_A(gA);
    auto tBgB     = thr_mma.partition_B(gB);
    auto tCgC     = thr_mma.partition_C(gC);

    auto tArA     = thr_mma.partition_fragment_A(gA);
    auto tBrB     = thr_mma.partition_fragment_B(gB);
    auto tCrC     = thr_mma.partition_fragment_C(gC);

    int num_tile_k = k / kTileK;

    if (thread0()) {
        printf(" num_tile_k: %d\n", num_tile_k);
    }

    for (int i = 0; i < num_tile_k; ++i) {
        copy(tAgA, tArA);
        copy(tBgB, tBrB);
        gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
    }

    copy(tCrC, tCgC);

#if 0
    if (thread0()) {
        print(" mC:"); print_tensor(mC); print("\n");
    }
#endif
}

/*
 * Just use one tensor core(SM89_16x8x32_F32E4M3E4M3F32_TN) calculate mma
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
        printf("This case only show the example of the M=16, N=8, K=32 shape, please use the correct inputs\n");
        return -1;
    }

    printf("m: %d, n: %d, k: %d\n", m, n, k);

    using TypeA = cute::float_e4m3_t;
    using TypeB = cute::float_e4m3_t;
    using TypeC = float;

    thrust::host_vector<TypeA> h_A(m*k);
    thrust::host_vector<TypeB> h_B(n*k);
    thrust::host_vector<TypeC> h_C(m*n);

    srand(42);
    for (int j = 0; j < m*k; ++j) h_A[j] = __nv_fp8_e4m3( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < n*k; ++j) h_B[j] = __nv_fp8_e4m3( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<TypeC>( 2*(rand() / double(RAND_MAX)) - 1 );

    thrust::device_vector<TypeA> d_A = h_A;
    thrust::device_vector<TypeB> d_B = h_B;
    thrust::device_vector<TypeC> d_C = h_C;

    using mma_op      = SM89_16x8x32_F32E4M3E4M3F32_TN;
    using mma_traits  = MMA_Traits<mma_op>;
    using mma_atom    = MMA_Atom<mma_traits>;
    using TiledMMA    = decltype(make_tiled_mma(mma_atom{}));

    dim3 block(32);
    dim3 grid(1);

    printf("grid: (%d, %d, %d), block: (%d, %d, %d)\n",
           grid.x, grid.y, grid.z,
           block.x, block.y, block.z);

    gemm_kernel<TypeA, TypeB, TypeC, 16, 8, 32, TiledMMA>
        <<<grid, block>>>(d_C.data().get(), d_A.data().get(), d_B.data().get(), m, n, k);

    CUTE_CHECK_LAST();

    thrust::host_vector<TypeC> cute_result = d_C;

    return 0;
}