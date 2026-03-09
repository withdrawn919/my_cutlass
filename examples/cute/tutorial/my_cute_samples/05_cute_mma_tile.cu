#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>
#include <cstdint>

using namespace cute;

template <typename TC, typename TA, typename TB,
          int kTileM, int kTileN, int kTileK,
          typename TiledMMA>
__global__ void gemm_kernel(TC *Cptr, TA *Aptr, TB *Bptr, int m, int n, int k) {
    int tid = threadIdx.x;

    Tensor mA = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor mB = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor mC = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

    auto tiler = make_tile(Int<kTileM>{}, Int<kTileN>{}, Int<kTileK>{});
    auto coord = make_coord(0, 0, 0);

    Tensor gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{}); // (kTileM, kTileK)
    Tensor gB = local_tile(mB, tiler, coord, Step<X, _1, _1>{}); // (kTileN, kTileK)
    Tensor gC = local_tile(mC, tiler, coord, Step<_1, _1, X>{}); // (kTileM, kTileN)

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

    // copy from globalMem to regMem
    auto copy_atom = AutoVectorizingCopy{};
    copy(copy_atom, tCgA, tCrA);
    copy(copy_atom, tCgB, tCrB);

    // mma compute
    gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

    // copy from regMem to globalMem
    copy(copy_atom, tCrC, tCgC);

    if(thread0()){
        print("mC: ");print_tensor(mC);print("\n");
    }
}
/*
* Expand tile shape, use more tensor core(SM80_16x8x16_F32F16F16F32_TN) calculate mma.
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

    if (m != 32 || n != 32 || k != 16) {
        printf("This case only show the example of the M=32, N=32, K=16 shape, please use the correct inputs\n");
        return -1;
    }

    printf("m: %d, n: %d, k: %d\n", m, n, k);

    using TA = cute::half_t;
    using TB = cute::half_t;
    using TC = float;

    thrust::host_vector<TA> h_A(m*k);
    thrust::host_vector<TB> h_B(n*k);
    thrust::host_vector<TC> h_C(m*n);

    srand(42);
    for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<TC>(-1);

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;

    using mma_op = SM80_16x8x16_F32F16F16F32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    using TiledMMA = decltype(make_tiled_mma(mma_atom{},
                                             Layout<Shape<_2, _4, _1>>{},
                                             Tile<_32, _32, _16>{}));

    dim3 grid(1);
    dim3 block(size(TiledMMA{}));
    static constexpr int kShmSize = 0;

    printf("========================grid: (%d, %d, %d), block: (%d, %d, %d)========================\n",
           grid.x, grid.y, grid.z,
           block.x, block.y, block.z);

    gemm_kernel<TC, TA, TB, 32, 32, 16, TiledMMA>
        <<<grid, block>>>(d_C.data().get(), d_A.data().get(), d_B.data().get(), m, n, k);

    CUTE_CHECK_LAST();

    thrust::host_vector<TC> cute_result = d_C;

    return 0;
}