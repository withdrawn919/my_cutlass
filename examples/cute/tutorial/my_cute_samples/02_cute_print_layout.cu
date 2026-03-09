#include <cute/tensor.hpp>
#include <fstream>
#include <sstream>

using namespace cute;

int main() {
    using mma_op = SM89_16x8x32_F32E4M3E5M2F32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    using TiledMMA = decltype(make_tiled_mma(mma_atom{}));

    using Copy_op = SM75_U16x2_LDSM_T;
    using CopyA_atom = Copy_Atom<Copy_op, cute::half_t>;
    using TiledCopyA = decltype(make_tiled_copy_A(CopyA_atom{}, TiledMMA{}));

#if 1
    print_latex(TiledCopyA{});
#else
    print_svg(TiledMMA{});
#endif
    
    return 0;
}