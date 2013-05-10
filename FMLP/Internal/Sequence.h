#ifndef FMLP_INTERNAL_SEQUENCE_H_
#define FMLP_INTERNAL_SEQUENCE_H_

namespace FMLP {
    namespace Internal {
        template <int... I>
        class Sequence {};
        
        template<int N, int... S>
        struct GenerateSequence : GenerateSequence<N-1, N-1, S...> { };
        
        template<int... S>
        struct GenerateSequence<0, S...> {
            typedef Sequence<S...> type;
        };
        
        template<bool, int, typename, typename>
        struct FirstNImpl;
        
        template <int Max, int Next, int... Remaining, int... Built>
        struct FirstNImpl<true, Max, Sequence<Next, Remaining...>, Sequence<Built...>>
        : FirstNImpl<(sizeof...(Built) < Max), Max, Sequence<Remaining...>, Sequence<Built..., Next>> {};
        
        template <int Max, int... Remaining, int... Built>
        struct FirstNImpl<false, Max, Sequence<Remaining...>, Sequence<Built...>> {
            typedef Sequence<Built...> type;
        };
        
        template <int Max, int... List>
        struct FirstN : FirstNImpl<(Max > 0), Max - 1, Sequence<List...>, Sequence<>> {};
        
        template <int L, int... Rest>
        struct Last : Last<Rest...> {};
        
        template <int L>
        struct Last<L> {
            const static int value = L;
        };
    }
}
#endif
