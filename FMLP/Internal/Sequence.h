#ifndef FMLP_INTERNAL_SEQUENCE_H_
#define FMLP_INTERNAL_SEQUENCE_H_

namespace FMLP {
    namespace Internal {
        
        /**
         * @todo Document
         */
        template <int... I>
        class Sequence {};
        
        /**
         * @todo Document
         */
        template<int N, int... S>
        struct GenerateSequence : GenerateSequence<N-1, N-1, S...> { };
        
        /**
         * @todo Document
         */
        template<int... S>
        struct GenerateSequence<0, S...> {
            typedef Sequence<S...> type;
        };
        
        // Hide the unspecialized class from doxygen
        /// @cond HIDDEN
        template<bool, int, typename, typename> struct FirstNImpl; /// @endcond
        
        /**
         * @todo Document
         */
        template <int Max, int Next, int... Remaining, int... Built>
        struct FirstNImpl<true, Max, Sequence<Next, Remaining...>, Sequence<Built...>>
        : FirstNImpl<(sizeof...(Built) < Max), Max, Sequence<Remaining...>, Sequence<Built..., Next>> {};
        
        /**
         * @todo Document
         */
        template <int Max, int... Remaining, int... Built>
        struct FirstNImpl<false, Max, Sequence<Remaining...>, Sequence<Built...>> {
            typedef Sequence<Built...> type;
        };
        
        /**
         * @todo Document
         */
        template <int Max, int... List>
        struct FirstN : FirstNImpl<(Max > 0), Max - 1, Sequence<List...>, Sequence<>> {};
        
        /**
         * @todo Document
         */
        template <int L, int... Rest>
        struct Last : Last<Rest...> {};
        
        /**
         * @todo Document
         */
        template <int L>
        struct Last<L> {
            const static int value = L;
        };
    }
}
#endif
