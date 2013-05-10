#ifndef FMLP_INTERNAL_UTILITY_H_
#define FMLP_INTERNAL_UTILITY_H_

namespace FMLP {
    namespace Internal {
        
        /**
         * @todo Document
         */
        template <typename... TTypes>
        inline void unpack(TTypes...) {
        }
        
        /**
         * @todo Document
         */
        template <typename TValue>
        inline TValue sum(TValue v) {
            return v;
        }
        
        /**
         * @todo Document
         */
        template <typename TFirst, typename... TValues>
        inline TFirst sum(TFirst f, TValues... v) {
            return f + sum(v...);
        }
    }
}

#endif
