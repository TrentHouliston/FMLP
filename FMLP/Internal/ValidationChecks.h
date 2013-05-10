#ifndef FMLP_INTERNAL_VALIDATIONCHECKS_H_
#define FMLP_INTERNAL_VALIDATIONCHECKS_H_

#include <type_traits>

namespace FMLP {
    namespace Internal {
        
        /**
         * @todo Document
         */
        template <typename TType>
        struct IsRatio : std::false_type {
        };
        
        /**
         * @todo Document
         */
        template <int Num, int Den>
        struct IsRatio<std::ratio<Num, Den>> : std::true_type {
        };
    }
}
#endif
