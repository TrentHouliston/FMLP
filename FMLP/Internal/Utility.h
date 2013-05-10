#ifndef FMLP_INTERNAL_UTILITY_H_
#define FMLP_INTERNAL_UTILITY_H_

#include <random>
#include <sstream>

namespace FMLP {
    namespace Internal {
        
        template <typename... TTypes>
        inline void unpack(TTypes...) {
        }
        
        template <typename... TTypes, int... I>
        std::string printTuple(std::tuple<TTypes...> tuple, Sequence<I...>) {
            std::stringstream s;
            char chars[100];
            
            unpack((sprintf(chars, "%8.5f, ", std::get<I>(tuple)),(std::cout << chars),0)...);
            
            return s.str();
        }
        
        template <typename... TTypes>
        std::string printTuple(std::tuple<TTypes...> tuple) {
            return printTuple(tuple, typename GenerateSequence<sizeof...(TTypes)>::type());
        }
        
        template <typename TValue>
        inline TValue sum(TValue v) {
            return v;
        }
        
        template <typename TFirst, typename... TValues>
        inline TFirst sum(TFirst f, TValues... v) {
            return f + sum(v...);
        }
    }
}

#endif
