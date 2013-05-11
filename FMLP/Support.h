#ifndef FMLP_SUPPORT_H_
#define FMLP_SUPPORT_H_

#include <cmath>
#include <random>

namespace FMLP {
    
    /**
     * @todo Document
     */
    struct HyperbolicTan {
        static inline double func(const double input) {
            return tanh(input);
        }
        
        static inline double dfunc(const double input) {
            return (1 - tanh(input)) * (1 + tanh(input));
        }
    };
    
    /**
     * @todo Document
     */
    struct Sigmoid {
        static inline double func(const double input) {
            return 1 / (1 + exp(-input));
        }
        
        static inline double dfunc(const double input) {
            return func(input) * (1 - func(input));
        }
    };
    
    /**
     * @todo Document
     */
    struct Gaussian {
        static inline double func(const double input) {
            return exp(-pow(input, 2));
        }
        
        static inline double dfunc(const double input) {
            return -2 * func(input) * input;
        }
    };
    
    /**
     * @todo Document
     */
    struct Sin {
        static inline double func(const double input) {
            return sin(input);
        }
        
        static inline double dfunc(const double input) {
            return cos(input);
        }
    };
    
    /**
     * @todo Document
     */
    struct DefaultRNG {
        static inline double rand() {
            std::random_device rd;
            std::uniform_real_distribution<> dist(-0.5, 0.5);
            return dist(rd);
        }
    };
}

#endif
