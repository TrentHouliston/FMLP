#ifndef FMLP_SUPPORT_H_
#define FMLP_SUPPORT_H_

namespace FMLP {
    
    struct HyperbolicTan {
        static inline double func(double input) {
            return tanh(input);
        }
        
        static inline double dfunc(double input) {
            return (1 - tanh(input)) * (1 + tanh(input));
        }
    };
    
    struct Sigmoid {
        static inline double func(double input) {
            return 1 / (1 + exp(-input));
        }
        
        static inline double dfunc(double input) {
            return func(input) * (1 - func(input));
        }
    };
    
    struct DefaultRNG {
        static inline double rand(...) {
            std::random_device rd;
            std::uniform_real_distribution<> dist(-0.5, 0.5);
            //std::normal_distribution<> dist(0, 0.25);
            return dist(rd);
        }
    };
}

#endif
