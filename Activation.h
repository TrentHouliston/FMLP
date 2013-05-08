#ifndef NEURALNETWORK_ACTIVATION_H
#define NEURALNETWORK_ACTIVATION_H

struct HyperbolicTan {
    static inline double func(double input) {
        return std::tanh(input);
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
        return func(1 - func(input));
    }
};

#endif
