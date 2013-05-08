#ifndef NEURALNETWORK_UTILITY_H
#define NEURALNETWORK_UTILITY_H

#include <random>

inline double getRand(int value) {
    
    std::random_device rd;
    std::uniform_int_distribution<int> dist(0, 1000);
    return (static_cast<double>(dist(rd)) / static_cast<double>(10000)) - 0.05;
}

template <typename TValue>
inline TValue sum(TValue v) {
    return v;
}

template <typename TFirst, typename... TValues>
inline TFirst sum(TFirst f, TValues... v) {
    return f + sum(v...);
}

#endif
