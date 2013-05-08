#ifndef NEURALNETWORK_UTILITY_H
#define NEURALNETWORK_UTILITY_H

#include <random>
#include <sstream>

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

inline double getRand(int value) {
    
    std::random_device rd;
    std::uniform_real_distribution<> dist(-0.5, 0.5);
    //std::normal_distribution<> dist(0, 0.25);
    return dist(rd);
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
