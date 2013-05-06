#ifndef NEURALNETWORK_UTILITY_H
#define NEURALNETWORK_UTILITY_H



template <typename TValue>
inline TValue sum(TValue v) {
    return v;
}

template <typename TFirst, typename... TValues>
inline TFirst sum(TFirst f, TValues... v) {
    return f + sum(v...);
}

#endif
