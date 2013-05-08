#ifndef NEURALNETWORK_NEURON_H
#define NEURALNETWORK_NEURON_H

#include <cmath>
#include <tuple>
#include "Utility.h"

template <typename>
class Neuron;

template <int... Values>
class Neuron<Sequence<Values...>> {
public:
    
    inline double func(double input) {
        return std::tanh(input);
    }
    
    inline double dfunc(double input) {
        return 1 - (tanh(input) * tanh(input));
    }
    
    std::tuple<decltype(double(Values))...> weights = std::make_tuple(getRand(Values)...);
    
    double operator()(std::tuple<decltype(double(Values))...> input) {
        return func(sum((std::get<Values>(weights) * std::get<Values>(input))...));
    }
    
    std::tuple<decltype(double(Values))...> operator()(std::tuple<decltype(double(Values))...> input, double childError) {
        double derivitive = dfunc(sum((std::get<Values>(weights) * std::get<Values>(input))...));
        
        double error = derivitive * childError;
        
        auto ret = std::make_tuple((error * std::get<Values>(weights))...);
        
        std::make_tuple((std::get<Values>(weights) += error * std::get<Values>(input) * 0.05)...);
        
        return ret;
    }
    
    // Learning
    
    // Calculate my error for this layer and push it up
    //    // Error is calculated as (derivitive * child error * weights)...)
    
    // Have a learning rate (between 0 and 1, normally 0.05ish)
    
    // Calculate our delta
    //    // Our delta is calculated as for each weight, derivative * myerror * parent[i] * learning rate
    
    // TODO last layer doesn't do a tanh
    //    // Implement this along with picking the functions, setup so that the final function is an empty function
    
    // TODO implement momentum (keeping some of our delta)
    
    // TODO if possible make the function and derivitive function it uses editable
    
    // TODO if possible make the function it uses for each layer specific to that layer
    //    // TODO this is possible by using another partial specialization which allows setting per layer
    
    
    
    // you can use std::get<int>(weights) = bla in order to set the value in the tuple
};

#endif
