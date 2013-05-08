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
        auto delta = std::make_tuple((error * std::get<Values>(input) * 0.15)...);
        
        /*std::cout << "\tNeuron" << std::endl;
        std::cout << "\t\tChild Error:  " << printTuple(std::make_tuple(childError)) << std::endl;
        std::cout << "\t\tInputs:       " << printTuple(input) << std::endl;
        std::cout << "\t\tWeights:      " << printTuple(weights) << std::endl;
        std::cout << "\t\tDot Product:  " << printTuple(std::make_tuple(sum((std::get<Values>(weights) * std::get<Values>(input))...))) << std::endl;
        std::cout << "\t\tDerivitive:   " << printTuple(std::make_tuple(dfunc(sum((std::get<Values>(weights) * std::get<Values>(input))...)))) << std::endl;
        std::cout << "\t\tPush Back:    " << printTuple(ret) << std::endl;
        std::cout << "\t\tDelta:        " << printTuple(delta) << std::endl;*/
        
        unpack((std::get<Values>(weights) -= std::get<Values>(delta))...);
        
        //std::cout << "\t\tAdjusted:     " << printTuple(weights) << std::endl;
        
        // Unpack the calls to adjust our weights
        
        //std::cout << std::endl;
        
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
