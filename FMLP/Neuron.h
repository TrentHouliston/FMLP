#ifndef NEURALNETWORK_NEURON_H
#define NEURALNETWORK_NEURON_H

#include <cmath>
#include <tuple>
#include "Utility.h"

template <typename, typename, typename, typename>
class Neuron;

template <int... Values, typename TActivation, int LearningNumerator, int LearningDenominator, int MomentumNumerator, int MomentumDenominator>
class Neuron<Sequence<Values...>, TActivation, std::ratio<LearningNumerator, LearningDenominator>, std::ratio<MomentumNumerator, MomentumDenominator>> {
private:
    std::tuple<decltype(double(Values))...> weights = std::make_tuple(getRand(Values)...);
    std::tuple<decltype(double(Values))...> deltas = std::make_tuple(static_cast<double>(Values - Values)...);
    
    constexpr double fraction(int numerator, int denominator) {
        return static_cast<double>(numerator) / static_cast<double>(denominator);
    }
    
public:

    
    double operator()(std::tuple<decltype(double(Values))...> input) {
        return TActivation::func(sum((std::get<Values>(weights) * std::get<Values>(input))...));
    }
    
    std::tuple<decltype(double(Values))...> operator()(const std::tuple<decltype(double(Values))...>& input, const double childError) {
        
        const double derivitive = TActivation::dfunc(sum((std::get<Values>(weights) * std::get<Values>(input))...));
        const double error = derivitive * childError;
        const auto ret = std::make_tuple((error * std::get<Values>(weights))...);
        const auto delta = std::make_tuple((error * std::get<Values>(input) * fraction(LearningNumerator, LearningDenominator))...);
        
        std::cout << "\tNeuron " << this << std::endl;
        std::cout << "\t\tChild Error:  " << printTuple(std::make_tuple(childError)) << std::endl;
        std::cout << "\t\tInputs:       " << printTuple(input) << std::endl;
        std::cout << "\t\tWeights:      " << printTuple(weights) << std::endl;
        std::cout << "\t\tDot Product:  " << printTuple(std::make_tuple(sum((std::get<Values>(weights) * std::get<Values>(input))...))) << std::endl;
        std::cout << "\t\tDerivitive:   " << printTuple(std::make_tuple(TActivation::dfunc(sum((std::get<Values>(weights) * std::get<Values>(input))...)))) << std::endl;
        std::cout << "\t\tPush Back:    " << printTuple(ret) << std::endl;
        std::cout << "\t\tDelta:        " << printTuple(delta) << std::endl;
        
        unpack((std::get<Values>(deltas) -= std::get<Values>(delta))...);
        
        std::cout << "\t\tDeltas:       " << printTuple(deltas) << std::endl << std::endl;
        
        return ret;
    }
    
    bool applyLearning() {
        std::cout << "Applying Learning" << std::endl;
        std::cout << "\tOriginal: " << printTuple(weights) << std::endl;
        std::cout << "\tDeltas:   " << printTuple(deltas) << std::endl;
        unpack((std::get<Values>(weights) += std::get<Values>(deltas))...);
        std::cout << "\tNew:      " << printTuple(weights) << std::endl;
        std::cout << std::endl;
        
        // Apply momentum to the deltas
        unpack((std::get<Values>(deltas) *= fraction(MomentumNumerator, MomentumDenominator))...);
        return true;
    }
    
    // TODO last layer doesn't do a tanh
    //    // Implement this along with picking the functions, setup so that the final function is an empty function
    
    // TODO if possible make the function and derivitive function it uses editable
    
    // TODO if possible make the function it uses for each layer specific to that layer
    //    // TODO this is possible by using another partial specialization which allows setting per layer
    
    
    
    // you can use std::get<int>(weights) = bla in order to set the value in the tuple
};

#endif
