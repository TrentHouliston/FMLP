#ifndef NEURALNETWORK_NEURONLAYER_H
#define NEURALNETWORK_NEURONLAYER_H

#include <tuple>
#include "Neuron.h"

template <typename, typename>
class NeuronLayerImpl;

template <int... Width, int... InputWidth>
class NeuronLayerImpl<Sequence<Width...>, Sequence<InputWidth...>> {
public:
    std::tuple<decltype(Width, Neuron<Sequence<InputWidth...>>())...> neurons;
    
    std::tuple<decltype(double(Width))...> operator()(std::tuple<decltype(double(InputWidth))...> input) {
        return std::make_tuple(std::get<Width>(neurons)(input)...);
    }
    
    template <typename TTuples, int I>
    double sumColumn(TTuples input) {
        return sum(std::get<I>(std::get<Width>(input))...);
    }
    
    std::tuple<decltype(double(InputWidth))...> operator()(std::tuple<decltype(double(InputWidth))...> input, std::tuple<decltype(double(Width))...> error) {
        
        auto data = std::make_tuple(std::get<Width>(neurons)(input, std::get<Width>(error))...);
        return std::make_tuple(sumColumn<decltype(data), InputWidth>(data)...);
    }
};

template <int Layer, int Input>
class NeuronLayer : public NeuronLayerImpl<typename GenerateSequence<Layer>::type, typename GenerateSequence<Input>::type> {};

#endif
