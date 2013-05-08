#ifndef NEURALNETWORK_NEURONLAYER_H
#define NEURALNETWORK_NEURONLAYER_H

#include <tuple>
#include "Neuron.h"

template <typename, typename, typename>
class NeuronLayerImpl;

template <int... Width, int... InputWidth, int BiasNeuron>
class NeuronLayerImpl<Sequence<Width...>, Sequence<InputWidth...>, Sequence<BiasNeuron>> {
public:
    std::tuple<decltype(Width, Neuron<Sequence<InputWidth..., BiasNeuron>>())...> neurons;
    
    std::tuple<decltype(double(Width))...> operator()(std::tuple<decltype(double(InputWidth))...> input) {
        return std::make_tuple(std::get<Width>(neurons)(std::tuple_cat(input, std::make_tuple(1)))...);
    }
    
    template <typename TTuples, int I>
    double sumColumn(TTuples input) {
        return sum(std::get<I>(std::get<Width>(input))...);
    }
    
    std::tuple<decltype(double(InputWidth))...> operator()(std::tuple<decltype(double(InputWidth))...> input, std::tuple<decltype(double(Width))...> error) {
        //std::cout << "In: " << sizeof...(InputWidth) << " Out: " << sizeof...(Width) << std::endl;
        //std::cout << "Error Vector: " << printTuple(error) << std::endl;
        
        auto data = std::make_tuple(std::get<Width>(neurons)(std::tuple_cat(input, std::make_tuple(1)), std::get<Width>(error))...);
        return std::make_tuple(sumColumn<decltype(data), InputWidth>(data)...);
    }
};

template <int Layer, int Input>
class NeuronLayer : public NeuronLayerImpl<typename GenerateSequence<Layer>::type, typename GenerateSequence<Input>::type, Sequence<Input>> {};

#endif
