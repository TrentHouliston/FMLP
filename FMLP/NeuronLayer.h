#ifndef NEURALNETWORK_NEURONLAYER_H
#define NEURALNETWORK_NEURONLAYER_H

#include <tuple>
#include <ratio>
#include "Neuron.h"
#include "Activation.h"

template <typename, typename, typename>
class NeuronLayerImpl;

template <int... Width, int... InputWidth, int BiasNeuron>
class NeuronLayerImpl<Sequence<Width...>, Sequence<InputWidth...>, Sequence<BiasNeuron>> {
public:
    std::tuple<decltype(Width, Neuron<Sequence<InputWidth..., BiasNeuron>, HyperbolicTan, std::ratio<1, 20>, std::ratio<1, 10>>())...> neurons;
    
    std::tuple<decltype(double(Width))...> operator()(const std::tuple<decltype(double(InputWidth))...>& input) {
        return std::make_tuple(std::get<Width>(neurons)(std::tuple_cat(input, std::make_tuple(1.0)))...);
    }
    
    template <typename TTuples, int I>
    double sumColumn(const TTuples& input) {
        return sum(std::get<I>(std::get<Width>(input))...);
    }
    
    std::tuple<decltype(double(InputWidth))...> operator()(const std::tuple<decltype(double(InputWidth))...>& input, const std::tuple<decltype(double(Width))...>& error) {
        //std::cout << "In: " << sizeof...(InputWidth) << " Out: " << sizeof...(Width) << std::endl;
        //std::cout << "Error Vector: " << printTuple(error) << std::endl;
        
        auto data = std::make_tuple(std::get<Width>(neurons)(std::tuple_cat(input, std::make_tuple(1.0)), std::get<Width>(error))...);
        return std::make_tuple(sumColumn<decltype(data), InputWidth>(data)...);
    }
    
    void applyLearning() {
        unpack(std::get<Width>(neurons).applyLearning()...);
    }
};

template <int Layer, int Input>
class NeuronLayer : public NeuronLayerImpl<typename GenerateSequence<Layer>::type, typename GenerateSequence<Input>::type, Sequence<Input>> {};

#endif
