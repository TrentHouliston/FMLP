#ifndef FMLP_INTERNAL_NEURONLAYER_H_
#define FMLP_INTERNAL_NEURONLAYER_H_

#include <tuple>
#include <ratio>
#include "Neuron.h"
#include "Support.h"

namespace FMLP {
    namespace Internal {
        
        template <typename, typename, typename, typename, typename, typename, typename>
        class NeuronLayerImpl;
        
        template <int... InputWidth, int... OutputWidth, int BiasNeuron, typename Activation, typename Learning, typename Momentum, typename RNG>
        class NeuronLayerImpl<Sequence<InputWidth...>, Sequence<OutputWidth...>, Sequence<BiasNeuron>, Activation, Learning, Momentum, RNG> {
        public:
            std::tuple<decltype(OutputWidth, Neuron<Sequence<InputWidth..., BiasNeuron>, Activation, Learning, Momentum, RNG>())...> neurons;
            
            std::tuple<decltype(double(OutputWidth))...> operator()(const std::tuple<decltype(double(InputWidth))...>& input) {
                return std::make_tuple(std::get<OutputWidth>(neurons)(std::tuple_cat(input, std::make_tuple(1.0)))...);
            }
            
            template <typename TTuples, int I>
            double sumColumn(const TTuples& input) {
                return sum(std::get<I>(std::get<OutputWidth>(input))...);
            }
            
            std::tuple<decltype(double(InputWidth))...> operator()(const std::tuple<decltype(double(InputWidth))...>& input, const std::tuple<decltype(double(OutputWidth))...>& error) {
                //std::cout << "In: " << sizeof...(InputWidth) << " Out: " << sizeof...(Width) << std::endl;
                //std::cout << "Error Vector: " << printTuple(error) << std::endl;
                
                auto data = std::make_tuple(std::get<OutputWidth>(neurons)(std::tuple_cat(input, std::make_tuple(1.0)), std::get<OutputWidth>(error))...);
                return std::make_tuple(sumColumn<decltype(data), InputWidth>(data)...);
            }
            
            void applyLearning() {
                unpack(std::get<OutputWidth>(neurons).applyLearning()...);
            }
        };
        
        template <typename LayerConfig>
        class NeuronLayer : public NeuronLayerImpl<
        typename GenerateSequence<LayerConfig::inputWidth>::type,
        typename GenerateSequence<LayerConfig::outputWidth>::type,
        Sequence<LayerConfig::inputWidth>,
        typename LayerConfig::activation,
        typename LayerConfig::learning,
        typename LayerConfig::momentum,
        typename LayerConfig::rng
        > {};
    }
}

#endif
