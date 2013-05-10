#ifndef FMLP_INTERNAL_NEURONLAYER_H_
#define FMLP_INTERNAL_NEURONLAYER_H_

#include <tuple>
#include "Neuron.h"

namespace FMLP {
    namespace Internal {
        
        // Hide the unspecialized class from doxygen
        /// @cond HIDDEN
        template <typename, typename, typename, typename, typename, typename, typename> class NeuronLayerImpl; // @endcond
        
        /**
         * @todo Document
         */
        template <int... InputWidth, int... OutputWidth, int BiasNeuron, typename Activation, typename Learning, typename Momentum, typename RNG>
        class NeuronLayerImpl<Sequence<InputWidth...>, Sequence<OutputWidth...>, Sequence<BiasNeuron>, Activation, Learning, Momentum, RNG> {
        private:
            /// @todo document
            std::tuple<decltype(OutputWidth, Neuron<Sequence<InputWidth..., BiasNeuron>, Activation, Learning, Momentum, RNG>())...> neurons;
            
            /**
             * @todo Document
             */
            template <typename TTuples, int I>
            double sumColumn(const TTuples& input) {
                return sum(std::get<I>(std::get<OutputWidth>(input))...);
            }
            
        public:
            /**
             * @todo Document
             */
            std::tuple<decltype(double(OutputWidth))...> operator()(const std::tuple<decltype(double(InputWidth))...>& input) {
                return std::make_tuple(std::get<OutputWidth>(neurons)(std::tuple_cat(input, std::make_tuple(1.0)))...);
            }

            /**
             * @todo Document
             */
            std::tuple<decltype(double(InputWidth))...> operator()(const std::tuple<decltype(double(InputWidth))...>& input, const std::tuple<decltype(double(OutputWidth))...>& error) {
                
                auto data = std::make_tuple(std::get<OutputWidth>(neurons)(std::tuple_cat(input, std::make_tuple(1.0)), std::get<OutputWidth>(error))...);
                return std::make_tuple(sumColumn<decltype(data), InputWidth>(data)...);
            }
            
            /**
             * @todo Document
             */
            void applyLearning() {
                unpack(std::get<OutputWidth>(neurons).applyLearning()...);
            }
        };
        
        /**
         * @todo Document
         */
        template <typename LayerConfig>
        class NeuronLayer : public NeuronLayerImpl<
            /// @todo document
            typename GenerateSequence<LayerConfig::inputWidth>::type,
            /// @todo document
            typename GenerateSequence<LayerConfig::outputWidth>::type,
            /// @todo document
            Sequence<LayerConfig::inputWidth>,
            /// @todo document
            typename LayerConfig::activation,
            /// @todo document
            typename LayerConfig::learning,
            /// @todo document
            typename LayerConfig::momentum,
            /// @todo document
            typename LayerConfig::rng
        > {};
    }
}

#endif
