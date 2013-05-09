#ifndef NEURALNETWORK_NEURALNETWORK_H
#define NEURALNETWORK_NEURALNETWORK_H

#include <tuple>
#include "Sequence.h"
#include "Utility.h"
#include "NeuronLayer.h"

namespace FMLP {
    
    // Forward declaring LayerConfig as it is used in the transform
    template <int, typename>
    struct LayerConfig;
    
    // Anonymous namespace to hide implementation details
    namespace {
        
        /**
         * @todo Document
         *
         * This class is the recursive expansion to classify, it runs through all layers recursivly
         */
        template <int I, int Length = I>
        struct Classify {
            template <typename... TTypes, typename... TInputs>
            static inline auto call(std::tuple<TTypes...>& layers, const std::tuple<TInputs...>& input)
            -> decltype(Classify<I - 1, Length>::call(layers, std::get<Length - I>(layers)(input))) {
                return Classify<I - 1, Length>::call(layers, std::get<Length - I>(layers)(input));
            }
        };
        
        /**
         * @todo Document
         *
         * This class is the specialization which ends the recursion by returning the final result
         */
        template <int Length>
        struct Classify<1, Length> {
            template <typename... TTypes, typename... TInputs>
            static inline auto call(std::tuple<TTypes...>& layers, const std::tuple<TInputs...>& input) -> decltype(std::get<Length - 1>(layers)(input)) {
                return std::get<Length - 1>(layers)(input);
            }
        };

        /**
         * @todo Document
         *
         * This class is the recursive expansion to learn, it goes through all layers recursivly, returning their backprop
         */
        template <int I, int Length = I>
        struct Learn {
            template <typename... TTypes, typename... TInputs, typename... TTarget>
            static inline auto call(std::tuple<TTypes...>& layers, const std::tuple<TInputs...>& input, const std::tuple<TTarget...>& target)
            -> decltype(std::get<Length - I>(layers)(input, Learn<I - 1, Length>::call(layers, std::get<Length - I>(layers)(input), target))) {
                return std::get<Length - I>(layers)(input, Learn<I - 1, Length>::call(layers, std::get<Length - I>(layers)(input), target));
            }
        };
        
        /**
         * @todo Document
         *
         * This class is the specialization which ends the recursion by getting the final error for the backprop
         */
        template <int Length>
        struct Learn<1, Length> {
            
            template <typename... TTypes, int... I>
            static inline std::tuple<decltype(double(I))...> error(const std::tuple<TTypes...>& actual, const std::tuple<TTypes...>& target, Sequence<I...>) {
                return std::make_tuple((std::get<I>(actual) - std::get<I>(target))...);
            }
            
            template <typename... TTypes>
            static inline auto error(const std::tuple<TTypes...>& actual, const std::tuple<TTypes...>& target)
            -> decltype(error(actual, target, typename GenerateSequence<sizeof...(TTypes)>::type())) {
                return error(actual, target, typename GenerateSequence<sizeof...(TTypes)>::type());
            }
            
            template <typename... TTypes, typename... TInputs, typename... TTarget>
            static inline auto call(std::tuple<TTypes...>& layers, const std::tuple<TInputs...>& input, const std::tuple<TTarget...>& target)
            -> decltype(std::get<Length - 1>(layers)(input, error(std::get<Length - 1>(layers)(input), target))){
                return std::get<Length - 1>(layers)(input, error(std::get<Length - 1>(layers)(input), target));
            }
        };
        
        /**
         * @todo Document
         *
         * This class is the recursive expansion to apply learned weights to the system
         */
        template <int I, int Length = I>
        struct ApplyLearning {
            template <typename... TTypes, typename... TInputs>
            static inline void call(std::tuple<TTypes...>& layers) {
                std::get<Length - I>(layers).applyLearning();
                ApplyLearning<I - 1, Length>::call(layers);
            }
        };
        
        /**
         * @todo Document
         *
         * This class is the specialization which ends the recursion, applying the final layers learning
         */
        template <int Length>
        struct ApplyLearning<1, Length> {
            template <typename... TTypes, typename... TInputs>
            static inline void call(std::tuple<TTypes...>& layers) {
                std::get<Length - 1>(layers).applyLearning();
            }
        };
        
        /**
         * @todo Document
         *
         * This class a hidden class which implements everything needed to configure a layer (including it's input size)
         */
        template <int InputWidth, int OutputWidth, typename Activation>
        struct FullLayerConfig {
            typedef Activation activation;
            static const int inputWidth = InputWidth;
            static const int outputWidth = OutputWidth;
        };
        
        /**
         * @todo Document
         *
         * This class is the final result of the meta functions which contains our actual functions exposed to the end user
         */
        template <typename...> class FMLPImpl;
        template <int... Input, int... Output, int... Layers, typename... Activations, int... LayerInputs>
        class FMLPImpl<Sequence<Input...>, Sequence<Output...>, FullLayerConfig< LayerInputs, Layers, Activations>...> {
            public:
                // This builds all of our neuron layers except our first layer (which is simply an input so does not need one)
                std::tuple<NeuronLayer<Layers, LayerInputs>...> layers;
                
                std::tuple<decltype(double(Output))...> classify(const std::tuple<decltype(double(Input))...>& input) {
                    return Classify<sizeof...(Layers)>::call(layers, input);
                }
                
                void learn(const std::tuple<decltype(double(Input))...>& input, const std::tuple<decltype(double(Output))...>& target) {
                    Learn<sizeof...(Layers)>::call(layers, input, target);
                }
                
                void applyLearning() {
                    ApplyLearning<sizeof...(Layers)>::call(layers);
                }
        };
        
        /**
         * @todo Document
         *
         * This class a helper class which performs some expansion and operations on the provided arguments to allow them to be used in the system
         */
        template <typename...> class FMLPTransform;
        template <int... PreviousInputs, int InputWidth, typename InputActivation, int... Widths, typename... Activations>
        class FMLPTransform<Sequence<PreviousInputs...>, LayerConfig<InputWidth, InputActivation>, LayerConfig<Widths, Activations>...> :
        public FMLPImpl <
            // This is our sequence for our input vector width
            typename GenerateSequence<InputWidth>::type,
            // This is our sequence for our output vector's width
            typename GenerateSequence<Last<Widths...>::value>::type,
            // This is our final layer objects with their input widths integrated
            FullLayerConfig<InputWidth, InputWidth, InputActivation>,
            FullLayerConfig<PreviousInputs, Widths, Activations>...
        > {};
        
    }
    
    /**
     * @todo Document
     *
     * This class a public class which exposes the values that the end user can configure
     */
    template <int Width, typename Activation>
    struct LayerConfig {};
    
    template <typename...> class FMLPAdvanced;
    template <int InputWidth, typename InputActivation, int... Widths, typename... Activations>
    class FMLPAdvanced<LayerConfig<InputWidth, InputActivation>, LayerConfig<Widths, Activations>...> :
    public FMLPTransform<
        typename FirstN<sizeof...(Widths), InputWidth, Widths...>::type,
        LayerConfig<InputWidth, InputActivation>,
        LayerConfig<Widths, Activations>...
    > {};
    
    
    template <int... Widths>
    class FMLP :
    public FMLPAdvanced<LayerConfig<Widths, Sigmoid>...> {};
    
}
/*
 
 public FMLPLImpl<
 // This is our sequence for our input vector width
 typename GenerateSequence<InputWidth>::type,
 // This is our sequence for our output vector's width
 typename GenerateSequence<Last<Widths...>::value>::type,
 // This is the width value of each of our network layers
 Sequence<Widths...>,
 // This is the width of each network layers previous layer (therefore it's input width)
 typename FirstN<sizeof...(Widths), InputWidth, Widths...>::type
 > {};*/

#endif
