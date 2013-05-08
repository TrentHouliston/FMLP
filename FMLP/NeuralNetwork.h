#ifndef NEURALNETWORK_NEURALNETWORK_H
#define NEURALNETWORK_NEURALNETWORK_H

#include <tuple>
#include "Sequence.h"
#include "Utility.h"
#include "NeuronLayer.h"

template <int I, int Length = I>
struct Classify {
    template <typename... TTypes, typename... TInputs>
    static inline auto call(std::tuple<TTypes...>& layers, const std::tuple<TInputs...>& input)
    -> decltype(Classify<I - 1, Length>::call(layers, std::get<Length - I>(layers)(input))) {
        return Classify<I - 1, Length>::call(layers, std::get<Length - I>(layers)(input));
    }
};

template <int Length>
struct Classify<1, Length> {
    template <typename... TTypes, typename... TInputs>
    static inline auto call(std::tuple<TTypes...>& layers, const std::tuple<TInputs...>& input) -> decltype(std::get<Length - 1>(layers)(input)) {
        return std::get<Length - 1>(layers)(input);
    }
};

template <int I, int Length = I>
struct ApplyLearning {
    template <typename... TTypes, typename... TInputs>
    static inline void call(std::tuple<TTypes...>& layers) {
        std::get<Length - I>(layers).applyLearning();
        ApplyLearning<I - 1, Length>::call(layers);
    }
};

template <int Length>
struct ApplyLearning<1, Length> {
    template <typename... TTypes, typename... TInputs>
    static inline void call(std::tuple<TTypes...>& layers) {
        std::get<Length - 1>(layers).applyLearning();
    }
};

template <int I, int Length = I>
struct Learn {    
    template <typename... TTypes, typename... TInputs, typename... TTarget>
    static inline auto call(std::tuple<TTypes...>& layers, const std::tuple<TInputs...>& input, const std::tuple<TTarget...>& target)
    -> decltype(std::get<Length - I>(layers)(input, Learn<I - 1, Length>::call(layers, std::get<Length - I>(layers)(input), target))) {
        return std::get<Length - I>(layers)(input, Learn<I - 1, Length>::call(layers, std::get<Length - I>(layers)(input), target));
    }
};

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

template <typename, typename, typename, typename>
class NeuralNetImpl;

template <int... Input, int... Output, int... Layers, int... LayerInputs>
class NeuralNetImpl<Sequence<Input...>, Sequence<Output...>, Sequence<Layers...>, Sequence<LayerInputs...>> {

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

template <int InputWidth, int... Widths>
class NeuralNet :
public NeuralNetImpl<
    // This is our sequence for our input vector width
    typename GenerateSequence<InputWidth>::type,
    // This is our sequence for our output vector's width
    typename GenerateSequence<Last<Widths...>::value>::type,
    // This is the width value of each of our network layers
    Sequence<Widths...>,
    // This is the width of each network layers previous layer (therefore it's input width)
    typename FirstN<sizeof...(Widths), InputWidth, Widths...>::type
> {};

#endif
