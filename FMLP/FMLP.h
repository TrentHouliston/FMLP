#ifndef FMLP_FMLP_H_
#define FMLP_FMLP_H_

#include <ratio>
#include "Support.h"
#include "Internal/NeuralNetwork.h"
#include "Internal/ValidationChecks.h"

namespace FMLP {
    
    /**
     * @brief Declares the configuration options for a layer in the FMLP.
     *
     * @details
     *  @todo DOCUMENT
     *
     * @tparam Width        The width of this layer (the number of neurons).
     * @tparam Activation   The activation function object that holds static functions func(double) and dfunc(double) which
     *                          are used in the activation of the neurons of this layer. Defaults to Sigmoid
     * @tparam LearningRate An std::ratio object expressing the learning rate as a fraction. defaults to 1/20 (0.05)
     * @tparam Momentum     An std::ratio object expressing the momentum kept for the next delta as a fraction. defaults to 1/10 (0.1)
     * @tparam RNG          A class with a static function rand() that returns a random number for this layers inital widths. defaults to DefaultRNG
     */
    template <int Width, typename Activation = Sigmoid, typename LearningRate = std::ratio<1, 20>, typename Momentum = std::ratio<1, 10>, typename RNG = DefaultRNG>
    struct LayerConfig {
        
        // This static assertation checks that the LearningRate is an std::ratio object
        static_assert(Internal::IsRatio<LearningRate>::value, "The learning rate must be expressed as an std::ratio object");
        // This static assertation checks that the Momentum is an std::ratio object
        static_assert(Internal::IsRatio<Momentum>::value, "The momentum must be expressed as an std::ratio object");
        
        // TODO add static assertations to make sure that the activation function has a double func(double) and double dfunc(double) methods
        // TODO add static asserts to make sure that RNG has a double rand(...) function
    };
    
    // Hide our base template class from doxygen
    /// @cond HIDE
    template <typename...> class FMLPAdvanced; // @endcond
    
    /**
     * @brief Allows advanced configuration of the options for each layer in the neural network.
     *
     * @details
     *  @todo DOCUMENT
     *
     * @code{.cpp}
     *  
     * @endcode
     *
     * @tparam InputWidth
     * @tparam InputActivation
     * @tparam Widths
     * @tparam Activations
     */
    template <int InputWidth, typename InputActivation, int... Widths, typename... Activations>
    class FMLPAdvanced<LayerConfig<InputWidth, InputActivation>, LayerConfig<Widths, Activations>...> :
    public Internal::FMLPTransform<
        /// @todo Document
        typename Internal::FirstN<sizeof...(Widths), InputWidth, Widths...>::type,
        /// @todo Document
        LayerConfig<InputWidth, InputActivation>,
        /// @todo Document
        LayerConfig<Widths, Activations>...
    > {};
    
    /**
     * @brief The basic options of the neural network, allows
     *
     * @details
     *  @todo DOCUMENT
     *
     * @code{.cpp}
     *  FMLP::FMLP<2, 3, 1> net;
     * @endcode
     *
     * @tparam Widths   The width of each of the layers in the system, starting with the input layer and ending with the output.
     */
    template <int... Widths>
    class FMLP :
    public FMLPAdvanced<LayerConfig<Widths>...> {};
    
}

#endif
