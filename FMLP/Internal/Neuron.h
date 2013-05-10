#ifndef FMLP_INTERNAL_NEURON_H_
#define FMLP_INTERNAL_NEURON_H_

#include <tuple>
#include "Utility.h"

namespace FMLP {
    namespace Internal {
        
        // Hide the unspecialized class from doxygen
        /// @cond HIDDEN
        template <typename, typename, typename, typename, typename> class Neuron; /// @endcond
        
        /**
         * @todo Document
         */
        template <int... Values, typename TActivation, typename LearningRate, typename Momentum, typename RNG>
        class Neuron<Sequence<Values...>, TActivation, LearningRate, Momentum, RNG> {
        private:
            /// @todo Document
            std::tuple<decltype(double(Values))...> weights = std::make_tuple((static_cast<void>(Values), RNG::rand())...);
            /// @todo Document
            std::tuple<decltype(double(Values))...> deltas = std::make_tuple(static_cast<double>(Values - Values)...);
            
            /**
             * @todo Document
             */
            constexpr double fraction(int numerator, int denominator) {
                return static_cast<double>(numerator) / static_cast<double>(denominator);
            }
            
        public:
            
            /**
             * @todo Document
             */
            double operator()(std::tuple<decltype(double(Values))...> input) {
                return TActivation::func(sum((std::get<Values>(weights) * std::get<Values>(input))...));
            }
            
            /**
             * @todo Document
             */
            std::tuple<decltype(double(Values))...> operator()(const std::tuple<decltype(double(Values))...>& input, const double childError) {
                
                const double derivitive = TActivation::dfunc(sum((std::get<Values>(weights) * std::get<Values>(input))...));
                const auto ret = std::make_tuple((childError * std::get<Values>(weights))...);
                const auto delta = std::make_tuple((childError * derivitive * std::get<Values>(input) * fraction(LearningRate::num, LearningRate::den))...);
                unpack((std::get<Values>(deltas) -= std::get<Values>(delta))...);
                
                return ret;
            }
            
            /**
             * @todo Document
             */
            bool applyLearning() {
                
                // Apply our sum deltas to the weights
                unpack((std::get<Values>(weights) += std::get<Values>(deltas))...);
                
                // Apply momentum to the deltas
                unpack((std::get<Values>(deltas) *= fraction(Momentum::num, Momentum::den))...);
                
                // This is to simplify the unpacking of this function
                return true;
            }
        };
    }
}

#endif
