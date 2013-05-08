#include <iostream>
#include "FMLP/NeuralNetwork.h"

int main(int argc, const char * argv[])
{

    NeuralNet<2, 3, 1> net;
    
    for(int i = 0; i < 50; i++) {
        std::cout << "Iteration: " << i + 1 << std::endl;
        //std::cout << "Learning 0,0" << std::endl;
        net.learn(std::make_tuple(-1,-1), std::make_tuple(-1));
        //std::cout << std::endl << "Learning 0,1" << std::endl;
        net.learn(std::make_tuple(-1,1), std::make_tuple(1));
        //std::cout << std::endl << "Learning 1,0" << std::endl;
        net.learn(std::make_tuple(1,-1), std::make_tuple(1));
        //std::cout << std::endl << "Learning 1,1" << std::endl;
        net.learn(std::make_tuple(1,1), std::make_tuple(-1));
        //std::cout << std::endl;
        
        net.applyLearning();
        
        std::cout << "xor(0, 0) = " << std::get<0>(net.classify(std::make_tuple(0, 0))) << std::endl;
        std::cout << "xor(0, 1) = " << std::get<0>(net.classify(std::make_tuple(0, 1))) << std::endl;
        std::cout << "xor(1, 0) = " << std::get<0>(net.classify(std::make_tuple(1, 0))) << std::endl;
        std::cout << "xor(1, 1) = " << std::get<0>(net.classify(std::make_tuple(1, 1))) << std::endl;
        std::cout << std::endl;
    }
}

