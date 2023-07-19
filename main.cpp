#include "neuralnetwork.hpp"
#include "utils.hpp"

/*

    Author: Omar DIASSE
    Location : Dakar, Senegal
    Contact  : diasseomar10@gmail.com
    
    It's my neural network class with c++ language

    Compilation: g++ main.cpp neuralnetwork.cpp utils.cpp -Wall -std=c++17 -o prog

*/


int main()
{
    NeuralNetwork n(3, 4, 1, .2);

    // Input data
    std::vector<std::vector<float>> input =
    {
        {1.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, 0.0f},
        {1.0f, 1.0f, 1.0f}
    };

    // Output data
    std::vector<std::vector<float>> target = 
    {
        {0.0f},
        {1.0f},
        {1.0f},
        {0.0f}
    };

    const int epochs = 3000;
    for(int e = 0; e < epochs; e++)
    {
        std::cout << "Epochs: " << e+1 << std::endl;
        for(int i = 0; i < (int)input.size(); i++)
        {
            n.train(make2d(input[i]), make2d(target[i]));
        }
    }

    for(int i = 0; i < (int)input.size(); i++)
    {
        std::cout << "Input" << std::endl;
        n.showMat(input[i]);

        std::vector<std::vector<float>> val = n.query(make2d(input[i]), make2d(target[i]));
        std::cout << "Output" << std::endl;
        n.showMat(val);
    }

    // Perfomance

    std::cin.get();
    return 0;
}