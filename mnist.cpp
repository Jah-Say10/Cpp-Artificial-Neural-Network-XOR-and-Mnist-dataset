#include "neuralnetwork.hpp"
#include "utils.hpp"

/*

    Author   : Omar DIASSE
    Location : Dakar, Senegal
    Contact  : diasseomar10@gmail.com

    It's my neural network class with c++ language

    Compilation: g++ mnist.cpp neuralnetwork.cpp utils.cpp -Wall -std=c++17 -o prog

*/


int main()
{
    NeuralNetwork n(784, 100, 10, .1, "relu", "sigmoid");

    // Input data
    std::vector<std::vector<float>> mninst = getFromCSV("data/mnist_train_100.csv", ',');
    std::vector<std::vector<float>> input = getInputFromMat(mninst);
    std::vector<std::vector<float>> target = getTargetFromMat(mninst, 0, 10);

    for(int i = 0; i < (int)input.size(); i++)
    {
        for(int j = 0; j < (int)input[0].size(); j++)
        {
            input[i][j] = (input[i][j] / 255.0 * .99) + .01;
        }
    }

    for(int i = 0; i < (int)target.size(); i++)
    {
        for(int j = 0; j < (int)target[0].size(); j++)
        {
            target[i][j] = .01;
            target[i][(int)mninst[i][0]] = .99;
        }
    }

    const int epochs = 5;
    for(int e = 0; e < epochs; e++)
    {
        std::cout << "Epochs: " << e+1 << std::endl;
        for(int i = 0; i < (int)input.size(); i++)
        {
            n.train(make2d(input[i]), make2d(target[i]));
        }
    }

    std::vector<std::vector<float>> mninstTest = getFromCSV("data/mnist_test_10.csv", ',');
    std::vector<std::vector<float>> inputTest = getInputFromMat(mninstTest);
    std::vector<std::vector<float>> targetTest = getTargetFromMat(mninstTest, 0, 10);

    for(int i = 0; i < (int)inputTest.size(); i++)
    {
        for(int j = 0; j < (int)inputTest[0].size(); j++)
        {
            inputTest[i][j] = (inputTest[i][j] / 255.0 * .99) + .01;
        }
    }

    for(int i = 0; i < (int)targetTest.size(); i++)
    {
        for(int j = 0; j < (int)targetTest[0].size(); j++)
        {
            targetTest[i][j] = .01;
            targetTest[i][(int)mninstTest[i][0]] = .99;
        }
    }

    for(int i = 0; i < (int)inputTest.size(); i++)
    {
        std::cout << "Target" << std::endl;
        n.showMat(targetTest[i]);

        std::vector<std::vector<float>> val = n.query(make2d(inputTest[i]), make2d(targetTest[i]));
        std::cout << "Output" << std::endl;
        n.showMat(val);
    }

    std::cin.get();
    return 0;
}