#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <string>

class NeuralNetwork
{
    private:
        int m_numInput;
        int m_numHidden;
        int m_numOutput;
        float m_lr = .1f;

        std::string m_hiddenActivation;
        std::string m_outputActivation;

        std::vector<std::vector<float>> m_wih;
        std::vector<std::vector<float>> m_who;

        float sigmoid(float x);
        float sigmoidDerivative(float x);
        float relu(float x);
        float reluDerivative(float x);
        void sigmoid(std::vector<float> &m);
        void sigmoid(std::vector<std::vector<float>> &m);
        void sigmoidDerivative(std::vector<float> &m);
        void sigmoidDerivative(std::vector<std::vector<float>> &m);
        void relu(std::vector<std::vector<float>> &m);
        void reluDerivative(std::vector<std::vector<float>> &m);

    public:
        NeuralNetwork(int input, int hidden, int output, float lr, std::string hiddenActivation, std::string outputActivation);

        void train(std::vector<std::vector<float>> input, std::vector<std::vector<float>> target);
        std::vector<std::vector<float>> query(std::vector<std::vector<float>> input, std::vector<std::vector<float>> target);
        void backpropagation(std::vector<std::vector<float>> &weight, std::vector<std::vector<float>> error, std::vector<std::vector<float>> output, std::vector<std::vector<float>> input, float lr, std::string activation);

        std::vector<std::vector<float>> dotProduct(std::vector<std::vector<float>> m1, std::vector<std::vector<float>> m2);
        std::vector<float> dotProduct(std::vector<std::vector<float>> m, std::vector<float> v);
        std::vector<float> dotProduct(std::vector<float> v, std::vector<std::vector<float>> m);
        float dotProduct(std::vector<float> v1, std::vector<float> v2);
        std::vector<std::vector<float>> elementWise(std::vector<std::vector<float>> m1, std::vector<std::vector<float>> m2);
        std::vector<float> elementWise(std::vector<float> v1, std::vector<float> v2);
        void broadcast(std::vector<std::vector<float>> &m, float x);
        void broadcast(std::vector<float> &m, float x);
        std::vector<std::vector<float>> matOp(std::vector<std::vector<float>> m1, std::vector<std::vector<float>> m2, char op);
        std::vector<float> matOp(std::vector<float> v1, std::vector<float> v2, char op);
        std::vector<std::vector<float>> transpose(std::vector<std::vector<float>> m);
        
        std::vector<float> getColumnFromMat(std::vector<std::vector<float>> mat, int index);
        std::vector<std::vector<float>> getWeight(char w);

        void showMat(std::vector<std::vector<float>> m);
        void showMat(std::vector<float> v);
};