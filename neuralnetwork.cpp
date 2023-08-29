#include "neuralnetwork.hpp"

NeuralNetwork::NeuralNetwork(int input, int hidden, int output, float lr, std::string hiddenActivation, std::string outputActivation) :
    m_numInput(input), m_numHidden(hidden), m_numOutput(output), m_lr(lr), m_hiddenActivation(hiddenActivation), m_outputActivation(outputActivation)
{
    // Weight initialization
    srand((unsigned)time(NULL));

    m_wih.resize(m_numInput, std::vector<float> (m_numHidden));
    for(int i = 0; i < (int)m_wih.size(); i++)
    {
        for(int j = 0; j < (int)m_wih[0].size(); j++)
        {
            m_wih[i][j] = ((float)rand() / RAND_MAX) - .5;
        }
    }

    m_who.resize(m_numHidden, std::vector<float> (m_numOutput));
    for(int i = 0; i < (int)m_who.size(); i++)
    {
        for(int j = 0; j < (int)m_who[0].size(); j++)
        {
            m_who[i][j] = ((float)rand() / RAND_MAX) - .5;
        }
    }
}

void NeuralNetwork::train(std::vector<std::vector<float>> input, std::vector<std::vector<float>> target)
{
    // Forward pass
    std::vector<std::vector<float>> hidden = dotProduct(input, m_wih);
    if(m_hiddenActivation == "sigmoid")
        sigmoid(hidden);
    else if(m_hiddenActivation == "relu")
        relu(hidden);

    // Output
    std::vector<std::vector<float>> output = dotProduct(hidden, m_who);
    if(m_outputActivation == "sigmoid")
        sigmoid(output);
    else if(m_outputActivation == "relu")
        relu(output);
    std::cout << "Outputs" << std::endl;
    showMat(output);

    // Error
    std::vector<std::vector<float>> error = matOp(target, output, '-');
    std::cout << "Error: " << std::endl;
    showMat(error);

    // Hidden errors
    std::vector<std::vector<float>> hiddenError = dotProduct(m_who, transpose(error));

    // Backpropagation
    backpropagation(m_who, error, output, hidden, m_lr, m_outputActivation); // Output
    backpropagation(m_wih, transpose(hiddenError), hidden, input, m_lr, m_hiddenActivation); // Input

    std::cout << "----------------" << std::endl;
}

std::vector<std::vector<float>> NeuralNetwork::query(std::vector<std::vector<float>> input, std::vector<std::vector<float>> target)
{
    std::vector<std::vector<float>> hidden = dotProduct(input, m_wih);
    if(m_hiddenActivation == "sigmoid")
        sigmoid(hidden);
    else if(m_hiddenActivation == "relu")
        relu(hidden);
    
    std::vector<std::vector<float>> output = dotProduct(hidden, m_who);
    if(m_outputActivation == "sigmoid")
        sigmoid(output);
    else if(m_outputActivation == "relu")
        relu(output);

    return output;
}

void NeuralNetwork::backpropagation(std::vector<std::vector<float>> &weight, std::vector<std::vector<float>> error, std::vector<std::vector<float>> output, std::vector<std::vector<float>> input, float lr, std::string activation)
{
   std::cout << activation << std::endl;

    if(activation == "sigmoid")
        sigmoidDerivative(output);
    else if(activation == "relu")
        reluDerivative(output);

    std::vector<std::vector<float>> tmpConst = elementWise(error, output);

    std::vector<std::vector<float>> der = weight;
    for(int i = 0; i < (int)der.size(); i++)
    {
        for(int j = 0; j < (int)der[0].size(); j++)
        {
            der[i][j] = transpose(input)[i][0] * transpose(tmpConst)[j][0];
        };
    }
    broadcast(der, lr);

    weight = matOp(weight, der, '+');
}

float NeuralNetwork::sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}

float NeuralNetwork::sigmoidDerivative(float x)
{
    return (x * (1 - x));
}

float NeuralNetwork::relu(float x)
{
    return (x < 0) ? 0.0f : x;
}

float NeuralNetwork::reluDerivative(float x)
{
    return (x < 0) ? 0.0f : 1;
}

void NeuralNetwork::sigmoid(std::vector<float> &m)
{
    for(int i = 0; i < (int)m.size(); i++)
    {
        m[i] = sigmoid(m[i]);
    }
}

void NeuralNetwork::sigmoid(std::vector<std::vector<float>> &m)
{
    for(int i = 0; i < (int)m.size(); i++)
    {
        for(int j = 0; j < (int)m[0].size(); j++)
        {
            m[i][j] = sigmoid(m[i][j]);
        }
    }
}

void NeuralNetwork::sigmoidDerivative(std::vector<float> &m)
{
    for(int i = 0; i < (int)m.size(); i++)
    {
        m[i] = sigmoidDerivative(m[i]);
    }
}

void NeuralNetwork::sigmoidDerivative(std::vector<std::vector<float>> &m)
{
    for(int i = 0; i < (int)m.size(); i++)
    {
        for(int j = 0; j < (int)m[0].size(); j++)
        {
            m[i][j] = sigmoidDerivative(m[i][j]);
        }
    }
}

void NeuralNetwork::relu(std::vector<std::vector<float>> &m)
{
    for(int i = 0; i < (int)m.size(); i++)
    {
        for(int j = 0; j < (int)m[0].size(); j++)
        {
            m[i][j] = relu(m[i][j]);
        }
    }
}

void NeuralNetwork::reluDerivative(std::vector<std::vector<float>> &m)
{
    for(int i = 0; i < (int)m.size(); i++)
    {
        for(int j = 0; j < (int)m[0].size(); j++)
        {
            m[i][j] = reluDerivative(m[i][j]);
        }
    }
}

// 2 matrices
std::vector<std::vector<float>> NeuralNetwork::dotProduct(std::vector<std::vector<float>> m1, std::vector<std::vector<float>> m2)
{
    if((int)m1[0].size() != (int)m2.size())
    {
        return {{-1.0f}};
    }

    std::vector<std::vector<float>> res;
    res.resize((int)m1.size(), std::vector<float> ((int)m2[0].size()));

    for(int i = 0; i < (int)res.size(); i++)
    {
        for(int j = 0; j < (int)res[0].size(); j++)
        {
            res[i][j] = dotProduct(m1[i], getColumnFromMat(m2, j));
        }
    }

    return res;
}

// a matrice and a vector
std::vector<float> NeuralNetwork::dotProduct(std::vector<std::vector<float>> m, std::vector<float> v)
{
    if((int)m[0].size() != (int)v.size())
    {
        return {{-1.0f}};
    }

    std::vector<float> res = v;

    for(int i = 0; i < (int)res.size(); i++)
    {
        res[i] = dotProduct(m[i], v);
    }

    return res;
}

// a vector and matrice
std::vector<float> NeuralNetwork::dotProduct(std::vector<float> v, std::vector<std::vector<float>> m)
{
    if((int)m[0].size() != (int)v.size())
    {
        return {{-1.0f}};
    }

    std::vector<float> res = v;

    for(int i = 0; i < (int)res.size(); i++)
    {
        res[i] = dotProduct(v, getColumnFromMat(m, i));
    }

    return res;
}

// 2 vectors
float NeuralNetwork::dotProduct(std::vector<float> v1, std::vector<float> v2)
{
    if((int)v1.size() != (int)v2.size())
    {
        return -1.0f;
    }

    float val = 0.0f;
    for(int i = 0; i < (int)v1.size(); i++)
    {
        val += v1[i] * v2[i];
    }

    return val;
}

std::vector<std::vector<float>> NeuralNetwork::elementWise(std::vector<std::vector<float>> m1, std::vector<std::vector<float>> m2)
{
    std::vector<std::vector<float>> res = m1;

    for(int i = 0; i < (int)res.size(); i++)
    {
        for(int j = 0; j < (int)res[0].size(); j++)
        {
            res[i][j] = m1[i][j] * m2[i][j];
        }
    }

    return res;
}

std::vector<float> NeuralNetwork::elementWise(std::vector<float> v1, std::vector<float> v2)
{
    std::vector<float> res = v1;

    for(int i = 0; i < (int)res.size(); i++)
    {
        res[i] = v1[i] * v2[i];
    }

    return res;
}

void NeuralNetwork::broadcast(std::vector<std::vector<float>> &m, float x)
{
    for(int i = 0; i < (int)m.size(); i++)
    {
        for(int j = 0; j < (int)m[0].size(); j++)
        {
            m[i][j] = m[i][j] * x;
        }
    }
}

void NeuralNetwork::broadcast(std::vector<float> &m, float x)
{
    for(int i = 0; i < (int)m.size(); i++)
    {
        m[i] = m[i] * x;
    }
}

std::vector<std::vector<float>> NeuralNetwork::matOp(std::vector<std::vector<float>> m1, std::vector<std::vector<float>> m2, char op)
{
    std::vector<std::vector<float>> res = m1;

    if(op == '+')
    {
        for(int i = 0; i < (int)res.size(); i++)
        {
            for(int j = 0; j < (int)res[0].size(); j++)
            {
                res[i][j] = m1[i][j] + m2[i][j];
            }
        }
    }
    else if(op == '-')
    {
        for(int i = 0; i < (int)res.size(); i++)
        {
            for(int j = 0; j < (int)res[0].size(); j++)
            {
                res[i][j] = m1[i][j] - m2[i][j];
            }
        }
    }

    return res;
}

std::vector<float> NeuralNetwork::matOp(std::vector<float> v1, std::vector<float> v2, char op)
{
    std::vector<float> res = v1;

    if(op == '+')
    {
        for(int i = 0; i < (int)res.size(); i++)
        {
            res[i] = v1[i] + v2[i];
        }
    }
    else if (op == '-')
    {
        for(int i = 0; i < (int)res.size(); i++)
        {
            res[i] = v1[i] - v2[i];
        }
    }

    return res;
}

std::vector<std::vector<float>> NeuralNetwork::transpose(std::vector<std::vector<float>> m)
{
    std::vector<std::vector<float>> d;

    for(int i = 0; i < (int)m[0].size(); i++)
    {
        std::vector<float> s;
        for(int j = 0; j < (int)m.size(); j++)
        {
            s.push_back(m[j][i]);
        }
        d.push_back(s);
    }

    return d;
}

std::vector<float> NeuralNetwork::getColumnFromMat(std::vector<std::vector<float>> mat, int index)
{
    std::vector<float> column;

    for(int i = 0; i < (int)mat.size(); i++)
    {
        column.push_back(mat[i][index]);
    }

    return column;
}

std::vector<std::vector<float>> NeuralNetwork::getWeight(char w)
{
    if(w == 'h')
        return m_wih;
    else if(w == 'o')
        return m_who;

    return {{-1}};
}


void NeuralNetwork::showMat(std::vector<std::vector<float>> m)
{
    for(int i = 0; i < (int)m.size(); i++)
    {
        for(int j = 0; j < (int)m[0].size(); j++)
        {
            std::cout << m[i][j] << "|";
        }
        std::cout << std::endl;
    }
}

void NeuralNetwork::showMat(std::vector<float> v)
{
    for(int i = 0; i < (int)v.size(); i++)
    {
        std::cout << v[i] << "|";
    }
    std::cout << std::endl;
}