#include "utils.hpp"

std::vector<std::vector<float>> make2d(std::vector<float> array)
{
    std::vector<std::vector<float>> mat;
    mat.push_back(array);
    return mat;
}

std::vector<std::vector<float>> getFromCSV(std::string file, char separator)
{
    std::vector<std::vector<float>> data;
    std::string handle;

    std::ifstream openFile(file);

    if(openFile.is_open())
    {
        while(openFile.good())
        {
            openFile >> handle;
            std::vector<float> tmpVector;
            
            std::stringstream ss(handle);
            while(ss.good())
            {
                std::string tmp;
                getline(ss, tmp, separator);
                tmpVector.push_back(std::stof(tmp));
            }

            data.push_back(tmpVector);
        }
    }
    data.pop_back();
    openFile.close();

    return data;
}

std::vector<std::vector<float>> getInputFromMat(std::vector<std::vector<float>> mat)
{
    std::vector<std::vector<float>> res = mat;

    for(int i = 0; i < (int)res.size(); i++)
    {
        res[i].erase(res[i].begin());
    }

    return res;
}

std::vector<std::vector<float>> getTargetFromMat(std::vector<std::vector<float>> mat, int index, int size)
{
    std::vector<std::vector<float>> res;
    res.resize((int)mat.size(), std::vector<float> (size));

    for(int i = 0; i < (int)mat.size(); i++)
    {
        res[i][0] = mat[i][index];
    }

    return res;
}

int getMaxIndex(std::vector<float> array)
{
    float max = array[0];
    int index = 0;

    for(int i = 1; i < (int)array.size(); i++)
    {
        if(array[i] > max)
        {
            max = array[i];
            index = i;
        }
    }

    return index;
}

int sum(std::vector<int> array)
{
    int sum = 0;
    for(int i = 0; i < (int)array.size(); i++)
        sum += array[i];

    return sum;
}