#pragma once

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

std::vector<std::vector<float>> make2d(std::vector<float> array);

std::vector<std::vector<float>> getFromCSV(std::string file, char separator);

std::vector<std::vector<float>> getInputFromMat(std::vector<std::vector<float>> mat);

std::vector<std::vector<float>> getTargetFromMat(std::vector<std::vector<float>> mat, int index, int size);

int getMaxIndex(std::vector<float> array);

int sum(std::vector<int> array);