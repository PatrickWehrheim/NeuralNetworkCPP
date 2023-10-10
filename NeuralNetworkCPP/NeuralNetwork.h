#pragma once
#include <cstdint>
#include <string>
#include <memory>
#include <bitset>
#include <tuple>
#include <array>
#include <vector>
#include "Layer.h"

using namespace std;

class NeuralNetwork
{
private:
	Layer* _input = new Layer(0, 784);
	Layer* _hiddenLayer0 = new Layer(784, 1568);
	Layer* _hiddenLayer1 = new Layer(1568, 196);
	Layer* _output = new Layer(196, 10);

	shared_ptr<tuple<vector<int>*, vector<vector<bitset<CHAR_BIT>>>*>> _trainDataSet;
	shared_ptr<tuple<vector<int>*, vector<vector<bitset<CHAR_BIT>>>*>> _validationDataSet;

public:
	NeuralNetwork();
	~NeuralNetwork();

	void Train(int epochs, int steps);
	void Validation(int steps);
	void Backpropagation(int loss);
	uint32_t Run(vector<bitset<CHAR_BIT>> image);

	void SaveModel();
	void LoadModel();

private:
	shared_ptr<tuple<vector<int>*, vector<vector<bitset<CHAR_BIT>>>*>> ReadMNIST(string imagePath, string labelPath);
	uint32_t SwapEndian(uint32_t val);
};
