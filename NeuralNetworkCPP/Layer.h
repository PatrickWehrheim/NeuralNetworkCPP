#pragma once
#include <vector>
#include <random>
#include <bitset>

class Layer
{
public:
	Layer(const int inputSize, const int layerSize);

	void CalculateLayerValuesReLU(std::vector<float>* input);
	void CalculateLayerValuesSigmoid(std::vector<float>* input);
	void CalculateLayerValuesSoftmax(std::vector<float>* input);

	void SetLayerValues(std::vector<std::bitset<CHAR_BIT>> image);
	void SetLayerValues(std::vector<float> values);

	void SetWeightValues(std::vector<float> weights);

	auto GetLayerInputValues() const -> std::vector<float>*;
	auto GetLayerOutputValues() const -> std::vector<float>*;

	void Backpropagation(float loss, float sizeLastLayer);

private:
	std::vector<float>* _layerInputValues;
	std::vector<float>* _layerOutputValues;
	std::vector<float>* _weightValues;
};

