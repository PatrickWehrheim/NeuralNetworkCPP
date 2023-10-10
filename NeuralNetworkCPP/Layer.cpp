#pragma once
#include "Layer.h"
#include "Utilities.h"
#include <bitset>
#include <powerbase.h>

const float LR = 0.001f;
const float E = 2.71828f;

Layer::Layer(const int inputSize, const int layerSize)
{
	_layerInputValues = new std::vector<float>();
	_layerOutputValues = new std::vector<float>();
	_weightValues = new std::vector<float>();

	for (int i = 0; i < layerSize; i++)
	{
		_layerOutputValues->push_back(0);
		_layerInputValues->push_back(0);
	}

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> distr(-1, 1);

	for (int i = 0; i < inputSize * layerSize; i++)
	{
		_weightValues->push_back(distr(gen));
	}
}

/// <summary>
/// Calculates the new weights and gets the output with the activationfunction ReLu
/// </summary>
/// <param name="input">Outputs of the previous layer as input</param>
void Layer::CalculateLayerValuesReLU(std::vector<float>* input)
{
	int weightCount = 0;
	for (int layerCount = 0; layerCount < _layerOutputValues->size(); layerCount++)
	{
		float value = 0;
		for (int inpCount = 0; inpCount < input->size(); inpCount++)
		{
			value += input->at(inpCount) * _weightValues->at(weightCount);
			weightCount++;
		}
		_layerInputValues->at(layerCount) = value;
		_layerOutputValues->at(layerCount) = Utilities::ReLU(value);
	}
}

/// <summary>
/// Calculates the new weights and gets the output with the activationfunction Sigmoid
/// </summary>
/// <param name="input">Outputs of the previous layer as input</param>
void Layer::CalculateLayerValuesSigmoid(std::vector<float>* input)
{
	int weightCount = 0;
	for (int layerCount = 0; layerCount < _layerOutputValues->size(); layerCount++)
	{
		float value = 0;
		for (int inpCount = 0; inpCount < input->size(); inpCount++)
		{
			value += input->at(inpCount) * _weightValues->at(weightCount);
			weightCount++;
		}
		_layerInputValues->at(layerCount) = value;
		_layerOutputValues->at(layerCount) = Utilities::Sigmoid(value);
	}
}

/// <summary>
/// Calculates the new weights and gets the output with the activationfunction Softmax
/// </summary>
/// <param name="input">Outputs of the previous layer as input</param>
void Layer::CalculateLayerValuesSoftmax(std::vector<float>* input)
{
	int weightCount = 0;
	for (int layerCount = 0; layerCount < _layerOutputValues->size(); layerCount++)
	{
		float value = 0;
		float softmaxSum = 0;
		for (int inpCount = 0; inpCount < input->size(); inpCount++)
		{
			value += input->at(inpCount) * _weightValues->at(weightCount);
			softmaxSum += std::powf(E, value);
			weightCount++;
		}
		_layerInputValues->at(layerCount) = value;
		_layerOutputValues->at(layerCount) = Utilities::Softmax(value, softmaxSum);
	}
}

/// <summary>
/// Sets the layerOutputValues based on all pixels in the given image
/// </summary>
/// <param name="image">Image with 784 Pixels</param>
void Layer::SetLayerValues(std::vector<std::bitset<CHAR_BIT>> image)
{
	for (int j = 0; j < _layerOutputValues->size(); j++)
	{
		float pixel = image.at(j).to_ulong();
		_layerOutputValues->at(j) = pixel / 255.f;
	}
}

/// <summary>
/// Sets the layerOutputValues
/// </summary>
/// <param name="values">New values</param>
void Layer::SetLayerValues(std::vector<float> values)
{
	for (int i = 0; i < _layerOutputValues->size(); i++)
	{
		_layerOutputValues->at(i) = values.at(i);
	}
}

/// <summary>
/// Sets the weights of the model
/// </summary>
/// <param name="values">New values</param>
void Layer::SetWeightValues(std::vector<float> weights)
{
	for (int i = 0; i < _layerOutputValues->size(); i++)
	{
		_weightValues->at(i) = weights.at(i);
	}
}

/// <summary>
/// Gets the layerOutputValues
/// </summary>
/// <returns>Current outputValues</returns>
auto Layer::GetLayerOutputValues() const -> std::vector<float>*
{
	return _layerOutputValues;
}

/// <summary>
/// Gets the layerInputValues
/// </summary>
/// <returns>Current inputValues</returns>
auto Layer::GetLayerInputValues() const -> std::vector<float>*
{
	return _layerInputValues;
}

/// <summary>
/// Calculates the new weight values based on the loss
/// </summary>
/// <param name="loss">The current error value</param>
/// <param name="sizeLastLayer">The size of the previous layer</param>
void Layer::Backpropagation(float loss, float sizeLastLayer)
{
	float lernParam = LR * loss;
	float learnValue = 0;

	int weightCount = 0;
	for (int layerIndex = 0; layerIndex < _layerOutputValues->size(); layerIndex++)
	{
		for (int i = 0; i < sizeLastLayer; i++)
		{
			learnValue = _layerOutputValues->at(layerIndex) * lernParam;
			_weightValues->at(layerIndex) += learnValue;
			weightCount++;
		}
	}
}
