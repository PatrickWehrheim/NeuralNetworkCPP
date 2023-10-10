#pragma once
#include "NeuralNetwork.h"
#include "Utilities.h"
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <limits.h>
#include <Windows.h>
#include <string>
#include <direct.h>
#include <tuple>
#include <bitset>
#include <vector>
#include <random>

NeuralNetwork::NeuralNetwork(){}

NeuralNetwork::~NeuralNetwork()
{
	delete _output;
	_output = nullptr;
	delete _hiddenLayer1;
	_hiddenLayer1 = nullptr;
	delete _hiddenLayer0;
	_hiddenLayer0 = nullptr;
	delete _input;
	_input = nullptr;
}

/// <summary>
/// Train a new Model in the MNIST Dataset.
/// The training runs on the CPU, so one epoch needs
/// about 45 to 60 Minutes!
/// </summary>
/// <param name="epochs">Number of full train circles</param>
/// <param name="steps">Number of single images (max at 60.000)</param>
void NeuralNetwork::Train(int epochs, int steps)
{
	char buff[_MAX_PATH];
	char* _ = _getcwd(buff, _MAX_PATH);
	string imageTrainPath = buff + static_cast<string>("/mnistDataset/train-images.idx3-ubyte");
	string labelTrainPath = buff + static_cast<string>("/mnistDataset/train-labels.idx1-ubyte");
	string imageValPath = buff + static_cast<string>("/mnistDataset/t10k-images.idx3-ubyte");
	string labelValPath = buff + static_cast<string>("/mnistDataset/t10k-labels.idx1-ubyte");

	cout << "Begin MNIST Read" << endl;

	_trainDataSet = ReadMNIST(imageTrainPath, labelTrainPath);
	_validationDataSet = ReadMNIST(imageValPath, labelValPath);

	cout << "End MNIST Read" << endl;

	vector<int>* trainLabels = get<0>(*_trainDataSet);
	vector<vector<bitset<CHAR_BIT>>>* trainImages = get<1>(*_trainDataSet);

	for (int epoch = 0; epoch < epochs; epoch++)
	{
		cout << "Epoch started: " << epoch << endl;

		for (int step = 0; step < steps; step++)
		{
			int label = trainLabels->at(step);
			vector<bitset<CHAR_BIT>> image = trainImages->at(step);

			_input->SetLayerValues(image);
			_hiddenLayer0->CalculateLayerValuesReLU(_input->GetLayerOutputValues());
			_hiddenLayer1->CalculateLayerValuesSigmoid(_hiddenLayer0->GetLayerOutputValues());
			_output->CalculateLayerValuesSoftmax(_hiddenLayer1->GetLayerOutputValues());

			int output = 0;
			float maxFloat = 0.f;

			int counter = 0;
			float crossSum = 0.f;
			for (const float& f : *_output->GetLayerOutputValues())
			{
				crossSum += (label * log(f)) + ((1 - label) * log((1 - f)));
				if (f > maxFloat)
				{
					output = counter;
					maxFloat = f;
				}
				counter++;
			}

			float realLoss = 0.5f * (label - output) * (label - output);
			float crossEntropyLoss = (static_cast<float>(1) / _output->GetLayerOutputValues()->size()) * crossSum;
			float loss = label - output;

			Backpropagation(crossEntropyLoss);

			float progress = static_cast<float>(step) / steps * 100;
			cout << "Progress: " << progress << "% (" << step << ")" << " Epoch: " << epoch << endl;
		}

		cout << "Epoch ended: " << epoch << endl;
	}

	Validation(10000);
}

/// <summary>
/// Validation of the Model
/// </summary>
/// <param name="steps">Number of single images (max at 10.000)</param>
void NeuralNetwork::Validation(int steps)
{
	if (!_validationDataSet)
	{
		char buff[_MAX_PATH];
		char* _ = _getcwd(buff, _MAX_PATH);

		string imageValPath = buff + static_cast<string>("/mnistDataset/t10k-images.idx3-ubyte");
		string labelValPath = buff + static_cast<string>("/mnistDataset/t10k-labels.idx1-ubyte");
		_validationDataSet = ReadMNIST(imageValPath, labelValPath);
	}
	vector<int>* validLabels = get<0>(*_validationDataSet);
	vector<vector<bitset<CHAR_BIT>>>* validImages = get<1>(*_validationDataSet);

	int correct = 0;

	cout << "Start Validation" << endl;
	for (int i = 0; i < steps; i++)
	{
		cout << "Epoch started: " << i << endl
			<< "Label is: " << validLabels->at(i) << endl;

		int label = validLabels->at(i);
		vector<bitset<CHAR_BIT>> image = validImages->at(i);

		_input->SetLayerValues(image);
		_hiddenLayer0->CalculateLayerValuesReLU(_input->GetLayerOutputValues());
		_hiddenLayer1->CalculateLayerValuesSigmoid(_hiddenLayer0->GetLayerOutputValues());
		_output->CalculateLayerValuesSoftmax(_hiddenLayer1->GetLayerOutputValues());

		for (int inpCount = 0; inpCount < _output->GetLayerOutputValues()->size(); inpCount++)
		{
			cout << "Result for Number " << inpCount << ": " << _output->GetLayerOutputValues()->at(inpCount) << endl;
		}

		int output = 0;
		float maxFloat = 0.f;

		int counter = 0;
		float crossSum = 0.f;
		for (const float& f : *_output->GetLayerOutputValues())
		{
			crossSum += (label * log(f)) + ((1 - label) * log((1 - f)));
			if (f > maxFloat)
			{
				output = counter;
				maxFloat = f;
			}
			counter++;
		}

		float realLoss = 0.5f * (label - output) * (label - output);
		float crossEntropyLoss = (static_cast<float>(1) / _output->GetLayerOutputValues()->size()) * crossSum;
		float loss = label - output;

		if (loss == 0)
		{
			correct++;
		}

		cout << "Validation Results:" << endl;
		cout << "Output: " << output << endl
			<< "Loss: " << loss << "(" << realLoss << ")" << endl << endl;

		float progress = static_cast<float>(i) / steps * 100;
		cout << "Progress: " << progress << "% (" << i << ")" << endl;
	}

	cout << "Correct Answers: " << correct << " of " << steps << endl;
}

/// <summary>
/// The neurons gets updated based on the loss
/// </summary>
/// <param name="loss">Calculated error</param>
void NeuralNetwork::Backpropagation(int loss)
{
	for (int i = 0; i < _output->GetLayerOutputValues()->size(); i++)
	{
		float crossEntropyDeri = -1 * ((i * (1 / _output->GetLayerOutputValues()->at(i))) + (1 - i) * (static_cast<float>(1) / 1 - _output->GetLayerOutputValues()->at(i)));
	}

	_hiddenLayer0->Backpropagation(loss, _input->GetLayerOutputValues()->size());
	_hiddenLayer1->Backpropagation(loss, _hiddenLayer0->GetLayerOutputValues()->size());
	_output->Backpropagation(loss, _hiddenLayer1->GetLayerOutputValues()->size());
}

/// <summary>
/// Pass an image in the Model to get a prediction. 
/// This don't have to be 100% correct since the model is
/// NOT optimized!
/// </summary>
/// <param name="image">The image passed in to get a prediction</param>
/// <returns>Returns the Result of the Model</returns>
uint32_t NeuralNetwork::Run(vector<bitset<CHAR_BIT>> image)
{
	_input->SetLayerValues(image);
	_hiddenLayer0->CalculateLayerValuesReLU(_input->GetLayerOutputValues());
	_hiddenLayer1->CalculateLayerValuesSigmoid(_hiddenLayer0->GetLayerOutputValues());
	_output->CalculateLayerValuesSoftmax(_hiddenLayer1->GetLayerOutputValues());

	for (int inpCount = 0; inpCount < _output->GetLayerOutputValues()->size(); inpCount++)
	{
		cout << "Result for Number " << inpCount << ": " << _output->GetLayerOutputValues()->at(inpCount) << endl;
	}

	int output = 0;
	float maxFloat = 0.f;

	int counter = 0;
	for (const float& f : *_output->GetLayerOutputValues())
	{
		if (f > maxFloat)
		{
			output = counter;
			maxFloat = f;
		}
		counter++;
	}

	cout << "Output: " << output << endl;
	return output;
}

/// <summary>
/// The Model is Saved as a file in "../save/model.h1"
/// </summary>
void NeuralNetwork::SaveModel()
{
	char buff[_MAX_PATH];
	char* _ = _getcwd(buff, _MAX_PATH);
	string saveFilePath = buff + static_cast<string>("/save/model.h1");

	ofstream saveFile(saveFilePath, ios::out | ios::binary);

	string input = "";
	int counter = 0;
	for (int i = 0; i < _input->GetLayerOutputValues()->size(); i++)
	{
		input += _input->GetLayerOutputValues()->at(i);
		if (i < _input->GetLayerOutputValues()->size() - 1)
		{
			input += " ";
		}
		counter++;
	}
	for (int i = 0; i < _hiddenLayer0->GetLayerOutputValues()->size(); i++)
	{
		input += _input->GetLayerOutputValues()->at(i);
		if (i < _input->GetLayerOutputValues()->size() - 1)
		{
			input += " ";
		}
		counter++;
	}
	for (int i = 0; i < _hiddenLayer1->GetLayerOutputValues()->size(); i++)
	{
		input += _input->GetLayerOutputValues()->at(i);
		if (i < _input->GetLayerOutputValues()->size() - 1)
		{
			input += " ";
		}
		counter++;
	}

	saveFile.write(input.c_str(), counter);
}

/// <summary>
/// The model gets loaded from "../save/model.h1"
/// </summary>
void NeuralNetwork::LoadModel()
{
	char buff[_MAX_PATH];
	char* _ = _getcwd(buff, _MAX_PATH);
	string saveFilePath = buff + static_cast<string>("/save/model.h1");

	ifstream saveFile(saveFilePath, ios::in | ios::binary);

	char output;
	vector<float>* weights = new vector<float>();

	for (int i = 0; _input->GetLayerOutputValues()->size(); i++)
	{
		saveFile.read(reinterpret_cast<char*>(&output), 1);
		weights->push_back(output);
	}
	_input->SetWeightValues(*weights);
	weights->clear();

	for (int i = 0; _hiddenLayer0->GetLayerOutputValues()->size(); i++)
	{
		saveFile.read(reinterpret_cast<char*>(&output), 1);
		weights->push_back(output);
	}
	_hiddenLayer0->SetWeightValues(*weights);
	weights->clear();

	for (int i = 0; _hiddenLayer1->GetLayerOutputValues()->size(); i++)
	{
		saveFile.read(reinterpret_cast<char*>(&output), 1);
		weights->push_back(output);
	}
	_hiddenLayer1->SetWeightValues(*weights);
	weights->clear();
}

/// <summary>
/// Reading MNIST Datasets
/// </summary>
shared_ptr<tuple<vector<int>*, vector<vector<bitset<CHAR_BIT>>>*>> NeuralNetwork::ReadMNIST(string imagePath, string labelPath)
{
	ifstream imageFile(imagePath, ios::in | ios::binary);
	ifstream labelFile(labelPath, ios::in | ios::binary);

	int magic;
	imageFile.read(reinterpret_cast<char*>(&magic), 4);
	magic = SwapEndian(magic);

	labelFile.read(reinterpret_cast<char*>(&magic), 4);
	magic = SwapEndian(magic);

	uint32_t itemCount = 0, labelCount = 0, rows = 0, cols = 0;
	imageFile.read(reinterpret_cast<char*>(&itemCount), 4);
	itemCount = SwapEndian(itemCount);
	labelFile.read(reinterpret_cast<char*>(&labelCount), 4);
	labelCount = SwapEndian(labelCount);

	imageFile.read(reinterpret_cast<char*>(&rows), 4);
	rows = SwapEndian(rows);
	imageFile.read(reinterpret_cast<char*>(&cols), 4);
	cols = SwapEndian(cols);

	cout << "image and label num is: " << itemCount << endl;
	cout << "image rows: " << rows << ", cols: " << cols << endl;

	char label;
	char pixels;
	vector<bitset<CHAR_BIT>> pixelBinary = vector<bitset<CHAR_BIT>>(rows * cols * CHAR_BIT);

	vector<int>* labels = new vector<int>(itemCount);
	vector<vector<bitset<CHAR_BIT>>>* images = new vector<vector<bitset<CHAR_BIT>>>(itemCount);

	for (int i = 0; i < itemCount; i++) 
	{
		labelFile.read(&label, 1);

		for (int p = 0; p < rows * cols; p++)
		{
			imageFile.read(&pixels, 1);
			bitset<CHAR_BIT> binary = bitset<CHAR_BIT>(pixels);
			for (int b = 0; b < CHAR_BIT; b++)
			{
				pixelBinary.at(p)[b] = binary[b];
			}
		}
		
		images->at(i) = pixelBinary;
		labels->at(i) = static_cast<int>(label);
	}

	cout << "Image 200 is: " << endl;

	vector<bitset<CHAR_BIT>> image = images->at(200);

	for (int j = 0; j < _input->GetLayerOutputValues()->size(); j++)
	{
		cout << " " << image.at(j).to_ulong();
	}
	cout << endl;

	imageFile.close();
	labelFile.close();

	tuple<vector<int>*, vector<vector<bitset<CHAR_BIT>>>*> result = make_tuple(labels, images);

	return make_shared<tuple<vector<int>*, vector<vector<bitset<CHAR_BIT>>>*>>(result);
}

/// <summary>
/// The Endian get swaped from Base8 to Base16
/// Do this to get the real numbers from a char
/// </summary>
/// <param name="val">Value to swap</param>
/// <returns>The real Number in Base16</returns>
uint32_t NeuralNetwork::SwapEndian(uint32_t val)
{
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
	return (val << 16) | (val >> 16);
}