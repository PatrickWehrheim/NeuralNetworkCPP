#pragma once
#include <string>
#include "Layer.h"

class Utilities
{
public:
	static auto ReLU(float number) -> const float;
	static auto Sigmoid(float number) -> const float;
	static auto Softmax(float number, float sumNumber) -> const float;
	static std::string GetFile(std::string filePath);
	static void SaveFile(std::string filePath, std::string content);
};

