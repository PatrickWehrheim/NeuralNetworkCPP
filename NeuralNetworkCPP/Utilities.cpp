#pragma once
#include "Utilities.h"
#include <fstream>
#include <iostream>
#include <powerbase.h>

const float E = 2.71828f;

/// <summary>
/// Activation function | The values in the neurons can't be less then 0
/// </summary>
/// <param name="number">The current value</param>
/// <returns>Returns the given number or 0</returns>
auto Utilities::ReLU(float number) -> const float
{
	if (number > 0)
	{
		return number;
	}

	return 0;
}

/// <summary>
/// Activation function | The values in the neurons can't be less then 0 
/// </summary>
/// <param name="number">The current value</param>
/// <returns>Returns the given number or 0</returns>
auto Utilities::Sigmoid(float number) -> const float
{
	return 1 / (1 + std::powf(E, -number));
}

/// <summary>
/// Activation function | The values are set to the percantage of the biggest number
/// </summary>
/// <param name="number">The current value</param>
/// <param name="sumNumber">The sum of all values</param>
/// <returns>The percentage of the number based on all values</returns>
auto Utilities::Softmax(float number, float sumNumber) -> const float
{
	return std::pow(E, number) / sumNumber;
}

/// <summary>
/// Get the file at the given path
/// </summary>
/// <param name="filePath">The filepath</param>
/// <returns>The loaded file as a string</returns>
std::string Utilities::GetFile(std::string filePath)
{
	std::string content;
	auto stream = std::fstream{ };

	stream.open(filePath, std::ios::in);

	if (!stream.is_open())
	{
		std::cout << "Could not read file! " << filePath << " File does not exist!" << std::endl;

		return std::string();
	}

	std::string line;

	while (!stream.eof())
	{
		std::getline(stream, line);
		content.append(line + "\n");
	}

	stream.close();

	return content;
}

/// <summary>
/// Saves the File at the given path
/// </summary>
/// <param name="filePath">The filepath</param>
/// <param name="content">The string to save as a file</param>
void Utilities::SaveFile(std::string filePath, std::string content)
{
	auto stream = std::fstream{ };

	stream.open(filePath, std::ios::out);

	if (!stream.is_open())
	{
		std::cout << "Could not read file! " << filePath << " File does not exist!" << std::endl;
	}

	stream << content;

	stream.close();
}
