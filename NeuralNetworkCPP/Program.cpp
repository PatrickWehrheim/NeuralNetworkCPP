#pragma once
#include "NeuralNetwork.h"
#include "Console.h"
#include <iostream>

int main() {

	NeuralNetwork* nn = new NeuralNetwork();

	std::cout << "Hello" << std::endl;
	if (Console::WriteYNQuestion("Would you like to train the model?"))
	{
		nn->Train(1, 60000);
	}

	while (true)
	{
		if (Console::WriteYNQuestion("Do you want to draw a number?"))
		{
			Console::DrawQuad();
			Console::UserDrawNumber();
			auto result = Console::GetPixels();
			nn->Run(result);
		}
		else 
		{
			break;
		}
	}

	delete nn;
	nn = nullptr;
}
