#pragma once
#include <string>
#include <vector>
#include <bitset>

class Console
{
public:
	static bool WriteYNQuestion(std::string question);
	static std::vector<std::bitset<CHAR_BIT>> GetPixels();
	static void ShowProgressOfEpoch();
	static void DrawQuad();
	static void UserDrawNumber();
};

