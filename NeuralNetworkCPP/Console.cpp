#pragma once
#include "Console.h"
#include <iostream>
#include <cmath>
#include <windows.h>
#include <stdio.h>
#include <conio.h>
#include <vector>
#include <bitset>

#define WHITE_COLOR 112
#define BLACK_COLOR 7

#define gotoxy(x, y) printf("\033[%d;%dH", (y), (x))
#define clear() printf("\033[H\033[J")

using namespace std;

/// <summary>
/// Prints a Yes or No question in the console and awaits the userinput
/// </summary>
/// <param name="question">The question to ask</param>
/// <returns>Boolean based on the answer</returns>
bool Console::WriteYNQuestion(string question)
{
	string answer;
	cout << question << " (y/n)" << endl;

	cin >> answer;

	if (answer == "y" || answer == "Y")
	{
		return true;
	}
	else if (answer == "n" || answer == "N")
	{
		return false;
	}

	return false;
}

/// <summary>
/// Gets the pixels in a 30 by 30 quad from the console
/// </summary>
/// <returns>Returns a full Image with the pixels</returns>
vector<bitset<CHAR_BIT>> Console::GetPixels()
{	
	int width = 30;
	int height = 30;

	HWND window = GetConsoleWindow();
	HDC dc = GetDC(window);
	HDC captureDC = CreateCompatibleDC(dc);
	HBITMAP captureBitmap = CreateCompatibleBitmap(dc, width, height);
	SelectObject(captureDC, captureBitmap);

	BITMAPINFO bmi = { 0 };
	bmi.bmiHeader.biSize = sizeof(bmi.bmiHeader);
	bmi.bmiHeader.biWidth = width;
	bmi.bmiHeader.biHeight = height;
	bmi.bmiHeader.biPlanes = 1;
	bmi.bmiHeader.biBitCount = 32;
	bmi.bmiHeader.biCompression = BI_RGB;

	RGBQUAD* pixels = new RGBQUAD[width * height];

	GetDIBits(captureDC,
		captureBitmap,
		0,
		height,
		pixels,
		&bmi,
		DIB_RGB_COLORS);

	vector<bitset<CHAR_BIT>> image = vector<bitset<CHAR_BIT>>();

	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			int index = (height - 1 - y - 1) * width - 1 + x;
			bitset<CHAR_BIT> bit;
			if (pixels[index].rgbRed > 0) 
			{
				bit = 0;
			}
			else
			{
				bit = 255;
			}
			image.push_back(bit);
		}
	}

	delete[] pixels;

	ReleaseDC(window, dc);
	DeleteDC(captureDC);
	DeleteObject(captureBitmap);

	return image;
}

void Console::ShowProgressOfEpoch()
{
}

/// <summary>
/// Draws a 30 by 30 quad in the console
/// </summary>
void Console::DrawQuad()
{
	// Clear dosent work for the full Window, only for the shown window lines. 
	// Everything above and below the shown screen is ignored.
	clear();
	HWND window = GetConsoleWindow();
	HDC dc = GetDC(window);

	SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), WHITE_COLOR);

	for (int i = 0; i < 30; i++)
	{
		cout << " ";
	}
	cout << endl;
	for (int i = 0; i < 28; i++)
	{
		cout << " ";
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), BLACK_COLOR);
		for (int i = 0; i < 28; i++)
		{
			cout << " ";
		}
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), WHITE_COLOR);
		cout << " " << endl;
	}
	for (int i = 0; i < 30; i++)
	{
		cout << " ";
	}

	SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), BLACK_COLOR);

	cout << endl;
	cout << endl;
}

/// <summary>
/// Awaits the input of the user
/// The user can drag the mouse over the console to draw
/// </summary>
void Console::UserDrawNumber()
{
	HWND window = GetConsoleWindow();
	HDC dc = GetDC(window);

	while (GetKeyState(VK_LBUTTON) >= 0) 
	{
		cout << "";
	}
	SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), WHITE_COLOR);
	while (GetKeyState(VK_LBUTTON) < 0)
	{
		POINT mousePosition;
		GetCursorPos(&mousePosition);
		gotoxy(mousePosition.x / 25, mousePosition.y / 30);

		cout << " ";
	}
	SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), BLACK_COLOR);
	gotoxy(0, 32);
}
