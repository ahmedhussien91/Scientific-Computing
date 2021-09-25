// MatrixMul_CPU.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define  MAX_NUM_OF_ROWS 10000
#define	 MAX_NUM_OF_COLS 10000

float Matrix_1[MAX_NUM_OF_ROWS][MAX_NUM_OF_COLS];
float Matrix_2[MAX_NUM_OF_ROWS][MAX_NUM_OF_COLS];
float Matrix_output[MAX_NUM_OF_ROWS][MAX_NUM_OF_COLS];

int no_of_rows_1 = 4;
int no_of_rows_2 = 4;
int	no_of_cols_1 = 4;
int	no_of_cols_2 = 4;

void ReadMatrix_1_2(void);
void multiply(void);

int main()
{

	// read 2 Matrix from Files
	ReadMatrix_1_2();

	// take time snap before multiplication
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	//CPU Multiplication MAtrix_1 * Matrix_2
	multiply();
	// take time snap after multiplication
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	
	// print the Time taken to Multiply two Matrices 
	auto duration = duration_cast<microseconds>(t2 - t1).count();
	cout <<"Multiplication Time CPU(us):" << duration << "\n";

	//print the output matrix for testing
//	for (int i = 0; i < no_of_cols_1; i++) {
//		for (int j = 0; j < no_of_rows_2; j++)
//		{
//			cout << Matrix_output[i][j] << "	";
//		}
//		cout << "\n";
//	}
	return 0;
}
// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file


// this function read the two matrices from two files with the dimensions given in the global variables set at the begining of the file
void ReadMatrix_1_2(void) {
	int x, y;
	ifstream in_1("Matrix_1.txt", std::ifstream::in);
	ifstream in_2("Matrix_2.txt", std::ifstream::in);

	if (!in_1 || !in_2) {
		cout << "Error! Cannot open file.\n";
		return;
	}
	else if (no_of_cols_1 != no_of_rows_2) {
		cout << "Error! Matrix dimensions is not valid for multiplication.\n";
		return;
	}

	for (y = 0; y < no_of_cols_1; y++) {
		for (x = 0; x < no_of_rows_1; x++) {
			in_1 >> Matrix_1[x][y];
		}
	}
	for (y = 0; y < no_of_cols_2; y++) {
		for (x = 0; x < no_of_rows_2; x++) {
			in_2 >> Matrix_2[x][y];
		}
	}

	in_1.close();
	in_2.close();
}

// This function multiplies  
// Matrix_1[][] and Matrix_2[][], and  
// stores the result in Matrix_output[][]
// all Values is Global Value
void multiply(void)
{
	int i, j, k;
	for (i = 0; i < no_of_rows_1; i++)
	{
		for (j = 0; j < no_of_cols_2; j++)
		{
			Matrix_output[i][j] = 0;
			for (k = 0; k < no_of_cols_1; k++)
				Matrix_output[i][j] += Matrix_1[i][k] *
				Matrix_2[k][j];
		}
	}
}