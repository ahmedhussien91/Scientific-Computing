#include "gaussseidel.h"
#include <iostream>
#include <cmath>

#define MAX_ITERATIONS 100

GaussSeidel::GaussSeidel()
{
}

char GaussSeidel::solveEquations(double **coefficients, double *forcingFunctions, int size, double *roots)
{
    /* run time mesurment for Gauss sidel */
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    int *rowIndexArr = new int [size];

    if(!avoidZeroDiagonalAndMatchConvergenceCriterion(coefficients, rowIndexArr, size))
    {
        std::cout << "warning!: Gauss Seidel pivot element is less than summation of row elements. It may not converge\n";
        for (int i =0; i<size;i++)
        {
            rowIndexArr[i] = i;
        }
    }
    /* order coeffiecents to avoid zero diagonal */
    for(int i=0; i<size; i++)
    {
        if (coefficients[rowIndexArr[i]][i] == 0)
        {
            for(int j =i+1; j<size; j++)
            {
                if (coefficients[rowIndexArr[j]][i] !=0)
                {
                    int tmp_idx = rowIndexArr[j];
                    rowIndexArr[j] = rowIndexArr[i];
                    rowIndexArr[i] = tmp_idx;
                }
            }
        }
    }
    /* check that all the diagonal elements are not zero */
    for(int i=0; i<size; i++)
    {
        if (coefficients[rowIndexArr[i]][i] == 0)
        {
            std::cout << "Error!: Gauss Seidel pivot element is =0 after reordering. It will not converge\n";
            return -1;
        }
     }


    for (int i = 0; i < size; ++i)
    {
        roots[i] = 0.0;
    }

    int i = 0;
    bool toleranceLimitReached = false;
    do
    {
        toleranceLimitReached = iteration(coefficients, forcingFunctions, rowIndexArr, roots, size);

        ++i;
    } while(i < MAX_ITERATIONS && !toleranceLimitReached);

    /* run time mesurment for Gauss sidel */
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

    std::cout << "\nGausssidel take: "<<duration<<" us\n\n\n";
    return 0;
}

bool GaussSeidel::avoidZeroDiagonalAndMatchConvergenceCriterion(double **coeffiecients, int *indices, int size)
{
    for (int i = 0; i < size; ++i)
    {
        indices[i] = -1;
    }

    int rowIndex;
    for(int i = 0; i < size; ++i)
    {
        rowIndex = getTheOrderToConverge(coeffiecients, size, i);

        if(rowIndex == -1 || indices[rowIndex] != -1)
        {
            return false;
        }

        indices[rowIndex] = i;
    }

    return true;
}

int GaussSeidel::getTheOrderToConverge(double **matrix, int size, int rowIndex)
{
    for (int targetRowIndex = 0; targetRowIndex < size; ++targetRowIndex)
    {
        if(convergenceCriterion(matrix, rowIndex, targetRowIndex, size))
        {
            return targetRowIndex;
        }
    }
    return -1;
}

bool GaussSeidel::convergenceCriterion(double **matrix, int i, int diagonalCoeffienientToCheck, int size)
{
    double sum = 0.0;
    for (int j = 0; j < size; ++j)
    {
        if(j != diagonalCoeffienientToCheck)
        {
            sum += fabs(matrix[i][j]);
        }
    }

    return fabs(matrix[i][diagonalCoeffienientToCheck]) > sum;
}

bool GaussSeidel::iteration(double **coefficients, double *forcingFunctions, int *rowIndexArr, double *roots, int size)
{
    double errorPercentages[size];
    for (int i = 0; i < size; ++i)
    {
        errorPercentages[i] = 0;
    }

    for (int i = 0; i < size; ++i)
    {
        calculateXi(coefficients, forcingFunctions, rowIndexArr, roots, errorPercentages, size, i);
    }

    return isToleranceValueReached(errorPercentages, size);
}

void GaussSeidel::calculateXi(double **coefficients, double *forcingFunctions, int *rowIndexArr, double *roots, double *errorPercentages, int size, int i)
{
    double sum = 0.0, xiOld;
    for (int j = 0; j < size; ++j)
    {
        if(j != i)
        {
            sum += coefficients[rowIndexArr[i]][j] * roots[j];
        }
    }
    xiOld = roots[i];
    roots[i] = (forcingFunctions[rowIndexArr[i]] - sum) / coefficients[rowIndexArr[i]][i];
    errorPercentages[i] = calculateXiErrorPercentage(roots[i], xiOld);
}

double GaussSeidel::calculateXiErrorPercentage(double xi, double xiOld)
{
    return fabs((xi - xiOld) / xi) * 100;
}

bool GaussSeidel::isToleranceValueReached(double *errorPercentages, int size)
{
    bool isReached = true;

    for(int i = 0; i < size; ++i)
    {
        isReached &= (errorPercentages[i] <= m_tolerance);
    }

    return isReached;
}
