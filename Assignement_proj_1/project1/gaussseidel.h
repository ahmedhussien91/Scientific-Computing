#ifndef GAUSSSEIDEL_H
#define GAUSSSEIDEL_H

#include "iequationssolver.h"

class GaussSeidel : public IEquationSolver
{
public:
    GaussSeidel();

public:
    char solveEquations(double **coefficients, double *forcingFunctions, int size, double *roots);

    void setToleranceValue(double tolerance) {m_tolerance = tolerance;}

private:
    /* ordering functions */
    bool avoidZeroDiagonalAndMatchConvergenceCriterion(double **coeffiecients, int *indices, int size);
    int getTheOrderToConverge(double **matrix, int size, int rowIndex);
    bool convergenceCriterion(double **matrix, int i, int diagonalCoeffienientToCheck, int size);
    /* end of ordering functions */

    /* roots calculation fuctions */
    bool iteration(double **coefficients, double *forcingFunctions, int *rowIndexArr, double *roots, int size);
    void calculateXi(double **coefficients, double *forcingFunctions, int *rowIndexArr, double *roots, double *errorPercentages, int size, int i);
    double calculateXiErrorPercentage(double xi, double xiOld);
    bool isToleranceValueReached(double *errorPercentages, int size);
    /* end of roots calculation fuctions */

    double m_tolerance = 0.0;   /* default tolerance value */
};

#endif // GAUSSSEIDEL_H
