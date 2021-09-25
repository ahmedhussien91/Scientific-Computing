#ifndef GAUSSELIMINATION_H
#define GAUSSELIMINATION_H
#include "iostream"
#include <cmath>
#include "iequationssolver.h"

class GaussElimination : public IEquationSolver
{
public:
    GaussElimination();
    ~GaussElimination();
    /*
     *
     *  This will be the main interface to caluclating roots of a system of linear Equations using Gauss Elimination
     *  a00 x1 + a01 x2 = b0
     *  a10 x1 + a11 x2 = b1
     *  i/p: coefficients       -> 2 dimensional array[row][col] that carry matrix of the equations coeffcients
     *                          ex. [ a00, a01, a10, a11]
     *       size           -> size of rows & col & forcingFunctions array & roots array
     *                          ex. 2
     *       forcingFunctions -> pointer of array of forcing coeffiecents
     *                          ex. [b0, b1]
     *  o/p: roots        -> pointer of array of output roots of system of equations
     *                          ex. [x1, x2]
     * */
    char solveEquations(double **coefficients, double * forcingFunctions, int size, double * roots);

    void setToleranceValue(double tolerance){m_tolerance = tolerance;}
private:
    char elimainate(double **coefficients, double * forcingFunctions, int size, int *rowIndex, double *arrayOfRowBiggest);
    void pivot(double **coefficients, int size, int currentIndex, double *arrayOfRowBiggest, int *rowsPtrArr);
    void substituteBack(double **upperTriMatrix, double * forcingFunctions, int size, double * roots, int *rowsIndexArr);


    double m_tolerance = 1e-15;   /* default tolerance value */
};

#endif // GAUSSELIMINATION_H
