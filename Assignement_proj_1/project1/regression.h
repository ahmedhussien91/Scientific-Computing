#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H
#include <iostream>
#include<cmath>
#include "iequationssolver.h"


enum Regression_type {Linear,Polynomial};

class Regression
{
public:
    Regression(IEquationSolver * s);

    /* given points values for
     * double dim array of independent variables [(x10,x20,...,xk0,y0),(x11,x21,...,xk1,y1),...., (x1n,x2n,...,xkn,yn)]
     * number of points -> n &
     * order -> k (no of independent variables)
     * calculate coefficents a0,a1,a2,....ak
     *  */
    char calculateCoeffs(double **x, double *y,int no_of_points, int order, double *coeffs, Regression_type reg_type);

private:

    IEquationSolver * m_eqsSolver;
};

#endif // LINEARREGRESSION_H
