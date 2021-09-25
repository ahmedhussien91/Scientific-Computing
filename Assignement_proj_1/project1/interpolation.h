#ifndef INTERPOLATION_H
#define INTERPOLATION_H

#include "iequationssolver.h"
#include <string.h>
#include <limits>
#include <vector>
#include "iostream"

class Interpolation
{
public:
    Interpolation();

    /* given points values for independent variables x1,x2,...,xn & y & num_points
     * calculate coefficents ai,bi,ci,di, where i is from 0 to num_points - 1
     *  */
    char CSplineCalculateCoeffs(std::vector<std::vector<double>> dataList, double *coeffs);

    /* given points values for independent variables x1,x2,...,xn & y & num_points
     * calculate coefficents ai,bi,ci,di, where i is from 0 to num_points - 1
     *  */
    char NewtonsCalcInterpolatingPoly(std::vector<std::vector<double>> dataList, int order,
                                      double xi, double *yint, double *e, double *finite_diff_1st_row);

    /* must be called before calling calculateCoeffs in order to
     * define the strategy to solve system the system of linear equations
     *   */
    void setEqSolverStrategy (IEquationSolver * s){m_eqsSolver = s;}

private:
    double m_tolerance = 1e-15;   /* default tolerance value */
    IEquationSolver * m_eqsSolver;
};

#endif // INTERPOLATION_H
