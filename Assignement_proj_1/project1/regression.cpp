#include "regression.h"

Regression::Regression(IEquationSolver *s)
{
    m_eqsSolver = s;
}

char Regression::calculateCoeffs(double **x, double *y, int no_of_points, int order, double *coeffs, Regression_type reg_type){

    /* check if regression generate x^2,x^3....,x^order and extend the double array x with them to be able to
     * use the same algorithim to calculate both linear and polynomial */
    if (reg_type == Polynomial)
    {
        /* check that the given order is valid */
        if(no_of_points < order+1)
        {
            std::cout << "Error! the given polynomial order is invalid";
            return -1; /* error */
        }
        /* generate x's by the order length*/
        double **powers_of_x_arr = new double *[no_of_points];
        for (int i =0; i<no_of_points; i++)
        {
            double *powers_of_x_point = new double [order];
            powers_of_x_arr[i] = powers_of_x_point;
            for (int j =0; j<order ; j++)
            {
                  powers_of_x_arr[i][j] = pow(x[i][0],(j+1));
            }
        }

        int size = order + 1;
        double **equations_coeffs = new double *[size];
        double equations_Forcing_coeff[size];

        /* define a space for the equations_coeffs (2 dimensional array)*/
        for (int i =0; i<size; i++)
        {
            double *eq = new double [size];
            equations_coeffs[i] = eq;
        }

        /* calculate the equations coeffeicents for system of linear equations */
        for (int x_idx1=1; x_idx1 <= size; x_idx1++)
        {
            for (int x_idx2 =1; x_idx2 <= x_idx1; x_idx2++)
            {
                double sum=0;
                for (int point_idx=0; point_idx<no_of_points; point_idx++)
                {
                    if (((x_idx1-2) != -1) && ((x_idx2-2) != -1))
                        sum = sum + powers_of_x_arr[point_idx][x_idx1-2] * powers_of_x_arr[point_idx][x_idx2-2];
                     else if ((x_idx1-2) != -1)
                        sum = sum + powers_of_x_arr[point_idx][x_idx1-2];
                    else if ((x_idx2-2) != -1)
                        sum = sum + powers_of_x_arr[point_idx][x_idx2-2];
                    else
                        sum = sum + 1;
                }
                equations_coeffs[x_idx1-1][x_idx2-1] = sum;
                equations_coeffs[x_idx2-1][x_idx1-1] = sum;
            }

            double sum =0;
            for (int point_idx=0; point_idx<no_of_points; point_idx++)
            {
                if ((x_idx1-2) != -1)
                    sum = sum + y[point_idx] * powers_of_x_arr[point_idx][x_idx1-2];
                else
                    sum = sum + y[point_idx];
            }
            equations_Forcing_coeff[x_idx1-1] = sum;
        }

        /*solve the system of linear equations */
        m_eqsSolver->solveEquations(equations_coeffs, equations_Forcing_coeff, size, coeffs);

        for (int i =0; i<size; i++)
        {
            delete [] equations_coeffs[i];
        }
        delete [] equations_coeffs;


    }
    else if (reg_type == Linear)
    {
        int size = order + 1;
        double **equations_coeffs = new double *[size];
        double equations_Forcing_coeff[size];

        /* define a space for the equations_coeffs (2 dimensional array)*/
        for (int i =0; i<size; i++)
        {
            double *eq = new double [size];
            equations_coeffs[i] = eq;
        }

        /* calculate the equations coeffeicents for system of linear equations */
        for (int x_idx1=1; x_idx1 <= size; x_idx1++)
        {
            for (int x_idx2 =1; x_idx2 <= x_idx1; x_idx2++)
            {
                double sum=0;
                for (int point_idx=0; point_idx<no_of_points; point_idx++)
                {
                    if (((x_idx1-2) != -1) && ((x_idx2-2) != -1))
                        sum = sum + x[point_idx][x_idx1-2] * x[point_idx][x_idx2-2];
                     else if ((x_idx1-2) != -1)
                        sum = sum + x[point_idx][x_idx1-2];
                    else if ((x_idx2-2) != -1)
                        sum = sum + x[point_idx][x_idx2-2];
                    else
                        sum = sum + 1;
                }
                equations_coeffs[x_idx1-1][x_idx2-1] = sum;
                equations_coeffs[x_idx2-1][x_idx1-1] = sum;
            }

            double sum =0;
            for (int point_idx=0; point_idx<no_of_points; point_idx++)
            {
                if ((x_idx1-2) != -1)
                    sum = sum + y[point_idx] * x[point_idx][x_idx1-2];
                else
                    sum = sum + y[point_idx];
            }
            equations_Forcing_coeff[x_idx1-1] = sum;
        }

        /*solve the system of linear equations */
        m_eqsSolver->solveEquations(equations_coeffs, equations_Forcing_coeff, size, coeffs);

        for (int i =0; i<size; i++)
        {
            delete [] equations_coeffs[i];
        }
        delete [] equations_coeffs;
    }
    return 0;
}
