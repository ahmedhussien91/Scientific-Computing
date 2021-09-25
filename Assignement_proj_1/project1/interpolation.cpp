#include "interpolation.h"

Interpolation::Interpolation()
{

}

char Interpolation::CSplineCalculateCoeffs(std::vector<std::vector<double>> dataList, double *coeffs){
    char error = 0;
    int num_points = dataList.size();
    const int size = num_points - 1;
    double **equations_coeffs = new double *[4*size];
    double equations_Forcing_coeff[4*size];

    for (int i =0; i<4*size; i++)
    {
        double *eq = new double [4*size];
        (void)memset(eq, 0, sizeof(double)*4*size);
        equations_coeffs[i] = eq;
    }

    (void)memset(equations_Forcing_coeff, 0, sizeof(double)*4*size);

    /* calculate the equations coeffeicents for system of linear equations */
    /*The first and last functions must pass through endpoints (2 cond.) */
    equations_coeffs[0][0] = dataList[0][0] * dataList[0][0] * dataList[0][0];
    equations_coeffs[0][1] = dataList[0][0] * dataList[0][0];
    equations_coeffs[0][2] = dataList[0][0];
    equations_coeffs[0][3] = 1;
    equations_Forcing_coeff[0] = dataList[0][1];

    equations_coeffs[1][(4*size) - 4] = dataList[size][0] * dataList[size][0] * dataList[size][0];
    equations_coeffs[1][(4*size) - 3] = dataList[size][0] * dataList[size][0];
    equations_coeffs[1][(4*size) - 2] = dataList[size][0];
    equations_coeffs[1][(4*size) - 1] = 1;
    equations_Forcing_coeff[1] = dataList[size][1];

    /*The second derivatives at endpoints are zero (2 cond.) */
    equations_coeffs[2][0] = dataList[0][0] * 6;
    equations_coeffs[2][1] = 2;

    equations_coeffs[3][(4*size) - 4] = dataList[size][0] * 6;
    equations_coeffs[3][(4*size) - 3] = 2;

    int row_cnt = 3;
    /* Function values must be equal at interior knots (2n‐2 conditions) */
    for(int i = 1; i < num_points - 1; i++)
    {
        for(int j = i-1; j < i+1; j++)
        {
            int col_idx = 4 * j;
            equations_coeffs[row_cnt + j + 1][col_idx] = dataList[i][0] * dataList[i][0] * dataList[i][0];
            equations_coeffs[row_cnt + j + 1][col_idx + 1] = dataList[i][0] * dataList[i][0];
            equations_coeffs[row_cnt + j + 1][col_idx + 2] = dataList[i][0];
            equations_coeffs[row_cnt + j + 1][col_idx + 3] = 1;
            equations_Forcing_coeff[row_cnt + j + 1] = dataList[i][1];
        }

        row_cnt++;
    }

    row_cnt = row_cnt + num_points - 1;

    for(int i = 1; i < num_points - 1; i++)
    {
        int col_idx = 4 * (i-1);

        /* First derivatives at internal knots must be equal (n‐1 cond.) */
        equations_coeffs[row_cnt][col_idx] = 3 * dataList[i][0] * dataList[i][0];
        equations_coeffs[row_cnt][col_idx + 1] = 2 * dataList[i][0];
        equations_coeffs[row_cnt][col_idx + 2] = 1;

        equations_coeffs[row_cnt][col_idx + 4] = -3 * dataList[i][0] * dataList[i][0];
        equations_coeffs[row_cnt][col_idx + 5] = -2 * dataList[i][0];
        equations_coeffs[row_cnt][col_idx + 6] = -1;

        row_cnt++;

        /* Second derivatives at internal points must be equal (n‐1 cond.) */
        equations_coeffs[row_cnt][col_idx] = 6 * dataList[i][0];
        equations_coeffs[row_cnt][col_idx + 1] = 2 ;

        equations_coeffs[row_cnt][col_idx + 4] = -6 * dataList[i][0];
        equations_coeffs[row_cnt][col_idx + 5] = -2;

        row_cnt++;
    }

//    for(int j = 0; j < 4*size; j++)
//    {
//        for(int i = 0; i < 4*size; i++)
//        {
//            std::cout<<equations_coeffs[j][i]<<" ";
//        }
//        std::cout<<"-->"<<equations_Forcing_coeff[j];
//        std::cout<<"\n";
//    }
//    std::cout<<"\n";

    /*solve the system of linear equations */
    m_eqsSolver->solveEquations(equations_coeffs, equations_Forcing_coeff, 4*size, coeffs);


    for (int i =0; i<4*size; i++)
    {
        delete[] equations_coeffs[i];
    }

    delete[] equations_coeffs;

    return error;
}

char Interpolation::NewtonsCalcInterpolatingPoly(std::vector<std::vector<double>> dataList,
                                                  int order, double xi, double *yint,
                                                  double *e, double *finite_diff_1st_row)
{
    int num_points = dataList.size();
    char error = 0;
    int order_plus_1 = order + 1;
    double **finite_diffs = new double *[order_plus_1];

    /* set to invalid values */
    *yint = std::numeric_limits<double>::max();

    finite_diffs[0] = finite_diff_1st_row;
    for (int i =1; i<order_plus_1; i++)
    {
        double *eq = new double [order_plus_1];
        (void)memset(eq, 0, sizeof(double)*order_plus_1);
        finite_diffs[i] = eq;
    }

    if(order < num_points)
    {
        for(int i = 0; i < order_plus_1; i++)
        {
            finite_diffs[i][0] = dataList[i][1];
        }

        for(int j = 1; j < order_plus_1; j++)
        {
            for(int i = 0; i < (order_plus_1 - j); i++)
            {
                if((dataList[i+j][0] -dataList[i][0]) != 0)
                {
                    finite_diffs[i][j] = (finite_diffs[i+1][j-1] - finite_diffs[i][j-1]) /
                        (dataList[i+j][0] -dataList[i][0]);
                }
                else
                {
                    finite_diffs[i][j] = (finite_diffs[i+1][j-1] - finite_diffs[i][j-1]) /
                            m_tolerance;
                }
            }
        }

        double xterm = 1;
        double yint_loc[order_plus_1];

        yint_loc[0] = finite_diffs[0][0];


        for(int i = 1; i < order_plus_1; i++)
        {
            xterm = xterm * (xi - dataList[i-1][0]);
            yint_loc[i] = yint_loc[i-1] + (finite_diffs[0][i] * xterm);
            e[i-1] = yint_loc[i] - yint_loc[i-1];
        }
        *yint = yint_loc[order_plus_1 -1];
    }
    else
    {
        error = 1;
    }

    for (int i =1; i<order_plus_1; i++)
    {
        delete[] finite_diffs[i];
    }

    return error;
}

