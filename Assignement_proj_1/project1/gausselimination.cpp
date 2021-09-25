#include "gausselimination.h"

GaussElimination::GaussElimination()
{

}

char GaussElimination::solveEquations(double **coefficients, double * forcingFunctions, int size, double * roots)
{
    /* run time mesurment for Gauss elimination */
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    char error = 0;
    int *rowIndexArr = new int [size];
    double *arrayOfRowBiggest = new double [size];

    /* generate row index arr*/
    for (int i =0; i<size; i++)
    {
        rowIndexArr[i] = i; /* will be used in ordering the rows if there is a change */
    }
    /* get the max element in each row in an array of biggest element in a row */
    for (int row_i =0; row_i<size; row_i++)
    {
        arrayOfRowBiggest[row_i] = fabs(coefficients[row_i][0]);
        for (int col_i=1; col_i<size; col_i++)
        {
            if (arrayOfRowBiggest[row_i]<coefficients[row_i][col_i])
                arrayOfRowBiggest[row_i] = fabs(coefficients[row_i][col_i]);
        }
    }

    /* elimination step */
    error = elimainate(coefficients, forcingFunctions, size, rowIndexArr, arrayOfRowBiggest);

    /* subistitute back only if no errors in the eliminate step */
    if (error == 0)
        substituteBack(coefficients, forcingFunctions, size, roots, rowIndexArr);

    /* run time mesurment for Gauss elimination */
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

    std::cout << "\nGauss elimination take: "<<duration<<" us\n\n\n";
    return  error;
}

char GaussElimination::elimainate(double **coefficients, double * forcingFunctions, int size, int *rowsIndexArr, double *arrayOfRowBiggest){
    char error =0;

    /* elimination for loop*/
    for (int pivotRow_i =0; pivotRow_i < size-1; pivotRow_i++)
    {
        /* get pivot element and pivot equation and rearrange the array of elements */
        pivot(coefficients, size, pivotRow_i, arrayOfRowBiggest, rowsIndexArr);
        /* error handling */
        if (fabs(coefficients[rowsIndexArr[pivotRow_i]][pivotRow_i]/arrayOfRowBiggest[rowsIndexArr[pivotRow_i]]) < m_tolerance)
        {
            error = -1;
            std::cout<<"Error!: Gauss elimination: pivot is below tolerance\n";
            break;
        }

        /* elimination step from the rest of the equations */
        for (int rows_i= pivotRow_i+1; rows_i<size; rows_i++)
        {
            double factor = coefficients[rowsIndexArr[rows_i]][pivotRow_i]/coefficients[rowsIndexArr[pivotRow_i]][pivotRow_i];
            for (int col_i=pivotRow_i+1; col_i<size; col_i++)
            {
                coefficients[rowsIndexArr[rows_i]][col_i] = coefficients[rowsIndexArr[rows_i]][col_i] - factor* coefficients[rowsIndexArr[pivotRow_i]][col_i];
            }
            forcingFunctions[rowsIndexArr[rows_i]] = forcingFunctions[rowsIndexArr[rows_i]] - factor*forcingFunctions[rowsIndexArr[pivotRow_i]];
        }
    }
    if (fabs(coefficients[rowsIndexArr[size-1]][size-1]/arrayOfRowBiggest[rowsIndexArr[size-1]]) < m_tolerance)
    {
        error = -1;
        std::cout<<"Error!: Gauss elimination: pivot is below tolerance\n";
    }
    /* end of elimination for loop */

    return  error;
}

void GaussElimination::pivot(double **coefficients, int size, int currentIndex, double *arrayOfRowBiggest, int *rowsPtrArr)
{
    int biggestElementIdx = currentIndex;
    double biggest_element = fabs(coefficients[rowsPtrArr[currentIndex]][currentIndex]/arrayOfRowBiggest[rowsPtrArr[currentIndex]]);
    /* find the biggest pivot element */
    for (int row_i = currentIndex+1; row_i<size; row_i++)
    {
        double tmp = fabs(coefficients[rowsPtrArr[row_i]][currentIndex]/arrayOfRowBiggest[rowsPtrArr[row_i]]);
        if (tmp > biggest_element)
        {
            biggest_element = tmp;
            biggestElementIdx = row_i;
        }
    }
    /* replace the current row index with the row index that contain the biggest pivot element */
    int tmp = rowsPtrArr[currentIndex];
    rowsPtrArr[currentIndex] = rowsPtrArr[biggestElementIdx];
    rowsPtrArr[biggestElementIdx] = tmp;
}

void GaussElimination::substituteBack(double **upperTriMatrix, double * forcingFunctions, int size, double * roots, int *rowsIndexArr)
{
    /* solution of the last element X[n][n] = b[n]/a[n][n] */
    roots[size-1] = forcingFunctions[rowsIndexArr[size-1]]/upperTriMatrix[rowsIndexArr[size-1]][size-1];

    /* equation for row n-1 -> a[n-1][n-1]*x[n-1]+a[n-1][n]*x[n] = b[n-1]
     * we know x[n] and w need to get x[n-1]
     * x[n-1]= (b[n-1] - a[n-1][n]*x[n])/a[n-1][n-1]
     * */
    for(int row_i=size-2; row_i>=0; row_i--)
    {
        double sum =0;
        for (int col_i=row_i+1; col_i<size; col_i++)
        {
            sum = sum + upperTriMatrix[rowsIndexArr[row_i]][col_i] * roots[col_i];
        }
        roots[row_i] = (forcingFunctions[rowsIndexArr[row_i]] - sum)/upperTriMatrix[rowsIndexArr[row_i]][row_i];
    }

}


GaussElimination::~GaussElimination()
{

}
