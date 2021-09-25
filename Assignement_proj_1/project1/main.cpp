#include "iostream"
#include "gausselimination.h"
#include "gaussseidel.h"
#include "regression.h"
#include "interpolation.h"
#include "csvreader.h"
#include <string>
#include<iomanip>


std::string convert(float value)
{
  std::stringstream ss;
  ss << std::setprecision(std::numeric_limits<float>::digits10+1);
  ss << value;
  return ss.str();
}

int main(int argc, char *argv[])
{
    IEquationSolver *Gauss_elmination = new GaussElimination();
//    /* gauss seidel testing */
//    IEquationSolver *Gauss_test_obj = new GaussSeidel();
//    int size = 3;
//    double equ1[3]= {0.1, 7.0, -0.3};
//    double equ2[3]= {0.3, -0.2, 10};
//    double equ3[3]= {3.0, -0.1, -0.2};
//    double forcingF[3] = {-19.3, 71.4, 7.85};
//    double equroots[3] = {0.0, 0.0, 0.0};

//    double **equations = new double *[size];
//    equations[0] = equ1;
//    equations[1] = equ2;
//    equations[2] = equ3;

//    Gauss_test_obj->setToleranceValue(0);
//    Gauss_test_obj->solveEquations(equations,forcingF,size,equroots);
//    std::cout << "x0=" << equroots[0] << ", x1=" << equroots[1] << ", x2=" << equroots[2] << "\n";
//    /* expected output values x0=3, x1=-2.5, x2=7 */
//    delete [] equations;

//    /* gauss elimination testing */
////    IEquationSolver *Gauss_test_obj = new GaussElimination();

////    int size = 4;
////    double equ1[4]= {1,-1,2,1};
////    double equ2[4]= {3,2,1,4};
////    double equ3[4]= {5,8,6,3};
////    double equ4[4]= {4,2,5,3};
////    double forcingF[4] = {1,1,1,-1};
////    double equroots[4];

////    double **equations = new double *[size];
////    equations[0] = equ1;
////    equations[1] = equ2;
////    equations[2] = equ3;
////    equations[3] = equ4;

////    Gauss_test_obj->solveEquations(equations,forcingF,size,equroots);
////    std::cout<<"x0="<<equroots[0]<<", x1="<<equroots[1]<<", x2="<<equroots[2]<<", x3="<<equroots[3]<<"\n";
////    /* expected output values x0=8.59412, x1=34.4118, x2=36.7647 */
////    delete [] equations;
////    /* gauss elimination testing end */

//    /* linear regression testing */
//    int order = 2;
//    double point1[2]={0,0};
//    double point2[2]={2,1};
//    double point3[2]={2.5,2};
//    double point4[2]={1,3};
//    double point5[2]={4,6};
//    double point6[2]={7,2};
//    double **x = new double *[6];
//    x[0] = point1;
//    x[1] = point2;
//    x[2] = point3;
//    x[3] = point4;
//    x[4] = point5;
//    x[5] = point6;
//    int no_of_points = 6;

//    double y[6];
//    y[0] = 5;
//    y[1] = 10;
//    y[2] = 9;
//    y[3] = 0;
//    y[4] = 3;
//    y[5] = 27;

//    double output_coeffs[3];

//    Regression test_obj_1(Gauss_test_obj);
//    test_obj_1.calculateCoeffs(x,y,no_of_points,order,output_coeffs,Linear);
//    std::cout<<"a0="<<output_coeffs[0]<<", a1="<<output_coeffs[1]<<", a2="<<output_coeffs[2]<<"\n";
//    /* expected output values a0=5, a1=4, a2=-3 */
//    /* linear regression testing End */

//    {
//        /* Test the newton interpolation */
//        std::vector<std::vector<double> > dataList = {{1,0},{4,1.386294},{6,1.791759},{5,1.609438}};

//        double y_of_x, error[3], finite_coeff[4] ;

//        int order = 3;
//        Interpolation test_obj_1;
//        test_obj_1.setEqSolverStrategy(Gauss_test_obj);
//        test_obj_1.NewtonsCalcInterpolatingPoly(dataList,
//                                                order, 2, &y_of_x, &error[0],finite_coeff);
//        std::cout<<"f"<<order<<"(2)="<<y_of_x<<", R"<<order-1<<"="<<error<<"\n";
//        std::cout<< "finite coeffs:"<<finite_coeff[0]<<" , "<<finite_coeff[1]<<" , "<<finite_coeff[2]<<" , "<<finite_coeff[3]<<" \n";
//    }

//    {
//        /* test cubic spline */
//        double coeffs[4];
//        std::vector<std::vector<double> > dataList = {{3,2.5},{4.5,1},{7,2.5},{9,0.5}};

//        Interpolation test_obj_1;
//        test_obj_1.setEqSolverStrategy(Gauss_test_obj);
//        test_obj_1.CSplineCalculateCoeffs(dataList, coeffs);
//        for(int i =0; i < 12; i+=4)
//        {
//            std::cout<<"coeffs->>"<<i/4<< "  "<<"a = "<<coeffs[i]<< "  "<<"b="<<coeffs[i+1]<< "  "<<"c="<<coeffs[i+2]<< "  "<<"d="<<coeffs[i+3]<<"\n";
//        }
//    }

//    {
//        /* test csv reader */
//        // Creating an object of CSVWriter
//        CSVReader reader("C:/Users/mhassoub/Desktop/Nile masters stuff/3rd term/sceintific computing/assignements/assignement_1/coursework_1/part three datasets/sp1.csv",",");

//        // Get the data from CSV File
//        std::vector<std::vector<double> > dataList = reader.getData();

//        // Print the content of row by row on screen
//        for(std::vector<double> vec : dataList)
//        {
//            for(double data : vec)
//            {
//                std::cout<<"data out"<<data << " , ";
//            }
//            std::cout<<std::endl;
//        }
//        std::cout<<"std sting test"<<std::to_string(12341254) + " sadf sdf\n";
//    }

//    {
//        /* test output in csv file */
//        std::ofstream myfile;
//        myfile.open ("example.csv");
//        myfile << "This is the first cell in the first column.\n";
//        myfile << "a,b,c\n";
//        myfile << "c,s,v\n";
//        myfile << "1,a,3.456\n";
//        myfile << "semi;colon";
//        myfile.close();

//        //std::cout<<str" , " + "fskjsf" + "4345980345";
//    }
/***************** Run Part 2 exercise using elmination ***********************/
    /*************************** Linear ****************************************/
    {
        /* Run Part 2 exercise Gauss sidel */
        IEquationSolver *Gauss_sidel = new GaussSeidel();
        Regression test_obj_linear_elmination(Gauss_sidel);

        std::cout<< "/****************************************************/\n" << "Executing Linear with Gauss Sidel:\n";
        for(int i = 1; i <=3; i++)
        {
            CSVReader reader("part_two_datasets/reg" + std::to_string(i) + ".csv",",");
            std::cout<<  "\nFile: \"part_two_datasets/reg" + std::to_string(i) + ".csv\"\n";

            // Get the data from CSV File
            std::vector<std::vector<double>> dataList = reader.getData();

            // input variables
            int order = (dataList[0].size()-1);
            int no_of_points = dataList.size();
            double y[no_of_points];
            double **x = new double *[no_of_points];

            for (int idx=0; idx<no_of_points; idx++)
            {
                double *x_tmp = new double[order];
                x[idx]  = x_tmp;
            }

            for (int idx=0; idx<no_of_points; idx++)
            {
               y[idx] = dataList[idx][order];
               for(int col=0; col<order; col++){
                    x[idx][col] = dataList[idx][col];
               }
            }


            // output variables
            double Linear_coeffs[order+1];

            // Execute the Linear regrission
            test_obj_linear_elmination.calculateCoeffs(x, y, no_of_points, order, Linear_coeffs, Linear);
            std::cout<<"a0="<<convert(Linear_coeffs[0])<<", a1="<<convert(Linear_coeffs[1])<<", a2="<<convert(Linear_coeffs[2])<<"\n";

//            // write output of linear Reg to CSV file
            std::ofstream myfile_linearReg;
            myfile_linearReg.open ("part_two_datasets/reg"+ std::to_string(i) + "_linear_sidel"+ ".csv");

            for(int j =0; j < (order+1); j+=1)
            {
                if(j != order)
                     myfile_linearReg << 'a' + std::to_string(j) + "," ;
                else
                     myfile_linearReg << 'a' + std::to_string(j) + "\n" ;
            }

            for(int j =0; j < (order+1); j+=1)
            {
                if(j != order)
                    myfile_linearReg << convert(Linear_coeffs[j])+ ",";
                else
                    myfile_linearReg << convert(Linear_coeffs[j])+ "\n";
            }
            myfile_linearReg.close();
        }
    }
    {
        /* Run Part 2 exercise Gauss Elimination*/
        IEquationSolver *Gauss_elmination = new GaussElimination();
        Regression test_obj_linear_elmination(Gauss_elmination);
         std::cout<< "/****************************************************/\n" << "Executing Linear with Gauss elimination:\n";
        for(int i = 1; i <=3; i++)
        {
            CSVReader reader("part_two_datasets/reg" + std::to_string(i) + ".csv",",");
            std::cout<<  "\nFile: \"part_two_datasets/reg" + std::to_string(i) + ".csv\"\n";

            // Get the data from CSV File
            std::vector<std::vector<double>> dataList = reader.getData();

            // input variables
            int order = (dataList[0].size()-1);
            int no_of_points = dataList.size();
            double y[no_of_points];
            double **x = new double *[no_of_points];

            for (int idx=0; idx<no_of_points; idx++)
            {
                double *x_tmp = new double[order];
                x[idx]  = x_tmp;
            }

            for (int idx=0; idx<no_of_points; idx++)
            {
               y[idx] = dataList[idx][order];
               for(int col=0; col<order; col++){
                    x[idx][col] = dataList[idx][col];
               }
            }


            // output variables
            double Linear_coeffs[order+1];

            // Execute the Linear regrission
            test_obj_linear_elmination.calculateCoeffs(x, y, no_of_points, order, Linear_coeffs, Linear);
            std::cout<<"a0="<<convert(Linear_coeffs[0])<<", a1="<<convert(Linear_coeffs[1])<<", a2="<<convert(Linear_coeffs[2])<<"\n";

//            // write output of linear Reg to CSV file
            std::ofstream myfile_linearReg;
            myfile_linearReg.open ("part_two_datasets/reg"+ std::to_string(i) + "_linear_elimination"+ ".csv");

            for(int j =0; j < (order+1); j+=1)
            {
                if(j != order)
                     myfile_linearReg << 'a' + std::to_string(j) + "," ;
                else
                     myfile_linearReg << 'a' + std::to_string(j) + "\n" ;
            }

            for(int j =0; j < (order+1); j+=1)
            {
                if(j != order)
                    myfile_linearReg << convert(Linear_coeffs[j])+ ",";
                else
                    myfile_linearReg << convert(Linear_coeffs[j])+ "\n";
            }
            myfile_linearReg.close();
        }
    }
    /*************************** Linear END ****************************************/
    /*************************** Polynomial  ****************************************/
    {
        /* Run Part 2 exercise Gauss sidel */
        IEquationSolver *Gauss_sidel = new GaussSeidel();
        Regression test_obj_linear_elmination(Gauss_sidel);
        std::cout<< "/****************************************************/\n" << "Executing Polynomial with Gauss Sidel:\n";
        for(int i = 1; i <=2; i++)
        {
            CSVReader reader("part_two_datasets/reg" + std::to_string(i) + ".csv",",");
            std::cout<<  "\nFile: \"part_two_datasets/reg" + std::to_string(i) + ".csv\"\n";
            // Get the data from CSV File
            std::vector<std::vector<double>> dataList = reader.getData();

            // input variables
            int order = (dataList.size()-2);
            int no_of_points = dataList.size();
            double y[no_of_points];
            double **x = new double *[no_of_points];

            for (int idx=0; idx<no_of_points; idx++)
            {
                double *x_tmp = new double[1];
                x[idx]  = x_tmp;
            }

            for (int idx=0; idx<no_of_points; idx++)
            {
               y[idx] = dataList[idx][1];
               for(int col=0; col<1; col++){
                    x[idx][col] = dataList[idx][col];
               }
            }


            // output variables
            double Linear_coeffs[order+1];

            // Execute the Linear regrission
            test_obj_linear_elmination.calculateCoeffs(x, y, no_of_points, order, Linear_coeffs, Polynomial);
//            std::cout<<"a0="<<convert(Linear_coeffs[0])<<", a1="<<convert(Linear_coeffs[1])<<", a2="<<convert(Linear_coeffs[2])<<"\n";

//            // write output of linear Reg to CSV file
            std::ofstream myfile_linearReg;
            myfile_linearReg.open ("part_two_datasets/reg"+ std::to_string(i) + "_polynomial_sidel"+ ".csv");

            for(int j =0; j < (order+1); j+=1)
            {
                if(j != order)
                     myfile_linearReg << 'a' + std::to_string(j) + "," ;
                else
                     myfile_linearReg << 'a' + std::to_string(j) + "\n" ;
            }

            for(int j =0; j < (order+1); j+=1)
            {
                if(j != order)
                    myfile_linearReg << convert(Linear_coeffs[j])+ ",";
                else
                    myfile_linearReg << convert(Linear_coeffs[j])+ "\n";
            }
            myfile_linearReg.close();
        }
    }
    {
        /* Run Part 2 exercise Gauss Elimination*/
        IEquationSolver *Gauss_elmination = new GaussElimination();
        Regression test_obj_linear_elmination(Gauss_elmination);
        std::cout<< "/****************************************************/\n" << "Executing Polynomial with Gauss elimination:\n";
        for(int i = 1; i <=2; i++)
        {
            CSVReader reader("part_two_datasets/reg" + std::to_string(i) + ".csv",",");
            std::cout<<  "\nFile: \"part_two_datasets/reg" + std::to_string(i) + ".csv\"\n";
            // Get the data from CSV File
            std::vector<std::vector<double>> dataList = reader.getData();

            // input variables
            int order = (dataList.size()-2);
            int no_of_points = dataList.size();
            double y[no_of_points];
            double **x = new double *[no_of_points];

            for (int idx=0; idx<no_of_points; idx++)
            {
                double *x_tmp = new double[1];
                x[idx]  = x_tmp;
            }

            for (int idx=0; idx<no_of_points; idx++)
            {
               y[idx] = dataList[idx][1];
               for(int col=0; col<1; col++){
                    x[idx][col] = dataList[idx][col];
               }
            }


            // output variables
            double Linear_coeffs[order+1];

            // Execute the Linear regrission
            test_obj_linear_elmination.calculateCoeffs(x, y, no_of_points, order, Linear_coeffs, Polynomial);
//            std::cout<<"a0="<<convert(Linear_coeffs[0])<<", a1="<<convert(Linear_coeffs[1])<<", a2="<<convert(Linear_coeffs[2])<<"\n";

//            // write output of linear Reg to CSV file
            std::ofstream myfile_linearReg;
            myfile_linearReg.open ("part_two_datasets/reg"+ std::to_string(i) + "_polynomial_elimination"+ ".csv");

            for(int j =0; j < (order+1); j+=1)
            {
                if(j != order)
                     myfile_linearReg << 'a' + std::to_string(j) + "," ;
                else
                     myfile_linearReg << 'a' + std::to_string(j) + "\n" ;
            }

            for(int j =0; j < (order+1); j+=1)
            {
                if(j != order)
                    myfile_linearReg << convert(Linear_coeffs[j])+ ",";
                else
                    myfile_linearReg << convert(Linear_coeffs[j])+ "\n";
            }
            myfile_linearReg.close();
        }
    }
    /*************************** Polynomial  END ****************************************/


/**************************  Run Part 3 exercise using elimination ***********************************/
    {
        /* Run Part 3 exercise */
        Interpolation test_obj_int;
        test_obj_int.setEqSolverStrategy(Gauss_elmination);
        std::cout<< "/****************************************************/\n" << "Executing newton and spline[with Gauss elimination]:\n";
        for(int i = 1; i <=4; i++)
        {

            CSVReader reader("part_three_datasets/sp" + std::to_string(i) + ".csv",",");
            std::cout<<  "\nFile: \"part_three_datasets/sp" + std::to_string(i) + ".csv\"\n";
            // Get the data from CSV File
            std::vector<std::vector<double>> dataList = reader.getData();

            double y_of_x, error[dataList.size()-1] ;

            double *coeffs;
            double *finite_coeff;

//            sort(dataList.begin(), dataList.end());

            coeffs = new double [4*(dataList.size()-1)];
            finite_coeff = new double [dataList.size()];

            test_obj_int.CSplineCalculateCoeffs(dataList, coeffs);

            test_obj_int.NewtonsCalcInterpolatingPoly(dataList,
                                                    dataList.size()-1,
                                                    2, &y_of_x, &error[0],finite_coeff); // 2 dummy value

            std::ofstream myfile_spline;
            myfile_spline.open ("part_three_datasets/sp"+ std::to_string(i) + "_spline_elimination"+ ".csv");
            myfile_spline << "a0,a1,a2,a3,st_range,end_range\n";

            for(int j =0; j < (4*(dataList.size()-1)); j+=4)
            {
                myfile_spline<< convert(coeffs[j+3])+ "," + convert(coeffs[j+2])+ "," + convert(coeffs[j+1])
                        + "," + convert(coeffs[j])+ "," +
                        convert(dataList[j/4][0])+ "," + convert(dataList[(j/4)+1][0]) + "\n";
            }
            myfile_spline.close();
            delete[] coeffs;

            std::ofstream myfile_newton;
            myfile_newton.open ("part_three_datasets/sp"+ std::to_string(i) + "_newton_elimination"+ ".csv");

            for(int j =0; j < dataList.size(); j++)
            {
                myfile_newton << "b" + std::to_string(j) + ",";
            }

            myfile_newton << "order";

            myfile_newton << "\n";

            for(int j =0; j < dataList.size(); j++)
            {
                myfile_newton<<convert(finite_coeff[j]) + ",";
            }
            myfile_newton<< std::to_string(dataList.size()-1);
            myfile_newton.close();
            delete[] finite_coeff;

            std::ofstream myfile_newton_error;
            myfile_newton_error.open ("part_three_datasets/sp"+ std::to_string(i) + "_newton_error_elimination"+ ".csv");

            for(int j = 0; j < (dataList.size()-1); j++)
            {
                myfile_newton_error<<convert(error[j]) + ",";
            }
            myfile_newton_error.close();
        }
    }

/**************************  Run Part 3 exercise using seidel ***********************************/
    {
        /* Run Part 3 exercise using seidel*/
        Interpolation test_obj_int_seidel;
        /* gauss seidel testing */
        IEquationSolver *Gauss_test_obj_seidel = new GaussSeidel();

        test_obj_int_seidel.setEqSolverStrategy(Gauss_test_obj_seidel);
        std::cout<< "/****************************************************/\n" << "Executing newton and spline[with Gauss sidel]:\n";
        for(int i = 1; i <=4; i++)
        {

            CSVReader reader("part_three_datasets/sp" + std::to_string(i) + ".csv",",");
            std::cout<<  "\nFile: \"part_three_datasets/sp" + std::to_string(i) + ".csv\"\n";
            // Get the data from CSV File
            std::vector<std::vector<double>> dataList = reader.getData();

            double y_of_x, error[dataList.size()-1] ;

            double *coeffs;
            double *finite_coeff;

//            sort(dataList.begin(), dataList.end());

            coeffs = new double [4*(dataList.size()-1)];
            finite_coeff = new double [dataList.size()];

            test_obj_int_seidel.CSplineCalculateCoeffs(dataList, coeffs);

            test_obj_int_seidel.NewtonsCalcInterpolatingPoly(dataList,
                                                    dataList.size()-1,
                                                    2, &y_of_x, &error[0],finite_coeff); // 2 dummy value

            std::ofstream myfile_spline;
            myfile_spline.open ("part_three_datasets/sp"+ std::to_string(i) + "_spline_seidel"+ ".csv");
            myfile_spline << "a0,a1,a2,a3,st_range,end_range\n";

            for(int j =0; j < (4*(dataList.size()-1)); j+=4)
            {
                myfile_spline<< convert(coeffs[j+3])+ "," + convert(coeffs[j+2])+ "," + convert(coeffs[j+1])
                        + "," + convert(coeffs[j])+ "," +
                        convert(dataList[j/4][0])+ "," + convert(dataList[(j/4)+1][0]) + "\n";
            }
            myfile_spline.close();
            delete[] coeffs;

            std::ofstream myfile_newton;
            myfile_newton.open ("part_three_datasets/sp"+ std::to_string(i) + "_newton_seidel"+ ".csv");

            for(int j =0; j < dataList.size(); j++)
            {
                myfile_newton << "b" + std::to_string(j) + ",";
            }

            myfile_newton << "order";

            myfile_newton << "\n";

            for(int j =0; j < dataList.size(); j++)
            {
                myfile_newton<<convert(finite_coeff[j]) + ",";
            }
            myfile_newton<< std::to_string(dataList.size()-1);
            myfile_newton.close();
            delete[] finite_coeff;

            std::ofstream myfile_newton_error;
            myfile_newton_error.open ("part_three_datasets/sp"+ std::to_string(i) + "_newton_error_seidel"+ ".csv");

            for(int j = 0; j < (dataList.size()-1); j++)
            {
                myfile_newton_error<<convert(error[j]) + ",";
            }
            myfile_newton_error.close();
        }
    }

//    delete Gauss_test_obj;
//    Gauss_test_obj = NULL;

    return 0;

}
