__author__ = 'asabry'

import sys
import csv
from os import  walk
from numpy import *
from matplotlib.pyplot import plot, show, figure, axes
from mpl_toolkits.mplot3d import axes3d, Axes3D


# read CSV file and define column size & row size &
def read_points_from_csv(file_name):
    with open(file_name, 'r') as fileObj:
        reader = csv.reader(fileObj)
        csv_list = list(reader)

    csv_list.pop(0)  # remove the headder
    col_size = len(csv_list[0])
    row_size = len(csv_list)  # remove the header
    # reorder the list to be easier to access
    arr_of_elements = [[]]
    for idx in range(0, col_size):
        arr_of_elements.append([float(x[idx]) for x in csv_list])
    arr_of_elements.pop(0)  # remove the first element that is due to initialization & is empty

    return arr_of_elements


# read CSV file and define column size & row size &
def read_equation_coeff_from_csv(file_name):
    with open(file_name, 'r') as fileObj:
        reader = csv.reader(fileObj)
        csv_list = list(reader)

    csv_list.pop(0)  # remove the headder

    return csv_list


def plot_3d_points(arr_of_elements,ax):


    return


def plot_points(arr_of_elements,ax=0, color='r'):
    if len(arr_of_elements) == 2:
        plot(arr_of_elements[0], arr_of_elements[1], color + 'o')
    # elif len(arr_of_elements) == 3:
        # plot_3d_points(arr_of_elements,ax)  # give this a 3D option
    return


# equation_coeff : a0,a1,a2,a3 -> a3X**3+a2X**2+a1X**1+a0
# equation_type : order
# data_range: [start of interval, end of interval]
# step_size : give if data step size is very small
def plot_polynomial_equation(equation_coeff, data_range, step_size=0.01,option ='g'):
    x = arange(float(data_range[0]), float(data_range[1]),
               step_size)  # get values between -10 and 10 with 0.01 step and set to y
    i = 0
    y = float(equation_coeff[0]) * x ** i
    i = i + 1
    for idx in range(1, len(equation_coeff)):
        y = y + float(equation_coeff[idx]) * x ** i
        i = i + 1

    plot(x, y, option)
    return


def plot_3d_equation(arr_of_elements,ax):


    # Data for a three-dimensional line
    ax.plot3D(arr_of_elements[0], arr_of_elements[1], arr_of_elements[2], 'gray')
    return


# equation_coeff : a0,a1,a2,a3 -> a2X2+a1X1+a0 or a1X1+a0
# equation_type : order 1,2
# data_range: [start of interval, end of interval]
# step_size : give if data step size is very small
def plot_linear_equation(equation_coeff, points, step_size=0.01, data_range=[-1000, 1000], data_range2=[-1000, 1000]):
    if (len(equation_coeff) == 2):
        x = arange(float(data_range[0]), float(data_range[1]),
                   step_size)  # get values between -10 and 10 with 0.01 step and set to y
        y = float(equation_coeff[0]) + x * float(equation_coeff[1])

        plot(x, y)
    elif (len(equation_coeff) == 3):
        x1 = arange(float(data_range[0]), float(data_range[1]),
                    step_size)  # get values between -10 and 10 with 0.01 step and set to y
        # calculate the step size for the 2nd line to be equal the frist line
        step_size2 = (float(data_range2[1]) - float(data_range2[0])) / (
        (float(data_range[1]) - float(data_range[0])) / step_size)
        x2 = arange(float(data_range2[0]), float(data_range2[1]), step_size2)
        y = float(equation_coeff[0]) + x1 * float(equation_coeff[1]) + x2 * float(equation_coeff[1])

        ax = axes(projection='3d')
        plot_3d_equation([x1, x2, y],ax)
        # Data for a three-dimensional line
        ax.plot3D(x1, x2, y, 'gray')
        ax.scatter3D(points[0], points[1], points[2], c=points[2], cmap='Greens');
    return


# equation_coeff : a0,a1,a2,a3,...,an -> a0+a1(x-x0)+a2(x-x0)(x-x1)+...+an-1(x-x0)(x-x1)..(x-xn-1)
# x_points : x0,x1,x2,...xn
# data_range: [start of interval, end of interval]
# step_size : give if data step size is very small
def plot_newton_equation(equation_coeff, x_points, data_range, step_size=0.01,option='r'):
    x = arange(float(data_range[0]), float(data_range[1]),
               step_size)  # get values between data_range[1] and data_range[1] with 0.01 step and set to y
    y = float(equation_coeff[0])
    x_fun =1
    for idx in range(1, len(equation_coeff)):
        x_fun *= (x - x_points[idx-1])
        y += float(equation_coeff[idx]) * x_fun
    plot(x, y, option)

    return

def plot_newton_equation_points(equation_coeff, x_points, inc_points,option='r'):
    x=[]
    for idx in range(0,len(x_points)-1):
        step_size = (x_points[idx+1]-x_points[idx])/inc_points
        if step_size != 0:
            x.extend(arange(x_points[idx], x_points[idx+1],
               step_size))
        else:

            print("Error!: step size = 0, two repeated points in the data set ")

    y = float(equation_coeff[0])
    x_fun = 1
    for idx in range(1, len(equation_coeff)):
        x_fun *= (asarray(x) - x_points[idx - 1])
        y += float(equation_coeff[idx]) * x_fun
    plot(x, y, option)

    return


################ Main ########################
# testing all but newton function
# arrayOfElements = read_points_from_csv("reg1.csv")
# arrayOfElements1 = read_points_from_csv("reg2.csv")
# arrayOfElements2 = read_points_from_csv("reg3.csv")
# eq_coeff_range = read_equation_coeff_from_csv("reg3_sp1.csv")
# figure()
# plot_polynomial_equation(eq_coeff_range[0][0:-2], eq_coeff_range[0][-2:len(eq_coeff_range[0])])
# plot_linear_equation(eq_coeff_range[0][0:-2], eq_coeff_range[0][-2:len(eq_coeff_range[0])])
#
# # plot_polynomial_equation([1,1],[-10,10])
# # plot_polynomial_equation([1,1,1],[-10,10])
# figure()
# plot_points(arrayOfElements)
# plot_points(arrayOfElements1, 'b')
#
# figure()
# plot_points(arrayOfElements2, 'g')
#
# # show reset the figure index and show the previous figures
# show()

#loop on the files inside a folder and draw the figname_function.csv
path = "./project1/part_three_datasets/"
for (dirpath, dirnames, filenames) in walk(path):
    for filename in filenames:
        fig_function = filename.split('_')
        filename = path + filename
        # draw raw data on newton and spline
        if len(fig_function) == 1:
            figure(fig_function[0].split('.')[0]+'_newton')
            points = read_points_from_csv(filename)
            plot_points(points,'b')
            figure(fig_function[0].split('.')[0] + '_newton_2x')
            plot_points(points, 'b')
            figure(fig_function[0].split('.')[0] + '_newton_4x')
            plot_points(points, 'b')
            figure(fig_function[0].split('.')[0] + '_newton_8x')
            plot_points(points, 'b')
            figure(fig_function[0].split('.')[0] + '_spline')
            plot_points(points, 'b')
            figure(fig_function[0].split('.')[0] + '_spline_2x')
            plot_points(points, 'b')
            figure(fig_function[0].split('.')[0] + '_spline_4x')
            plot_points(points, 'b')
            figure(fig_function[0].split('.')[0] + '_spline_8x')
            plot_points(points, 'b')
            figure(fig_function[0].split('.')[0] + '_newton_s')
            points = read_points_from_csv(filename)
            plot_points(points, 'b')
            figure(fig_function[0].split('.')[0] + '_newton_2x_s')
            plot_points(points, 'b')
            figure(fig_function[0].split('.')[0] + '_newton_4x_s')
            plot_points(points, 'b')
            figure(fig_function[0].split('.')[0] + '_newton_8x_s')
            plot_points(points, 'b')
            figure(fig_function[0].split('.')[0] + '_spline_s')
            plot_points(points, 'b')
            figure(fig_function[0].split('.')[0] + '_spline_2x_s')
            plot_points(points, 'b')
            figure(fig_function[0].split('.')[0] + '_spline_4x_s')
            plot_points(points, 'b')
            figure(fig_function[0].split('.')[0] + '_spline_8x_s')
            plot_points(points, 'b')
        elif len(fig_function) == 3:
            #plot Newton data
            if "newton" in fig_function[1]:
                if "elimination" in fig_function[2]:
                    figure(fig_function[0]+'_newton')
                    row_arr = read_equation_coeff_from_csv(filename)
                    points = read_points_from_csv(path+fig_function[0]+'.csv')
                    step_size = (max(points[0])- min(points[0]))/1000 # equation
                    figure(fig_function[0] + '_newton')
                    plot_newton_equation(row_arr[0][0:-1], points[0], [min(points[0]),max(points[0])],step_size)
                    figure(fig_function[0] + '_newton_2x')
                    plot_newton_equation_points(row_arr[0][0:-1], points[0],2,'rs')
                    figure(fig_function[0] + '_newton_4x')
                    plot_newton_equation_points(row_arr[0][0:-1], points[0], 4, 'rs')
                    figure(fig_function[0] + '_newton_8x')
                    plot_newton_equation_points(row_arr[0][0:-1], points[0], 8, 'rs')
                elif "seidel" in fig_function[2]:
                    # figure(fig_function[0]+'_newton_s')
                    row_arr = read_equation_coeff_from_csv(filename)
                    points = read_points_from_csv(path+fig_function[0]+'.csv')
                    step_size = (max(points[0])- min(points[0]))/1000 # equation
                    figure(fig_function[0] + '_newton_s')
                    plot_newton_equation(row_arr[0][0:-1], points[0], [min(points[0]),max(points[0])],step_size,'c')
                    figure(fig_function[0] + '_newton_2x_s')
                    plot_newton_equation_points(row_arr[0][0:-1], points[0],2,'cs')
                    figure(fig_function[0] + '_newton_4x_s')
                    plot_newton_equation_points(row_arr[0][0:-1], points[0], 4, 'cs')
                    figure(fig_function[0] + '_newton_8x_s')
                    plot_newton_equation_points(row_arr[0][0:-1], points[0], 8, 'cs')
                else :
                    print("error\n")
            #plot spline data
            elif "spline" in fig_function[1]:
                if "elimination" in fig_function[2]:
                    row_arr = read_equation_coeff_from_csv(filename)
                    for row in row_arr:
                        # plot equation
                        step_size = (float(row[-1]) - float(row[-2]))/1000
                        # plot 2x
                        step_size_2x = (float(row[-1]) - float(row[-2])) / 2
                        # plot 4x
                        step_size_4x = (float(row[-1]) - float(row[-2])) / 4
                        # plot 8x
                        step_size_8x = (float(row[-1]) - float(row[-2])) / 8
                        if step_size == 0.0:
                            print("range are equal in file: ",filename,"\nrow: ",row)
                        else:
                            figure(fig_function[0] + '_spline')
                            plot_polynomial_equation(row[0:-2], row[-2:len(row)], step_size)
                            figure(fig_function[0] + '_spline_2x')
                            plot_polynomial_equation(row[0:-2], row[-2:len(row)], step_size_2x,'ys')
                            figure(fig_function[0] + '_spline_4x')
                            plot_polynomial_equation(row[0:-2], row[-2:len(row)], step_size_4x,'ys')
                            figure(fig_function[0] + '_spline_8x')
                            plot_polynomial_equation(row[0:-2], row[-2:len(row)], step_size_8x,'ys')
                elif "seidel" in fig_function[2]:
                    row_arr = read_equation_coeff_from_csv(filename)
                    for row in row_arr:
                        # plot equation
                        step_size = (float(row[-1]) - float(row[-2]))/1000
                        # plot 2x
                        step_size_2x = (float(row[-1]) - float(row[-2])) / 2
                        # plot 4x
                        step_size_4x = (float(row[-1]) - float(row[-2])) / 4
                        # plot 8x
                        step_size_8x = (float(row[-1]) - float(row[-2])) / 8
                        if step_size == 0.0:
                            print("range are equal in file: ",filename,"\nrow: ",row)
                        else:
                            figure(fig_function[0] + '_spline_s')
                            plot_polynomial_equation(row[0:-2], row[-2:len(row)], step_size,'c')
                            figure(fig_function[0] + '_spline_2x_s')
                            plot_polynomial_equation(row[0:-2], row[-2:len(row)], step_size_2x,'cs')
                            figure(fig_function[0] + '_spline_4x_s')
                            plot_polynomial_equation(row[0:-2], row[-2:len(row)], step_size_4x,'cs')
                            figure(fig_function[0] + '_spline_8x_s')
                            plot_polynomial_equation(row[0:-2], row[-2:len(row)], step_size_8x,'cs')
                else:
                    print("error")

            else:
                print("wrong function in file name")
        else:
            print('Error wrong file name many _')
show()
#draw part two
path = "./project1/part_two_datasets/"
for (dirpath, dirnames, filenames) in walk(path):
    for filename in filenames:
        fig_function = filename.split('_')
        filename = path + filename
        # draw raw data on newton and spline
        if len(fig_function) == 1:
            points = read_points_from_csv(filename)
            figure(fig_function[0].split('.')[0] + '_linear_elimination')
            plot_points(points, 'b')
            figure(fig_function[0].split('.')[0] + '_linear_sidel')
            plot_points(points, 'b')
            figure(fig_function[0].split('.')[0] + '_polynomial_elimination')
            plot_points(points, 'b')
            figure(fig_function[0].split('.')[0] + '_polynomial_sidel')
            plot_points(points, 'b')
        elif len(fig_function) == 3:
            #plot Newton data
            points = read_points_from_csv(fig_function[0]+'.csv')
            if "linear" in fig_function[1]:
                if "elimination" in fig_function[2]:
                    row_arr = read_equation_coeff_from_csv(filename)
                    if len(row_arr[0]) == 2:
                        step_size = (max(points[0]) - min(points[0])) / 1000  # equation
                        figure(fig_function[0] + '_linear_elimination')
                        plot_linear_equation(row_arr[0],points,step_size,[min(points[0]),max(points[0])])
                    elif len(row_arr[0]) == 3:
                        step_size = (max(points[0]) - min(points[0])) / 1000  # equation
                        step_size = (max(points[1]) - min(points[1])) / 1000  # equation
                        figure(fig_function[0] + '_linear_elimination')
                        plot_linear_equation(row_arr[0],points, step_size, [min(points[0]), max(points[0])], [min(points[1]), max(points[1])])
                    else:
                        print("error! _linear_elimination")
                elif "sidel" in fig_function[2]:
                    row_arr = read_equation_coeff_from_csv(filename)
                    if len(row_arr[0]) == 2:
                        step_size = (max(points[0]) - min(points[0])) / 1000  # equation
                        figure(fig_function[0] + '_linear_sidel')
                        plot_linear_equation(row_arr[0],points,step_size,[min(points[0]),max(points[0])])
                    elif len(row_arr[0]) == 3:
                        step_size = (max(points[0]) - min(points[0])) / 1000  # equation
                        step_size = (max(points[1]) - min(points[1])) / 1000  # equation
                        figure(fig_function[0] + '_linear_sidel')
                        plot_linear_equation(row_arr[0],points, step_size, [min(points[0]), max(points[0])], [min(points[1]), max(points[1])])
                    else:
                        print("error! _sidel_elimination")
            elif "polynomial" in fig_function[1]:
                if "elimination" in fig_function[2]:
                    row_arr = read_equation_coeff_from_csv(filename)
                    step_size = (max(points[0]) - min(points[0])) / 1000  # equation
                    figure(fig_function[0] + '_polynomial_elimination')
                    plot_polynomial_equation(row_arr[0],[min(points[0]),max(points[0])],step_size)

                elif "sidel" in fig_function[2]:
                    row_arr = read_equation_coeff_from_csv(filename)
                    step_size = (max(points[0]) - min(points[0])) / 1000  # equation
                    figure(fig_function[0] + '_polynomial_sidel')
                    plot_polynomial_equation(row_arr[0],[min(points[0]),max(points[0])],step_size)

show()