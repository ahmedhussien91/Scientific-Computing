__author__ = 'asabry'

import sys
import csv
from os import  walk
from numpy import *
from matplotlib.pyplot import plot, show, figure, axes
from mpl_toolkits.mplot3d import axes3d, Axes3D

def read_file_into_array(FileName, OutArrays):
    with open(FileName, 'r') as FileObj:
        file_array = FileObj.readlines()
        time_arr = []
        particle_arr = [[]]
        for line in file_array:
            if "t" in line:
                time_arr.append(float(line.split('=')[1].strip('\n')))
            elif "P" in line:
                particle_arr[int(line.split('=')[0].strip('P'))].append(float(line.split('=')[1].strip('\n')))
                if (len(particle_arr) <= 63):
                    particle_arr.append([])

        OutArrays[0] = time_arr
        OutArrays.extend(particle_arr)

out_arr_ref = [[]]
out_arr_Euler = [[]]
out_arr_MidPoint = [[]]
out_arr_Heun = [[]]
out_arr_RK = [[]]
out_arr_RK_A = [[]]
read_file_into_array("test_currentSys_RK_ref_001.txt", out_arr_ref)
read_file_into_array("test_currentSys_euler_0.1.txt", out_arr_Euler)
read_file_into_array("test_currentSys_MidPoint_0.1.txt", out_arr_MidPoint)
read_file_into_array("test_currentSys_Heun_0.1.txt", out_arr_Heun)
read_file_into_array("test_currentSys_RK_0.1.txt", out_arr_RK)
read_file_into_array("test_currentSys_RK_A_0.1.txt", out_arr_RK_A)
figure(1)
# plot all methods of 2nd particle in the cloth peace against time
particaleNum = 2
plot(out_arr_ref[0],out_arr_ref[particaleNum],'k-')
plot(out_arr_Euler[0],out_arr_Euler[particaleNum],'b-')
figure(2)
plot(out_arr_ref[0],out_arr_ref[particaleNum],'k-')
plot(out_arr_MidPoint[0],out_arr_MidPoint[particaleNum],'g-')
figure(3)
plot(out_arr_ref[0],out_arr_ref[particaleNum],'k-')
plot(out_arr_RK[0],out_arr_RK[particaleNum],'r-')
figure(4)
plot(out_arr_ref[0],out_arr_ref[particaleNum],'k-')
plot(out_arr_RK_A[0],out_arr_RK_A[particaleNum],'y-')
figure(5)
plot(out_arr_ref[0],out_arr_ref[particaleNum],'k-')
plot(out_arr_Heun[0],out_arr_Heun[particaleNum],'g-')

maxError_euler = 0
maxError_MidPoint = 0
maxError_Heun = 0
maxError_RK = 0
maxError_RK_A = 0
maxError_euler = 0

for out_arr_Euler_index,out_arr_Euler_time in enumerate(out_arr_Euler[0]):
    for out_arr_ref_index,out_arr_ref_time in enumerate(out_arr_ref[0]):
        if(out_arr_Euler_time == out_arr_ref_time):
            if(maxError_euler < abs(out_arr_ref[particaleNum][out_arr_ref_index] - out_arr_Euler[particaleNum][out_arr_Euler_index])):
                maxError_euler = abs(out_arr_ref[particaleNum][out_arr_ref_index] - out_arr_Euler[particaleNum][out_arr_Euler_index])

for out_arr_MidPoint_index,out_arr_MidPoint_time in enumerate(out_arr_MidPoint[0]):
    for out_arr_ref_index,out_arr_ref_time in enumerate(out_arr_ref[0]):
        if(out_arr_MidPoint_time == out_arr_ref_time):
            if(maxError_MidPoint < abs(out_arr_ref[particaleNum][out_arr_ref_index] - out_arr_MidPoint[particaleNum][out_arr_MidPoint_index])):
                maxError_MidPoint = abs(out_arr_ref[particaleNum][out_arr_ref_index] - out_arr_MidPoint[particaleNum][out_arr_MidPoint_index])


for out_arr_Heun_index,out_arr_Heun_time in enumerate(out_arr_Heun[0]):
    for out_arr_ref_index,out_arr_ref_time in enumerate(out_arr_ref[0]):
        if(out_arr_Heun_time == out_arr_ref_time):
            if(maxError_Heun < abs(out_arr_ref[particaleNum][out_arr_ref_index] - out_arr_Heun[particaleNum][out_arr_Heun_index])):
                maxError_Heun = abs(out_arr_ref[particaleNum][out_arr_ref_index] - out_arr_Heun[particaleNum][out_arr_Heun_index])

for out_arr_RK_index,out_arr_RK_time in enumerate(out_arr_RK[0]):
    for out_arr_ref_index,out_arr_ref_time in enumerate(out_arr_ref[0]):
        if(out_arr_RK_time == out_arr_ref_time):
            if(maxError_RK < abs(out_arr_ref[particaleNum][out_arr_ref_index] - out_arr_RK[particaleNum][out_arr_RK_index])):
                maxError_RK = abs(out_arr_ref[particaleNum][out_arr_ref_index] - out_arr_RK[particaleNum][out_arr_RK_index])

for out_arr_RK_A_index,out_arr_RK_A_time in enumerate(out_arr_RK_A[0]):
    for out_arr_ref_index,out_arr_ref_time in enumerate(out_arr_ref[0]):
        if(out_arr_RK_A_time == out_arr_ref_time):
            if(maxError_RK_A < abs(out_arr_ref[particaleNum][out_arr_ref_index] - out_arr_RK_A[particaleNum][out_arr_RK_A_index])):
                maxError_RK_A = abs(out_arr_ref[particaleNum][out_arr_ref_index] - out_arr_RK_A[particaleNum][out_arr_RK_A_index])

print("Max Error:\n")
print("\tEuler = ",maxError_euler)
print("\tMidpoint = ",maxError_MidPoint)
print("\tHeun = ",maxError_Heun)
print("\tRK = ",maxError_RK)
print("\tRK_A= ",maxError_RK_A)
show()