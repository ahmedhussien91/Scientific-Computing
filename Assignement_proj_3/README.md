# Scientific-Computing
this project contain code for those two requirements:
1. Matrix Multiplication on GPU
a) Implement the matrix multiplication operation on the graphics processing unit (GPU)
i. with the use of shared memory
ii. without the use of shared memory (check nVidia CUDA Programming Guide).
b) Compare obtained results with a CPU implementation of the same operation.
c) Provide explanation for obtained results.
Notes:
Matrix Multiplication Operation: https://en.wikipedia.org/wiki/Matrix_multiplication

2. Prewitt Filter on GPU
Prewitt 3x3 filter performs edge detection by finding the boundaries of the image where 
there is significant difference as compared to neighboring pixels.

Pseudo Code for Prewitt Filter
Provided with this assignment a simple Prewitt Filter code that runs on CPU and 
performs edge detection on PPM image files. You are required to:
a) Implement Prewitt filter on GPU
i. with the use of shared memory
ii. without the use of shared memory
b) Compare obtained results with a CPU implementation for different image sizes and 
estimate the expected speedup for different image sizes.
Notes:
You can open, edit, and convert image files to PPM format using the free image editor 
GIMP https://www.gimp.org/
Prewitt Operator: https://en.wikipedia.org/wiki/Prewitt _operator

Deliverables
source code for the project including your implementation files.
This exection time for different image sizes and using different techniques for Matrix multiplication (./MatrixMul/Execution Time.txt)
This exection time for different image sizes and using different techniques for Perwitt (./Perwitt/PERWIT/Execution Time.txt)