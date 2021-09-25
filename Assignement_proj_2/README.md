# Scientific-Computing
Description
Enhance a cloth simulator that is based on mass-spring model by adding improved
integrators. The basic simulator program with two integrators, Euler and Midpoint, is
provided. reimpelemnt the same simulation with integrators based on: (1) Heun’s method with iterations
(propose stopping criteria), (2) Runge-Kutta order 4, and (3) Adaptive RK methods with step
halving.
code is added to the CPhysEnv class.
To launch the simulation, use the attached OpenGL library and follow the instructions in
readme.txt to link OpenGL with Visual Studio.
Deliverables
You will need to electronically submit the source code for the project including your
implementation files as well as a functional executable.
A detailed report(in Mass-Spring+Simulation\Error Calculation\) 
explain how the program works and include justifications on how the new integrators lead to improved simulations. 
comparison of used integrators and propose metrics for error. 
investigate the effect of modifying the step size and spring stiffness (spring constant) on
simulation behaviour and stability for each integrator.