Documentation of python file - [[Extended_Data]].py


#### Libraries:
1) torch
2) math
3) os


#### Variables:
1) dev - torch device variable signifying CPU/GPU availability and usage

2) N_E - number of training examples
3) N_CV - number of cross validation examples
4) N_T - number of testing examples
5) T - number of timesteps or sequence length for a linear training case
6) T_test - number of timesteps or sequence length for a linear testing case
7) r2 - identity tensor of order 1
8) r - observation noise tensor, which is the square root of r2
9) vdB - additive white Gaussian noise of -20 dB
10) v - 10 to the power of (vdB/10)
11) q2 - product of v and r2
12) q - process noise tensor, which is the square root of q2
13) F10 - state evolution tensor of order 10
14) H10 - observation tensor of order 10
15) m, n - dimensions of state evolution matrix
16) F - state evolution tensor based on dimensions m and n
17) H - observation identity tensor based on dimensions m and n
18) m1_0 - tensor of zeroes of dimension mx1
19) m2_0 - product of 0 and identity tensor of order m

20) alpha_degree - angle value for matrix rotation
21) rotate_alpha - tensor of order 1 with value based on alpha_degree and torch.pi
22) cos_alpha - cosine of rotate_alpha
23) sin_alpha - sine of rotate_alpha
24) rotate_matrix - tensor of order 2 with values based on combinations of cos_alpha and sin_alpha


#### Methods:
1) __DataGen_True__:
	-  Input parameters:
		- SysModel_data - Object of class SystemModel of python file [[Linear_sysmdl]].
		- fileName - a string denoting the path to the .pt file.
		- T - number of timesteps or sequence length for a linear training case.
	- Functioning:
		- 
