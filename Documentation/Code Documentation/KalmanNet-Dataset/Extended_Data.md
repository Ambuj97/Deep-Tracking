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
8) r - observation noise, which is the square root of r2
9) vdB - additive white Gaussian noise of -20 dB
10) v - 10 to the power of (vdB/10)
11) q2 - product of v and r2
12) q - process noise, which is the square root of q2
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
	- Purpose:
		- The purpose of this method is to check data generation process and save it as a torch model at the path specified.
	- Functioning:
		- It calls the __GenerateBatch__ method of the SysModel_data object with parameters number of examples = 1, T, randomInit = False.
		- SysModel_data object's instance variable 'Input' is assigned to test_input variable.
		- SysModel_data object's instance variable 'Target' is assigned to test_target variable.
		- Finally, the two values of test_input and test_target are stored in a vector and saved as a torch model at the path (fileName) provided.

2) __DataGen__:
	- Input parameters:
		- SysModel_data - Object of class SystemModel of python file [[Linear_sysmdl]].
		- fileName - a string denoting the path to the .pt file.
		- T - number of timesteps or sequence length for a linear training case.
		- T_test - number of timesteps or sequence length for a linear testing case.
		- randomInit - flag indication random initialization, set as False.
	- Purpose:
		- The purpose of this method is to generate training, cross validation, and testing dataset and save it as a torch model at the path specified.
	- Functioning:
		- It calls the __GenerateBatch__ method of the SysModel_data object with parameters number of examples, T/T_test, randomInit = False.  Number of examples is set as either N_E or N_CV or N_T depending upon the generation of training, cross validation or testing dataset. The timesteps is set to either T (for training and cross validation) or T_test (for testing)
		- Corresponding instance variables of the object 'Input' and 'Target' are assigned to corresponding training, cross validation, and testing input and target variables.
		- Finally, all the training, cross validation, and testing variables are stored in a vector and saved as a torch .pt model at the specified path (fileName) provided.

3) __DataLoader__:
	- Input parameters:
		- fileName - path at which .pt model is stored.
	- Purpose:
		- The purpose of this function is to load the data stored inside the .pt model onto the CPU.
	- Functioning:
		- Loads the data from the .pt model stored at the provided path using torch's load() method into respective training, cross validation, and testing variables.
	- Return values:
		- It returns a vector of all the training, cross validation, and testing input and target values.

4) __DataLoader_GPU__:
	- Input parameters:
		- fileName - path at which .pt model is stored.
	- Purpose:
		- The purpose of this function is to load the data stored inside the .pt model onto the device available (stored in the 'dev' variable).
	- Functioning:
		- Loads the data from the .pt model stored at the provided path using torch's load() method, creates a DataLoader object and stores the values in respective training, cross validation, and testing variables.
		- The pin_memory = False signifies that the data will not be pinned in memory. Pinning memory can improve data transfer performance, especially when using CUDA, but it might also consume additional memory.
		- The map_location specifies the device where the data will be loaded, in this case onto the device available (stored in the 'dev' variable).
		- The .squeeze() torch method is used to removed unwanted singleton dimensions.
	- Return values:
		- It returns a vector of all the training, cross validation, and testing input and target values.



Note: Documentation of DecimateData, Decimate_and_perturbate_Data, getObs, Short_Traj_Split methods is remaining.