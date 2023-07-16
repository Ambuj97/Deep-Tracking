Documentation of python file - [[Linear_sysmdl]].py


#### Libraries:
1) torch


#### Variable Declarations:
1) dev - torch device variable signifying CPU/GPU availability and usage


#### Classes:
1) SystemModel:
	1) Methods:
		1) init():
			- Input parameters:
				- F - state evolution matrix.
				- q - process noise.
				- H - observation matrix.
				- r - observation noise.
				- T - number of timesteps or sequence length for a linear training case.
				- T_test - number of timesteps or sequence length for a linear testing case.
				- prior_Q - prior process noise covariance matrix (also used for initialization).
				- prior_Sigma - prior Sigma value (also used for initialization).
				- prior_S - prior S value (also used for initialization).
			- Purpose:
				- The purpose of this method is to initialize the instance variables.
			- Functioning:
				- It initializes the instance variables based on the input parameters.
				- Instance variables m and n are set according to the dimensions of F and K matrices respectively.
				- Process noise covariance matrix, Q, is initialized according to the process noise 'q' and identity tensor of dimension 'm' and observation noise matrix, R, is initialized according to the dimension of observation noise 'r' and identity tensor of dimension 'n'.
				- If prior_Q value is None, it is set to an identity tensor of order m, else it is set to the value passed in the parameter.
				- If prior_Sigma value is None, it is set to a tensor of zeroes of order mxn, else it is set to the value passed in the parameter.
				- If prior_S value is None, it is set to an identity tensor of order n, else it is set to the value passed in the parameter.


