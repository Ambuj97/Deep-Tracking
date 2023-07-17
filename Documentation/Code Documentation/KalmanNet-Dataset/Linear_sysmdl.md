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
		2) f():
			- Input parameters:
				- x - the matrix storing previous posteriori state estimates.
			- Purpose:
				- The purpose of this method is to get the priori state estimates using previous posteriori state estimates.
			- Functioning:
				- It returns the matrix multiplication between instance variable 'F' and x matrix.
			- Return values:
				- It returns the priori state estimate.
		3) h():
			- Input parameters:
				- x - the matrix storing priori state estimates.
			- Purpose:
				- The purpose of this method is to get the moments of the observations using priori state estimates.
			- Functioning:
				- It returns the matrix multiplication between instance variable 'H' and x matrix.
			- Return values:
				- It returns the moments of the observations.
		4) InitSequence():
			- Input parameters:
				- m1x_0 - initial state
				- m2x_0 - ??
			- Purpose:
				- The purpose of this method is to initialize the sequence as it begins by initializing some important instance variables.
			- Functioning:
				- It assigns the values to instance variables m1x_0, x_prev, and m2x_0 based on the parameters passed to the method.
		5) GenerateSequence():
			- Input parameters:
				- Q_gen - process noise matrix
				- R_gen - observation noise matrix
				- T - number of timesteps or sequence length
			- Purpose:
				- The purpose of this method is to use linear model logic along with part of Kalman filter to generate a sequence of inputs and targets as part of the synthetic dataset.
			- Functioning:
				- Pre-allocate empty tensors for current state and current observation based on dimensions m and n respectively.
				- Set x_prev to m1x_0 and current state, xt, to x_prev (initial step).
				- Looping over the number of timesteps, T, to create input and target trajectories.
					- First step inside the loop is to predict the evolution of the state and add multivariate normal noise to the state prediction if the process noise, q, is non zero.
					- Second step inside the loop is to predict the observation value and add multivariate normal noise to the prediction if the observation noise, r, is non zero.
					- Current state and current observation are saved to the empty tensors initialized earlier, at the timestep index.
					- Finally, current state, xt, is saved to previous state, x_prev for the next iteration.
		6) GenerateBatch():
			- Input parameters:
				- size - size of the batch (size/number of examples for training, cross validation, and testing)
				- T - number of timesteps or sequence length for a linear training case.
				- randomInit - random initialization (default set to False)
				- seqInit - sequence initialization (default set to False)
				- T_test - number of timesteps or sequence length for a linear testing case (default set to 0).


