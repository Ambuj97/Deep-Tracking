Documentation of python file - [[KalmanNet_nn]].py


#### Libraries:
1) torch
2) torch.nn
3) torch.nn.functional
4) torch.nn.init


#### Classes:
1) KalmanNetNN:
	1) Methods:
		1) __init()__:
			- Inheriting functionalities from the torch.nn.Module class of PyTorch
		2) __NNBuild()__:
			1) Input parameters:
				- SysModel - object of the class SystemModel (motion model and observation model system)
				- args - object containing the argument values
			2) Purpose:
				- The purpose of this method is to initiate the process of building the neural network architecture
			3) Functioning:
				- Based on the value of args.cuda, value of instance variable self.device is set to cuda or gpu.
				- Calling self.InitSystemDynamics method of the class and passing SysModel's f, h methods along with m, n dimensions of state evolution and observation matrix.
				- Calling self.InitKGainNet method of the class and passing SysModel's prior_Q, prior_Sigma, and prior_S values along with args object.
		3) __InitKGainNet()__:
			1) Input parameters:
				- prior_Q - initial value of process noise covariance matrix
				- prior_Sigma - initial value of Sigma covariance matrix (2nd order moments of x)
				- prior_S - initial value of S covariance matrix (2nd order moments of y)
				- args - object containing the argument values
			2) Purpose:
				- The purpose of this method to initialize Kalman Gain GRU based network. This network uses the power of RNNs, specifically GRUs to learn Kalman Gain from the data. The entire NN architecture (architecture #2 of the research paper) is defined in this method.
			3) Functioning:
				- First of all, instance variables are defined. self.seq_len_input is set to 1 as the KalmanNet calculates the KG time step by time step. 
