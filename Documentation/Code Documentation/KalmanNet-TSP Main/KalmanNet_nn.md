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
				- First of all, instance variables are defined where self.seq_len_input is set to 1 as the KalmanNet calculates the KG time step by time step, self.batch_size is set to batch size mentioned in the args.n_batch argument, self.prior_Q is set to prior_Q parameter passed to the method, self.prior_Sigma is set to prior_Sigma parameter, self.prior_S is set to prior_S parameter
				- The next steps define the NN for KG computation.
				- The GRU for tracking process noise covariance matrix, Q, is defined along with input feature dimensions required and hidden nodes. self.d_input_Q is set to product of the dimension of state evolution matrix, self.m and args.in_mult_KNet value. self.d_hidden_Q is set to square of self.m. Finally, the GRU layer, self.GRU_Q is created using torch's nn.GRU with number of input nodes and hidden nodes mentioned as arguments and is loaded onto the device available.
				- The GRU for tracking Sigma values is defined along with input feature dimensions required and hidden nodes. self.d_input_Sigma is set to the sum of nodes in hidden layer of previous GRU and product of the dimension of state evolution matrix, self.m and args.in_mult_KNet value. self.d_hidden_Sigma is set to square of self.m. Finally, the GRU layer, self.GRU_Sigma is created using torch's nn.GRU with number of input nodes and hidden nodes mentioned as arguments and is loaded onto the device available.
				- The final GNU layer is used to track the S values and is defined along with input feature dimensions, self.d_input_S required and hidden nodes, self.d_hidden_S. The GRU layer, self.GRU_S is created using torch's nn.GRU with number of input nodes and hidden nodes mentioned as arguments and is loaded onto the device available.
