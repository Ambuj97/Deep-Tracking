Documentation of python file - [[Linear_KF]].py


#### Libraries:
1) torch


#### Classes:
1) KalmanFilter:
	1) Methods:
		1) __init()__:
			- Input parameters:
				- SystemModel - object of the class SystemModel (motion model and observation model system)
				- args - object containing the argument values
			- Purpose:
				- The purpose of the init method is to initialize the instance variables.
			- Functioning:
				-  Instance variable self.device is set to cuda if args.use_cuda is True, else it is set to cpu.
				- State evolution matrix instance variable, self.F, is set to SystemModel.F, the state evolution matrix part of SystemModel.
				- Dimension instance variable of the state evolution matrix, self.m, is set to SystemModel.m, the dimension of the state evolution matrix part of SystemModel.
				- Process noise covariance matrix instance variable, self.Q, is set to SystemModel.Q and the tensor is moved on to the device specified in self.device.
				- Observation matrix instance variable, self.H, is set to SystemModel.H.
				- Dimension instance variable of the observation matrix, self.n, is set to SystemModel.n, the dimension of the observation matrix part of SystemModel.
				- Observation noise covariance matrix instance variable, self.R, is set to SystemModel.R and the tensor is moved on to the device specified in self.device.
				- Training time sequence instance variable, self.T, is set to SystemModel.T, training time sequence part of SystemModel.
				- Testing time sequence instance variable, self.T_test, is set to SystemModel.T_test, testing time sequence part of SystemModel.
		1) __Predict()__:
			- Purpose:
				- The purpose of this method is to predict the 1st order and the 2nd order moments of x and y.
			- Functioning:
				- The first step is to calculate 1st order statistical moments or priori moments of x by doing a batched matrix multiplication of self.batched_F with self.m1x_posterior tensors and store it in self.m1x_prior.
				- The second step is to calculate 2nd order statistical moments of x by doing a batched matrix multiplication of self.batched_F with self.m2x_posterior and store it in self.m2x_prior. The self.m2x_prior calculated is further used used in batch matrix multiplication with self.batched_F_T and process noise, self.Q is added to the product to get the final self.m2x_prior.
				- The third step is to calculate the 1st order statistical moment of y by doing a batched matrix multiplication of self.batched_H with self.m1x_prior tensors and store it in self.m1y.
				- The final step is to calculate 2nd order statistical moments of y by doing a batched matrix multiplication of self.batched_H with self.m2x_prior and store it in self.m2y. The self.m2y calculated is further used used in batch matrix multiplication with self.batched_H_T and observation noise, self.R is added to the product to get the final self.m2y.
		1) __KGain()__:
			- Purpose:
			- Functioning:
		2) __Innovation()__:
			- Input parameters:
			- Purpose:
			- Functioning:
		3) __Correct()__:
			- Input parameters:
			- Purpose:
			- Functioning:
		4) __Update()__:
			- Input parameters:
			- Purpose:
			- Functioning:
			- Return values:
		5) __Init_batched_sequence()__:
			- Input parameters:
			- Purpose:
			- Functioning:
		6) __GenerateBatch()__:
			- Input parameters:
			- Purpose:
			- Functioning: