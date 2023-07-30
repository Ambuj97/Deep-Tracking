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
				- The purpose of this method is to predict the 1st order and the 2nd order moments of x and y and carry out the predict step in Kalman filtering.
			- Functioning:
				- The first step is to calculate 1st order statistical moments or priori moments of x by doing a batched matrix multiplication of self.batched_F with self.m1x_posterior tensors and store it in self.m1x_prior.
				- The second step is to calculate 2nd order statistical moments of x by doing a batched matrix multiplication of self.batched_F with self.m2x_posterior and store it in self.m2x_prior. The self.m2x_prior calculated is further used used in batch matrix multiplication with self.batched_F_T and process noise, self.Q is added to the product to get the final self.m2x_prior.
				- The third step is to calculate the 1st order statistical moment of y by doing a batched matrix multiplication of self.batched_H with self.m1x_prior tensors and store it in self.m1y.
				- The final step is to calculate 2nd order statistical moments of y by doing a batched matrix multiplication of self.batched_H with self.m2x_prior and store it in self.m2y. The self.m2y calculated is further used used in batch matrix multiplication with self.batched_H_T and observation noise, self.R is added to the product to get the final self.m2y.
		1) __KGain()__:
			- Purpose:
				- The purpose of this method is to carry out the Kalman Gain computation.
			- Functioning:
				- First, self.KG is calculated by doing a batch matrix multiplication between self.m2x_prior and self.batched_H_T.
				- Second, final self.KG is calculated by doing a batch matrix multiplication between self.KG calculated in the previous step and inverse of the self.m2y tensor.
		1) __Innovation()__:
			- Input parameters:
				- y - predicted value
			- Purpose:
				- The purpose of this method is to calculate the innovation value.
			- Functioning:
				- The innovation value, self.dy, is the difference between predicted value, y, and the estimated value, self.m1y.
		1) __Correct()__:
			- Purpose:
				- The purpose of this method is to perform correction by computing the new posteriori moments
			- Functioning:
				- The first step computes the new posteriori moments, self.m1x_posterior using the priori moments, self.m1x_prior and batch matrix multiplication between the computed Kalman Gain, self.KG and the innovation value, self.dy.
				- The next step is to compute the second posteriori moment, self.m2x_posterior. A batch matrix multiplication is done between self.m2y and transpose of self.KG along the first and the second dimensions. Finally, self.m2x_posterior is computed by subtracting the output of batch matrix multiplication between selg.KG and previously computed self.m2x_posterior from self.m2x_prior.
		1) __Update()__:
			- Input parameters:
				- y - predicted value
			- Purpose:
				- The purpose of this method is to carry out the update step in Kalman filtering.
			- Functioning:
				- Call the Predict class method.
				- Call the KGain class method.
				- Call the Innovation class method.
				- Call the Correct class method.
			- Return values:
				- self.m1x_posterior, the new 1st posteriori moment
				- self.m2x_posterior, the new 2md posteriori moment
		1) __Init_batched_sequence()__:
			- Input parameters:
				- m1x_0_batch - initial state values
				- m2x_0_batch - initial covariance values
			- Purpose:
				- The purpose of this method is to initialize the instance variables self.m1x_0_batch and self.m2x_0_batch.
			- Functioning:
				- self.m1x_0_batch is set to m1x_0_batch parameter received.
				- self.m2x_0_batch is set to m2x_0_batch parameter received.
		1) __GenerateBatch()__:
			- Input parameters:
			- Purpose:
			- Functioning: