Documentation of python file - [[KalmanFilter_test]].py


#### Libraries:
1) torch
2) torch.nn
3) time


#### Dependencies:
1) KalmanFilter class from [[Linear_KF]] python file


#### Methods:
1) __KFTest()__:
	- Input parameters:
		- args - object containing the argument values
		- SysModel - object of the class SystemModel (motion model and observation model system)
		- test_input - testing input parameters
		- test_target - testing target values
		- allStates - loss on all states, set as True by default (if false: only calculate loss on position)
		- randomInit - random initialization condition, set as False
		- test_init - known testing random initial state
		- test_lengthMask - mask to handle input with varying lengths
	- Purpose:
		- The purpose of this section is to evaluate linear kalman filter on the testing data and observe the results of traditional kalman filtering.
	- Functioning:
		- The first step defines the loss_fn variable, which is torch.nn module's MSE loss function, where reduction = 'mean' indicates that the MSE loss will be averaged across all elements in the batch to obtain a single scalar value representing the mean MSE loss.
		- Initializing a tensor, MSE_KF_linear_arr, of zeroes of size args.N_T which represent the number of test sequences. It allocates memory for loss values for every test sequence
		- Initializing a tensor, KF_out, of zeroes of 3 dimensions (args.N_T, SysModel.m, args.T_test) to store output of Kalman Filtering process.
		- If allStates is set as False, loss will not be calculated on all states except positions. Depending on the number of states, loc tensor is initialized with True values for position and False values for velocity/acceleration.
		- To record execution time, start variable is set to current time using time.time().
		- The next step is to create an object, KF, of the KalmanFilter class with SysModel and args passed as init parameters.
		- Following the initialization, the initialization and forward computation of Kalman Filters is carried out. If the randomInit value is set to True, Init_batched_sequence() method of KF object is called with (test_init, SysModel.m2x_0.view(1, SysModel.m, SysModel.m).expand(args.N_T, -1, -1)) passed as parameters, where test_init is the known testing random initial state and m2x_0 tensor is reshaped using view() to the dimensions mentioned and expanded using expand() along the first dimension based on the number of testing sequences.<br>Else, the Init_batched_sequence() method is called with parameters (SysModel.m1x_0.view(1, SysModel.m, 1).expand(args.N_T, -1, -1), SysModel.m2x_0.view(1, SysModel.m, SysModel.m).expand(args.N_T, -1, -1)), where m1x)0 and m2x_0 are reshaped and expanded in their respective manner.
		- To test the Kalman Filter, GenerateBatch() method of KF object is called with test_input as parameter.
		- Execution time is recorded by setting end variable as current time and subtracting start from end. It is stored in a variable, t.
		- The output of the Kalman Filtering is stored in
		- KF_out.
		- Next step is to calculate the loss. A loop is applied for number of testing sequences, args.N_T. If allStates is set to True, the loss is calculated for all the states else it is only calculated for position. Additionally, if args.randomLength is set to True, test_lengthMask is used to handle the sequences. The loss is calculated using loss_fn defined at the start and the corresponding predicted value and target value are passed to the function. The loss is then stored inside the MSE_KF_linear_arr tensor defined above.
		- Once, losses for all testing sequences are calculated, MSE_KF_linear_avg stores the average of all the losses.
		- Also, MSE_KF_dB_avg stores the average loss in decibels.
		- MSE_KF_linear_std stores the standard deviation of all the losses using the parameter unbiased set as True, which calculates the unbiased estimate of the standard deviation using Bessel's correction to adjust for the bias introduced by using a sample instead of the whole population.
		- The confidence interval in decibels is also calculated and store in KF_std_dB variable.
		- Finally, the values MSE_KF_dB_avg, KF_std_dB, and t are displayed.
	- Return Values:
		- The function returns a list/array of MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, and KF_out.