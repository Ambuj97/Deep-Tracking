Documentation of python file - [[main_linear_CA]].py


#### Libraries:
1) torch
2) datetime


#### Dependencies:
1) SystemModel class from [[Linear_sysmdl]] python file.
2) general_settings method from [[config]] python file.
3) DataGen method from [[utils]] python file.
4) F_gen, F_CV, H_identity, H_onlyPos, Q_gen, Q_CV, R_3, R_2, R_onlyPos, m, m_cv, R_onlyPosBB, R_7, Q_bb, F_genbb, m_bb variables from [[parameters]] python file.
5) KFTest method from [[KalmanFilter_test]] python file.
6) KalmanNetNN class from [[KalmanNet_nn]] python file.
7) Pipeline_EKF class from [[Pipeline_EKF]] python file.
8) Plot_KF class from [[Plot]] python file.


#### Variable Declarations and Definitions:
1) today - current date
2) now - current time
3) strToday - formatted date in string
4) strNow - formatted time in string
5) strTime - concatenated string of strToday and strNow

6) path_results -  path where results are stored
7) offset - init condition of dataset (set as 0)
8) args - parse_args object containing the argument values
9) args.N_E - set as1000
10) args.N_CV - set as 100
11) args.N_T - set as 200
12) args.randomInit_train - set as True
13) args.randomInit_cv - set as True
14) args.randomInit_test - set as True
15) args.T - set as 100
16) args.T_test - set as 100
17) args.use_cuda - set as True
18) args.n_steps - set as 500
19) args.n_batch - set as 10
20) args.lr - set as 1e-4
21) args.wd - set as 1e-4

22) device - torch device variable signifying CPU/GPU availability and usage

23) KnownRandInit_train - if true, use known random init for training, else model is agnostic to random init. (set as True)
24) KnownRandInit_cv - if true, use known random init for cross validation, else model is agnostic to random init. (set as True)
25) KnownRandInit_test - if true, use known random init for testing, else model is agnostic to random init. (set as True)

26) std_gen - 0 or 1 value depending on condition if(args.randomInit_train or args.randomInit_cv or args.randomInit_test)
27) std_feed - 0 or 1 depending on condition if(KnownRandInit_train or KnownRandInit_cv or KnownRandInit_test)

28) m1x_0 - initial state matrix, set as a tensor of zeroes based on dimension m/m_bb
29) m1x_0_cv - initial state matrix for constant velocity, set as a tensor of zeroes based on dimension m_cv
30) m2x_0 - initial Covariance for feeding to filters and KNet, set as an identity tensor based on dimension m/m_bb multiplied by std_feed^2
31) m2x_0_gen - initial Covariance for generating dataset, set as an identity tensor based on dimension m/m_bb multiplied by std_gen^2
32) m2x_0_cv - initial Covariance for constant velocity, set as an identity tensor based on dimension m_cv multiplied by std_feed^2

33) Loss_On_AllState - boolean variable signifying whether to only calculate loss on position or not. Default set as False
34) Train_Loss_On_AllState - boolean variable signifying whether to only calculate training loss on position or not. Default set as False
35) CV_model - boolean variable signifying whether to use CV model or CA model. Default set as False

36) DatafolderName - path to the data folder
37) DatafileName - file name of the pytorch model

38) sys_model_gen - object of the class SystemModel imported from [[Linear_sysmdl]] python file
39) H_onlyPos - a tensor of order 2, defining the observation matrix considering position only
40) sys_model - another object of the class SystemModel imported from [[Linear_sysmdl]] python file


#### Purpose:
[[main_linear_CA]].py file is designed to create a linear dataset of constant velocity or constant acceleration.


#### Functioning:
Important steps involved:
1) Initialization of configuration variables/arguments mentioned under variables section.
2) Initialization of state variables, covariance matrices, and other dataset generation variables mentioned under the variables section.
3) An object, sys_model_gen, of the __'SystemModel'__ class is created. The class init() method initializes the instance variables based on the values (F_genbb, Q_bb, H_onlyPos, R_onlyPosBB, args.T, args.T_test) passed during object creation. __'InitSequence'__ method of class SystemModel is called with values (m1x_0, m2x_0_gen). The InitSequence method initializes further instance variables based on the values passed as parameters to the method.
4) If constant velocity model is used, i.e. CV_model is set to True, sys_model object of the __'SystemModel'__ class is created with class init() method initializing the instance variables based on the values (F_CV, Q_CV, H_onlyPos, R_onlyPos, args.T, args.T_test) passed during object creation. __'InitSequence'__ method of class SystemModel is called with values (m1x_0_cv, m2x_0_cv). The InitSequence method initializes further instance variables based on the values passed as parameters to the method.<br>
	Else, sys_model object of the __'SystemModel'__ class is created with class init() method initializing the instance variables based on the values (F_genbb, Q_bb, H_onlyPos, R_onlyPosBB, args.T, args.T_test) passed during object creation. __'InitSequence'__ method of class SystemModel is called with values (m1x_0, m2x_0). The InitSequence method initializes further instance variables based on the values passed as parameters to the method.
1) __DataGen()__ method of [[utils]] python file is called to generate training, cross-validation, and testing input, target, and init datasets. Parameters passed to the method are (args, sys_model_gen, DatafolderName+DatafileName) where args
