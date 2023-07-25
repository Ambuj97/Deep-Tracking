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


#### Variable Declarations:
1) today - current date
2) now - current time
3) strToday - formatted date in string
4) ) strNow - formatted time in string
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