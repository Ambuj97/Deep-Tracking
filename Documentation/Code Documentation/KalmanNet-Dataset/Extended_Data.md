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
8) r - observation noise tensor, which is the square root of 
9) q - process noise tensor
10) 