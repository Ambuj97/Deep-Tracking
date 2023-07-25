Documentation of python file - [[config]].py


#### Libraries:
1) argparse


#### Methods:
1) __general_settings()__:
	- Purpose:
		- The purpose of this method is to set the dataset settings, training settings, Kalman net settings for the simulation.
	- Functioning:
		- It creates an ArgumentParser object, parser, which is used to pass define and parse command-line arguments.
		- Various command line arguments are added using add_argument() method of the object. Their argument names, data types, default values, meta variable name, and help messages are mentioned for every argument.
		- Finally, the arguments are processed using parse_args() method of the object and stored in args variable.
	- Arguments:
		- Dataset size settings:
			- N_E - input training dataset size (# of sequences)
			- N_CV - input cross validation dataset size (# of sequences)
			- N_T - input test dataset size (# of sequences)
			- T - input sequence length
			- T_test - input test sequence length
		- Dataset random length settings:
			- randomLength - if True, consider random sequence length
			- T_max - if random sequence length is set to True, input max sequence length
			- T_min -  if random sequence length is set to True, input min sequence length
		- Dataset random initial state settings:
			- randomInit_train - if True, consider random initial state for training set.
			- randomInit_cv - if True, consider random initial state for cross validation set
			- randomInit_test - if True, consider random initial state for test set
			- variance - input variance for the random initial state with uniform distribution
			- distribution - input distribution for the random initial state (uniform/normal)
		- Training settings:
			- use_cuda - if True, use CUDA
			- n_steps - number of training steps (default: 1000)
			- n_batch - input batch size for training (default: 20)
			- lr - learning rate (default: 1e-3)
			- wd - weight decay (default: 1e-4)
			- CompositionLoss - if True, use composition loss
			- alpha - input alpha (0,1) for the composition loss
		- KalmanNet settings:
			- in_mult_KNet - input dimension multiplier for KNet
			- out_mult_KNet - output dimension multiplier for KNet
	- Return values:
		- args - processed command-line commands using parse_args() method. It is an object containing the argument values.
