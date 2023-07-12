Documentation of python file - [[main_linear]].py


#### Libraries:
1) torch
2) datetime


#### Dependencies:
1) SystemModel class from [[Linear_sysmdl]] python file.
2) DataGen, DataLoader, DataLoader_GPU, Decimate_and_perturbate_Data, Short_Traj_Split classes from [[Extended_data]] python file.
3) F, H, T, T_test, m1_0, m2_0, r2, q2, r, q variables from [[Extended_data]] python file.


#### Variable Declarations:
1) torch.pi - value of pi (3.1415927410125732)
2) dev - torch device variable signifying CPU/GPU availability and usage

3) today - current date
4) now - current time
5) strToday - formatted date in string
6) strNow - formatted time in string
7) strTime - concatenated string of strToday and strNow

8) path_results - path where results are stored
9) dataFolderName - name of the folder where .pt model has to be stored
10) dataFileName - name of the .pt file


#### Objects:
1) sys_model - object of the SystemModel class


#### Functioning:
Responsible for creating a dataset of linear form.
	Important steps involved:
		1) An object, sys_model, of the __'SystemModel'__ class is created. The class init() method initializes the instance variables based on the values (F, q, H, r, T, T_test) passed during object creation.
		2) __'InitSequence'__ method of class SystemModel is called with values (m1_0, m2_0). The InitSequence method initializes further instance variables based on the values passed as parameters to the method.
		3) Data generation is started by calling the __'DataGen'__ method of [[Extended_data]] python file. Parameters passed to the method are (sys_model, dataFolderName + dataFileName, T, T_test, randomInit = False). 
		5) __'DataLoader_GPU'__ method of [[Extended_data]] is called with parameters (dataFolderName + dataFileName) to load the data from the mentioned directory, which is then displayed finally at the end of the [[main_linear]] file.