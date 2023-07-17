# importing necessary libraries
import torch
import torch.nn as nn
from datetime import datetime


torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

# importing SystemModel class from Linear_sysmdl.py
from Linear_sysmdl import SystemModel

# importing dataset variables and functions from Extended_data.py
from Extended_data import DataGen,DataLoader,DataLoader_GPU, Decimate_and_perturbate_Data,Short_Traj_Split
from Extended_data import F, H, T, T_test, m1_0, m2_0, r2, q2, r, q
# from Extended_data import , F_rotated, H_rotated

# checking if gpu is available or not
if torch.cuda.is_available():
   dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
   print("Running on the GPU")
else:
   dev = torch.device("cpu")
   print("Running on the CPU")


print("Pipeline Start")

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)
path_results = 'RTSNet/'

####################
### System Model ###
####################

print("1/r2 [dB]: ", 10 * torch.log10(1/r2[0]))
print("1/q2 [dB]: ", 10 * torch.log10(1/q2[0]))


sys_model = SystemModel(F, q, H, r, T, T_test)
sys_model.InitSequence(m1_0, m2_0)

##############################
### Generate and load data ###
##############################
dataFolderName = 'KalmanNet-Dataset-main/Simulations/Linear_canonical/Experiments/'
dataFileName = '7x7_rq020_T100_1.pt' # rq are named in dB
print("Start Data Gen")
DataGen(sys_model, dataFolderName + dataFileName, T, T_test,randomInit=False)
print("Data Load")
[train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader_GPU(dataFolderName + dataFileName)
print("trainset size:",train_target.size())
print("cvset size:",cv_target.size())
print("testset size:",test_target.size())



