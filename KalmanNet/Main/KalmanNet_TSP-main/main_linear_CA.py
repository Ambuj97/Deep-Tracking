# importing python libraries
import torch
from datetime import datetime
import time
# importing dependencies from other python files from the project
# Simulations folder
from Simulations.Linear_sysmdl import SystemModel
import Simulations.config as config
import Simulations.utils as utils
from Simulations.Linear_CA.parameters import F_gen,F_CV,H_identity,H_onlyPos,\
   Q_gen,Q_CV,R_3,R_2,R_onlyPos,\
   m,m_cv, R_onlyPosBB, R_7, Q_bb, F_genbb, m_bb

# Filters folder
from Filters.KalmanFilter_test import KFTest

# KNet folder
from KNet.KalmanNet_nn import KalmanNetNN

#Pipelines folder
from Pipelines.Pipeline_EKF import Pipeline_EKF as Pipeline

# Plot.py file
from Plot import Plot_KF as Plot
from preprocess_MOT import trainInput, trainOutput


################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

# results path
path_results = 'KNet'

print("Pipeline Start")
####################################
### Generative Parameters For CA ###
####################################
args = config.general_settings()
### Dataset parameters
args.N_E = 83
args.N_CV = 83
args.N_T = 83
offset = 0 ### Init condition of dataset
args.randomInit_train = True
args.randomInit_cv = True
args.randomInit_test = True

args.T = 599
args.T_test = 599
### training parameters
KnownRandInit_train = False # if true: use known random init for training, else: model is agnostic to random init
KnownRandInit_cv = False
KnownRandInit_test = False

args.use_cuda = True # use GPU or not
args.n_steps = 250
args.n_batch = 10
args.lr = 1e-4
args.wd = 1e-4

if args.use_cuda:
   if torch.cuda.is_available():
      device = torch.device('cuda')
      print("Using GPU")
      trainInput = trainInput.to('cuda:0')
      trainOutput = trainOutput.to('cuda:0')
   else:
      raise Exception("No GPU found, please set args.use_cuda = False")
else:
    device = torch.device('cpu')
    print("Using CPU")

if(args.randomInit_train or args.randomInit_cv or args.randomInit_test):
   std_gen = 1
else:
   std_gen = 0

if(KnownRandInit_train or KnownRandInit_cv or KnownRandInit_test):
   std_feed = 0
else:
   std_feed = 1


m1x_0 = torch.zeros(m_bb)
m1x_0_cv = torch.zeros(m_cv) # Initial State for CV
m2x_0 = std_feed * std_feed * torch.eye(m_bb) # Initial Covariance for feeding to filters and KNet
m2x_0_gen = std_gen * std_gen * torch.eye(m_bb) # Initial Covariance for generating dataset
m2x_0_cv = std_feed * std_feed * torch.eye(m_cv) # Initial Covariance for CV

#############################
###  Dataset Generation   ###
#############################
### PVA or P
Loss_On_AllState = False # if false: only calculate loss on position
Train_Loss_On_AllState = False # if false: only calculate training loss on position
CV_model = False # if true: use CV model, else: use CA model

DatafolderName = 'Simulations/Linear_CA/data/'
DatafileName = '7x7_T50.pt'


####################
### System Model ###
####################
sys_model_gen = SystemModel(F_genbb, Q_bb, H_identity, R_7, args.T, args.T_test)
sys_model_gen.InitSequence(m1x_0, m2x_0_gen)# x0 and P0

if CV_model:
   H_onlyPos = torch.tensor([[1, 0]]).float()
   sys_model = SystemModel(F_CV, Q_CV, H_identity, R_7, args.T, args.T_test)
   sys_model.InitSequence(m1x_0_cv, m2x_0_cv)# x0 and P0
else:
   sys_model = SystemModel(F_genbb, Q_bb, H_identity, R_7, args.T, args.T_test)
   sys_model.InitSequence(m1x_0, m2x_0)# x0 and P0

# print("Start Data Gen")
# utils.DataGen(args, sys_model_gen, DatafolderName+DatafileName)

# print("Load Original Data")
# [train_input, train_target, cv_input, cv_target, test_input, test_target, train_init, cv_init, test_init] = torch.load(DatafolderName+DatafileName, map_location=device)

# print("Data Shape")
# print("testset state x size:",test_target.size())
# print("testset observation y size:",test_input.size())
# print("trainset state x size:",train_target.size())
# print("trainset observation y size:",train_input.size())
# print("cvset state x size:",cv_target.size())
# print("cvset observation y size:",cv_input.size())

# print("Compute Loss on All States (if false, loss on position only):", Loss_On_AllState)

##############################
### Evaluate Kalman Filter ###
##############################
print("Evaluate Kalman Filter")
if args.randomInit_test and KnownRandInit_test:
   [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out] = KFTest(args, sys_model, trainInput, trainOutput, allStates=Loss_On_AllState, randomInit = True, test_init=test_init)
else: 
   [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out] = KFTest(args, sys_model, trainInput, trainOutput, allStates=Loss_On_AllState)

##########################
### Evaluate KalmanNet ###
##########################
# Build Neural Network
KNet_model = KalmanNetNN()
KNet_model.NNBuild(sys_model, args)
print("Number of trainable parameters for KNet pass 1:",sum(p.numel() for p in KNet_model.parameters() if p.requires_grad))
## Train Neural Network
KNet_Pipeline = Pipeline(strTime, "KNet", "KNet")
KNet_Pipeline.setssModel(sys_model)
KNet_Pipeline.setModel(KNet_model)
KNet_Pipeline.setTrainingParams(args)

trainingStartTime = time.time()
if (KnownRandInit_train):
   print("Train KNet with Known Random Initial State")
   print("Train Loss on All States (if false, loss on position only):", Train_Loss_On_AllState)
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(sys_model, trainInput, trainOutput, trainInput, trainOutput, path_results, MaskOnState=not Train_Loss_On_AllState, randomInit = True, cv_init=cv_init,train_init=train_init)
else:
   print("Train KNet with Unknown Initial State")
   print("Train Loss on All States (if false, loss on position only):", Train_Loss_On_AllState)
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KNet_Pipeline.NNTrain(sys_model, trainInput, trainOutput, trainInput, trainOutput, path_results, MaskOnState=not Train_Loss_On_AllState)

trainingEndTime = time.time()

trainingTime = trainingEndTime - trainingStartTime

print('Time taken for trainig: ', trainingTime)
if (KnownRandInit_test): 
   print("Test KNet with Known Random Initial State")
   ## Test Neural Network
   print("Compute Loss on All States (if false, loss on position only):", Loss_On_AllState)
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,KNet_out,RunTime] = KNet_Pipeline.NNTest(sys_model, trainInput, trainOutput, path_results,MaskOnState=not Loss_On_AllState,randomInit=True,test_init=test_init)
else: 
   print("Test KNet with Unknown Initial State")
   ## Test Neural Network
   print("Compute Loss on All States (if false, loss on position only):", Loss_On_AllState)
   [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,KNet_out,RunTime] = KNet_Pipeline.NNTest(sys_model, trainInput, trainOutput, path_results,MaskOnState=not Loss_On_AllState)

      
####################
### Plot results ###
####################
# PlotfolderName = "Figures/Linear_CA/"
# PlotfileName0 = "TrainPVA_position.png"
# PlotfileName1 = "TrainPVA_velocity.png"
# PlotfileName2 = "TrainPVA_acceleration.png"

# Plot = Plot(PlotfolderName, PlotfileName0)
# print("Plot")
# Plot.plotTraj_CA(test_target, KF_out, KNet_out, dim=0, file_name=PlotfolderName+PlotfileName0)#Position
# Plot.plotTraj_CA(test_target, KF_out, KNet_out, dim=1, file_name=PlotfolderName+PlotfileName1)#Velocity
# Plot.plotTraj_CA(test_target, KF_out, KNet_out, dim=2, file_name=PlotfolderName+PlotfileName2)#Acceleration