import torch
import torch.nn as nn
import random
import time
from Plot import Plot


class Pipeline_KF:

    def __init__(self, Time, folderName, modelName):
        super().__init__()
        self.Time = Time
        self.folderName = folderName + '/'
        self.modelName = modelName
        self.modelFileName = self.folderName + "model_" + self.modelName 
        self.PipelineName = self.folderName + "pipeline_" + self.modelName 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def save(self):
        torch.save(self, self.PipelineName)
        torch.save(self, self.modelFileName)

    def setssModel(self, ssModel):
        self.ssModel = ssModel

    def setModel(self, model):
        self.model = model

    def setTrainingParams(self, n_Epochs, n_Batch, learningRate, weightDecay):
        self.N_Epochs = n_Epochs  # Number of Training Epochs
        self.N_B = n_Batch # Number of Samples in Batch
        self.learningRate = learningRate # Learning Rate
        self.weightDecay = weightDecay # L2 Weight Regularization - Weight Decay

        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)


    def NNTrain(self, n_Examples, train_input, train_target, n_CV, cv_input, cv_target):

        self.N_E = n_Examples
        self.N_CV = n_CV

        MSE_cv_linear_batch = torch.empty([self.N_CV])
        self.MSE_cv_linear_epoch = torch.empty([self.N_Epochs])
        self.MSE_cv_dB_epoch = torch.empty([self.N_Epochs])

        MSE_train_linear_batch = torch.empty([self.N_B])
        self.MSE_train_linear_epoch = torch.empty([self.N_Epochs])
        self.MSE_train_dB_epoch = torch.empty([self.N_Epochs])

        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_dB_opt = 10
        self.MSE_cv_idx_opt = 0

        for ti in range(0, self.N_Epochs):

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()

            for j in range(0, self.N_CV):
                y_cv = cv_input[j, :, :]
                self.model.InitSequence(self.ssModel.m1x_0)

                x_out_cv = torch.empty(self.ssModel.m, self.ssModel.T)
                for t in range(0, self.ssModel.T):
                    x_out_cv[:, t] = self.model(y_cv[:, t])

                # Compute Training Loss
                MSE_cv_linear_batch[j] = self.loss_fn(x_out_cv, cv_target[j, :, :]).item()

            # Average
            self.MSE_cv_linear_epoch[ti] = torch.mean(MSE_cv_linear_batch)
            self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

            if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                self.MSE_cv_idx_opt = ti
                torch.save(self.model, self.modelFileName)

            ###############################
            ### Training Sequence Batch ###
            ###############################

            # Training Mode
            self.model.train()

            # Init Hidden State
            self.model.init_hidden()

            Batch_Optimizing_LOSS_sum = 0

            for j in range(0, self.N_B):
                n_e = random.randint(0, self.N_E - 1)

                y_training = train_input[n_e, :, :]
                self.model.InitSequence(self.ssModel.m1x_0)

                x_out_training = torch.empty(self.ssModel.m, self.ssModel.T)
                for t in range(0, self.ssModel.T):
                    x_out_training[:, t] = self.model(y_training[:, t])

                # Compute Training Loss
                LOSS = self.loss_fn(x_out_training, train_target[n_e, :, :])
                MSE_train_linear_batch[j] = LOSS.item()

                Batch_Optimizing_LOSS_sum = Batch_Optimizing_LOSS_sum + LOSS

            # Average
            self.MSE_train_linear_epoch[ti] = torch.mean(MSE_train_linear_batch)
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])

            ##################
            ### Optimizing ###
            ##################

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            self.optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / self.N_B
            Batch_Optimizing_LOSS_mean.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()

            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :", self.MSE_cv_dB_epoch[ti],
                  "[dB]")

            if (ti > 1):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")

    # def NNTest(self, n_Test, test_input):

    #     self.N_T = n_Test

    #     self.MSE_test_linear_arr = torch.empty([self.N_T])

    #     # MSE LOSS Function
    #     # loss_fn = nn.MSELoss(reduction='mean')
    #     # print(self.modelFileName)
    #     # self.modelFileName = 'SORT-KalmanNet/new_sort/KNet/model_KNet_7x7_rq020_T100_mean_0_vdb_20_changed_initial_x_y.pt'
    #     self.model = torch.load(self.modelFileName, map_location=self.device)
    #     # print(self.model)
    #     self.model.eval()

    #     torch.no_grad()
        
    #     # start = time.time()

    #     for j in range(0, self.N_T):
            
    #         y_mdl_tst = test_input[j, :, :]

    #         self.model.InitSequence(self.ssModel.m1x_0)

    #         x_out_test = torch.empty(self.ssModel.m, self.ssModel.T)

    #         for t in range(0, self.ssModel.T):
    #             x_out_test[:, t] = self.model(y_mdl_tst[:, t])

    #     #     self.MSE_test_linear_arr[j] = loss_fn(x_out_test, test_target[j, :, :]).item()

    #     # end = time.time()
    #     # t = end - start

    #     # # Average
    #     # self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
    #     # self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

    #     # # Standard deviation
    #     # self.MSE_test_dB_std = torch.std(self.MSE_test_linear_arr, unbiased=True)
    #     # self.MSE_test_dB_std = 10 * torch.log10(self.MSE_test_dB_std)

    #     # # Print MSE Cross Validation
    #     # str = self.modelName + "-" + "MSE Test:"
    #     # print(str, self.MSE_test_dB_avg, "[dB]")
    #     # str = self.modelName + "-" + "STD Test:"
    #     # print(str, self.MSE_test_dB_std, "[dB]")
    #     # # Print Run Time
    #     # print("Inference Time:", t)

    #     return x_out_test

    def NNTest(self, test_input, MaskOnState=False,\
     randomInit=False,test_init=None,load_model=False,load_model_path=None,\
        test_lengthMask=None):
        # print('Test input shape', test_input.shape)
        # Load model
        if load_model:
            self.model = torch.load(load_model_path, map_location=self.device) 
        else:
            self.model = torch.load(self.modelFileName, map_location=self.device)
        
        # print('\n\n')
        # print('NNTEST input\n')
        # print(test_input)
        self.N_T = test_input.shape[0]
        self.ssModel.T_test = test_input.size()[-1]
        self.MSE_test_linear_arr = torch.zeros([self.N_T])
        x_out_test = torch.zeros([self.N_T, self.ssModel.m, self.ssModel.T_test]).to(self.device)

        if MaskOnState:
            #3x3 model
            mask = torch.tensor([True,False,False])
            
            if self.ssModel.m == 2: 
                mask = torch.tensor([True,False])
            elif self.ssModel.m == 7: 
                mask = torch.tensor([True, True, True, True, False, False, False])

        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')

        # Test mode
        self.model.eval()
        self.model.batch_size = self.N_T
        # Init Hidden State
        self.model.init_hidden_KNet()
        torch.no_grad()

        start = time.time()

        if (randomInit):
            self.model.InitSequence(test_init, self.ssModel.T_test)               
        else:
            self.model.InitSequence(self.ssModel.m1x_0.reshape(1,self.ssModel.m,1).repeat(self.N_T,1,1), self.ssModel.T_test)         
        
        for t in range(0, self.ssModel.T_test):
            x_out_test[:,:, t] = torch.squeeze(self.model(torch.unsqueeze(test_input[:,:, t],2)))
        
        end = time.time()
        t = end - start

        # MSE loss
        # for j in range(self.N_T):# cannot use batch due to different length and std computation  
        #     if(MaskOnState):
        #         if self.args.randomLength:
        #             self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j,mask,test_lengthMask[j]], test_target[j,mask,test_lengthMask[j]]).item()
        #         else:
        #             self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j,mask,:], test_target[j,mask,:]).item()
        #     else:
        #         if self.args.randomLength:
        #             self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j,:,test_lengthMask[j]], test_target[j,:,test_lengthMask[j]]).item()
        #         else:
        #             self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j,:,:], test_target[j,:,:]).item()
        
        # # Average
        # self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        # self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

        # # Standard deviation
        # self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)

        # # Confidence interval
        # self.test_std_dB = 10 * torch.log10(self.MSE_test_linear_std + self.MSE_test_linear_avg) - self.MSE_test_dB_avg

        # # Print MSE and std
        # str = self.modelName + "-" + "MSE Test:"
        # print(str, self.MSE_test_dB_avg, "[dB]")
        # str = self.modelName + "-" + "STD Test:"
        # print(str, self.test_std_dB, "[dB]")
        # Print Run Time
        print("Inference Time:", t)

        return x_out_test
    
    def PlotTrain_KF(self, MSE_KF_linear_arr, MSE_KF_dB_avg):

        self.Plot = Plot(self.folderName, self.modelName)

        self.Plot.NNPlot_epochs(self.N_Epochs, MSE_KF_dB_avg,
                                self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

       # self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)