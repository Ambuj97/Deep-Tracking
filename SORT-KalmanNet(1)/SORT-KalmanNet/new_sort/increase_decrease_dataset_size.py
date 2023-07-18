# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 19:01:34 2022

@author: ndasilv1
"""
import torch

####################
### decrease dim ###
####################

# data = torch.load('Simulations/Linear_canonical/H=I/10x10_rq020_T100.pt', map_location=torch.device('cpu'))
# count = 0
# print (data[0])
# for i in range(len(data)):
#     data[i] = data[i][:,0:7,:]


# torch.save(data,'Simulations/Linear_canonical/H=I/7x7_rq020_T100.pt')


####################
### increase dim ###
####################

data = torch.load('Simulations/Linear_canonical/H=I/5x5_rq020_T100.pt', map_location=torch.device('cpu'))

eye7 = torch.eye(7)
expand_matriz = torch.zeros(7,100)
expand_matriz[0:7,0:7] += eye7


data7 = torch.cat([data[0][1], expand_matriz[5:7,:]], axis=0)

final_data = []
for i in range(len(data)):
    new_data = torch.empty(size=(data[i].shape[0],7,data[i].shape[2]))
    for j in range(data[i].shape[0]):
        final = torch.cat((data[i][j], expand_matriz[5:7,:]), axis=0)
        new_data[j]=final
    final_data.append(new_data)
    
torch.save(final_data,'Simulations/Linear_canonical/H=I/7x7_rq020_T100.pt')