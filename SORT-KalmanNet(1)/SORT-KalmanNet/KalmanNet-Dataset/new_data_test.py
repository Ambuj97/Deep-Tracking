# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 15:23:34 2022

@author: ndasilv1
"""
from Extended_data import DataGen,DataLoader,DataLoader_GPU
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
##############################
### Generate and load data ###
##############################
# dataFolderName = 'Simulations/Linear_canonical' + '/'
# dataFileName = 'data_as_input_add_noise_for_target.pt' # rq are named in dB
# dataFileName = '7x7_rq020_T100_mean_0_vdb_20_changed_initial_x_y.pt' # rq are named in dB
## dataFolderName = 'Simulations/Lorenz_Atractor/data/T100' + '/'
## dataFileName = 'data_lor_v20_rq020_T100.pt' # rq are named in dB
dataFolderName = '../'
dataFileName = 'synthetic_bb.pt'

print("Data Load")
[train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader_GPU(dataFolderName + dataFileName)
print("trainset:",train_target.shape)
print("cvset:",cv_target.shape)
print("testset:",test_target.shape)

index = list(range(0,100))
  
# plot lines

trajectory = 7
w = np.sqrt(train_input[trajectory][3]*train_input[trajectory][2])
h = train_input[trajectory][2]/w
x_bottom_left = train_input[trajectory][0] - w/2
y_bottom_left = train_input[trajectory][1] - h/2


w_target = np.sqrt(train_target[trajectory][3]*train_target[trajectory][2])
h_target = train_target[trajectory][2]/w
x_bottom_left_target = train_target[trajectory][0] - w/2
y_bottom_left_target = train_target[trajectory][1] - h/2

fig, ax = plt.subplots()
ax.plot(train_input[trajectory][0], train_input[trajectory][1], label = "trajectory input", linestyle="-")
ax.plot(train_target[trajectory][0], train_target[trajectory][1], label = "trajectory input", linestyle="--", color='red')
for i in range(0, train_input.shape[2]):
    step = i
    ax.add_patch(Rectangle((x_bottom_left[step], y_bottom_left[step]), w[step], h[step],color="orange", fill=False))
    #ax.add_patch(Rectangle((x_bottom_left_target[step], y_bottom_left_target[step]), w_target[step], h_target[step],color="green", fill=False))
plt.show()

# fig, ax = plt.subplots()
# ax.plot(train_input[trajectory][0], train_input[trajectory][1], label = "input", linestyle="-")
# ax.add_patch(Rectangle((x_bottom_left[1], y_bottom_left[1]), w[1], h[1],color="yellow"))
# plt.legend()
# plt.xlabel("X-AXIS")
# plt.ylabel("Y-AXIS")
# plt.title("PLOT-1")
# plt.show()
# plt.plot(train_input[1][0], train_input[1][1], label = "input", linestyle="-")
# plt.plot(train_target[1][0], train_target[1][1], label = "target", linestyle="--")
# plt.plot(index, train_input[1][3], label = "input", linestyle="-")
# plt.plot(index, train_target[1][3], label = "target", linestyle="--")
# plt.plot(x, np.sin(x), label = "curve 1", linestyle="-.")
# plt.plot(x, np.cos(x), label = "curve 2", linestyle=":")
#plt.legend()
# plt.show()

