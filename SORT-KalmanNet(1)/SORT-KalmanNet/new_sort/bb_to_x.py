# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 00:27:52 2022

@author: ndasilv1
"""
import os
import glob
import numpy as np
import torch


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))

pattern = os.path.join('data', 'train', '*', 'det', 'det.txt')
for seq_dets_fn in glob.glob(pattern):
    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
    seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0] #get file name

    trajectory = np.empty((4,0)) # initiate trajectories
    final_data = []
    line_before = 1.0 #bb id

    for line in seq_dets:
        
        #create trajectories for bb with same id
        if line_before == line[0]:
            
            line = convert_bbox_to_z(line)
         
            trajectory = np.concatenate((trajectory, line), axis=1)
            final_x_tensor = torch.from_numpy(trajectory)
            
         
        else:
  
            final_data.append(final_x_tensor)

            line_before += 1

            trajectory = np.empty((4,0))
            
            # Save first trajectory of next bb
            line = convert_bbox_to_z(line)
            trajectory = np.concatenate((trajectory, line), axis=1)
            final_x_tensor = torch.from_numpy(trajectory)
            
            
    final_data.append(final_x_tensor) #save last bb trajectories
    torch.save(final_data,'tensor_dataset_'+seq+'.pt')
