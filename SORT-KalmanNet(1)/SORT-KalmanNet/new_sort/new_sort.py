# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 17:48:56 2022

@author: ndasilv1
"""

from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
# from filterpy.kalman import KalmanFilter
import torch

from datetime import datetime

torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
# import torch.nn as nn
from Linear_sysmdl import SystemModel
# from Extended_data import DataGen,DataLoader,DataLoader_GPU, Decimate_and_perturbate_Data,Short_Traj_Split
#from Extended_data import N_E, N_CV, N_T, F, H, T, T_test, m1_0, m2_0, m, n
from Extended_data import N_T, F, H, T, T_test, m1_0, m2_0
from Pipeline_KF import Pipeline_KF
from KalmanNet_nn import KalmanNetNN

np.random.seed(0)


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(o) 

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


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """

  w = np.sqrt(x[2] * x[3])

  h = x[2] / w
  if(score==None):

    #test = np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    #print (test)
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))

  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  dataFileName = ['7x7_rq020_T100_mean_0_vdb_20_changed_initial_x_y.pt']
  modelFolder = 'SORT-KalmanNet/new_sort/KNet' + '/'
  today = datetime.today()
  now = datetime.now()
  strToday = today.strftime("%m.%d.%y")
  strNow = now.strftime("%H:%M:%S")
  strTime = strToday + "_" + strNow
  r2 = torch.tensor([0.1], dtype=torch.float32)
  vdB = -20 # ratio v=q2/r2
  v = 10**(vdB/10)
  q2 = torch.mul(v,r2)

#for index in range(0,len(r2)):

  print("1/r2 [dB]: ", 10 * torch.log10(1/r2[0]))
  print("1/q2 [dB]: ", 10 * torch.log10(1/q2[0]))

  # True model
  r = torch.sqrt(r2[0])
  q = torch.sqrt(q2[0])
  sys_model = SystemModel(F, q, H, r, 1, 1)
  sys_model.InitSequence(m1_0, m2_0)
  
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.KNet_Pipeline = Pipeline_KF(KalmanBoxTracker.strTime, "SORT-KalmanNet/new_sort/KNet", "KNet_"+ KalmanBoxTracker.dataFileName[0])
    self.KNet_Pipeline.setssModel(KalmanBoxTracker.sys_model)
    self.KNet_model = KalmanNetNN()
    self.KNet_model.Build(KalmanBoxTracker.sys_model)
    self.KNet_Pipeline.setModel(self.KNet_model)
    
    self.KNet_Pipeline.model = torch.load(KalmanBoxTracker.modelFolder+"model_KNet_7x7_rq020_T100_mean_0_vdb_20_changed_initial_x_y.pt", map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    self.extra_dim = np.zeros((3,1))
    #self.x = convert_bbox_to_z(bbox) 
    self.x = np.concatenate((convert_bbox_to_z(bbox), self.extra_dim), axis=0)

    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
  def update(self,bbox):
      """
      Updates the state vector with observed bbox.
      """
      self.time_since_update = 0
      self.history = []
      self.hits += 1
      print ('hits ', self.hits)
      self.hit_streak += 1
      
      #self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
      """
      Advances the state vector and returns the predicted bounding box estimate.
      """
      if((self.x[6]+self.x[2])<=0):
        self.x[6] *= 0.0
     # self.kf.predict()
      self.x = self.KNet_Pipeline.NNTest(1, torch.from_numpy(np.reshape(self.x, (1,) + self.x.shape)).float()).cpu().detach().numpy()
      print ('x ', self.x)
      self.age += 1
      if(self.time_since_update>0):
        self.hit_streak = 0
      self.time_since_update += 1
      self.history.append(convert_x_to_bbox(self.x))
      return self.history[-1]
    
  def get_state(self):
     """
     Returns the current bounding box estimate.
     """
     return convert_x_to_bbox(self.x)
 
def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  iou_matrix = iou_batch(detections, trackers)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  def update(self, dets=np.empty((0, 5))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      print ('pos ', pos)
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    print ('dets ', dets)
    print ('trks ', trks)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)
    print ('matched ', matched)
    #print ('unmatched dets ', unmatched_dets)
    #print ('unmatched trks', unmatched_trks)
    # update matched trackers with assigned detections
    for m in matched:
      print (m)
      self.trackers[m[1]].update(dets[m[0], :])
      print ('trackers ',  self.trackers[m[1]])

    # create and initialise new trackers for unmatched detections
    # print ('unmatch ', unmatched_dets)
    for i in unmatched_dets:
       # print('i ', i)
       # print('dets ', dets[i,:])
        trk = KalmanBoxTracker(dets[i,:])
       # print ('after kalman')
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        #print ('get state ', d)
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    # print('return: ', ret, 'len ', len(ret))
    if(len(ret)>0):
        
      return np.concatenate(ret)
    return np.empty((0,5))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
  # all train
  args = parse_args()
  display = args.display
  phase = args.phase
  total_time = 0.0
  total_frames = 0
  colours = np.random.rand(32, 3) #used only for display
  if(display):
    if not os.path.exists('MOT15'):
      print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
      exit()
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')

  if not os.path.exists('output'):
    os.makedirs('output')
  pattern = os.path.join('SORT-KalmanNet/new_sort', args.seq_path, phase, 'KITTI-13', 'det', 'det.txt')
  # print(pattern)
  for seq_dets_fn in glob.glob(pattern):
    print("KITTI-13")
    mot_tracker = Sort(max_age=args.max_age, 
                       min_hits=args.min_hits,
                       iou_threshold=args.iou_threshold) #create instance of the SORT tracker
    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
    seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]
    # print ('seq ', seq)
    # print ('seq_dets ', seq_dets)
    
    with open(os.path.join('output', '%s.txt'%(seq)),'w') as out_file:
      print("Processing %s."%(seq))
      for frame in range(int(seq_dets[:,0].max())):
        #print ('frame ', frame)  
        frame += 1 #detection and frame numbers begin at 1
        dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
        dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        total_frames += 1

        if(display):
          fn = os.path.join('MOT15', phase, seq, 'img1', '%06d.jpg'%(frame))
          im =io.imread(fn)
          ax1.imshow(im)
          plt.title(seq + ' Tracked Targets')

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        for d in trackers:
          print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
          if(display):
            d = d.astype(np.int32)
            # print (d)
            ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))

        if(display):
          fig.canvas.flush_events()
          plt.draw()
          ax1.cla()

  # print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

  if(display):
    print("Note: to get real runtime results run without the option: --display")
