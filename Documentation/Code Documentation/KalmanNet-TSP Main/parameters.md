Documentation of python file - [[parameters]].py


#### Libraries:
1) torch


#### Variable Declarations:
1) m - number of dimensions in state vector for constant acceleration model
2) m_cv - number of dimensions in state vector for constant velocity model
3) m_bb - number of dimensions in state vector for bounding box model
4) delta_t_gen - time difference
5) F_gen - state evolution matrix for constant acceleration model
6) F_CV - state evolution matrix for constant velocity model
7) F_genbb - state evolution matrix for bounding box model
8) H_identity - full observation matrix
9) H_onlyPos - observation matrix for only position
10) r2 - identity tensor of order 1
11) q2 - identity tensor of order 1
12) Q_gen - process noise matrix for constant acceleration model
13) Q_CV - process noise matrix for constant velocity model
14) Q_bb - process noise matrix for bounding box model
15) R_3 - observation noise for constant acceleration model
16) R_2 - observation noise for constant velocity model
17) R_7 - observation noise for bounding box model
18) R_onlyPos - observation noise for only position
19) R_onlyPosBB - observation noise for only position in the bounding box model


#### Purpose:
The purpose of this python file is to set the parameters for KalmanNet and Kalman Filters.