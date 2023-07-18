# SORT-KalmanNet
Implementation of KalmanNet in SORT.

# Envs
* ```conda env create -f environment.yml``` for linux with gpu (cuda 11.0) and ``` conda env create -f environment_windows_cpu.yml ``` as the name suggests.
* For other versions of cuda [get the right version of pytorch](https://pytorch.org/get-started/locally/).

# new_sort
* data dir: txt used for tracking from MOT15.
* Model trained from kalmanNet in KNet folder.
* Extended_data, KalmanNet_nn, Linear_sysmdl, Pipeline_KF and Plot files are used to load kalmanNet model and predict the state.
* sort.py is the original SORT code.
* new_sort.py is the modified version of SORT with KalmanNet implemented.

Download [MOT15 dataset](https://motchallenge.net/data/MOT15/), save inside new_sort dir and run ```python new_sort.py --display``` for visualise the bounding boxes.

# KalmanNet-Dataset

Where the dataset 7x7 with only positive values is being generated.

# KalmanNet_TSP

KalmanNet trained using the main_linear.py for now. The file extended_data was modified to run the 7x7 model with F and H evolution matrices from SORT.

