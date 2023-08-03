Baseline model:
- Generated KalmanNet synthetic dataset based on equations of motion for a 7x7 model (x, y, s, r, x', y', s') to incorporate the bounding box notion with Kalman Net.
- Trained KalmanNet model on the new dataset to estimate position of the bounding box.
- Integrated with SORT to test the trained model on KITTI-13 dataset.

Innovation:
- The main aspect of the thesis is to try to create a KalmanNet based tracking within SORT instead of Kalman Filter based tracking and propose a data-based KG computation and remove  complex matrix inversions from the KG computation.

Results:
- The KalmanNet training results are in the experimentation section.
- The SORT tracking and association on KITTI-13 dataset with Kalman Net model trained is tracking the bounding boxes but the results are not satisfactory.

Todo coming days:
- Keep working on improving tracking.