# Temporal-Uncertainty-Estimates
Instance segmentation with neural networks is an essential task in environment perception. However, the networks can predict false positive instances with high confidence values and true positives with low ones. Hence, it is important to accurately model the uncertainties of neural networks to prevent safety issues and foster interpretability. In applications such as automated driving the detection of road users like vehicles and pedestrians is of highest interest. We present a temporal approach to detect false positives and investigate uncertainties of instance segmentation networks. Since image sequences are available for online applications, we track instances over multiple frames and create temporal instance-wise aggregated metrics of uncertainty. The prediction quality is estimated by predicting the intersection over union as performance measure. Furthermore, we show how to use uncertainty information to replace the traditional score value from object detection and improve the overall performance of instance segmentation networks.

For further reading, please refer to https://arxiv.org/abs/2012.07504.

# Preparation:
We assume that the user is already using a neural network for instance segmentation and a corresponding dataset. For each image from the instance segmentation dataset, Temporal-Uncertainty-Estimates requires the following data where each video is in its own folder:

- the input image (height, width) as png
- the ground truth (height, width) as png
- a three-dimensional numpy array (num instances, height, width) that contains the predicted instance mask for the current image
- a four-dimensional numpy array (num instances, height, width, classes) that contains the softmax probabilities per pixel or a one-dimensional numpy array (num instances) that contains the softmax probabilities per instance computed for the current image
- a one-dimensional numpy array (num instances) that contains the score value computed for the current image

Before running Temporal-Uncertainty-Estimates, please edit all necessary paths stored in "global_defs.py". The code is CPU based and parts of of the code trivially parallize over the number of input images, adjust "NUM_CORES" in "global_defs.py" to make use of this. Also, in the same file, select the tasks to be executed by setting the corresponding boolean variable (True/False).

# Run Code:
```sh
./run.sh
```

# Networks and Datasets:
The results in https://arxiv.org/abs/2012.07504 have been obtained from two instance segmentation networks, the Mask R-CNN (https://github.com/matterport/Mask_RCNN) and the YOLACT network (https://github.com/dbolya/yolact), together with the KITTI and the MOT dataset (https://www.vision.rwth-aachen.de/page/mots).

# Author:
Kira Maag (University of Wuppertal)
