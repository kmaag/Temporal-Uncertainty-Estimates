#!/usr/bin/env python3
"""
script including
class object with global settings
"""

class CONFIG:
  
  #------------------#
  # select or define #
  #------------------#
  
  img_types             = ["kitti", "mot"]
  model_names           = ["mask_rcnn", "yolact"] 
  classification_models = ["LR_L1", "GB", "NN_L2"]
  regression_models     = ["LR", "LR_L1", "LR_L2", "GB", "NN_L1", "NN_L2"]
  
  IMG_TYPE             = img_types[0]
  MODEL_NAME           = model_names[0] 
  CLASSIFICATION_MODEL = classification_models[0]    
  REGRESSION_MODEL     = regression_models[0]   
  
  #---------------------#
  # set necessary path  #
  #---------------------#
  
  my_io_path  = "/home/user/object_tracking_io/" + IMG_TYPE + "/"
  
  #--------------------------------------------------------------------#
  # select tasks to be executed by setting boolean variable True/False #
  #--------------------------------------------------------------------#
  
  COMPUTE_TIME_SERIES_INSTANCES = False
  PLOT_TIME_SERIES_INSTANCES    = False 
  ANALYZE_TRACKING              = False
  COMPUTE_TIME_SERIES_METRICS   = False
  VISUALIZE_METRICS             = False
  COMPUTE_MEAN_AP               = False
  COMPUTE_MEAN_AP_METRICS       = False
  PLOT_MEAN_AP                  = False
  VISUALIZE_REGRESSION          = False
  VISUALIZE_CLASSIFICATION      = False
  ANALYZE_METRICS               = False
  PLOT_ANALYZE_METRICS          = False
  
  #-----------#
  # optionals #
  #-----------#
  
  SCORE_THRESHOLD   = '00'
  MAP_THRESHOLD     = '00'
  NUM_CORES         = 10
  EPS_MATCHING      = 100
  NUM_REG_MATCHING  = 5
  CLASS_COMPONENT   = "car" # car, person
  METRICS_COMPONENT = ("E", "S", "iou") 
  NUM_PREV_FRAMES   = 5
  NUM_RESAMPLING    = 10
  FLAG_CLASSIF      = 1
  FLAG_OBJ_SEG      = 1 # 0: object detection, 1: segmentation
  FLAG_NEW_METRICS  = 2 # 0: U^i, 1: U^is plus score and ratio, 2: V^i
  
  if IMG_TYPE == "kitti":
    NUM_IMAGES = 2981
    CLASSES = [1,2]
  elif IMG_TYPE == "mot":
    if MODEL_NAME == "mask_rcnn":
      NUM_IMAGES = 2862
    elif MODEL_NAME == "yolact":
      NUM_IMAGES = 2562
    NUM_RESAMPLING = min(NUM_RESAMPLING, 4)
    CLASSES = [2]
  
  IMG_DIR              = my_io_path + "inputimages/val/"
  GT_DIR               = my_io_path + "groundtruth/val/"
  HELPER_DIR           = my_io_path + "helpers/"                   + MODEL_NAME + str(SCORE_THRESHOLD) + "/"
  PRED_DIR             = my_io_path + "pred_instance/"             + MODEL_NAME + str(SCORE_THRESHOLD) + "/"
  SOFTMAX_DIR          = my_io_path + "pred_softmax/"              + MODEL_NAME + str(SCORE_THRESHOLD) + "/"
  INSTANCES_SMALL_DIR  = my_io_path + "instances_small/"           + MODEL_NAME + str(SCORE_THRESHOLD) + "/"
  SOFTMAX_SMALL_DIR    = my_io_path + "softmax_small/"             + MODEL_NAME + str(SCORE_THRESHOLD) + "/"
  TIME_SERIES_INST_DIR = my_io_path + "time_series_instances/"     + MODEL_NAME + str(SCORE_THRESHOLD) + "/"
  IMG_TIME_SERIES_DIR  = my_io_path + "img_time_series_instances/" + MODEL_NAME + str(SCORE_THRESHOLD) + "/"
  ANALYZE_TRACKING_DIR = my_io_path + "results_tracking/"          + MODEL_NAME + str(SCORE_THRESHOLD) + "/"
  METRICS_DIR          = my_io_path + "metrics/"                   + MODEL_NAME + str(SCORE_THRESHOLD) + "_os" + str(FLAG_OBJ_SEG) + "/"
  IMG_METRICS_DIR      = my_io_path + "img_metrics/"               + MODEL_NAME + str(SCORE_THRESHOLD) + "_os" + str(FLAG_OBJ_SEG) + "/"
  MEAN_AP_DIR          = my_io_path + "mean_ap/"                   + MODEL_NAME + str(MAP_THRESHOLD) + "/"
  MEAN_AP_METRICS_DIR  = my_io_path + "mean_ap_metrics/"           + MODEL_NAME + str(MAP_THRESHOLD) + "_os" + str(FLAG_OBJ_SEG) + "/"
  IMG_IOU_INST_DIR     = my_io_path + "img_iou_instances/"         + MODEL_NAME + str(SCORE_THRESHOLD) + "_os" + str(FLAG_OBJ_SEG) + "/npf" + str(NUM_PREV_FRAMES) + "_" + REGRESSION_MODEL + "_nm" + str(FLAG_NEW_METRICS) + "/"
  IMG_IOU0_INST_DIR    = my_io_path + "img_iou0_instances/"        + MODEL_NAME + str(SCORE_THRESHOLD) + "_os" + str(FLAG_OBJ_SEG) + "/npf" + str(NUM_PREV_FRAMES) + "_" + CLASSIFICATION_MODEL + "_nm" + str(FLAG_NEW_METRICS) + "/"
  ANALYZE_DIR          = my_io_path + "results_analyze/"           + MODEL_NAME + str(SCORE_THRESHOLD) + "_os" + str(FLAG_OBJ_SEG) + "/npf" + str(NUM_PREV_FRAMES) + "_runs" + str(NUM_RESAMPLING) + "_nm" + str(FLAG_NEW_METRICS) + "/"
  IMG_ANALYZE_DIR      = my_io_path + "img_results_analyze/"       + MODEL_NAME + str(SCORE_THRESHOLD) + "_os" + str(FLAG_OBJ_SEG) + "/npf" + str(NUM_PREV_FRAMES) + "_runs" + str(NUM_RESAMPLING) + "_nm" + str(FLAG_NEW_METRICS) + "/"
  
  
"""
In case of problems, feel free to contact: Kira Maag, kmaag@uni-wuppertal.de
"""
