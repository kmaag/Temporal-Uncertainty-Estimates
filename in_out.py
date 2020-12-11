#!/usr/bin/env python3
"""
script including
functions for handling input/output like loading/saving
"""

import os
import pickle
import numpy as np
from PIL import Image

from global_defs import CONFIG


def get_save_path_image_i( vid, i ):
  if CONFIG.IMG_TYPE == CONFIG.img_types[0]:
    return CONFIG.IMG_DIR + vid + "/" + str(i).zfill(6) +".png"
  elif CONFIG.IMG_TYPE == CONFIG.img_types[1]:
    return CONFIG.IMG_DIR + vid + "/" + str(i+1).zfill(6) +".jpg"


def get_save_path_gt_i( vid, i ):
  if CONFIG.IMG_TYPE == CONFIG.img_types[0]:
    return CONFIG.GT_DIR + vid + "/" + str(i).zfill(6) +".png"
  elif CONFIG.IMG_TYPE == CONFIG.img_types[1]:
    return CONFIG.GT_DIR + vid + "/" + str(i+1).zfill(6) +".png"


def get_save_path_instances_i( vid, i ):
  if CONFIG.IMG_TYPE == CONFIG.img_types[0]:
    return CONFIG.PRED_DIR + vid + "/" + str(i).zfill(6) + ".p"
  elif CONFIG.IMG_TYPE == CONFIG.img_types[1]:
    return CONFIG.PRED_DIR + vid + "/" + str(i+1).zfill(6) + ".p"


def get_save_path_softmax_i( vid, i ):
  if CONFIG.IMG_TYPE == CONFIG.img_types[0]:
    return CONFIG.SOFTMAX_DIR + vid + "/" + str(i).zfill(6) +".p"
  elif CONFIG.IMG_TYPE == CONFIG.img_types[1]:
    return CONFIG.SOFTMAX_DIR + vid + "/" + str(i+1).zfill(6) +".p"


def get_save_path_score_i( vid, i ):
  if CONFIG.IMG_TYPE == CONFIG.img_types[0]:
    return CONFIG.SOFTMAX_DIR + vid + "/" + str(i).zfill(6) +"score.p"
  elif CONFIG.IMG_TYPE == CONFIG.img_types[1]:
    return CONFIG.SOFTMAX_DIR + vid + "/" + str(i+1).zfill(6) +"score.p"


def get_save_path_instances_small_i( vid, i ):
  return CONFIG.INSTANCES_SMALL_DIR + vid + "/instances_small" + str(i).zfill(6) + ".p"


def get_save_path_softmax_small_i( vid, i ):
  return CONFIG.SOFTMAX_SMALL_DIR + vid + "/softmax_small" + str(i).zfill(6) + ".p"


def get_save_path_score_small_i( vid, i ):
  return CONFIG.SOFTMAX_SMALL_DIR + vid + "/score_small" + str(i).zfill(6) + ".p"


def get_save_path_time_series_instances_i( vid, i, eps, num_reg ):
  return CONFIG.TIME_SERIES_INST_DIR + vid + "/time_series_instances" + str(i).zfill(6) + "_eps" + str(eps) + "_num_reg" + str(num_reg) + ".p"


def get_save_path_time_series_metrics_i( vid, i, eps, num_reg, flag_3d=0 ):
  if flag_3d == 0:
    if vid == 'all':
      return CONFIG.METRICS_DIR + "time_series_metrics" + str(i).zfill(6) + "_eps" + str(eps) + "_num_reg" + str(num_reg) + ".p"
    else:
      return CONFIG.METRICS_DIR + vid + "/time_series_metrics" + str(i).zfill(6) + "_eps" + str(eps) + "_num_reg" + str(num_reg) + ".p"
  elif flag_3d == 1:
    return CONFIG.METRICS_DIR + vid + "_time_series_metrics_eps" + str(eps) + "_num_reg" + str(num_reg) + ".p"
    

def ground_truth_load( vid, i ):
  read_path = get_save_path_gt_i( vid, i )
  gt = np.asarray( Image.open(read_path) )
  return gt


def instances_load( vid, i ):
  read_path = get_save_path_instances_i( vid, i )
  instances = pickle.load( open( read_path, "rb" ) )
  return instances


def softmax_load( vid, i, ):
  read_path = get_save_path_softmax_i( vid, i, )
  softmax = pickle.load( open( read_path, "rb" ) )
  return softmax


def score_load( vid, i, ):
  read_path = get_save_path_score_i( vid, i, )
  score = pickle.load( open( read_path, "rb" ) )
  return score


def instances_small_dump( instances, vid, i ):
  dump_path = get_save_path_instances_small_i( vid, i )
  dump_dir  = os.path.dirname( dump_path )
  if not os.path.exists( dump_dir ):
    os.makedirs( dump_dir )
  pickle.dump( instances, open( dump_path, "wb" ) )
  

def instances_small_load( vid, i ):
  read_path = get_save_path_instances_small_i( vid, i )
  instances = pickle.load( open( read_path, "rb" ) )
  return instances


def softmax_small_dump( softmax, vid, i ):
  dump_path = get_save_path_softmax_small_i( vid, i )
  dump_dir  = os.path.dirname( dump_path )
  if not os.path.exists( dump_dir ):
    os.makedirs( dump_dir )
  pickle.dump( softmax, open( dump_path, "wb" ) )
  
  
def softmax_small_load( vid, i ):
  read_path = get_save_path_softmax_small_i( vid, i )
  softmax = pickle.load( open( read_path, "rb" ) )
  return softmax


def score_small_dump( score, vid, i ):
  dump_path = get_save_path_score_small_i( vid, i )
  dump_dir  = os.path.dirname( dump_path )
  if not os.path.exists( dump_dir ):
    os.makedirs( dump_dir )
  pickle.dump( score, open( dump_path, "wb" ) )
  
  
def score_small_load( vid, i ):
  read_path = get_save_path_score_small_i( vid, i )
  score = pickle.load( open( read_path, "rb" ) )
  return score


def time_series_instances_dump( instances, vid, i, eps, num_reg ):
  dump_path = get_save_path_time_series_instances_i( vid, i, eps, num_reg )
  dump_dir  = os.path.dirname( dump_path )
  if not os.path.exists( dump_dir ):
    os.makedirs( dump_dir )
  pickle.dump( instances, open( dump_path, "wb" ) )
  

def time_series_instances_load( vid, i, eps, num_reg ):
  read_path = get_save_path_time_series_instances_i( vid, i, eps, num_reg )
  instances = pickle.load( open( read_path, "rb" ) )
  return instances


def time_series_metrics_dump( time_series_metrics, vid, i, eps, num_reg, flag_3d=0 ):
  dump_path = get_save_path_time_series_metrics_i( vid, i, eps, num_reg, flag_3d )
  dump_dir  = os.path.dirname( dump_path )
  if not os.path.exists( dump_dir ):
    os.makedirs( dump_dir )
  pickle.dump( time_series_metrics, open( dump_path, "wb" ) )


def time_series_metrics_load( vid, i, eps, num_reg, flag_3d=0 ):
  read_path = get_save_path_time_series_metrics_i( vid, i, eps, num_reg, flag_3d )
  time_series_metrics = pickle.load( open( read_path, "rb" ) )
  return time_series_metrics
  
  
def write_analyzed_tracking( ):
  
  result_path = os.path.join(CONFIG.ANALYZE_TRACKING_DIR, "tracking_results_table.txt")
  with open(result_path, 'wt') as fi:
    
    tm = sorted(os.listdir( CONFIG.ANALYZE_TRACKING_DIR ))
    for t in tm:
      if '.p' in t:
        
        tracking_metrics  = pickle.load( open( CONFIG.ANALYZE_TRACKING_DIR + t, "rb" ) )        
        print(t, ':', file=fi )
        print("Recall & Precision & FAR & F  \\\\ ", file=fi )
        print( "{:.4f}".format(tracking_metrics['recall'][0]), "& {:.4f}".format(tracking_metrics['precision'][0]), "& {:.2f}".format(tracking_metrics['far'][0]), "& {:.4f} \\\\ ".format(tracking_metrics['f_measure'][0]), file=fi )
        print("GT & MT & PT & ML \\\\ ", file=fi )
        print( "{:.0f}".format(tracking_metrics['num_gt_ids'][0]), "& {:.0f}".format(tracking_metrics['mostly_tracked'][0]), "& {:.0f}".format(tracking_metrics['partially_tracked'][0]), "& {:.0f} \\\\ ".format(tracking_metrics['mostly_lost'][0]), file=fi )
        print("FP & FN & IDsw & FM \\\\ ", file=fi )
        print( "{:.0f}".format(tracking_metrics['fp'][0]), "& {:.0f}".format(tracking_metrics['misses'][0]), "& {:.0f}".format(tracking_metrics['switch_id'][0]), "({:.4f})".format(tracking_metrics['switch_id'][0] / tracking_metrics['gt_obj'][0]), "& {:.0f} \\\\ ".format(tracking_metrics['switch_tracked'][0]), file=fi )
        print("TP & MotA & MotP BB & MotB geo \\\\ ", file=fi )
        print( "{:.0f}".format(tracking_metrics['matches'][0]), "& {:.4f}".format(tracking_metrics['mot_a'][0]), "& {:.2f}".format(tracking_metrics['mot_p_bb'][0]), "& {:.2f} \\\\ ".format(tracking_metrics['mot_p_geo'][0]), file=fi )
        print(' ', file=fi)
        
        
def write_instances_info( metrics, mean_stats, std_stats ):
  
  num_prev_frames = CONFIG.NUM_PREV_FRAMES
  max_inst = int( np.load(CONFIG.HELPER_DIR + "max_inst.npy") )
  score_th = float(CONFIG.SCORE_THRESHOLD) / 100
  
  with open(CONFIG.ANALYZE_DIR + CONFIG.CLASSIFICATION_MODEL + '_instances_info.txt', 'wt') as fi:
    
    num_iou_0 = 0
    num_iou_b0 = 0
    for i in range(len(metrics['S'])):
      if metrics['S'][i] > 0 and metrics['score'][i] >= score_th:
        if metrics['iou'][i] >= 0.5:
          num_iou_b0 += 1
        elif metrics['iou'][i] < 0.5:
          num_iou_0 += 1
    
    print( "total number of instances greater score threshold (in the dataset): ", num_iou_0+num_iou_b0, file=fi )
    print( "IoU = 0: ", num_iou_0, file=fi )
    print( "IoU > 0: ", num_iou_b0, file=fi )
    print( " ", file=fi)
    
    num_iou_0 = 0
    num_iou_b0 = 0
    counter = 0
    list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))
    for vid in list_videos:
      images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + "/" ))
      for i in range(len(images_all)):
        if i >= num_prev_frames:
          for j in range(max_inst):
            if metrics['S'][counter] > 0 and metrics['score'][counter] >= score_th:
              if metrics['iou'][counter] >= 0.5:
                num_iou_b0 += 1
              elif metrics['iou'][counter] < 0.5:
                num_iou_0 += 1
            counter += 1
        else:
          counter += max_inst

    print( "number of instances: ", num_iou_0+num_iou_b0, file=fi )
    print( "IoU = 0: ", num_iou_0, file=fi )
    print( "IoU > 0: ", num_iou_b0, file=fi )
    print( " ", file=fi)
    
    M = sorted([ s for s in mean_stats if 'iou' in s ])    
    for i in range(CONFIG.NUM_PREV_FRAMES+1):
      print( "number of considered frames: ", i+1,  file=fi)
      for s in M: print( s, ": {:.0f}".format(mean_stats[s][i])+"($\pm${:.0f})".format(std_stats[s][i]), file=fi )
      print( " ", file=fi)
    
    
def write_min_max_file( max_r2_list, max_mse_list, max_auc_list, max_acc_list, min_r2_list, min_mse_list, min_auc_list, min_acc_list ):

  result_path = os.path.join(CONFIG.IMG_ANALYZE_DIR, "max_min_results.txt")
  #with open(result_path, 'a') as fi:
  with open(result_path, 'wt') as fi:
    
    print("max R^2:", max_r2_list[0], "std:", max_r2_list[1], "num frames:", max_r2_list[2], "type:", max_r2_list[3], file=fi)
    print("max sigma:", max_mse_list[0], "std:", max_mse_list[1], "num frames:", max_mse_list[2], "type:", max_mse_list[3], file=fi)
    print("max auroc:", max_auc_list[0], "std:", max_auc_list[1], "num frames:", max_auc_list[2], "type:", max_auc_list[3], file=fi)
    print("max accuracy:", max_acc_list[0], "std:", max_acc_list[1], "num frames:", max_acc_list[2], "type:", max_acc_list[3], file=fi)
    print(" ", file=fi)
    print("minimum with LR in regression/ LR_L1 in classification with 0 additional frames", file=fi)
    print("min R^2:", min_r2_list[0], "std:", min_r2_list[1], "num frames: 0", "type:", min_r2_list[2], file=fi)
    print("min sigma:", min_mse_list[0], "std:", min_mse_list[1], "num frames: 0", "type:", min_mse_list[2], file=fi)
    print("min auroc:", min_auc_list[0], "std:", min_auc_list[1], "num frames 0:", "type:", min_auc_list[2], file=fi)
    print("min accuracy:", min_acc_list[0], "std:", min_acc_list[1], "num frames 0:", "type:", min_acc_list[2], file=fi)


def write_table_timeline( ):
  
  num_prev_frames = CONFIG.NUM_PREV_FRAMES
  reg_list = CONFIG.regression_models
  cl_list = CONFIG.classification_models
  read_path1 = CONFIG.ANALYZE_DIR + 'stats/'
  
  with open(CONFIG.ANALYZE_DIR + 'LR_L1_instances_info.txt', 'r') as f:
    lines = f.read().strip().split('\n')
  baseline = max( float(lines[1].split(':')[1]), float(lines[2].split(':')[1]) ) / float(lines[0].split(':')[1])

  result_path = os.path.join(CONFIG.IMG_ANALYZE_DIR, "results_table.txt")
  with open(result_path, 'wt') as fi:
    
    print("Meta Classification $\IoU = 0 , > 0$", file=fi )
    print("Naive Baseline:", "& ACC = ", "${:.2f}\%".format(100*baseline), "& AUROC = $50.00\%$      \\ ", file=fi )
    
    stats = pickle.load( open(  read_path1 + "GB_CL_stats.p", "rb" ) )    
    print("Entropy Baseline:       & ACC = ", "${:.2f}\%".format(100*np.mean(stats['entropy_test_acc'][0], axis=0)), "(\pm{:.2f}\%)".format(100*np.std(stats["entropy_test_acc"][0], axis=0)), "& AUROC = " "${:.2f}\%".format(100*np.mean(stats["entropy_test_auroc"][0], axis=0))+"(\pm{:.2f}\%)".format(100*np.std(stats["entropy_test_auroc"][0], axis=0)), " \\ ", file=fi )
      
    print("Score Baseline:       & ACC = ", "${:.2f}\%".format(100*np.mean(stats['score_test_acc'][0], axis=0)), "(\pm{:.2f}\%)".format(100*np.std(stats["score_test_acc"][0], axis=0)), "& AUROC = " "${:.2f}\%".format(100*np.mean(stats["score_test_auroc"][0], axis=0))+"(\pm{:.2f}\%)".format(100*np.std(stats["score_test_auroc"][0], axis=0)), " \\ ", file=fi )
    
    print( "       & LR_L1    &          & GB       &          & NN_L2    &          \\\\ ", file= fi)
    print( "       & ACC      & AUROC    & ACC      & AUROC    & ACC      & AUROC    \\\\ ", file= fi)
      
    for cl_type in cl_list:
    
      read_path = read_path1 + cl_type + "_CL_stats.p"
      stats = pickle.load( open( read_path, "rb" ) )
      
      ind_acc = 0
      ind_auc = 0
        
      for num_frames in range(1,num_prev_frames+1):
        
        if np.mean(stats["penalized_test_acc"][ind_acc], axis=0) < np.mean(stats["penalized_test_acc"][num_frames], axis=0):
          ind_acc = num_frames
        if np.mean(stats["penalized_test_auroc"][ind_auc], axis=0) < np.mean(stats["penalized_test_auroc"][num_frames], axis=0):
          ind_auc = num_frames
      
      print( "${:.2f}\%".format(100*np.mean(stats["penalized_test_acc"][ind_acc], axis=0))+"(\pm{:.2f}\%)".format(100*np.std(stats["penalized_test_acc"][ind_acc], axis=0))+"^{:.0f}$".format(ind_acc+1), end=" & ", file=fi )
      print( "${:.2f}\%".format(100*np.mean(stats["penalized_test_auroc"][ind_auc], axis=0))+"(\pm{:.2f}\%)".format(100*np.std(stats["penalized_test_auroc"][ind_auc], axis=0))+"^{:.0f}$".format(ind_auc+1), end=" & ", file=fi )
      
    print("   \\\\ ", file=fi )
      
    print("Meta Regression $\IoU$", file=fi )
    
    stats = pickle.load( open(  read_path1 + "GB_stats.p", "rb" ) )  
    print("Entropy Baseline:       & $R^2$ = ", "${:.2f}\%".format(100*np.mean(stats["entropy_test_r2"][0], axis=0))+"(\pm{:.2f}\%)".format(100*np.std(stats["entropy_test_r2"][0], axis=0)), "& $\sigma$ = " "${:.3f}".format(np.mean(stats["entropy_test_mse"][0], axis=0)), "(\pm{:.3f})".format(np.std(stats["entropy_test_mse"][0], axis=0)), " \\ ", file=fi )
    
    print("Score Baseline:       & $R^2$ = ", "${:.2f}\%".format(100*np.mean(stats["score_test_r2"][0], axis=0))+"(\pm{:.2f}\%)".format(100*np.std(stats["score_test_r2"][0], axis=0)), "& $\sigma$ = " "${:.3f}".format(np.mean(stats["score_test_mse"][0], axis=0)), "(\pm{:.3f})".format(np.std(stats["score_test_mse"][0], axis=0)), " \\ ", file=fi )
    
    print( "       & LR       &          & LR_L1    &          & LR_L2    &          \\\\ ", file= fi)
    print( "       & $\sigma$ & $R^2$    & $\sigma$ & $R^2$    & $\sigma$ & $R^2$    \\\\ ", file= fi)

    for reg_type in reg_list[0:int(len(reg_list)/2)]:
      
      read_path = read_path1 + reg_type + "_stats.p"
      stats = pickle.load( open( read_path, "rb" ) )
      
      ind_sig = 0
      ind_r2 = 0
        
      for num_frames in range(1,num_prev_frames+1):
        
        if np.mean(stats["regr_test_mse"][ind_sig], axis=0) > np.mean(stats["regr_test_mse"][num_frames], axis=0):
          ind_sig = num_frames
        if np.mean(stats["regr_test_r2"][ind_r2], axis=0) < np.mean(stats["regr_test_r2"][num_frames], axis=0):
          ind_r2 = num_frames
      
      print( "${:.3f}".format(np.mean(stats["regr_test_mse"][ind_sig], axis=0))+"(\pm{:.3f})".format(np.std(stats["regr_test_mse"][ind_sig], axis=0))+"^{:.0f}$".format(ind_sig+1), end=" & ", file=fi )
      print( "${:.2f}\%".format(100*np.mean(stats["regr_test_r2"][ind_r2], axis=0))+"(\pm{:.2f}\%)".format(100*np.std(stats["regr_test_r2"][ind_r2], axis=0))+"^{:.0f}$".format(ind_r2+1), end=" & ", file=fi )
      
    print("   \\\\ ", file=fi )
        
    print( "       & GB       &          & NN_L1    &          & NN_L2    &          \\\\ ", file= fi)
    print( "       & $\sigma$ & $R^2$    & $\sigma$ & $R^2$    & $\sigma$ & $R^2$    \\\\ ", file= fi)
    
    reg_list_short = reg_list[int(len(reg_list)/2): len(reg_list)]
    for reg_type in reg_list_short:
      
      read_path = read_path1 + reg_type + "_stats.p"
      stats = pickle.load( open( read_path, "rb" ) )
      
      ind_sig = 0
      ind_r2 = 0
        
      for num_frames in range(1,num_prev_frames+1):
        
        if np.mean(stats["regr_test_mse"][ind_sig], axis=0) > np.mean(stats["regr_test_mse"][num_frames], axis=0):
          ind_sig = num_frames
        if np.mean(stats["regr_test_r2"][ind_r2], axis=0) < np.mean(stats["regr_test_r2"][num_frames], axis=0):
          ind_r2 = num_frames
      
      print( "${:.3f}".format(np.mean(stats["regr_test_mse"][ind_sig], axis=0))+"(\pm{:.3f})".format(np.std(stats["regr_test_mse"][ind_sig], axis=0))+"^{:.0f}$".format(ind_sig+1), end=" & ", file=fi )
      print( "${:.2f}\%".format(100*np.mean(stats["regr_test_r2"][ind_r2], axis=0))+"(\pm{:.2f}\%)".format(100*np.std(stats["regr_test_r2"][ind_r2], axis=0))+"^{:.0f}$".format(ind_r2+1), end=" & ", file=fi )
      
    print("   \\\\ ", file=fi )  


