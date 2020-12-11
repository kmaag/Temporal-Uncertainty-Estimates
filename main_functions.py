#!/usr/bin/env python3
"""
script including
class objects and functions called in main
"""

import os
import time
import pickle
import numpy as np
import matplotlib.colors as colors
from multiprocessing import Pool
from sklearn.metrics import r2_score, mean_squared_error, roc_curve, auc

from global_defs import CONFIG
from metrics     import compute_ts_instances, analyze_tracking_vid, comp_time_series_metrics
from in_out      import instances_load, ground_truth_load, softmax_load, score_load,\
                        instances_small_dump, softmax_small_dump, score_small_dump,\
                        get_save_path_instances_small_i, instances_small_load,\
                        get_save_path_time_series_instances_i, time_series_instances_load,\
                        get_save_path_time_series_metrics_i, softmax_small_load, score_small_load,\
                        time_series_metrics_dump, time_series_metrics_load, write_analyzed_tracking,\
                        write_instances_info, write_min_max_file, write_table_timeline
from plot        import plot_matching, plot_scatter_metric_iou, plot_metrics_per_component,\
                        plot_metrics_per_class, plot_metrics_per_class_all, plot_instances_shapes,\
                        plot_instances_shapes_flexible, plot_scatter_lifetime, plot_map,\
                        plot_map_models, visualize_regr_classif_i, plot_regression_scatter,\
                        plot_coef_timeline, plot_train_val_test_timeline, plot_r2_auc_timeline_data,\
                        plot_r2_auc_metrics, plot_baselines_vs_ours
from calculate   import compute_time_series_3d_metrics, comp_mean_average_precision,\
                        regression_fit_and_predict, classification_fit_and_predict
from helper      import time_series_metrics_to_nparray, instance_search,\
                        time_series_metrics_to_dataset, split_tvs_and_concatenate,\
                        concatenate_val_for_visualization
                     

#----------------------------#
class compute_time_series_instances(object):
#----------------------------#

  def __init__(self, num_cores=1):
    """
    object initialization
    :param num_cores: (int) number of cores used for parallelization
    """
    self.num_cores = num_cores if not hasattr(CONFIG, 'NUM_CORES') else CONFIG.NUM_CORES
    

  def compute_time_series_instances_per_image(self):
    """
    perform time series instances computation with matching algorithm
    """
    print('calculating time series instances')
    start = time.time()
    
    num_img_per_vid = []
    
    list_videos = sorted(os.listdir( CONFIG.PRED_DIR ))  
    for vid in list_videos:
      images_all = sorted(os.listdir( CONFIG.PRED_DIR + vid + '/' ))  
      
      if not os.path.exists( CONFIG.INSTANCES_SMALL_DIR + vid + '/' ):
        os.makedirs( CONFIG.INSTANCES_SMALL_DIR + vid + '/' )
      if not os.path.exists( CONFIG.SOFTMAX_SMALL_DIR + vid + '/' ):
        os.makedirs( CONFIG.SOFTMAX_SMALL_DIR + vid + '/' )
      
      p = Pool(self.num_cores)                                         
      p_args = [ (vid,k) for k in range(len(images_all)) ]
      p.starmap( self.delete_intersection_with_ignore_region, p_args ) 
      p.close()
      
      num_img_per_vid.append( len(images_all) )
    
    p_args = [ (list_videos[k],num_img_per_vid[k]) for k in range(len(list_videos)) ]
    Pool(self.num_cores).starmap( compute_ts_instances, p_args ) 
    
    print('Time needed:', time.time()-start)
    
    
  def delete_intersection_with_ignore_region(self,vid,i):
    """
    delete instances that have large intersection with the ignore region
    """    
    if not os.path.isfile( get_save_path_instances_small_i(vid,i) ):
      
      print(vid,i)
      
      instances = instances_load( vid, i )
      gt = ground_truth_load( vid, i )
      softmax = softmax_load( vid, i )
      score = score_load( vid, i )

      remaining_inst_list = []
      for j in range(instances.shape[0]):
        
        intersection = np.count_nonzero(instances[j,gt==10000])
        pixel_inst_j = np.count_nonzero(instances[j]>0)
        
        if pixel_inst_j > 0:
          
          # cars are not in the ignore regions, we have to delete cars (id 1)
          if CONFIG.IMG_TYPE == 'mot':
            class_j = int(instances[j].max() // 1000)
            if intersection/pixel_inst_j < 0.8 and class_j == 2:
              remaining_inst_list.append(j)
        
          elif intersection/pixel_inst_j < 0.8:
            remaining_inst_list.append(j)

      instances_new = np.zeros((len(remaining_inst_list), instances.shape[1], instances.shape[2]))
      if CONFIG.MODEL_NAME == 'mask_rcnn':
        softmax_new = np.zeros((len(remaining_inst_list), instances.shape[1], instances.shape[2], softmax.shape[3]))
      elif CONFIG.MODEL_NAME == 'yolact':
        softmax_new = np.zeros((len(remaining_inst_list), softmax.shape[1]))
      score_new = np.zeros((len(remaining_inst_list)))
      
      if remaining_inst_list == []:
        print('empty:', vid,i)
      
      counter = 0
      for j in range(instances.shape[0]):
        if j in remaining_inst_list:
          instances_new[counter] = instances[j]
          softmax_new[counter] = softmax[j]
          score_new[counter] = score[j]
          counter += 1
      
      instances_small_dump( instances_new, vid, i )
      softmax_small_dump( softmax_new, vid, i )
      score_small_dump( score_new, vid, i )
    


#----------------------------#
class plot_time_series_instances(object):
#----------------------------#
  
  def __init__(self, num_cores=1):
    """
    object initialization
    :param num_cores: (int)  number of cores used for parallelization
    """
    self.num_cores = num_cores if not hasattr(CONFIG, 'NUM_CORES') else CONFIG.NUM_CORES  
    
    
  def plot_time_series_instances_per_image(self):
    """
    plot time series instances 
    """
    print('plot time series instances')
    
    colors_list_tmp = list(colors._colors_full_map.values())  # 1163 colors
    colors_list = []
    for color in colors_list_tmp:
      if len(color) == 7:
        colors_list.append(color)
    
    list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))
    for vid in list_videos: 
      images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
      p = Pool(self.num_cores)
      p_args = [ (vid,k,colors_list) for k in range(len(images_all)) ]
      p.starmap( plot_matching, p_args ) 
      p.close()
      
      

#----------------------------#
class analyze_tracking_algo(object):
#----------------------------#

  def __init__(self, num_cores=1):
    """
    object initialization
    :param num_cores: (int) number of cores used for parallelization
    :param classes:   (int) classes of the dataset
    """
    self.num_cores = num_cores if not hasattr(CONFIG, 'NUM_CORES') else CONFIG.NUM_CORES
    self.classes   = CONFIG.CLASSES
    

  def analyze_tracking(self):
    """
    analyze results of the tracking algorithm
    """
    if os.path.isfile( CONFIG.ANALYZE_TRACKING_DIR + 'tracking_metrics.p' ):  
      print('skip calculating')
    
    else:
    
      print('calculating object tracking evaluation metrics')
      start = time.time()
      
      if not os.path.exists( CONFIG.ANALYZE_TRACKING_DIR ):
        os.makedirs( CONFIG.ANALYZE_TRACKING_DIR )
      
      for c in self.classes:
        print('class', c)
        
        list_gt_ids = []
        num_img_per_vid = []
        
        list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))   
        for vid in list_videos:
          images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
          num_img_per_vid.append( len(images_all) )
          
          list_gt_ids_tmp = []
          
          for n in range(len(images_all)):
            
            gt_image = ground_truth_load(vid, n)
            
            for k in np.unique(gt_image):
              if k != 0 and k!= 10000:
                
                gt_class = gt_image // 1000
                if c == gt_class[gt_image==k].max():
              
                  list_gt_ids_tmp.append(k)
            
          list_gt_ids_tmp = np.unique(list_gt_ids_tmp)
          list_gt_ids.append(list_gt_ids_tmp) 
        
          print(vid, list_gt_ids_tmp)

        p_args = [ (list_videos[k],num_img_per_vid[k],np.asarray(list_gt_ids[k]),c) for k in range(len(list_videos)) ]
        Pool(self.num_cores).starmap( analyze_tracking_vid, p_args ) 
        
      print('Start concatenate')
      
      for c in self.classes:
        print('class', c)
        
        tracking_metrics = { 'num_frames': np.zeros((1)), 'gt_obj': np.zeros((1)), 'fp': np.zeros((1)), 'misses': np.zeros((1)), 'mot_a': np.zeros((1)), 'dist_bb': np.zeros((1)), 'dist_geo': np.zeros((1)), 'matches': np.zeros((1)), 'mot_p_bb': np.zeros((1)), 'mot_p_geo': np.zeros((1)), 'far': np.zeros((1)), 'f_measure': np.zeros((1)), 'precision': np.zeros((1)), 'recall': np.zeros((1)), 'switch_id': np.zeros((1)), 'num_gt_ids': np.zeros((1)), 'mostly_tracked': np.zeros((1)), 'partially_tracked': np.zeros((1)), 'mostly_lost': np.zeros((1)), 'switch_tracked': np.zeros((1)) }
        
        list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))
        for vid in list_videos:
          
          tracking_metrics_i  = pickle.load( open( CONFIG.ANALYZE_TRACKING_DIR + 'tracking_metrics_' + vid + '_class' + str(c) + '.p', 'rb' ) )
          
          for tm in tracking_metrics:
            if tm not in ['mot_a', 'mot_p_bb', 'mot_p_geo', 'far', 'f_measure', 'precision', 'recall']:
              tracking_metrics[tm] += tracking_metrics_i[tm]
        
        tracking_metrics = self.comp_remaining_metrics(tracking_metrics)
        
        if len(self.classes) == 1:
          pickle.dump( tracking_metrics, open( CONFIG.ANALYZE_TRACKING_DIR + 'tracking_metrics.p', 'wb' ) )  
        else: 
          pickle.dump( tracking_metrics, open( CONFIG.ANALYZE_TRACKING_DIR + 'tracking_metrics_class' + str(c) + '.p', 'wb' ) )  
          
      if len(self.classes) > 1:
        
        for tm in tracking_metrics:
            tracking_metrics[tm] = 0
              
        for c in self.classes:
          tracking_metrics_i  = pickle.load( open( CONFIG.ANALYZE_TRACKING_DIR + 'tracking_metrics_class' + str(c) + '.p', 'rb' ) )
          
          for tm in tracking_metrics:
            if tm not in ['mot_a', 'mot_p_bb', 'mot_p_geo', 'far', 'f_measure', 'precision', 'recall']:
              tracking_metrics[tm] += tracking_metrics_i[tm]
              
        tracking_metrics = self.comp_remaining_metrics(tracking_metrics)
        pickle.dump( tracking_metrics, open( CONFIG.ANALYZE_TRACKING_DIR + 'tracking_metrics.p', 'wb' ) )  
          
      print('Time needed:', time.time()-start)
    write_analyzed_tracking( )
      
      
  def comp_remaining_metrics(self, tracking_metrics):
    """
    helper function for metrics calculation
    """ 
  
    tracking_metrics['mot_a'] = 1 - ((tracking_metrics['misses'] + tracking_metrics['fp'] + tracking_metrics['switch_id'])/tracking_metrics['gt_obj'])
    
    tracking_metrics['mot_p_bb'] = tracking_metrics['dist_bb'] / tracking_metrics['matches']
    
    tracking_metrics['mot_p_geo'] = tracking_metrics['dist_geo'] / tracking_metrics['matches']
    
    tracking_metrics['far'] = tracking_metrics['fp'] / tracking_metrics['num_frames'] * 100
    
    tracking_metrics['f_measure'] = (2 * tracking_metrics['matches']) / (2 * tracking_metrics['matches'] + tracking_metrics['misses'] + tracking_metrics['fp'])
    
    tracking_metrics['precision'] = tracking_metrics['matches'] / ( tracking_metrics['matches'] + tracking_metrics['fp'])
    
    tracking_metrics['recall'] = tracking_metrics['matches'] / ( tracking_metrics['matches'] + tracking_metrics['misses'])
    
    return tracking_metrics
    
 

#----------------------------#
class compute_time_series_metrics(object):
#----------------------------#

  def __init__(self, num_cores=1):
    """
    object initialization
    :param num_cores: (int) number of cores used for parallelization
    :param epsilon:   (int) used in matching algorithm
    :param num_reg:   (int) used in matching algorithm
    :param max_inst:  (int) maximum number of instances in all image sequences
    """
    self.num_cores = num_cores if not hasattr(CONFIG, 'NUM_CORES') else CONFIG.NUM_CORES
    self.epsilon   = CONFIG.EPS_MATCHING
    self.num_reg   = CONFIG.NUM_REG_MATCHING
    self.max_inst  = 0


  def compute_time_series_metrics_per_image(self):
    """
    perform time series metrics computation 
    """
    print('calculating time series metrics')
    start = time.time()
    
    if os.path.isfile( CONFIG.HELPER_DIR + 'max_inst.npy' ):
      
      self.max_inst = np.load(CONFIG.HELPER_DIR + 'max_inst.npy')
      
    else:
      
      list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))
      for vid in list_videos: 
        
        images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
        
        for i in range(len(images_all)):
          
          ts_inst = time_series_instances_load(vid, i, self.epsilon, self.num_reg)
          ts_inst[ts_inst<0] *= -1
          print(i, np.unique(ts_inst))
          ts_inst = ts_inst % 10000
          
          if ts_inst.shape[0] > 0:
        
            self.max_inst = max(self.max_inst, ts_inst.max())
            
        print('max:', self.max_inst)
      
      if not os.path.exists( CONFIG.HELPER_DIR ):
        os.makedirs( CONFIG.HELPER_DIR )
    
      np.save(os.path.join(CONFIG.HELPER_DIR, 'max_inst'), self.max_inst)
      print('max_inst', self.max_inst, 'calculated in {}s\r'.format( round(time.time()-start) ) )  
      
    print('maximal number of instances:', self.max_inst)
    
    if not os.path.isfile( get_save_path_time_series_metrics_i( 'all', '_2d', self.epsilon, self.num_reg ) ):
      
      list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))
      for vid in list_videos: 
        
        if not os.path.exists( CONFIG.METRICS_DIR + vid + "/" ):
          os.makedirs( CONFIG.METRICS_DIR + vid + "/" )
        
        images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
        p = Pool(self.num_cores)
        p_args = [ (vid,k) for k in range(len(images_all)) ]
        p.starmap( self.compute_time_series_metrics_i, p_args ) 
        p.close()
      
      metrics = time_series_metrics_load( list_videos[0], 0, self.epsilon, self.num_reg )
      for vid in list_videos: 
        images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
        for i in range(len(images_all)):
          if vid != list_videos[0] or i != 0:
            m = time_series_metrics_load( vid, i, self.epsilon, self.num_reg )
            for j in metrics:
              metrics[j] += m[j]
      
      print(len(metrics['S']) / self.max_inst)
      time_series_metrics_dump( metrics, 'all', '_2d', self.epsilon, self.num_reg )
      
    compute_time_series_3d_metrics(self.max_inst)
    

  def compute_time_series_metrics_i( self, vid, i ):
    """
    compute time series metrics 
    """
    if os.path.isfile( get_save_path_time_series_instances_i( vid, i, self.epsilon, self.num_reg ) ) and not os.path.isfile( get_save_path_time_series_metrics_i( vid, i, self.epsilon, self.num_reg ) ):
      
      start = time.time()
      
      instances = time_series_instances_load( vid, i, self.epsilon, self.num_reg )
      softmax = softmax_small_load( vid, i )
      score = score_small_load( vid, i )
      gt = ground_truth_load(vid, i)

      time_series_metrics = comp_time_series_metrics( instances, softmax, score, gt, self.max_inst )
      time_series_metrics_dump( time_series_metrics, vid, i, self.epsilon, self.num_reg ) 
      print('image', i, 'processed in {}s\r'.format( round(time.time()-start) ) )      
      


#----------------------------#
class visualize_time_series_metrics(object):
#----------------------------#

  def __init__(self):
    """
    object initialization
    :param num_imgs:          (int)    number of images to be processed
    :param epsilon:           (int)    used in matching algorithm
    :param num_reg:           (int)    used in matching algorithm
    :param comp_class_string: (string) considered class
    :param metrics_list:      (list)   considered metrics
    """
    self.num_imgs          = CONFIG.NUM_IMAGES
    self.epsilon           = CONFIG.EPS_MATCHING
    self.num_reg           = CONFIG.NUM_REG_MATCHING
    self.comp_class_string = CONFIG.CLASS_COMPONENT
    self.metrics_list      = CONFIG.METRICS_COMPONENT
    
  
  def visualize_metrics_vs_iou(self):
    """
    plot metrics vs iou as scatterplots
    """
    print('visualize metrics vs iou')
    
    if CONFIG.MODEL_NAME == 'mask_rcnn':
      list_metrics = ['S', 'S_in', 'S_bd', 'S_rel', 'S_rel_in', 'mean_x', 'mean_y', 'E', 'E_in', 'E_bd', 'E_rel', 'E_rel_in', 'score', 'survival', 'V', 'V_in', 'V_bd', 'V_rel', 'V_rel_in', 'ratio', 'deformation', 'M', 'M_in', 'M_bd', 'M_rel', 'M_rel_in', 'diff_mean', 'diff_size']
      
      plot_scatter_metric_iou(list_metrics)
      
      list_metrics = ['E', 'V', 'S', 'score']
      plot_scatter_metric_iou(list_metrics)
      
    elif CONFIG.MODEL_NAME == 'yolact':
      list_metrics = ['S', 'E', 'survival', 'S_in', 'E_rel', 'ratio', 'S_bd', 'score', 'deformation', 'S_rel', 'mean_x', 'diff_mean', 'S_rel_in', 'mean_y', 'diff_size']
      plot_scatter_metric_iou(list_metrics)
      
      list_metrics = ['E', 'E_rel', 'S', 'score']
      plot_scatter_metric_iou(list_metrics)
      
      list_metrics = ['deformation', 'survival', 'ratio']
      plot_scatter_metric_iou(list_metrics)
    
    
  def visualize_time_series_metrics_per_component(self):
    """
    plot time series metrics 
    """
    print('visualize time series metrics per component')
    
    max_id_comp_list = instance_search( self.comp_class_string )
    
    print(max_id_comp_list)
    
    list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))
    for i,vid in zip( range(len(list_videos)), list_videos ):
      
      max_id = max_id_comp_list[i]
      
      # component existent
      if max_id != -1:
        
        list0 = []
        list1 = []
        list2 = []
        list_iou = []
        list_img_num = []
        flag_start = 0
        
        images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
        for img in range(len(images_all)):
          
          time_series_metrics = time_series_metrics_load( vid, img, self.epsilon, self.num_reg )
      
          if time_series_metrics['S'][max_id-1] > 0:
            
            list0.append(time_series_metrics[self.metrics_list[0]][max_id-1])
            list1.append(time_series_metrics[self.metrics_list[1]][max_id-1])
            list2.append(time_series_metrics[self.metrics_list[2]][max_id-1])
            list_iou.append(time_series_metrics['iou'][max_id-1])
            list_img_num.append(img)
            
            if flag_start==0:
              flag_start = 1
            
          elif flag_start == 1:
            
            flag_reg = 0
            
            for k in range(img,img+self.num_reg-1):
              
              t_s_metrics = time_series_metrics_load( vid, k, self.epsilon, self.num_reg )
              
              if t_s_metrics['S'][max_id-1] > 0:
                flag_reg = 1
                
            if flag_reg == 1:
              list0.append(np.nan)
              list1.append(np.nan)
              list2.append(np.nan)
              list_iou.append(np.nan)
              list_img_num.append(img)
              
            else:
              break
        
        save_path  = CONFIG.IMG_METRICS_DIR + 'per_comp/' + str(vid) + '_metrics_class_' + self.comp_class_string + '_start' + str(list_img_num[0]) + '_end' + str(list_img_num[-1]) + '/'
        if not os.path.exists( save_path ):
          os.makedirs( save_path )
        
        for j in range(len(list_img_num)):
          plot_metrics_per_component(save_path, list_img_num, list0, list1, list2, list_iou, self.metrics_list, vid, j, max_id)


  def visualize_time_series_metrics_per_class(self):
    """
    plot time series metrics per class
    """
    print('visualize time series metrics per class')
    
    if not os.path.exists( CONFIG.IMG_METRICS_DIR + 'per_class/' ):
      os.makedirs( CONFIG.IMG_METRICS_DIR + 'per_class/' )
      
    max_inst = int( np.load(CONFIG.HELPER_DIR + 'max_inst.npy') )
    print('maximal number of instances:', max_inst)
    
    list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))
    
    # 3: metrics_class[0], S, iou
    metrics_class = np.zeros((self.num_imgs, max_inst*len(list_videos), 3))
    
    for i,vid in zip( range(len(list_videos)), list_videos ): 
      
      metrics_class_vid = plot_metrics_per_class(vid, self.comp_class_string, self.metrics_list[0], max_inst)
      
      metrics_class[0:metrics_class_vid.shape[0], max_inst*i:max_inst*(i+1),:] = metrics_class_vid
      
    plot_metrics_per_class_all(metrics_class, self.comp_class_string, self.metrics_list[0])
    
    
  def plot_time_series_instances_shapes(self):
    """
    plot time series components (three-dimensional) 
    """
    print('visualize time series instances shapes')
    max_id_comp_list = instance_search( self.comp_class_string )
    
    list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))
    for i,vid in zip( range(len(list_videos)), list_videos ):
      
      max_id = max_id_comp_list[i]
      # component existent
      if max_id != -1:
      
        plot_instances_shapes( vid, max_id, self.comp_class_string )
        plot_instances_shapes_flexible( vid, max_id, self.comp_class_string )
      
        
  def visualize_lifetime_mean(self):
    """
    plot components lifetime
    """
    print('visualize lifetime of components')
    
    size_cut = 1000
    
    max_inst = int( np.load(CONFIG.HELPER_DIR + 'max_inst.npy') )
    print('maximal number of instances:', max_inst)
    
    list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))
    
    lifetime_mean_size = np.zeros(( max_inst*len(list_videos), 2 ))
    
    for k,vid in zip( range(len(list_videos)), list_videos ): 
      
      print('video', vid)
      
      images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
      for img in range(len(images_all)):
        
        time_series_metrics = time_series_metrics_load ( vid, img, self.epsilon, self.num_reg )
      
        for i in range(0,max_inst):
        
          if time_series_metrics['S'][i] > 0:
          
            lifetime_mean_size[k*max_inst+i,0] += 1
            lifetime_mean_size[k*max_inst+i,1] += time_series_metrics['S'][i]
            
    for i in range(0,max_inst*len(list_videos)):  
      if lifetime_mean_size[i,0] > 0:
        lifetime_mean_size[i,1] = lifetime_mean_size[i,1] / lifetime_mean_size[i,0]
          
    # array with mean size and lifetime       
    idx1 = np.asarray(np.where(lifetime_mean_size[:,1] > 0))
    lifetime_mean_size_del = lifetime_mean_size[idx1[0,:],:]
    
    idx2 = np.asarray(np.where(lifetime_mean_size[:,1] > size_cut))
    lifetime_mean_size_del_cut = lifetime_mean_size[idx2[0,:],:]
    
    print(np.shape(lifetime_mean_size_del), np.shape(lifetime_mean_size_del_cut))
    
    mean_lifetime = np.zeros((2))
    mean_lifetime[0] = np.sum(lifetime_mean_size_del[:,0]) / lifetime_mean_size_del.shape[0]
    mean_lifetime[1] = np.sum(lifetime_mean_size_del_cut[:,0]) / lifetime_mean_size_del_cut.shape[0]
    
    plot_scatter_lifetime( mean_lifetime, lifetime_mean_size_del, size_cut, lifetime_mean_size_del_cut )
    
    result_path = os.path.join(CONFIG.IMG_METRICS_DIR + 'lifetime/', 'lifetime.txt')
    with open(result_path, 'wt') as fi:
      print('mean lifetime:', mean_lifetime[0], ', mean lifetime for instances > ', size_cut, ':', mean_lifetime[1], file=fi)
    
    
    
#----------------------------#
class compute_mean_ap(object):
#----------------------------#

  def __init__(self):
    """
    object initialization
    :param num_prev_frames: (int) number of previous considered frames
    """
    self.num_prev_frames = CONFIG.NUM_PREV_FRAMES
    

  def compute_map(self):
    """
    calculate mean average precision
    """
    if CONFIG.SCORE_THRESHOLD != '00':
      print('Error: wrong score threshold')
      exit()
    print('calculating mean average precision')
    start = time.time()

    result_path = os.path.join(CONFIG.MEAN_AP_DIR, 'results_mAP_npf' + str(self.num_prev_frames) + '.txt')
    if not os.path.exists( CONFIG.MEAN_AP_DIR ):
      os.makedirs( CONFIG.MEAN_AP_DIR )
    
    comp_mean_average_precision(result_path)
    
    print('Time needed:', time.time()-start)
    


#----------------------------#
class compute_mean_ap_metrics(object):
#----------------------------#

  def __init__(self):
    """
    object initialization
    :param epsilon:         (int) used in matching algorithm
    :param num_reg:         (int) used in matching algorithm
    :param runs:            (int) number of resamplings 
    :param num_prev_frames: (int) number of previous considered frames
    """
    self.epsilon         = CONFIG.EPS_MATCHING
    self.num_reg         = CONFIG.NUM_REG_MATCHING
    self.runs            = CONFIG.NUM_RESAMPLING
    self.num_prev_frames = CONFIG.NUM_PREV_FRAMES


  def compute_map_metrics(self):
    """
    calculate mean average precision with application of meta classification
    """
    if CONFIG.SCORE_THRESHOLD != '00':
      print('Error: wrong score threshold')
      exit()
    print('calculating mean average precision')
    start = time.time()
    
    score_map_th = float(CONFIG.MAP_THRESHOLD) / 100
    tvs = np.load(CONFIG.HELPER_DIR + 'tvs_runs' + str(self.runs ) + '.npy')
    
    if CONFIG.IMG_TYPE == 'kitti':
      self.runs = 3 
    
    max_inst = int( np.load(CONFIG.HELPER_DIR + "max_inst.npy") )
    num_imgs = CONFIG.NUM_IMAGES
    list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))
    y0a_pred_zero_val = np.zeros(( (num_imgs-(len(list_videos)*self.num_prev_frames)) * max_inst ))
    print('max instances', max_inst)
    counter_new = 0
      
    for run in range(self.runs): 
      
      train_val_test_string = tvs[run]
    
      metrics = time_series_metrics_load( tvs[run], 0, self.epsilon, self.num_reg, 1 ) 
      nclasses = np.max( metrics['class'] ) + 1
      
      Xa, classes, _, y0a, _, _ = time_series_metrics_to_dataset( metrics, nclasses, 0 )
      Xa = np.concatenate( (Xa,classes), axis=-1 )

      Xa_train, _, _, _, _, _, y0a_train, _, _ = split_tvs_and_concatenate( Xa, y0a, y0a, tvs[run] )
      
      Xa_val, y0a_val, y0a_zero_val, not_del_rows_val, _ = concatenate_val_for_visualization( Xa, y0a )
      
      print('shapes:', 'Xa train', np.shape(Xa_train), 'y0a train', np.shape(y0a_train), 'Xa val',  np.shape(Xa_val), 'y0a val', np.shape(y0a_val))
          
      y_train_pred, y_val_pred, _, _ = classification_fit_and_predict(Xa_train, y0a_train, Xa_val, y0a_val, Xa_val) 
      
      print(run, np.sum(y0a_val), np.sum(np.argmax(y_val_pred,axis=-1)), np.sum((y0a_val+np.argmax(y_val_pred,axis=-1)==2)))
      
      fpr, tpr, _ = roc_curve(y0a_train, y_train_pred[:,1])
      print('time series model auroc score (train):', auc(fpr, tpr) )
      fpr, tpr, _ = roc_curve(y0a_val, y_val_pred[:,1])
      print('time series model auroc score (val):', auc(fpr, tpr) )
      print(' ')
          
      y0a_pred = np.ones((y0a_zero_val.shape[0]))*-1
      y0a_calc = np.ones((y0a_zero_val.shape[0]))*-1 
      y_val_pred_argmax = [1 if y_val_pred[i,1]>(1-score_map_th) else 0 for i in range(y_val_pred.shape[0])]
      
      counter = 0
      for i in range(y0a_zero_val.shape[0]):
        if not_del_rows_val[i] == True:
          y0a_pred[i] = y_val_pred_argmax[counter]
          y0a_calc[i] = y0a_val[counter]
          counter += 1  

      counter = 0
      for vid,v in zip(list_videos, range(len(list_videos))):
        images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + "/" ))
        if train_val_test_string[v] == 'v' or train_val_test_string[v] == 's':
          y0a_pred_zero_val[counter_new*max_inst:(counter_new+len(images_all))*max_inst] = y0a_pred[counter*max_inst:(counter+len(images_all))*max_inst]
          
          print(vid, 'IoU = 0: (gt, pred)', np.sum(y0a_calc[counter*max_inst:(counter+len(images_all))*max_inst]==1), np.sum(y0a_pred[counter*max_inst:(counter+len(images_all))*max_inst]==1), np.sum(y0a_pred_zero_val[counter_new*max_inst:(counter_new+len(images_all))*max_inst]==1))
          print(vid, 'IoU > 0: (gt, pred)', np.sum(y0a_calc[counter*max_inst:(counter+len(images_all))*max_inst]==0), np.sum(y0a_pred[counter*max_inst:(counter+len(images_all))*max_inst]==0), np.sum(y0a_pred_zero_val[counter_new*max_inst:(counter_new+len(images_all))*max_inst]==0))
          
          counter_new += len(images_all)
        counter += len(images_all)

    print('FP:', np.sum(y0a_pred_zero_val==1))
    print('calculating mean average precision')
    
    result_path = os.path.join(CONFIG.MEAN_AP_METRICS_DIR, 'results_mAP_npf' + str(self.num_prev_frames) + '.txt')
    if not os.path.exists( CONFIG.MEAN_AP_METRICS_DIR ):
      os.makedirs( CONFIG.MEAN_AP_METRICS_DIR )
    
    comp_mean_average_precision(result_path, y0a_pred_zero_val)
    
    print('Time needed:', time.time()-start)
    


#----------------------------#
class plot_mean_average_precision(object):
#----------------------------#

  def __init__(self):
    """
    object initialization
    :param num_prev_frames: (int) number of previous considered frames
    :param models:          (list) models with different thresholds
    """
    self.num_prev_frames = CONFIG.NUM_PREV_FRAMES
    self.models          = sorted(os.listdir( CONFIG.MEAN_AP_DIR + '..' ))


  def plot_mean_ap(self):
    """
    plot mean average precision
    """
    print('plot mean average precision')
    counter = 0
    for model in self.models:
      print(model)
      if CONFIG.MODEL_NAME in model:
        counter += 1

    map_fp_fn_tp_classic = np.zeros((counter, 4))
    map_fp_fn_tp_meta = np.zeros((counter, 4))
    
    map_fp_fn_tp_classic, map_fp_fn_tp_meta = self.fill_mean_ap(map_fp_fn_tp_classic, map_fp_fn_tp_meta, CONFIG.MODEL_NAME)
    
    save_path = CONFIG.MEAN_AP_METRICS_DIR + '../images_' + CONFIG.MODEL_NAME + "_os" + str(CONFIG.FLAG_OBJ_SEG) + '/'
    if not os.path.exists( save_path ):
      os.makedirs( save_path )
    plot_map(save_path, map_fp_fn_tp_classic, map_fp_fn_tp_meta)
    
    for model in CONFIG.model_names:
      if model != CONFIG.MODEL_NAME:
        other_model = model
        
    map_fp_fn_tp_classic_other = np.zeros((counter, 4))
    map_fp_fn_tp_meta_other = np.zeros((counter, 4))
    
    map_fp_fn_tp_classic_other, map_fp_fn_tp_meta_other = self.fill_mean_ap(map_fp_fn_tp_classic_other, map_fp_fn_tp_meta_other, other_model)
    
    plot_map_models(save_path, map_fp_fn_tp_classic, map_fp_fn_tp_meta, map_fp_fn_tp_classic_other, map_fp_fn_tp_meta_other)
    
    
  def fill_mean_ap(self, map_fp_fn_tp_classic, map_fp_fn_tp_meta, model_name):
    """
    fill arrays with values
    """
    counter = 0
    for model in self.models:
      if model_name in model:
        print(model, 'mAP, FP, FN, TP')
        
        data_path = CONFIG.MEAN_AP_DIR + '../' + model + '/results_mAP_npf' + str(self.num_prev_frames) + '.txt' 
        with open(data_path, 'r') as f:
          dataset = f.read().strip().split('\n')
          print(dataset[-4].split('mAP:')[1], dataset[-2].split('FP')[1], dataset[-1].split('FN')[1], dataset[-3].split('TP')[1])
          map_fp_fn_tp_classic[counter, 0] = float( dataset[-4].split('mAP:')[1] )
          map_fp_fn_tp_classic[counter, 1] = int( dataset[-2].split('FP')[1] )
          map_fp_fn_tp_classic[counter, 2] = int( dataset[-1].split('FN')[1] )
          map_fp_fn_tp_classic[counter, 3] = int( dataset[-3].split('TP')[1] )
        
        data_path = CONFIG.MEAN_AP_METRICS_DIR + '/../' + model + "_os" + str(CONFIG.FLAG_OBJ_SEG) + '/results_mAP_npf' + str(self.num_prev_frames) + '.txt' 
        with open(data_path, 'r') as f:
          dataset = f.read().strip().split('\n')
          print(dataset[-4].split('mAP:')[1], dataset[-2].split('FP')[1], dataset[-1].split('FN')[1], dataset[-3].split('TP')[1])
          map_fp_fn_tp_meta[counter, 0] = float( dataset[-4].split('mAP:')[1] )
          map_fp_fn_tp_meta[counter, 1] = int( dataset[-2].split('FP')[1] )
          map_fp_fn_tp_meta[counter, 2] = int( dataset[-1].split('FN')[1] )
          map_fp_fn_tp_meta[counter, 3] = int( dataset[-3].split('TP')[1] )
         
        counter += 1
    
    print(model_name)
    print('max original mAP:', map_fp_fn_tp_classic[1:,0].max(), np.argmax(map_fp_fn_tp_classic[1:,0])+1)
    print('max meta mAP:', map_fp_fn_tp_meta[1:,0].max(), np.argmax(map_fp_fn_tp_meta[1:,0])+1)
    
    return map_fp_fn_tp_classic, map_fp_fn_tp_meta


    
#----------------------------#
class visualize_meta_prediction(object):
#----------------------------#
  
  def __init__(self, num_cores=1):
    """
    object initialization
    :param num_cores: (int) number of cores used for parallelization
    :param epsilon:   (int) used in matching algorithm
    :param num_reg:   (int) used in matching algorithm
    :param max_inst:  (int) maximal instances in all videos
    :param runs:      (int) number of resamplings 
    """
    self.num_cores = num_cores if not hasattr(CONFIG, 'NUM_CORES') else CONFIG.NUM_CORES
    self.epsilon   = CONFIG.EPS_MATCHING
    self.num_reg   = CONFIG.NUM_REG_MATCHING
    self.max_inst  = np.load(CONFIG.HELPER_DIR + 'max_inst.npy')
    self.runs      = CONFIG.NUM_RESAMPLING


  def visualize_regression_per_image(self):
    """
    perform metrics visualization
    """
    print('visualization running')
    t= time.time()
    
    tvs = np.load(CONFIG.HELPER_DIR + 'tvs_runs' + str(self.runs ) + '.npy')
    metrics = time_series_metrics_load( tvs[0], 0, self.epsilon, self.num_reg, 1 ) 
    nclasses = np.max( metrics['class'] ) + 1
    
    Xa, classes, ya, _, _, _ = time_series_metrics_to_dataset( metrics, nclasses, 0 )
    Xa = np.concatenate( (Xa,classes), axis=-1 )

    Xa_train, _, _, ya_train, _, _, _, _, _ = split_tvs_and_concatenate( Xa, ya, ya, tvs[0] )
      
    Xa_val, ya_val, ya_zero_val, not_del_rows_val, plot_image_list = concatenate_val_for_visualization( Xa, ya )
    
    print('shapes:', 'Xa train', np.shape(Xa_train), 'ya train', np.shape(ya_train), 'Xa val',  np.shape(Xa_val), 'ya val', np.shape(ya_val))
   
    y_train_pred, y_val_pred, _, _ = regression_fit_and_predict(Xa_train, ya_train, Xa_val, ya_val, Xa_val) 
    
    print('time series model r2 score (train):', r2_score(ya_train,y_train_pred) )
    print('time series model r2 score (val):', r2_score(ya_val,y_val_pred) )
    print(' ')
      
    ya_pred = np.zeros((ya_zero_val.shape[0]))
    counter = 0
    for i in range(ya_zero_val.shape[0]):
      if not_del_rows_val[i] == True:
        ya_pred[i] = y_val_pred[counter]
        counter += 1
    
    print('Start visualize time series metrics')
    colors_list_tmp = list(colors._colors_full_map.values())  # 1163 colors
    colors_list = []
    for color in colors_list_tmp:
      if len(color) == 7:
        colors_list.append(color)
        
    p_args = [ (ya_zero_val[self.max_inst*k:self.max_inst*(k+1)], ya_pred[self.max_inst*k:self.max_inst*(k+1)], i, j, k, colors_list, 0) for i,j,k in zip( list(zip(*plot_image_list))[0], list(zip(*plot_image_list))[1], list(zip(*plot_image_list))[2] ) ]
    Pool(self.num_cores).starmap( visualize_regr_classif_i, p_args )

    print('time needed ', time.time()-t)  
    
    
  
#----------------------------#
class visualize_IoU_prediction(object):
#----------------------------#
  
  def __init__(self, num_cores=1):
    """
    object initialization
    :param num_cores: (int) number of cores used for parallelization
    :param epsilon:   (int) used in matching algorithm
    :param num_reg:   (int) used in matching algorithm
    :param max_inst:  (int) maximal instances in all videos
    :param runs:      (int) number of resamplings 
    """
    self.num_cores = num_cores if not hasattr(CONFIG, 'NUM_CORES') else CONFIG.NUM_CORES
    self.epsilon   = CONFIG.EPS_MATCHING
    self.num_reg   = CONFIG.NUM_REG_MATCHING
    self.max_inst  = np.load(CONFIG.HELPER_DIR + 'max_inst.npy')
    self.runs      = CONFIG.NUM_RESAMPLING


  def visualize_classification_per_image(self):
    """
    perform metrics visualization
    """
    print('visualization of IoU=0/>0 running')
    t= time.time()
    
    tvs = np.load(CONFIG.HELPER_DIR + 'tvs_runs' + str(self.runs ) + '.npy')
    metrics = time_series_metrics_load( tvs[0], 0, self.epsilon, self.num_reg, 1 ) 
    nclasses = np.max( metrics['class'] ) + 1
    
    Xa, classes, _, y0a, _, _ = time_series_metrics_to_dataset( metrics, nclasses, 0 )
    Xa = np.concatenate( (Xa,classes), axis=-1 )
    
    Xa_train, _, _, _, _, _, y0a_train, _, _ = split_tvs_and_concatenate( Xa, y0a, y0a, tvs[0] )
      
    Xa_val, y0a_val, y0a_zero_val, not_del_rows_val, plot_image_list = concatenate_val_for_visualization( Xa, y0a )
    
    print('shapes:', 'Xa train', np.shape(Xa_train), 'y0a train', np.shape(y0a_train), 'Xa val',  np.shape(Xa_val), 'y0a val', np.shape(y0a_val))
        
    y_train_pred, y_val_pred, _, _ = classification_fit_and_predict(Xa_train, y0a_train, Xa_val, y0a_val, Xa_val) 
    
    fpr, tpr, _ = roc_curve(y0a_train, y_train_pred[:,1])
    print('time series model auroc score (train):', auc(fpr, tpr) )
    fpr, tpr, _ = roc_curve(y0a_val, y_val_pred[:,1])
    print('time series model auroc score (val):', auc(fpr, tpr) )
    print(' ')
      
    y0a_pred = np.zeros((y0a_zero_val.shape[0]))
    y_val_pred_argmax = np.argmax(y_val_pred,axis=-1)
    counter = 0
    for i in range(y0a_zero_val.shape[0]):
      if not_del_rows_val[i] == True:
        y0a_pred[i] = y_val_pred_argmax[counter]
        counter += 1
    
    print('Start visualize time series metrics')
    colors_list_tmp = list(colors._colors_full_map.values())  # 1163 colors
    colors_list = []
    for color in colors_list_tmp:
      if len(color) == 7:
        colors_list.append(color)
    
    # y0a_zero_val, y0a_pred: 0 : IoU>0, 1 : IoU=0
    y0a_zero_val = (y0a_zero_val-1)*-1
    y0a_pred = (y0a_pred-1)*-1
        
    p_args = [ (y0a_zero_val[self.max_inst*k:self.max_inst*(k+1)], y0a_pred[self.max_inst*k:self.max_inst*(k+1)], i, j, k, colors_list, 1) for i,j,k in zip( list(zip(*plot_image_list))[0], list(zip(*plot_image_list))[1], list(zip(*plot_image_list))[2] ) ]
    Pool(self.num_cores).starmap( visualize_regr_classif_i, p_args )

    print('time needed ', time.time()-t)  
    
  
  
#----------------------------#    
class analyze_metrics(object):
#----------------------------#

  def __init__(self, num_cores=1):
    """
    object initialization
    :param num_cores:       (int) number of cores used for parallelization
    :param epsilon:         (int) used in matching algorithm
    :param num_reg:         (int) used in matching algorithm
    :param num_prev_frames: (int) number of previous considered frames
    :param runs:            (int) number of resamplings 
    """
    self.num_cores       = num_cores if not hasattr(CONFIG, 'NUM_CORES') else CONFIG.NUM_CORES
    self.epsilon         = CONFIG.EPS_MATCHING
    self.num_reg         = CONFIG.NUM_REG_MATCHING
    self.num_prev_frames = CONFIG.NUM_PREV_FRAMES
    self.runs            = CONFIG.NUM_RESAMPLING
    
    
  def analyze_time_series_metrics( self ):
    """
    analze time series metrics
    """
    print('start analyzing')
    t= time.time()
    
    if not os.path.exists( CONFIG.ANALYZE_DIR+'stats/' ):
      os.makedirs( CONFIG.ANALYZE_DIR+'stats/' )
      os.makedirs( CONFIG.ANALYZE_DIR+'scatter/' )
      os.makedirs( CONFIG.ANALYZE_DIR+'train_val_test_timeline/' )
      os.makedirs( CONFIG.ANALYZE_DIR+'feature_importance/' )
    
    self.tvs = np.load(CONFIG.HELPER_DIR + 'tvs_runs' + str(self.runs ) + '.npy')
    
    metrics = time_series_metrics_load( self.tvs[0], 0, self.epsilon, self.num_reg, 1 ) 
    nclasses = np.max( metrics['class'] ) + 1
    _, _, _, _, X_names, class_names = time_series_metrics_to_dataset( metrics, nclasses, 0 )
    self.X_names = np.concatenate( (X_names,class_names), axis=-1 )
    print('Metrics: ', self.X_names)

    classification_stats = ['penalized_train_acc', 'penalized_train_auroc', 'penalized_val_acc', 'penalized_val_auroc', 'penalized_test_acc', 'penalized_test_auroc', 'entropy_train_acc','entropy_train_auroc', 'entropy_val_acc','entropy_val_auroc', 'entropy_test_acc','entropy_test_auroc', 'score_train_acc','score_train_auroc', 'score_val_acc','score_val_auroc', 'score_test_acc','score_test_auroc', 'iou0_found', 'iou0_not_found', 'not_iou0_found', 'not_iou0_not_found' ]
    regression_stats = ['regr_train_r2', 'regr_train_mse', 'regr_val_r2', 'regr_val_mse', 'regr_test_r2', 'regr_test_mse', 'entropy_train_r2', 'entropy_train_mse', 'entropy_val_r2', 'entropy_val_mse', 'entropy_test_r2', 'entropy_test_mse', 'score_train_r2', 'score_train_mse', 'score_val_r2', 'score_val_mse', 'score_test_r2', 'score_test_mse' ]
    
    stats = self.init_stats_frames( classification_stats, regression_stats )
    
    print('start runs')
    
    if 'LR' in CONFIG.REGRESSION_MODEL and (CONFIG.CLASSIFICATION_MODEL == 'LR_L1' or CONFIG.FLAG_CLASSIF == 0): 
        
      single_run_stats = self.init_stats_frames( classification_stats, regression_stats )

      p_args = [ ( single_run_stats, run ) for run in range(self.runs) ]
      single_run_stats = Pool(self.num_cores).starmap( self.fit_reg_cl_run_timeseries, p_args )
      
      for num_frames in range(self.num_prev_frames+1):
        for run in range(self.runs):
          for s in stats:
            if s not in ['metric_names']:
              stats[s][num_frames][run] = single_run_stats[run][s][num_frames][run]
      
    else:
      for run in range(self.runs):
        stats = self.fit_reg_cl_run_timeseries(stats, run)
    
    pickle.dump( stats, open( CONFIG.ANALYZE_DIR + 'stats/' + CONFIG.REGRESSION_MODEL + '_stats.p', 'wb' ) )
    if CONFIG.FLAG_CLASSIF == 1:
      pickle.dump( stats, open( CONFIG.ANALYZE_DIR + 'stats/' + CONFIG.CLASSIFICATION_MODEL + '_CL_stats.p', 'wb' ) ) 
    print('regression (and classification) finished')
    
    mean_stats = dict({})
    std_stats = dict({})
    
    for s in classification_stats:
      mean_stats[s] = 0.5*np.ones((self.num_prev_frames+1))
      std_stats[s] = 0.5*np.ones((self.num_prev_frames+1))
    for s in regression_stats:
      mean_stats[s] = np.zeros((self.num_prev_frames+1,))
      std_stats[s] = np.zeros((self.num_prev_frames+1,))
    mean_stats['coef'] = np.zeros((self.num_prev_frames+1, len(self.X_names)*(self.num_prev_frames+1)))
    std_stats['coef'] = np.zeros((self.num_prev_frames+1, len(self.X_names)*(self.num_prev_frames+1)))
    
    for num_frames in range(self.num_prev_frames+1):
      for s in stats:
        if s not in [ 'metric_names']:
          mean_stats[s][num_frames] = np.mean(stats[s][num_frames], axis=0)
          std_stats[s][num_frames]  = np.std( stats[s][num_frames], axis=0)

    if CONFIG.FLAG_CLASSIF == 1 and ('LR' in CONFIG.CLASSIFICATION_MODEL or 'GB' in CONFIG.CLASSIFICATION_MODEL):
      plot_coef_timeline(mean_stats, self.X_names)
    
    num_timeseries = np.arange(1, self.num_prev_frames+2)
      
    plot_train_val_test_timeline(num_timeseries, np.asarray(mean_stats['regr_train_r2']), np.asarray(std_stats['regr_train_r2']), np.asarray(mean_stats['regr_val_r2']), np.asarray(std_stats['regr_val_r2']), np.asarray(mean_stats['regr_test_r2']), np.asarray(std_stats['regr_test_r2']), 'r2')
        
    if CONFIG.FLAG_CLASSIF == 1:
          
      plot_train_val_test_timeline(num_timeseries, np.asarray(mean_stats['penalized_train_auroc']), np.asarray(std_stats['penalized_train_auroc']), np.asarray(mean_stats['penalized_val_auroc']), np.asarray(std_stats['penalized_val_auroc']), np.asarray(mean_stats['penalized_test_auroc']), np.asarray(std_stats['penalized_test_auroc']), 'auc')
        
      plot_train_val_test_timeline(num_timeseries, np.asarray(mean_stats['penalized_train_acc']), np.asarray(std_stats['penalized_train_acc']), np.asarray(mean_stats['penalized_val_acc']), np.asarray(std_stats['penalized_val_acc']), np.asarray(mean_stats['penalized_test_acc']), np.asarray(std_stats['penalized_test_acc']), 'acc')
      
      write_instances_info(metrics, mean_stats, std_stats)
    
    print('time needed ', time.time()-t)
    
    
  def init_stats_frames( self, classification_stats, regression_stats ):
  
    stats     = dict({})
                    
    for s in classification_stats:
      stats[s] = 0.5*np.ones((self.num_prev_frames+1, self.runs))
      
    for s in regression_stats:
      stats[s] = np.zeros((self.num_prev_frames+1, self.runs))
    
    stats['coef'] = np.zeros((self.num_prev_frames+1, self.runs, len(self.X_names)*(self.num_prev_frames+1)))
    stats['metric_names'] = self.X_names
    
    return stats 
  
  
  def fit_reg_cl_run_timeseries( self, stats, run ): 
    
    num_metrics = len(self.X_names)
    
    metrics = time_series_metrics_load( self.tvs[run], 0, self.epsilon, self.num_reg, 1 ) 
    nclasses = np.max( metrics['class'] ) + 1
    
    Xa, classes, ya, y0a, _, _ = time_series_metrics_to_dataset( metrics, nclasses, run )
    Xa = np.concatenate( (Xa,classes), axis=-1 )
    
    Xa_train_all, Xa_val_all, Xa_test_all, ya_train, ya_val, ya_test, y0a_train, y0a_val, y0a_test = split_tvs_and_concatenate( Xa, ya, y0a, self.tvs[run], run )
    
    for num_frames in range(self.num_prev_frames+1):
      
      Xa_train = Xa_train_all[:,0:(num_metrics * (num_frames+1))]
      Xa_val= Xa_val_all[:,0:(num_metrics * (num_frames+1))]
      Xa_test = Xa_test_all[:,0:(num_metrics * (num_frames+1))]
      
      print('run', run, self.tvs[run], 'num frames', num_frames, 'shapes:', 'Xa train', np.shape(Xa_train), 'Xa val', np.shape(Xa_val), 'Xa test', np.shape(Xa_test), 'ya train', np.shape(ya_train), 'ya val', np.shape(ya_val), 'ya test', np.shape(ya_test))
      
      # classification
      if CONFIG.FLAG_CLASSIF == 1:

        y0a_train_pred, y0a_val_pred, y0a_test_pred, coefs_model = classification_fit_and_predict( Xa_train, y0a_train, Xa_val, y0a_val, Xa_test, num_frames )

        stats['penalized_train_acc'][num_frames,run] = np.mean( np.argmax(y0a_train_pred,axis=-1)==y0a_train )
        stats['penalized_val_acc'][num_frames,run] = np.mean( np.argmax(y0a_val_pred,axis=-1)==y0a_val )
        stats['penalized_test_acc'][num_frames,run] = np.mean( np.argmax(y0a_test_pred,axis=-1)==y0a_test )

        fpr, tpr, _ = roc_curve(y0a_train, y0a_train_pred[:,1])
        stats['penalized_train_auroc'][num_frames,run] = auc(fpr, tpr)
        fpr, tpr, _ = roc_curve(y0a_val, y0a_val_pred[:,1])
        stats['penalized_val_auroc'][num_frames,run] = auc(fpr, tpr)
        fpr, tpr, _ = roc_curve(y0a_test, y0a_test_pred[:,1])
        stats['penalized_test_auroc'][num_frames,run] = auc(fpr, tpr)
        
        stats['iou0_found'][num_frames,run] = np.sum( np.logical_and(np.argmax(y0a_test_pred,axis=-1)  == 1, y0a_test == 1) ) \
                                            + np.sum( np.logical_and(np.argmax(y0a_val_pred,axis=-1) == 1, y0a_val == 1) ) \
                                            + np.sum( np.logical_and(np.argmax(y0a_train_pred,axis=-1) == 1, y0a_train == 1) )
        stats['iou0_not_found'][num_frames,run] = np.sum( np.logical_and(np.argmax(y0a_test_pred,axis=-1) == 0, y0a_test == 1) ) \
                                                + np.sum( np.logical_and(np.argmax(y0a_val_pred,axis=-1) == 0, y0a_val == 1) ) \
                                                + np.sum( np.logical_and(np.argmax(y0a_train_pred,axis=-1) == 0, y0a_train == 1) )
        stats['not_iou0_found'][num_frames,run] = np.sum( np.logical_and(np.argmax(y0a_test_pred,axis=-1) == 0, y0a_test == 0) ) \
                                                + np.sum( np.logical_and(np.argmax(y0a_val_pred,axis=-1) == 0, y0a_val == 0) ) \
                                                + np.sum( np.logical_and(np.argmax(y0a_train_pred,axis=-1) == 0, y0a_train == 0) )
        stats['not_iou0_not_found'][num_frames,run] = np.sum( np.logical_and(np.argmax(y0a_test_pred,axis=-1) == 1, y0a_test == 0) ) \
                                                    + np.sum( np.logical_and(np.argmax(y0a_val_pred,axis=-1) == 1, y0a_val == 0) ) \
                                                    + np.sum( np.logical_and(np.argmax(y0a_train_pred,axis=-1) == 1, y0a_train == 0) )

        coefs_tmp = np.zeros(((self.num_prev_frames+1)*num_metrics))
        coefs_tmp[0:(num_frames+1)*num_metrics] = coefs_model
        stats['coef'][num_frames,run] = np.asarray(coefs_tmp)

        
      # regression
      ya_train_pred, ya_val_pred, ya_test_pred, _ = regression_fit_and_predict( Xa_train, ya_train, Xa_val, ya_val, Xa_test, num_frames )

      stats['regr_train_mse'][num_frames,run] = np.sqrt( mean_squared_error(ya_train, ya_train_pred) )
      stats['regr_val_mse'][num_frames,run] = np.sqrt( mean_squared_error(ya_val, ya_val_pred) )
      stats['regr_test_mse'][num_frames,run] = np.sqrt( mean_squared_error(ya_test, ya_test_pred) )
      
      stats['regr_train_r2'][num_frames,run]  = r2_score(ya_train, ya_train_pred)
      stats['regr_val_r2'][num_frames,run]  = r2_score(ya_val, ya_val_pred)
      stats['regr_test_r2'][num_frames,run]  = r2_score(ya_test, ya_test_pred)

      if run == 0:
        plot_regression_scatter( Xa_test, ya_test, ya_test_pred, self.X_names, num_frames )
        
    # entropie baseline
    E_ind = 0
    for E_ind in range(len(self.X_names)):
      if self.X_names[E_ind] == 'E':
        break
    if CONFIG.FLAG_CLASSIF == 1:
    
      y0a_train_pred, y0a_val_pred, y0a_test_pred, _ = classification_fit_and_predict( Xa_train[:,E_ind].reshape((Xa_train.shape[0],1)), y0a_train, Xa_val[:,E_ind].reshape((Xa_val.shape[0],1)), y0a_val, Xa_test[:,E_ind].reshape((Xa_test.shape[0],1)) )
        
      stats['entropy_train_acc'][0,run] = np.mean( np.argmax(y0a_train_pred,axis=-1)==y0a_train )
      stats['entropy_val_acc'][0,run] = np.mean( np.argmax(y0a_val_pred,axis=-1)==y0a_val )
      stats['entropy_test_acc'][0,run] = np.mean( np.argmax(y0a_test_pred,axis=-1)==y0a_test )
      
      fpr, tpr, _ = roc_curve(y0a_train, y0a_train_pred[:,1])
      stats['entropy_train_auroc'][0,run] = auc(fpr, tpr)
      fpr, tpr, _ = roc_curve(y0a_val, y0a_val_pred[:,1])
      stats['entropy_val_auroc'][0,run] = auc(fpr, tpr)
      fpr, tpr, _ = roc_curve(y0a_test, y0a_test_pred[:,1])
      stats['entropy_test_auroc'][0,run] = auc(fpr, tpr)
      
    ya_train_pred, ya_val_pred, ya_test_pred, _ = regression_fit_and_predict( Xa_train[:,E_ind].reshape((Xa_train.shape[0],1)), ya_train, Xa_val[:,E_ind].reshape((Xa_val.shape[0],1)), ya_val, Xa_test[:,E_ind].reshape((Xa_test.shape[0],1)), 0 )

    stats['entropy_train_mse'][0,run] = np.sqrt( mean_squared_error(ya_train, ya_train_pred) )
    stats['entropy_val_mse'][0,run] = np.sqrt( mean_squared_error(ya_val, ya_val_pred) )
    stats['entropy_test_mse'][0,run] = np.sqrt( mean_squared_error(ya_test, ya_test_pred) )
    
    stats['entropy_train_r2'][0,run]  = r2_score(ya_train, ya_train_pred)
    stats['entropy_val_r2'][0,run]  = r2_score(ya_val, ya_val_pred)
    stats['entropy_test_r2'][0,run]  = r2_score(ya_test, ya_test_pred)
    
    # score baseline
    score_ind = 0
    for score_ind in range(len(self.X_names)):
      if self.X_names[score_ind] == 'score':
        break
    if CONFIG.FLAG_CLASSIF == 1:
    
      y0a_train_pred, y0a_val_pred, y0a_test_pred, _ = classification_fit_and_predict( Xa_train[:,score_ind].reshape((Xa_train.shape[0],1)), y0a_train, Xa_val[:,score_ind].reshape((Xa_val.shape[0],1)), y0a_val, Xa_test[:,score_ind].reshape((Xa_test.shape[0],1)) )
        
      stats['score_train_acc'][0,run] = np.mean( np.argmax(y0a_train_pred,axis=-1)==y0a_train )
      stats['score_val_acc'][0,run] = np.mean( np.argmax(y0a_val_pred,axis=-1)==y0a_val )
      stats['score_test_acc'][0,run] = np.mean( np.argmax(y0a_test_pred,axis=-1)==y0a_test )
      
      fpr, tpr, _ = roc_curve(y0a_train, y0a_train_pred[:,1])
      stats['score_train_auroc'][0,run] = auc(fpr, tpr)
      fpr, tpr, _ = roc_curve(y0a_val, y0a_val_pred[:,1])
      stats['score_val_auroc'][0,run] = auc(fpr, tpr)
      fpr, tpr, _ = roc_curve(y0a_test, y0a_test_pred[:,1])
      stats['score_test_auroc'][0,run] = auc(fpr, tpr)
      
    ya_train_pred, ya_val_pred, ya_test_pred, _ = regression_fit_and_predict( Xa_train[:,score_ind].reshape((Xa_train.shape[0],1)), ya_train, Xa_val[:,score_ind].reshape((Xa_val.shape[0],1)), ya_val, Xa_test[:,score_ind].reshape((Xa_test.shape[0],1)), 0 )

    stats['score_train_mse'][0,run] = np.sqrt( mean_squared_error(ya_train, ya_train_pred) )
    stats['score_val_mse'][0,run] = np.sqrt( mean_squared_error(ya_val, ya_val_pred) )
    stats['score_test_mse'][0,run] = np.sqrt( mean_squared_error(ya_test, ya_test_pred) )
    
    stats['score_train_r2'][0,run]  = r2_score(ya_train, ya_train_pred)
    stats['score_val_r2'][0,run]  = r2_score(ya_val, ya_val_pred)
    stats['score_test_r2'][0,run]  = r2_score(ya_test, ya_test_pred)
      
    return stats
    
    

#----------------------------#    
class visualize_analyzed_metrics(object):
#----------------------------#

  def __init__(self):
    """
    object initialization
    :param num_prev_frames: (int) number of previous frames
    :param reg_list:        (list) different regression models
    :param cl_list:         (list) different classification models
    """
    self.num_prev_frames = CONFIG.NUM_PREV_FRAMES
    self.reg_list        = CONFIG.regression_models
    self.cl_list         = CONFIG.classification_models
     
    
  def visualize_analyzed_time_series_metrics( self ):
    """
    visualize analzed time series metrics
    """  
    print('visualization running')
    
    if not os.path.exists( CONFIG.IMG_ANALYZE_DIR ):
      os.makedirs( CONFIG.IMG_ANALYZE_DIR )
    
    self.prepare_plot_timelines_and_min_max_file( )
    write_table_timeline( )
      
        
  def prepare_plot_timelines_and_min_max_file( self ):
    """
    preparation for plotting timelines as well as baselines
    """ 
    
    data_reg_list_mean = []
    data_reg_list_std = []
    data_cl_list_mean = []
    data_cl_list_std = []
    
    # r2/auc/acc, std, num_frames, reg/classif type
    max_r2_list = [0, 0, -1, 'empty']
    max_mse_list = [0, 0, -1, 'empty']
    max_auc_list = [0, 0, -1, 'empty']
    max_acc_list = [0, 0, -1, 'empty']
      
    read_path1 = CONFIG.ANALYZE_DIR + 'stats/'
          
    for reg_type in self.reg_list:
      
      read_path = read_path1 + reg_type + '_stats.p'
      stats = pickle.load( open( read_path, 'rb' ) )
      
      for num_frames in range(self.num_prev_frames+1):
        data_reg_list_mean.append( np.mean(stats['regr_test_r2'][num_frames], axis=0) )
        data_reg_list_std.append( np.std(stats['regr_test_r2'][num_frames], axis=0) )
        
        if max_r2_list[0] < np.mean(stats['regr_test_r2'][num_frames], axis=0):
          max_r2_list[0] = np.mean(stats['regr_test_r2'][num_frames], axis=0)
          max_r2_list[1] = np.std(stats['regr_test_r2'][num_frames], axis=0)
          max_r2_list[2] = num_frames
          max_r2_list[3] = str(reg_type)
          
        if max_mse_list[0] < np.mean(stats['regr_test_mse'][num_frames], axis=0):
          max_mse_list[0] = np.mean(stats['regr_test_mse'][num_frames], axis=0)
          max_mse_list[1] = np.std(stats['regr_test_mse'][num_frames], axis=0)
          max_mse_list[2] = num_frames
          max_mse_list[3] = str(reg_type)
      
      if reg_type == 'LR':
        min_r2_list = [np.mean(stats['regr_test_r2'][0], axis=0), np.std(stats['regr_test_r2'][0], axis=0), str(reg_type)]
        min_mse_list = [np.mean(stats['regr_test_mse'][0], axis=0), np.std(stats['regr_test_mse'][0], axis=0), str(reg_type)]
            
    for cl_type in self.cl_list:
    
      read_path = read_path1 + cl_type + '_CL_stats.p'
      stats = pickle.load( open( read_path, 'rb' ) )
      
      for num_frames in range(self.num_prev_frames+1):
        data_cl_list_mean.append( np.mean(stats['penalized_test_auroc'][num_frames], axis=0) )
        data_cl_list_std.append( np.std(stats['penalized_test_auroc'][num_frames], axis=0) )
        
        if max_auc_list[0] < np.mean(stats['penalized_test_auroc'][num_frames], axis=0):
          max_auc_list[0] = np.mean(stats['penalized_test_auroc'][num_frames], axis=0)
          max_auc_list[1] = np.std(stats['penalized_test_auroc'][num_frames], axis=0)
          max_auc_list[2] = num_frames
          max_auc_list[3] = str(cl_type) + '_CL'
        if max_acc_list[0] < np.mean(stats['penalized_test_acc'][num_frames], axis=0):
          max_acc_list[0] = np.mean(stats['penalized_test_acc'][num_frames], axis=0)
          max_acc_list[1] = np.std(stats['penalized_test_acc'][num_frames], axis=0)
          max_acc_list[2] = num_frames
          max_acc_list[3] = str(cl_type) + '_CL'
          
      if cl_type == 'LR_L1':
        min_auc_list = [np.mean(stats['penalized_test_auroc'][0], axis=0), np.std(stats['penalized_test_auroc'][0], axis=0), str(cl_type)+'_CL']
        min_acc_list = [np.mean(stats['penalized_test_acc'][0], axis=0), np.std(stats['penalized_test_acc'][0], axis=0), str(cl_type)+'_CL']
    
    write_min_max_file(max_r2_list, max_mse_list, max_auc_list, max_acc_list, min_r2_list, min_mse_list, min_auc_list, min_acc_list)
          
    plot_r2_auc_timeline_data(self.reg_list, np.asarray(data_reg_list_mean), np.asarray(data_reg_list_std), 'r2')
    plot_r2_auc_timeline_data(self.cl_list, np.asarray(data_cl_list_mean), np.asarray(data_cl_list_std), 'auc')
    
    input_path = CONFIG.IMG_ANALYZE_DIR.split('_nm')[0]
    metrics_compositions = ['_nm0/', '_nm1/', '_nm2/'] 
    if os.path.exists(input_path + metrics_compositions[0]) and os.path.exists(input_path + metrics_compositions[1]) and os.path.exists(input_path + metrics_compositions[2]):
      print('plot metrics in comparison')
      data_reg_list_mean = np.zeros((len(self.reg_list), self.num_prev_frames+1, len(metrics_compositions)))
      data_reg_list_std = np.zeros((len(self.reg_list), self.num_prev_frames+1, len(metrics_compositions)))
      data_cl_list_mean = np.zeros((len(self.cl_list), self.num_prev_frames+1, len(metrics_compositions)))
      data_cl_list_std = np.zeros((len(self.cl_list), self.num_prev_frames+1, len(metrics_compositions)))
      # entropy, score, metaseg, time-dynamic metaseg, our - R2/Auroc - standard deviation
      baseline_our = np.zeros((5,2,2))
      
      for i in range(len(metrics_compositions)):
        read_path1 = CONFIG.ANALYZE_DIR.split('_nm')[0] + metrics_compositions[i] + 'stats/'

        for reg_type, j in zip(self.reg_list, range(len(self.reg_list))):
      
          read_path = read_path1 + reg_type + '_stats.p'
          stats = pickle.load( open( read_path, 'rb' ) )
          for num_frames in range(self.num_prev_frames+1):
            data_reg_list_mean[j,num_frames,i] = np.mean(stats['regr_test_r2'][num_frames], axis=0)
            data_reg_list_std[j,num_frames,i] = np.std(stats['regr_test_r2'][num_frames], axis=0)
          
          if i == 2 and reg_type == 'GB':
            baseline_our[0,0,0] = np.mean(stats["entropy_test_r2"][0], axis=0)
            baseline_our[0,0,1] = np.std(stats["entropy_test_r2"][0], axis=0)
            baseline_our[1,0,0] = np.mean(stats["score_test_r2"][0], axis=0)
            baseline_our[1,0,1] = np.std(stats["score_test_r2"][0], axis=0)
            baseline_our[4,0,0] = data_reg_list_mean[j,:,i].max()
            baseline_our[4,0,1] = data_reg_list_std[j,np.argmax(data_reg_list_mean[j,:,i]),i]
          if i == 0 and reg_type == 'LR':
            baseline_our[2,0,0] = np.mean(stats["regr_test_r2"][0], axis=0)
            baseline_our[2,0,1] = np.std(stats["regr_test_r2"][0], axis=0)
          if i == 0 and reg_type == 'GB':
            baseline_our[3,0,0] = data_reg_list_mean[j,:,i].max()
            baseline_our[3,0,1] = data_reg_list_std[j,np.argmax(data_reg_list_mean[j,:,i]),i]
            
        for cl_type, j in zip(self.cl_list, range(len(self.cl_list))):
        
          read_path = read_path1 + cl_type + '_CL_stats.p'
          stats = pickle.load( open( read_path, 'rb' ) )
          for num_frames in range(self.num_prev_frames+1):
            data_cl_list_mean[j,num_frames,i] = np.mean(stats['penalized_test_auroc'][num_frames], axis=0)
            data_cl_list_std[j,num_frames,i] = np.std(stats['penalized_test_auroc'][num_frames], axis=0)
            
          if i == 2 and cl_type == 'GB':
            baseline_our[0,1,0] = np.mean(stats["entropy_test_auroc"][0], axis=0)
            baseline_our[0,1,1] = np.std(stats["entropy_test_auroc"][0], axis=0)
            baseline_our[1,1,0] = np.mean(stats["score_test_auroc"][0], axis=0)
            baseline_our[1,1,1] = np.std(stats["score_test_auroc"][0], axis=0)
            baseline_our[4,1,0] = data_cl_list_mean[j,:,i].max()
            baseline_our[4,1,1] = data_cl_list_std[j,np.argmax(data_cl_list_mean[j,:,i]),i]
          if i == 0 and cl_type == 'LR_L1':
            baseline_our[2,1,0] = np.mean(stats["penalized_test_auroc"][0], axis=0)
            baseline_our[2,1,1] = np.std(stats["penalized_test_auroc"][0], axis=0)
          if i== 0 and cl_type == 'GB':
            baseline_our[3,1,0] = data_cl_list_mean[j,:,i].max()
            baseline_our[3,1,1] = data_cl_list_std[j,np.argmax(data_cl_list_mean[j,:,i]),i]
               
      for reg_type, j in zip(self.reg_list, range(len(self.reg_list))):
        plot_r2_auc_metrics(reg_type, data_reg_list_mean[j], data_reg_list_std[j], 'r2')
        
      for cl_type, j in zip(self.cl_list, range(len(self.cl_list))):
        plot_r2_auc_metrics(cl_type, data_cl_list_mean[j], data_cl_list_std[j], 'auc')
            
      baseline_path = CONFIG.HELPER_DIR + 'baseline_os' + str(CONFIG.FLAG_OBJ_SEG) + '.npy'
      np.save(baseline_path, baseline_our)
      other_model = CONFIG.model_names[0] if CONFIG.MODEL_NAME == CONFIG.model_names[1] else CONFIG.model_names[1]
      other_model_path = baseline_path.replace(CONFIG.MODEL_NAME, other_model)
      other_model_path = other_model_path.replace(CONFIG.SCORE_THRESHOLD , CONFIG.MAP_THRESHOLD )
      print(other_model, other_model_path)
      if os.path.isfile( baseline_path ) and os.path.isfile( other_model_path ):
        baseline_our_other_model = np.load(other_model_path)
        plot_baselines_vs_ours(CONFIG.MODEL_NAME, baseline_our, other_model, baseline_our_other_model)
      
      
      
