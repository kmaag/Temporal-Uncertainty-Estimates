#!/usr/bin/env python3
"""
script including
functions that do calculations
"""

import os
import time
import numpy as np
import xgboost as xgb
from sklearn import linear_model
from scipy.stats import linregress
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d
from sklearn.neural_network import MLPClassifier

import keras
from keras import regularizers
from keras.models import Model
from keras.layers import Input, Conv1D, Flatten, Dense

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from global_defs import CONFIG
from in_out      import ground_truth_load, time_series_metrics_load, time_series_instances_load,\
                        score_small_load, get_save_path_time_series_metrics_i, time_series_metrics_dump
from metrics     import shifted_iou, compute_matches_gt_pred
from helper      import time_series_metrics_to_nparray

os.environ['CUDA_VISIBLE_DEVICES'] =  "0"  
if 'NN' in CONFIG.REGRESSION_MODEL:
  tf_config = tf.ConfigProto()
  tf_config.gpu_options.per_process_gpu_memory_fraction = 0.1 
  set_session(tf.Session(config=tf_config))  


def create_bb(mask):
  """
  create bounding boxes
  """
  
  # shape boxes: [num_instances, (y1, x1, y2, x2)], mask: [height, width, num_instances] with 0/1
  boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
  for i in range(mask.shape[-1]):
    m_i = mask[:, :, i]
    horizontal_indicies = np.where(np.any(m_i, axis=0))[0]
    vertical_indicies = np.where(np.any(m_i, axis=1))[0]
    if horizontal_indicies.shape[0]:
      x1, x2 = horizontal_indicies[[0, -1]]
      y1, y2 = vertical_indicies[[0, -1]]
      # x2 and y2 should not be part of the box. Increment by 1.
      x2 += 1
      y2 += 1
    else:
      x1, x2, y1, y2 = 0, 0, 0, 0
    boxes[i] = np.array([y1, x1, y2, x2])
  return boxes.astype(np.int32)


def compute_splitting( runs ):    
  """
  compute train/val/test splitting
  """
  print('compute splitting')
  
  if CONFIG.IMG_TYPE == 'kitti':
    train_val_test = [6,1,2]
  elif CONFIG.IMG_TYPE == 'mot':
    train_val_test = [3,1,0] 

  list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))
  tvs = np.zeros((len(list_videos)), dtype='int16')
  list_tvs = []
  
  if CONFIG.IMG_TYPE == 'kitti':
    list_tvs.append('ssvtttttt')
    list_tvs.append('tttvssttt')
    list_tvs.append('ttttttsvs')

  run = 0
  while True:
    
    np.random.seed( run )
    mask = np.random.rand(len(list_videos))
    sorted_mask = np.argsort(mask)
    
    counter = 0
    for t in range(train_val_test[0]):
      tvs[sorted_mask[counter]] = 0
      counter += 1
    for v in range(train_val_test[1]):
      tvs[sorted_mask[counter]] = 1
      counter += 1
    for s in range(train_val_test[2]):
      tvs[sorted_mask[counter]] = 2
      counter += 1
    
    tmp_name = ''
    for i in range(len(list_videos)):
      if tvs[i] == 0:
        tmp_name += 't'
      elif tvs[i] == 1:
        tmp_name += 'v'
      elif tvs[i] == 2:
        tmp_name += 's'
        
    if tmp_name not in list_tvs:
      list_tvs.append(tmp_name)
      
    if len(list_tvs) == runs:
      break
    
    run += 1

  print('train/val/test splitting', list_tvs)
  np.save(os.path.join(CONFIG.HELPER_DIR, 'tvs_runs' + str(runs)), list_tvs)
  

def compute_time_series_3d_metrics( max_inst ):
  """
  compute additional three-dim. time series metrics 
  """
  print('calculating 3d time series metrics')
  start = time.time()
  
  epsilon = CONFIG.EPS_MATCHING
  num_reg = CONFIG.NUM_REG_MATCHING
  classes = CONFIG.CLASSES
  runs = CONFIG.NUM_RESAMPLING
  num_images = CONFIG.NUM_IMAGES
  
  if not os.path.isfile( CONFIG.HELPER_DIR + 'tvs_runs' + str(runs ) + '.npy' ):
    compute_splitting(runs)
  tvs = np.load(CONFIG.HELPER_DIR + 'tvs_runs' + str(runs ) + '.npy')  
  
  metrics_2d = time_series_metrics_load( 'all', '_2d', epsilon, num_reg )
  Xa_names = sorted([ m for m in metrics_2d if m not in ['class','iou','iou0']]) 
  Xa = time_series_metrics_to_nparray( metrics_2d, Xa_names, normalize=True )
  
  survival_prev_frames = 5
  frames_regression = 5
  
  new_metrics = ['survival', 'ratio', 'deformation', 'diff_mean', 'diff_size']
    
  list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))
    
  matching_inst_gt = np.zeros((len(list_videos), num_images, max_inst), dtype='int16')
  gt_01 = np.zeros((len(list_videos), num_images, 3000), dtype='int16')
  
  # height / width
  size_ratio = np.zeros((len(list_videos), len(classes))) 
  counter_size_ratio = np.zeros((len(list_videos), len(classes))) 
  instances_size_ratio = np.zeros((len(list_videos), num_images, max_inst))
  
  instances_deformation = np.zeros((len(list_videos), num_images, max_inst))
  
  # mean x, mean y, size
  predicted_measures = np.zeros((len(list_videos), num_images, max_inst, 3))
  # distance mean predicted and calculated, difference size
  diff_measures = np.zeros((len(list_videos), num_images, max_inst, 2))
  
  counter = 0
  
  for vid, v in zip(list_videos, range(len(list_videos))): 
    print('video:', vid)
    images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
    for n in range(len(images_all)):
    
      gt = np.array(ground_truth_load(vid, n), dtype='int16')
      instance  = np.array( time_series_instances_load( vid, n, epsilon, num_reg ), dtype='int16')
      score = score_small_load( vid, n )
      
      pred_match, _ = compute_matches_gt_pred(gt, instance, score, 0.5, 1)
      
      instance[instance<0] *= -1
      instance = instance % 10000
      
      ## survival, ratio
      for j in range(instance.shape[0]):
        
        matching_inst_gt[v, n, instance[j].max()-1] = pred_match[j]

        coord_indicies = np.where(instance[j] > 0)
        if (coord_indicies[1].max()-coord_indicies[1].min()) != 0:
          instances_size_ratio[v, n, instance[j].max()-1] = (coord_indicies[0].max()-coord_indicies[0].min()) / (coord_indicies[1].max()-coord_indicies[1].min())
        
      for j in np.unique(gt):
        if j != 0 and j != 10000:
          
          gt_01[v, n, j] = 1
          
          coord_indicies = np.where(gt == j)
          if (gt[gt==j].max() // 1000) == classes[0]:
            size_ratio[v, 0] += (coord_indicies[0].max()-coord_indicies[0].min()) / (coord_indicies[1].max()-coord_indicies[1].min())
            counter_size_ratio[v, 0] += 1
          else:
            size_ratio[v, 1] += (coord_indicies[0].max()-coord_indicies[0].min()) / (coord_indicies[1].max()-coord_indicies[1].min())
            counter_size_ratio[v, 1] += 1   
      
      ## deformation
      if n >= 1:
        for j in range(instance.shape[0]):
          if instance[j].max() in np.unique(instance_t_1):
            
            instance_tmp = instance[j].copy()
            instance_tmp[instance_tmp > 0] = 1
            
            for k in range(instance_t_1.shape[0]):
              if instance_t_1[k].max() == instance[j].max():
                break
            instance_t_1_tmp = instance_t_1[k].copy()
            instance_t_1_tmp[instance_t_1_tmp > 0] = 1
            
            instances_deformation[v, n, instance[j].max()-1] = shifted_iou(instance_tmp, instance_t_1_tmp)
      instance_t_1 = instance.copy()
      
      ## diff_mean, diff_size
      reg_steps = min(frames_regression, n)
      for j in range(instance.shape[0]):
        
        id_instance = instance[j].max()-1
        mean_x_list = []
        mean_y_list = []
        size_list = []
        time_list = []

        for k in range(reg_steps):
          
          if metrics_2d['S'][max_inst*counter+id_instance - (max_inst*(reg_steps-k))] > 0:
            mean_x_list.append(metrics_2d['mean_x'][max_inst*counter+id_instance - (max_inst*(reg_steps-k))])
            mean_y_list.append(metrics_2d['mean_y'][max_inst*counter+id_instance - (max_inst*(reg_steps-k))])
            size_list.append(metrics_2d['S'][max_inst*counter+id_instance - (max_inst*(reg_steps-k))])
            time_list.append(k)
        if len(time_list) == 0:
          predicted_measures[v, n, id_instance, 0] = metrics_2d['mean_x'][max_inst*counter+id_instance]
          predicted_measures[v, n, id_instance, 1] = metrics_2d['mean_y'][max_inst*counter+id_instance]
          predicted_measures[v, n, id_instance, 2] = metrics_2d['S'][max_inst*counter+id_instance]
        elif len(time_list) == 1:
          predicted_measures[v, n, id_instance, 0] = mean_x_list[0]
          predicted_measures[v, n, id_instance, 1] = mean_y_list[0]
          predicted_measures[v, n, id_instance, 2] = size_list[0]   
        else:
          b_x, a_x, _, _, _ = linregress(time_list, mean_x_list)
          b_y, a_y, _, _, _ = linregress(time_list, mean_y_list)
          b_s, a_s, _, _, _ = linregress(time_list, size_list)
          predicted_measures[v, n, id_instance, 0] = a_x + b_x * reg_steps
          predicted_measures[v, n, id_instance, 1] = a_y + b_y * reg_steps
          predicted_measures[v, n, id_instance, 2] = a_s + b_s * reg_steps
        
        diff_measures[v, n, id_instance, 0] = ( (predicted_measures[v, n, id_instance, 0] - metrics_2d['mean_x'][max_inst*counter+id_instance])**2 + (predicted_measures[v, n, id_instance, 1] - metrics_2d['mean_y'][max_inst*counter+id_instance])**2 )**0.5
        diff_measures[v, n, id_instance, 1] = predicted_measures[v, n, id_instance, 2] - metrics_2d['S'][max_inst*counter+id_instance]
      counter += 1
  
  print('preparations done in {}s\r'.format( round(time.time()-start) ) )
  
  for run in range(runs):
    
    if not os.path.isfile( get_save_path_time_series_metrics_i(tvs[run], 0, epsilon, num_reg, 1) ):
      
      print('start run', run)
      start = time.time()
      
      metrics_3d = metrics_2d.copy()
      for m in list(new_metrics):
        metrics_3d[m] = list([])
      
      # create survival model
      Xa_train_surv = np.zeros(( num_images * max_inst, Xa.shape[1] * (survival_prev_frames+1)))
      ya_train_surv = np.zeros(( num_images* max_inst ))
      Xa_train = np.zeros(( num_images * max_inst, Xa.shape[1] * (survival_prev_frames+1)))
      ya_train = np.zeros(( num_images * max_inst ))
      Xa_val = np.zeros(( num_images * max_inst, Xa.shape[1] * (survival_prev_frames+1)))
      ya_val = np.zeros(( num_images * max_inst ))

      counter = 0
      counter_train_surv = 0
      counter_train = 0
      counter_val = 0

      for vid,v in zip(list_videos, range(len(list_videos))):
        images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
        for i in range(len(images_all)):
          
          for j in range(max_inst):
            
            flag_same_gt = 1
              
            if ~(Xa[counter]==0).all():
              
              tmp = np.zeros(( Xa.shape[1] * (survival_prev_frames+1) ))
              for k in range(0,survival_prev_frames+1):
                if counter-(max_inst*k) >= 0:
                  tmp[Xa.shape[1]*k:Xa.shape[1]*(k+1)] = Xa[counter-(max_inst*k)] 
                
                if i >= survival_prev_frames:
                  if matching_inst_gt[v, i, j] == 0 or matching_inst_gt[v, i, j] != matching_inst_gt[v, i-k, j]:
                    flag_same_gt = 0
                else:
                  flag_same_gt = 0 

              if tvs[run][v] == 't' and flag_same_gt == 1:
                Xa_train_surv[counter_train_surv,:] = tmp
                ya_train_surv[counter_train_surv] = np.sum( gt_01[v, (i+1):, matching_inst_gt[v, i, j]] )
                counter_train_surv +=1
              if tvs[run][v] == 't':
                Xa_train[counter_train,:] = tmp
                ya_train[counter_train] = np.sum( gt_01[v, (i+1):, matching_inst_gt[v, i, j]] )
                counter_train +=1
              if tvs[run][v] == 'v' or tvs[run][v] == 's':
                Xa_val[counter_val,:] = tmp
                ya_val[counter_val] = np.sum( gt_01[v, (i+1):, matching_inst_gt[v, i, j]] )
                counter_val +=1
            counter += 1

      # delete rows with zeros
      Xa_train_surv = Xa_train_surv[:counter_train_surv,:] 
      ya_train_surv = ya_train_surv[:counter_train_surv] 
      Xa_train = Xa_train[:counter_train,:] 
      ya_train = ya_train[:counter_train] 
      Xa_val = Xa_val[:counter_val,:] 
      ya_val = ya_val[:counter_val] 
      
      ya_train = np.squeeze(ya_train)
      ya_val = np.squeeze(ya_val)

      print('Shapes train survival: ', np.shape(Xa_train_surv), 'train: ', np.shape(Xa_train), 'val: ', np.shape(Xa_val) )
      y_train_surv_pred, y_train_pred, y_val_pred = survival_fit_and_predict(Xa_train_surv, ya_train_surv, Xa_train, Xa_val)
      
      print('survival model r2 score (train survival ):', r2_score(ya_train_surv,y_train_surv_pred) )
      print('survival model r2 score (train):', r2_score(ya_train,y_train_pred) )
      print('survival r2 score (val):', r2_score(ya_val,y_val_pred) )

      size_ratio_train = np.zeros((len(classes))) 
      counter_size_ratio_train = np.zeros((len(classes))) 
      for v in range(len(list_videos)):
        if tvs[run][v] == 't':
          size_ratio_train += size_ratio[v]
          counter_size_ratio_train += counter_size_ratio[v] 
      size_ratio_train /= counter_size_ratio_train
      print('size ratio', size_ratio_train)
        
      # add new metric survival
      counter = 0
      counter_train = 0
      counter_val = 0
      
      for vid,v in zip(list_videos, range(len(list_videos))):
        images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
        for i in range(len(images_all)):
          
          for j in range(max_inst):
            
            for m in list(new_metrics):
              metrics_3d[m].append( 0 )
              
            if ~(Xa[counter]==0).all():
              
              if metrics_3d['class'][counter] == classes[0]:
                metrics_3d['ratio'][-1] = float(instances_size_ratio[v, i, j] / size_ratio_train[0])
              else:
                metrics_3d['ratio'][-1] = float(instances_size_ratio[v, i, j] / size_ratio_train[1])
              
              if tvs[run][v] == 't':
                metrics_3d['survival'][-1] = y_train_pred[counter_train]
                counter_train +=1
              if tvs[run][v] == 'v' or tvs[run][v] == 's':
                metrics_3d['survival'][-1] = y_val_pred[counter_val]
                counter_val +=1
              
              metrics_3d['deformation'][-1] = instances_deformation[v, i, j]
              metrics_3d['diff_mean'][-1] = diff_measures[v, i, j, 0]
              metrics_3d['diff_size'][-1] = abs(diff_measures[v, i, j, 1])
            counter += 1
            
      print('len', len(metrics_3d['iou']), len(metrics_3d['survival']))
      time_series_metrics_dump( metrics_3d, tvs[run], 0, epsilon, num_reg, 1 ) 
      print('run', run, 'processed in {}s\r'.format( round(time.time()-start) ) )


def comp_iou_mask(masks1, masks2):
  """
  compute IoU
  """
  
  # shape: [height, hidth, num_instances]
  if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
    return np.zeros((masks1.shape[-1], masks2.shape[-1]))

  masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
  masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
  pixel1 = np.sum(masks1, axis=0)
  pixel2 = np.sum(masks2, axis=0)

  intersections = np.dot(masks1.T, masks2)
  union = pixel1[:, None] + pixel2[None, :] - intersections
  overlaps = intersections / union
  return overlaps
  
  
def match_gt_pred(gt_boxes, gt_class_ids, gt_masks, pred_boxes, pred_class_ids, pred_scores, pred_masks, img_num_gt, img_num_pred, iou_threshold=0.5):
  """
  match ground truth instances with predictions
  """
  
  # sort predictions by score from high to low
  indices = np.argsort(pred_scores)[::-1]
  pred_boxes = pred_boxes[indices]
  pred_class_ids = pred_class_ids[indices]
  pred_scores = pred_scores[indices]
  pred_masks = pred_masks[..., indices]
  img_num_pred = img_num_pred[indices]

  # compute IoU overlaps [pred_masks, gt_masks]
  overlaps = comp_iou_mask(pred_masks, gt_masks)

  # each prediction has the index of the matched gt
  pred_match = -1 * np.ones([pred_boxes.shape[0]])
  # each gt has the index of the matched prediction
  gt_match = -1 * np.ones([gt_boxes.shape[0]])
  
  for i in range(len(pred_boxes)):
    sorted_ixs = np.argsort(overlaps[i])[::-1]
    for j in sorted_ixs:
      if gt_match[j] > -1:
        continue
      if overlaps[i, j] < iou_threshold:
        break
      if (pred_class_ids[i] == gt_class_ids[j]) and (img_num_pred[i] == img_num_gt[j]):
        gt_match[j] = i
        pred_match[i] = j
        break

  return gt_match, pred_match, overlaps 
  
  
def compute_ap(gt_boxes, gt_class_ids, gt_masks, pred_boxes, pred_class_ids, pred_scores, pred_masks, img_num_gt, img_num_pred, iou_threshold=0.5):
  """
  compute average precision per class
  """

  gt_match, pred_match, overlaps = match_gt_pred(gt_boxes, gt_class_ids, gt_masks, pred_boxes, pred_class_ids, pred_scores, pred_masks, img_num_gt, img_num_pred, iou_threshold)

  # compute precision and recall at each prediction box step
  precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
  recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)
  # pad with start and end values to simplify the calculation
  precisions = np.concatenate([[0], precisions, [0]])
  recalls = np.concatenate([[0], recalls, [1]])
  # ensure precision values decrease but don't increase
  for i in range(len(precisions) - 2, -1, -1):
    precisions[i] = np.maximum(precisions[i], precisions[i + 1])
  # compute AP over recall range
  indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
  mAP = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
  return mAP, gt_match, pred_match


def comp_mean_average_precision(result_path, y0a_pred_zero_val=[]):
  """
  compute mean average precision 
  """
  
  epsilon = CONFIG.EPS_MATCHING
  num_reg = CONFIG.NUM_REG_MATCHING
  num_prev_frames = CONFIG.NUM_PREV_FRAMES
  score_map_th = float(CONFIG.MAP_THRESHOLD) / 100
  if len(y0a_pred_zero_val) > 0:
    max_inst = int( np.load(CONFIG.HELPER_DIR + "max_inst.npy") )
  else:
    max_inst = 0
  
  mAP = np.zeros((len(CONFIG.CLASSES)))    
  tp_fp_fn = np.zeros((3), dtype='int16')  
  
  with open(result_path, 'wt') as fi:
  
    for cl, j in zip(CONFIG.CLASSES, range(len(CONFIG.CLASSES)) ):
      print('class', cl, file=fi)

      instance_masks_gt = []
      gt_class_ids = []
      instance_masks_pred = []
      pred_class_ids = []
      pred_scores = []
      img_num_gt = []
      img_num_pred = []
      
      list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR )) 
      
      counter_y0a = 0
        
      for vid,v in zip(list_videos, range(len(list_videos))):
        
        print(vid)
        images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
        
        for k in range(len(images_all)):
          
          if k >= num_prev_frames:
            
            gt_image = ground_truth_load(vid, k)
            obj_ids = np.unique(gt_image)
            for i in range(len(obj_ids)):
              if obj_ids[i] != 0 and obj_ids[i] != 10000 and (obj_ids[i] // 1000) == cl:
                m_i = np.zeros(np.shape(gt_image))
                m_i[gt_image==obj_ids[i]] = 1
                instance_masks_gt.append(m_i)
                gt_class_ids.append(obj_ids[i] // 1000)
                img_num_gt.append(k+1000*v)
             
            inst_image = time_series_instances_load(vid, k, epsilon, num_reg)
            inst_image[inst_image<0] *= -1
            scores = score_small_load(vid, k)
            for i in range(inst_image.shape[0]):

              if (len(y0a_pred_zero_val) == 0 and scores[i] >= score_map_th) or (len(y0a_pred_zero_val) > 0 and y0a_pred_zero_val[counter_y0a+(inst_image[i].max()%10000)-1]==0):
              
                if (inst_image[i].max() // 10000) == cl:

                  m_i = np.zeros(( inst_image.shape[1], inst_image.shape[2] ))
                  m_i[inst_image[i,:,:]!=0] = 1
                  instance_masks_pred.append(m_i)
                  pred_class_ids.append(inst_image[i].max() // 10000)   
                  pred_scores.append(scores[i])
                  img_num_pred.append(k+1000*v)
            
            counter_y0a += max_inst
      # Pack instance masks into an array
      if len(instance_masks_gt) > 0:
        # [height, width, num_instances]
        gt_masks = np.stack(instance_masks_gt, axis=2).astype(np.bool)
        gt_class_ids = np.array(gt_class_ids, dtype=np.int32)
        # [num_instances, (y1, x1, y2, x2)]
        gt_boxes = create_bb(gt_masks)
        img_num_gt = np.array(img_num_gt)
      
      if len(instance_masks_pred) > 0:
        pred_masks = np.stack(instance_masks_pred, axis=2).astype(np.bool)
        pred_class_ids = np.array(pred_class_ids, dtype=np.int32)
        pred_scores = np.array(pred_scores)
        pred_boxes = create_bb(pred_masks)
        img_num_pred = np.array(img_num_pred) 

        print('number of gt instances:', gt_class_ids.shape[0], file=fi)
        print('number of predicted instances:', pred_class_ids.shape[0], file=fi)
        
        print('compute mAP')
        mAP[j], gt_match, pred_match = compute_ap(gt_boxes, gt_class_ids, gt_masks, pred_boxes, pred_class_ids, pred_scores, pred_masks, img_num_gt, img_num_pred)
        print('number of matches', np.count_nonzero(gt_match>-1), np.count_nonzero(pred_match>-1), file=fi)
        print('number of non-matches gt and pred', np.count_nonzero(gt_match==-1), np.count_nonzero(pred_match==-1), file=fi)
        print('class', cl, 'mAP', mAP[j], file=fi)
        
        tp_fp_fn[0] += np.count_nonzero(gt_match>-1)
        tp_fp_fn[1] += np.count_nonzero(pred_match==-1)
        tp_fp_fn[2] += np.count_nonzero(gt_match==-1)
    
    print('AP per class:', mAP, 'mAP:', np.sum(mAP)/len(CONFIG.CLASSES), file=fi)
    print('TP', tp_fp_fn[0], file=fi)
    print('FP', tp_fp_fn[1], file=fi)
    print('FN', tp_fp_fn[2], file=fi)
     

def survival_fit_and_predict( X_train, y_train, X_val, X_test ):
  """
  survival analysis
  """
  
  model = xgb.XGBRegressor(objective='survival:cox',
                        booster='gblinear',
                        base_score=40,
                        n_estimators=100)
  model.fit(X_train, y_train)
  y_train_pred = model.predict(X_train)
  y_val_pred = model.predict(X_val, output_margin=True)
  y_test_pred = model.predict(X_test, output_margin=True)
  
  return y_train_pred, y_val_pred, y_test_pred

  
def get_alphas( n_steps, min_pow, max_pow ):
  """
  compute alphas for linear models
  """
  
  m = interp1d([0,n_steps-1],[min_pow,max_pow])
  alphas = [10 ** m(i).item() for i in range(n_steps)]
  
  return alphas


def regression_fit_and_predict( X_train, y_train, X_val, y_val, X_test, num_prev_frames = CONFIG.NUM_PREV_FRAMES ):
  """
  fit regression models
  """
  
  if CONFIG.REGRESSION_MODEL == 'LR':
    
    model = linear_model.LinearRegression()
    model.fit(X_train,y_train)
    
  elif CONFIG.REGRESSION_MODEL == 'LR_L1':
    
    alphas = get_alphas( 50, min_pow = -4.2, max_pow = 0.8 )
    
    min_mse = 1000
    best_model = []
  
    for i in range(len(alphas)):
      model = linear_model.Lasso(alpha=alphas[i], max_iter=1e5, tol=1e-3)
      model.fit(X_train,y_train)
      
      y_val_pred = np.clip( model.predict(X_val), 0, 1 )
      tmp = 1.0/len(y_val) * np.sum( (y_val-y_val_pred)**2 )
      
      if tmp < min_mse:
        min_mse = tmp
        best_model = model
    model = best_model
    
  elif CONFIG.REGRESSION_MODEL == 'LR_L2':
    
    alphas = get_alphas( 50, min_pow = -4.2, max_pow = 0.8 )
    
    min_mse = 1000
    best_model = []
  
    for i in range(len(alphas)):
      model = linear_model.Ridge(alpha=alphas[i], max_iter=1000, tol=1e-3)
      model.fit(X_train,y_train)
      
      y_val_pred = np.clip( model.predict(X_val), 0, 1 )
      tmp = 1.0/len(y_val) * np.sum( (y_val-y_val_pred)**2 )
      
      if tmp < min_mse:
        min_mse = tmp
        best_model = model
    model = best_model
    
  elif CONFIG.REGRESSION_MODEL == 'GB':
    
    if False:
      
      choosen_metrics = 25
      print('number of metrics for gradient boosting:', choosen_metrics)
      coefs_model = np.zeros((X_train.shape[1]))
      
      num_metrics = int(X_train.shape[1]/(num_prev_frames+1))
      coef_metrics = np.zeros((num_metrics))
      
      for k in range(5):
        
        model = xgb.XGBRegressor(max_depth=5, colsample_bytree=0.5, n_estimators=100, reg_alpha=0.4, reg_lambda=0.4)
        model.fit( X_train, y_train )
        
        importance_metrics = abs(np.array(model.feature_importances_))
        for l in range(num_prev_frames+1):
          coef_metrics += importance_metrics[num_metrics*l:num_metrics*(l+1)]
          
      index_coefs = np.argsort(coef_metrics)[0:choosen_metrics]
      
      X_train_new = np.zeros((X_train.shape[0], choosen_metrics*(num_prev_frames+1)))
      X_val_new = np.zeros((X_val.shape[0], choosen_metrics*(num_prev_frames+1)))
      X_test_new = np.zeros((X_test.shape[0], choosen_metrics*(num_prev_frames+1)))
      
      counter = 0
      for k in range(num_metrics):
        if k in index_coefs:
          for l in range(num_prev_frames+1):
            X_train_new[:,counter] = X_train[:,num_metrics*l+k]
            X_val_new[:,counter] = X_val[:,num_metrics*l+k]
            X_test_new[:,counter] = X_test[:,num_metrics*l+k]
            counter += 1
            
      model = xgb.XGBRegressor(max_depth=5, colsample_bytree=0.5, n_estimators=100, reg_alpha=0.4, reg_lambda=0.4)
      model.fit( X_train_new, y_train )
      
      importance_metrics = np.array(model.feature_importances_)
      counter = 0
      for k in range(num_metrics):
        if k in index_coefs:
          for l in range(num_prev_frames+1):
            coefs_model[num_metrics*l+k] = importance_metrics[counter]
            counter += 1
          
      X_train = X_train_new.copy()
      X_val = X_val_new.copy()
      X_test = X_test_new.copy()
      
    else:
      model = xgb.XGBRegressor(max_depth=5, colsample_bytree=0.5, n_estimators=100, reg_alpha=0.4, reg_lambda=0.4)
      model.fit( X_train, y_train )
  
  elif CONFIG.REGRESSION_MODEL == 'NN_L1':
    
    num_metrics = int(X_train.shape[1]/(num_prev_frames+1))
    # (components, num_prev_frames+1, number of metrics)
    X_train = X_train.reshape(X_train.shape[0], num_prev_frames+1, num_metrics )
    X_val = X_val.reshape(X_val.shape[0], num_prev_frames+1, num_metrics )
    X_test = X_test.reshape(X_test.shape[0], num_prev_frames+1, num_metrics )
    
    print('X_train and X_val shape', X_train.shape, X_val.shape)
    
    input_shape  = (X_train.shape[1], X_train.shape[2])
    inp = Input(input_shape)
    weight=1e-4
    dropout=0.25
    
    y = inp
    y = Conv1D(filters=16, kernel_size=(5,), padding='same', strides=1,
              kernel_regularizer=regularizers.l1(weight), activation='relu')(inp)
    y = Flatten()(y)
    y = Dense( 50, kernel_regularizer=regularizers.l1(weight), activation='relu' )(y)
    y = Dense( 1 )(y)
    
    model = Model(inputs=inp,outputs=y)
    model.summary()

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[])
    model.fit(X_train, y_train, epochs=200, validation_data=(X_val,y_val), batch_size=128) 
    
  elif CONFIG.REGRESSION_MODEL == 'NN_L2':
    
    num_metrics = int(X_train.shape[1]/(num_prev_frames+1))
    # (components, num_prev_frames+1, number of metrics)
    X_train = X_train.reshape(X_train.shape[0], num_prev_frames+1, num_metrics )
    X_val = X_val.reshape(X_val.shape[0], num_prev_frames+1, num_metrics )
    X_test = X_test.reshape(X_test.shape[0], num_prev_frames+1, num_metrics )
    
    print('X_train and X_val shape', X_train.shape, X_val.shape)
    
    input_shape  = (X_train.shape[1], X_train.shape[2])
    inp = Input(input_shape)
    wdecay=1e-3
    dropout=0.25
    
    y = inp
    y = Conv1D(filters=16, kernel_size=(5,), padding='same', strides=1,
              kernel_regularizer=regularizers.l2(wdecay), activation='relu')(inp)
    y = Flatten()(y)
    y = Dense( 50, kernel_regularizer=regularizers.l2(wdecay), activation='relu' )(y)
    y = Dense( 1 )(y)
    
    model = Model(inputs=inp,outputs=y)
    model.summary()

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[])
    model.fit(X_train, y_train, epochs=200, validation_data=(X_val,y_val), batch_size=128) 
    print(model.summary())
     
  y_train_pred = np.clip( model.predict(X_train), 0, 1 )
  y_val_pred = np.clip( model.predict(X_val), 0, 1 )
  y_test_R_pred = np.clip( model.predict(X_test), 0, 1 )

  return y_train_pred, y_val_pred, y_test_R_pred, model


def classification_fit_and_predict( X_train, y_train, X_val, y_val, X_test, num_prev_frames = CONFIG.NUM_PREV_FRAMES ):
  """
  fit classification models
  """
  
  coefs_model = np.zeros((X_train.shape[1]))
  
  if CONFIG.CLASSIFICATION_MODEL == 'LR_L1':
    
    alphas = get_alphas( 50, min_pow = -4.2, max_pow = 0.8 )
    max_acc = -1
    best_model = []
    
    for i in range(len(alphas)):
      model = linear_model.LogisticRegression(C=alphas[i], penalty='l1', solver='saga', max_iter=1000, tol=1e-3 )
      model.fit( X_train, y_train )
      
      y_val_pred = model.predict_proba(X_val)
      tmp = np.mean( np.argmax(y_val_pred,axis=-1)==y_val )
            
      # take the result of the best alpha
      if tmp > max_acc:
        max_acc = tmp
        best_model = model
    model = best_model
    
    coefs_model = model.coef_
    
  elif CONFIG.CLASSIFICATION_MODEL == 'GB':
    
    if False:

      choosen_metrics = 25
      
      num_metrics = int(X_train.shape[1]/(num_prev_frames+1))
      coef_metrics = np.zeros((num_metrics))
      
      for k in range(5):
        
        model = xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, subsample=0.5, reg_alpha=0.5, reg_lambda=0.5) 
        model.fit( X_train, y_train )
        
        importance_metrics = abs(np.array(model.feature_importances_))
        
        for l in range(num_prev_frames+1):
          coef_metrics += importance_metrics[num_metrics*l:num_metrics*(l+1)]
          
      index_coefs = np.argsort(coef_metrics)[0:choosen_metrics]
      
      X_train_new = np.zeros((X_train.shape[0], choosen_metrics*(num_prev_frames+1)))
      X_val_new = np.zeros((X_val.shape[0], choosen_metrics*(num_prev_frames+1)))
      X_test_new = np.zeros((X_test.shape[0], choosen_metrics*(num_prev_frames+1)))
      
      counter = 0
      for k in range(num_metrics):
        if k in index_coefs:
          
          for l in range(num_prev_frames+1):
            
            X_train_new[:,counter] = X_train[:,num_metrics*l+k]
            X_val_new[:,counter] = X_val[:,num_metrics*l+k]
            X_test_new[:,counter] = X_test[:,num_metrics*l+k]
            counter += 1
            
      model = xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, subsample=0.5, reg_alpha=0.5, reg_lambda=0.5) 
      model.fit( X_train_new, y_train )
      
      importance_metrics = np.array(model.feature_importances_)
      counter = 0
      for k in range(num_metrics):
        if k in index_coefs:
          
          for l in range(num_prev_frames+1):
            
            coefs_model[num_metrics*l+k] = importance_metrics[counter]
            counter += 1
          
      X_train = X_train_new.copy()
      X_val = X_val_new.copy()
      X_test = X_test_new.copy()
      
    else:
      model = xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, subsample=0.5, reg_alpha=0.5, reg_lambda=0.5) 
      model.fit( X_train, y_train )
      
      coefs_model = np.array(model.feature_importances_)
    
  elif CONFIG.CLASSIFICATION_MODEL == 'NN_L2':
    model = MLPClassifier(hidden_layer_sizes=(50,), activation='logistic', solver='adam', alpha=0.4, batch_size=128, max_iter=200)
    model.fit( X_train, y_train )
    
  y_train_pred = model.predict_proba(X_train)
  y_val_pred = model.predict_proba(X_val)
  y_test_R_pred = model.predict_proba(X_test)
  
  return y_train_pred, y_val_pred, y_test_R_pred, coefs_model







