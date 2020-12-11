#!/usr/bin/env python3
"""
script including
functions for easy usage in main scripts
"""

import os
import subprocess
import numpy as np

from global_defs import CONFIG
from in_out      import time_series_metrics_load
import labels as labels


def name_to_latex( name ):
  """
  metric names in latex
  """
  
  for i in range(100):
    if name == "cprob"+str(i):
      return "$C_{"+str(i)+"}$"

  mapping = {'E': '$\\bar E$',
             'E_bd': '${\\bar E}_{bd}$',
             'E_in': '${\\bar E}_{in}$',
             'E_rel_in': '$\\tilde{\\bar E}_{in}$',
             'E_rel': '$\\tilde{\\bar E}$',
             'M': '$\\bar M$',
             'M_bd': '${\\bar M}_{bd}$',
             'M_in': '${\\bar M}_{in}$',
             'M_rel_in': '$\\tilde{\\bar M}_{in}$',
             'M_rel': '$\\tilde{\\bar M}$',
             'S': '$S$',
             'S_bd': '${S}_{bd}$',
             'S_in': '${S}_{in}$',
             'S_rel_in': '$\\tilde{S}_{in}$',
             'S_rel': '$\\tilde{S}$',
             'V': '$\\bar V$',
             'V_bd': '${\\bar V}_{bd}$',
             'V_in': '${\\bar V}_{in}$',
             'V_rel_in': '$\\tilde{\\bar V}_{in}$',
             'V_rel': '$\\tilde{\\bar V}$',
             'mean_x' : '${\\bar k}_{v}$',
             'mean_y' : '${\\bar k}_{h}$', 
             'C_p' : '${C}_{p}$',
             'iou' : '$IoU$',
             'score' : '$s$',
             'survival' : '$v$',
             'ratio' : '$r$', 
             'deformation' : '$f$',
             'diff_mean' : '$d_{c}$',
             'diff_size' : '$d_{s}$'}      
  if str(name) in mapping:
    return mapping[str(name)]
  else:
    return str(name) 
  
  
def name_to_latex_scatter_plot( name ):
  """
  metric names in latex for scatter plots
  """
  
  for i in range(100):
    if name == "cprob"+str(i):
      return "$C_{"+str(i)+"}$"

  mapping = {'E': '$\\bar E$',
             'E_bd': '${\\bar E}_{bd}$',
             'E_in': '${\\bar E}_{in}$',
             'E_rel_in': '$\\tilde{\\bar E}_{in}/\\tilde{\\bar E}_{in,max}$',
             'E_rel': '$\\tilde{\\bar E}/\\tilde{\\bar E}_{max}$',
             'M': '$\\bar M$',
             'M_bd': '${\\bar M}_{bd}$',
             'M_in': '${\\bar M}_{in}$',
             'M_rel_in': '$\\tilde{\\bar M}_{in}/\\tilde{\\bar M}_{in,max}$',
             'M_rel': '$\\tilde{\\bar M}/\\tilde{\\bar M}_{max}$',
             'V': '$\\bar V$',
             'V_bd': '${\\bar V}_{bd}$',
             'V_in': '${\\bar V}_{in}$',
             'V_rel_in': '$\\tilde{\\bar V}_{in}/\\tilde{\\bar V}_{in,max}$',
             'V_rel': '$\\tilde{\\bar V}/\\tilde{\\bar V}_{max}$',
             'S': '$S/S_{max}$',
             'S_bd': '$S_{bd}/S_{bd,max}$',
             'S_in': '$S_{in}/S_{in,max}$',
             'S_rel_in': '$\\tilde{S}_{in}/\\tilde{S}_{in,max}$',
             'S_rel': '$\\tilde{S}/\\tilde{S}_{max}$',
             'mean_x' : '${\\bar k}_{v}$',
             'mean_y' : '${\\bar k}_{h}$', 
             'C_p' : '${C}_{p}$',
             'iou' : '$IoU$',
             'score' : '$s$',
             'survival' : '$v$',
             'ratio' : '$r$', 
             'deformation' : '$f$',
             'diff_mean' : '$d_{c}$',
             'diff_size' : '$d_{s}$'}        
  if str(name) in mapping:
    return mapping[str(name)]
  else:
    return str(name) 


def instance_search( comp_class_string ):
  """
  search instance per video with largest lifetime
  """
    
  if os.path.isfile( CONFIG.HELPER_DIR + "list_max_id_inst_" + comp_class_string + ".npy" ):
    
    max_id_comp_list = np.load(CONFIG.HELPER_DIR + "list_max_id_inst_" + comp_class_string + ".npy")
    
  else:
    
    epsilon    = CONFIG.EPS_MATCHING
    num_reg    = CONFIG.NUM_REG_MATCHING
    
    named2label = { label.name : label for label in reversed(labels.kitti_labels) }
    comp_class = named2label[ comp_class_string ].trainId
    
    max_inst = int( np.load(CONFIG.HELPER_DIR + "max_inst.npy") )
    print('maximal number of instances:', max_inst)

    # list of the instance id with the biggest instance of class comp_class
    # value -1, if there is no instance in the image sequence
    max_id_comp_list = []
  
    # take from each sequence the instance that appears most frequently
    list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))
    
    for vid in list_videos: 
      print("video", vid)
    
      instances_i = np.zeros((max_inst+1))
      
      images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + "/" ))
      
      for img in range(len(images_all)):
        
        time_series_metrics = time_series_metrics_load( vid, img, epsilon, num_reg )
      
        for i in range(0,max_inst):
        
          if (time_series_metrics["S"][i] > 0) and (time_series_metrics["class"][i] == comp_class):
            
            instances_i[i+1] +=1
      
      if instances_i[int(np.argmax(instances_i))] > 0:
        max_id_comp_list.append( int(np.argmax(instances_i)) )
      else:
        max_id_comp_list.append(-1)
        
    np.save(os.path.join(CONFIG.HELPER_DIR, "list_max_id_inst_" + comp_class_string), max_id_comp_list)
    
  return max_id_comp_list
  

def time_series_metrics_to_nparray( metrics, names, normalize=False, all_metrics=[] ):
  """
  metrics to np array
  """
  
  I = range(len(metrics['S']))
  M_with_zeros = np.zeros((len(I), len(names)))
  I = np.asarray(metrics['S']) > 0
  
  M = np.asarray( [ np.asarray(metrics[ m ])[I] for m in names ] )
  MM = []
  if all_metrics == []:
    MM = M.copy()
  else:
    MM = np.asarray( [ np.asarray(all_metrics[ m ])[I] for m in names ] )
  
  # normalize: E = 0 and sigma = 1
  if normalize == True:
    for i in range(M.shape[0]):
      if names[i] != "class":
        M[i] = ( np.asarray(M[i]) - np.mean(MM[i], axis=-1 ) ) / ( np.std(MM[i], axis=-1 ) + 1e-10 )
  M = np.squeeze(M.T)
  
  counter = 0
  for i in range(M_with_zeros.shape[0]):
    if I[i] == True and M_with_zeros.shape[1]>1:
      M_with_zeros[i,:] = M[counter,:]
      counter += 1
    if I[i] == True and M_with_zeros.shape[1]==1:
      M_with_zeros[i] = M[counter]
      counter += 1
  
  return M_with_zeros


def time_series_metrics_to_dataset( metrics, nclasses, run, all_metrics=[] ):
  """
  normalized and 0s stay in (no of instances * no of images, no of metrics)
  """  
  
  epsilon = CONFIG.EPS_MATCHING
  num_reg = CONFIG.NUM_REG_MATCHING
  
  class_names = []
  class_names = [ "cprob"+str(i) for i in range(nclasses) if "cprob"+str(i) in metrics ]
    
  if CONFIG.FLAG_NEW_METRICS == 0:
    X_names = sorted([ m for m in metrics if m not in ["class","iou","iou0","score","survival","ratio","deformation","diff_mean","diff_size"] and "cprob" not in m  ]) 
  elif CONFIG.FLAG_NEW_METRICS == 1:
    X_names = sorted([ m for m in metrics if m not in ["class","iou","iou0","survival","deformation","diff_mean","diff_size"] and "cprob" not in m  ]) 
  elif CONFIG.FLAG_NEW_METRICS == 2:
    X_names = sorted([ m for m in metrics if m not in ["class","iou","iou0"] and "cprob" not in m ]) 
    
  elif CONFIG.FLAG_NEW_METRICS == 3:
    X_names = sorted([ m for m in metrics if m not in ["class","iou","iou0","score"] and "cprob" not in m ]) 
     
  print("create time series metrics (to dataset), score threshold:", CONFIG.SCORE_THRESHOLD, "metrics:", CONFIG.FLAG_NEW_METRICS)
  Xa      = time_series_metrics_to_nparray( metrics, X_names, normalize=True, all_metrics=all_metrics )
  classes = time_series_metrics_to_nparray( metrics, class_names, normalize=True, all_metrics=all_metrics )
  ya      = time_series_metrics_to_nparray( metrics, ["iou" ]   , normalize=False )
  y0a     = time_series_metrics_to_nparray( metrics, ["iou0"]   , normalize=False )

  return Xa, classes, ya, y0a, X_names, class_names  


def split_tvs_and_concatenate( Xa, ya, y0a, train_val_test_string, run=0 ):
  """
  0s will be sorted out, the metrics of the previous frames (NUM_PREV_FRAMES) will be included 
  """ 
  
  np.random.seed( run )
  num_images = CONFIG.NUM_IMAGES
  num_prev_frames = CONFIG.NUM_PREV_FRAMES
  max_inst = int( np.load(CONFIG.HELPER_DIR + "max_inst.npy") )
  
  print("Concatenate timeseries dataset and create train/val/test splitting")
    
  list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))
  
  ya = np.squeeze(ya)
  y0a = np.squeeze(y0a)
  
  Xa_train = np.zeros(( (num_images-(len(list_videos)*num_prev_frames)) * max_inst, Xa.shape[1] * (num_prev_frames+1)))
  ya_train = np.zeros(( (num_images-(len(list_videos)*num_prev_frames)) * max_inst ))
  y0a_train = np.zeros(( (num_images-(len(list_videos)*num_prev_frames)) * max_inst ))
  
  Xa_val = np.zeros(( (num_images-(len(list_videos)*num_prev_frames)) * max_inst, Xa.shape[1] * (num_prev_frames+1)))
  ya_val = np.zeros(( (num_images-(len(list_videos)*num_prev_frames)) * max_inst ))
  y0a_val = np.zeros(( (num_images-(len(list_videos)*num_prev_frames)) * max_inst ))
  
  Xa_test = np.zeros(( (num_images-(len(list_videos)*num_prev_frames)) * max_inst, Xa.shape[1] * (num_prev_frames+1)))
  ya_test = np.zeros(( (num_images-(len(list_videos)*num_prev_frames)) * max_inst ))
  y0a_test = np.zeros(( (num_images-(len(list_videos)*num_prev_frames)) * max_inst ))
  
  counter = 0
  counter_train = 0
  counter_val = 0
  counter_test = 0
  
  for vid,v in zip(list_videos, range(len(list_videos))):
    images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + "/" ))
    
    if CONFIG.IMG_TYPE == 'mot' and train_val_test_string[v] == 'v':
      split_point = int( len(images_all)/3 - num_prev_frames/2 )
      
    for i in range(len(images_all)):
      
      if i >= num_prev_frames:
          
        tmp = np.zeros(( max_inst, Xa.shape[1] * (num_prev_frames+1) ))
        for j in range(0,num_prev_frames+1):
          tmp[:,Xa.shape[1]*j:Xa.shape[1]*(j+1)] = Xa[max_inst*(counter-j):max_inst*(counter-j+1)] 
        
        if train_val_test_string[v] == 't':
        
          Xa_train[max_inst*counter_train:max_inst*(counter_train+1),:] = tmp
          ya_train[max_inst*counter_train:max_inst*(counter_train+1)] = ya[max_inst*counter:max_inst*(counter+1)]
          y0a_train[max_inst*counter_train:max_inst*(counter_train+1)] = y0a[max_inst*counter:max_inst*(counter+1)]
          counter_train +=1
        
        elif CONFIG.IMG_TYPE == 'kitti':
          
          if train_val_test_string[v] == 'v':
            
            Xa_val[max_inst*counter_val:max_inst*(counter_val+1),:] = tmp
            ya_val[max_inst*counter_val:max_inst*(counter_val+1)] = ya[max_inst*counter:max_inst*(counter+1)]
            y0a_val[max_inst*counter_val:max_inst*(counter_val+1)] = y0a[max_inst*counter:max_inst*(counter+1)]
            counter_val +=1
            
          elif train_val_test_string[v] == 's':
            
            Xa_test[max_inst*counter_test:max_inst*(counter_test+1),:] = tmp
            ya_test[max_inst*counter_test:max_inst*(counter_test+1)] = ya[max_inst*counter:max_inst*(counter+1)]
            y0a_test[max_inst*counter_test:max_inst*(counter_test+1)] = y0a[max_inst*counter:max_inst*(counter+1)]
            counter_test +=1
        
        elif CONFIG.IMG_TYPE == 'mot':
          
          if train_val_test_string[v] == 'v':
            
            if i <= split_point:
            
              Xa_val[max_inst*counter_val:max_inst*(counter_val+1),:] = tmp
              ya_val[max_inst*counter_val:max_inst*(counter_val+1)] = ya[max_inst*counter:max_inst*(counter+1)]
              y0a_val[max_inst*counter_val:max_inst*(counter_val+1)] = y0a[max_inst*counter:max_inst*(counter+1)]
              counter_val +=1
              
            elif i > split_point + num_prev_frames:
              
              Xa_test[max_inst*counter_test:max_inst*(counter_test+1),:] = tmp
              ya_test[max_inst*counter_test:max_inst*(counter_test+1)] = ya[max_inst*counter:max_inst*(counter+1)]
              y0a_test[max_inst*counter_test:max_inst*(counter_test+1)] = y0a[max_inst*counter:max_inst*(counter+1)]
              counter_test +=1     
      
      counter += 1
    
  # delete rows with only zeros in frame t
  not_del_rows_train = ~(Xa_train[:,0:Xa.shape[1]]==0).all(axis=1)
  Xa_train = Xa_train[not_del_rows_train]
  ya_train = ya_train[not_del_rows_train]
  y0a_train = y0a_train[not_del_rows_train]  
  
  # upsampling
  FAC_UPSAMPLING = 1
  
  if FAC_UPSAMPLING > 0:
    
    if not os.path.isfile( CONFIG.ANALYZE_DIR + "Xa_A_run" + str(run) + ".npy" ):
      
      print("create augmented data")
      
      np.save(CONFIG.ANALYZE_DIR+"Xa_train_run"+str(run)+".npy", Xa_train)
      np.save(CONFIG.ANALYZE_DIR+"ya_train_run"+str(run)+".npy", ya_train)
      
      subprocess.check_call(['Rscript', 'upsampling_smote.R',str(run), CONFIG.ANALYZE_DIR], shell=False)
    
    print("load augmented data")
    
    Xa_A = np.load(CONFIG.ANALYZE_DIR + "Xa_A_run" + str(run) + ".npy")
    ya_A = np.load(CONFIG.ANALYZE_DIR + "ya_A_run" + str(run) + ".npy")
    
    y0a_A = np.zeros(( len(ya_A) ))
    y0a_A[ya_A<0.5] = 1
    
    augmented_mask = np.random.rand(len(ya_A)) < float(len(ya_train)) / float(len(ya_A)) * FAC_UPSAMPLING
    
    Xa_train = np.concatenate( (Xa_train, Xa_A[augmented_mask]), axis = 0)
    ya_train = np.concatenate( (ya_train, ya_A[augmented_mask]), axis = 0)
    y0a_train = np.concatenate( (y0a_train, y0a_A[augmented_mask]), axis = 0)
  
  not_del_rows_val = ~(Xa_val[:,0:Xa.shape[1]]==0).all(axis=1)
  Xa_val = Xa_val[not_del_rows_val]
  ya_val = ya_val[not_del_rows_val]
  y0a_val = y0a_val[not_del_rows_val]  
  
  not_del_rows_test = ~(Xa_test[:,0:Xa.shape[1]]==0).all(axis=1)
  Xa_test = Xa_test[not_del_rows_test]
  ya_test = ya_test[not_del_rows_test]
  y0a_test = y0a_test[not_del_rows_test]  
  
  ya_train = np.squeeze(ya_train)
  y0a_train =np.squeeze(y0a_train)
  ya_val = np.squeeze(ya_val)
  y0a_val =np.squeeze(y0a_val)
  ya_test = np.squeeze(ya_test)
  y0a_test =np.squeeze(y0a_test)
  
  return Xa_train, Xa_val, Xa_test, ya_train, ya_val, ya_test, y0a_train, y0a_val, y0a_test

    
def concatenate_val_for_visualization( Xa, ya ):
  """
  concatenate validation set for visualization
  """ 
  
  num_imgs = CONFIG.NUM_IMAGES
  num_prev_frames = CONFIG.NUM_PREV_FRAMES
  max_inst = int( np.load(CONFIG.HELPER_DIR + "max_inst.npy") )
  
  ya = np.squeeze(ya)
    
  list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))
  
  #validation data 
  #prediction with components in num_prev_frames previous frames
  Xa_zero_val = np.zeros(( (num_imgs-(len(list_videos)*num_prev_frames)) * max_inst, Xa.shape[1] * (num_prev_frames+1)))
  ya_zero_val = np.zeros(( (num_imgs-(len(list_videos)*num_prev_frames)) * max_inst ))
  
  plot_image_list = []
  counter = 0
  counter_new = 0
  for vid in list_videos:
    images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + "/" ))
    for i in range(len(images_all)):
      
      if i >= num_prev_frames:
        
        plot_image_list.append( ( vid, i, counter_new ) )
          
        tmp = np.zeros(( max_inst, Xa.shape[1] * (num_prev_frames+1) ))
        for j in range(0,num_prev_frames+1):
          tmp[:,Xa.shape[1]*j:Xa.shape[1]*(j+1)] = Xa[max_inst*(counter-j):max_inst*(counter-j+1)] 
            
        Xa_zero_val[max_inst*counter_new:max_inst*(counter_new+1),:] = tmp
        ya_zero_val[max_inst*counter_new:max_inst*(counter_new+1)] = ya[max_inst*counter:max_inst*(counter+1)]
        counter_new +=1
      
      counter += 1
  
  # delete rows with only zeros in frame t
  not_del_rows_val = ~(Xa_zero_val[:,0:Xa.shape[1]]==0).all(axis=1)
  Xa_val = Xa_zero_val[not_del_rows_val]
  ya_val = ya_zero_val[not_del_rows_val]
  
  ya_val = np.squeeze(ya_val)
  ya_zero_val = np.squeeze(ya_zero_val)
    
  return Xa_val, ya_val, ya_zero_val, not_del_rows_val, plot_image_list   





