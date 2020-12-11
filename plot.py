#!/usr/bin/env python3
"""
script including
functions for visualizations
"""

import os
import sys
import time
import random
import colorsys
import numpy as np
import pandas as pd
import matplotlib as mpl 
import plotly.graph_objects as go
import matplotlib.colors as colors
from PIL import Image, ImageDraw
from scipy.stats import pearsonr
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('font', size=10, family='serif')
plt.rc("text", usetex=True)
from matplotlib.ticker import StrMethodFormatter

from global_defs import CONFIG
from in_out      import time_series_instances_load, get_save_path_image_i, ground_truth_load, time_series_metrics_load, score_small_load
from helper      import name_to_latex, name_to_latex_scatter_plot
from calculate   import create_bb
import labels as labels

trainId2label = { label.trainId : label for label in reversed(labels.kitti_labels) }
named2label   = { label.name : label for label in reversed(labels.kitti_labels) }


def hex_to_rgb(input1):
  value1 = input1.lstrip('#')
  return tuple(int(value1[i:i+2], 16) for i in (0, 2 ,4))


def colorFader(c1,c2,mix=0): 
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    if mix <= 1: #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
      return mpl.colors.to_hex((1-mix)*c1 + mix*c2)
    else:        # the other way round
      mix = mix - 1
      return mpl.colors.to_hex((1-mix)*c2 + mix*c1)


def plot_matching( vid, n, colors_list ):
  
  t = time.time()
  
  epsilon = CONFIG.EPS_MATCHING
  num_reg = CONFIG.NUM_REG_MATCHING
  
  input_image = np.asarray( Image.open(get_save_path_image_i(vid, n)) )
  gt_image = ground_truth_load(vid, n)
  inst_image  = time_series_instances_load( vid, n, epsilon, num_reg )

  instance_masks = []
  gt_class_ids = []
  obj_ids = np.unique(gt_image)
  for i in range(len(obj_ids)):
    if obj_ids[i] != 0 and obj_ids[i] != 10000:
        m_i = np.zeros(np.shape(gt_image))
        m_i[gt_image==obj_ids[i]] = 1
        instance_masks.append(m_i)
        gt_class_ids.append(obj_ids[i] // 1000)
  # Pack instance masks into an array
  if len(instance_masks) > 0:
    mask_tmp = np.stack(instance_masks, axis=2).astype(np.bool)
    # [num_instances, (y1, x1, y2, x2)]
    gt_bbox = create_bb(mask_tmp)
  
  instance_masks = []
  seg_class_ids = []
  for i in range(inst_image.shape[0]):
    m_i = np.zeros(( inst_image.shape[1], inst_image.shape[2] ))
    m_i[inst_image[i,:,:]!=0] = 1
    instance_masks.append(m_i)
    seg_class_ids.append(inst_image[i].max() // 10000)        
  if len(instance_masks) > 0:
    mask_tmp = np.stack(instance_masks, axis=2).astype(np.bool)
    seg_bbox = create_bb(mask_tmp)
  
  I1 = input_image.copy()
  tmp = np.zeros((3))
  for i in range(input_image.shape[0]):
    for j in range(input_image.shape[1]):
      if gt_image[i,j] > 0 and gt_image[i,j] < 10000:   
        tmp = np.asarray( hex_to_rgb( colors_list[ (gt_image[i,j]-1) % len(colors_list) ] ) )
        I1[i,j,:] = tmp * 0.6 + input_image[i,j,:] * 0.4
      elif gt_image[i,j] == 10000:
        tmp = np.asarray((255,255,255))
        I1[i,j,:] = tmp * 0.6 + input_image[i,j,:] * 0.4
  
  I2 = input_image.copy()
  for i in range(input_image.shape[0]):
    for j in range(input_image.shape[1]):
      tmp = np.zeros((3))
      counter = 0
      for k in range(inst_image.shape[0]):
        if inst_image[k,i,j] > 0: 
          tmp += np.asarray( hex_to_rgb( colors_list[ int(inst_image[k,i,j]-1) % len(colors_list) ] ) )
          counter += 1
        elif inst_image[k,i,j] < 0: 
          counter += 1   
      if counter > 0:
        tmp /= counter
        I2[i,j,:] = tmp * 0.6 + input_image[i,j,:] * 0.4
  
  img = np.concatenate( (I1,I2), axis=0 )
  image = Image.fromarray(img.astype('uint8'), 'RGB')
  
  draw = ImageDraw.Draw(image)
  if len(gt_class_ids) > 0:
    for i in range(gt_bbox.shape[0]):
      color_class = trainId2label[ gt_class_ids[i] ].color
      draw.rectangle([gt_bbox[i,1], gt_bbox[i,0], gt_bbox[i,3], gt_bbox[i,2]], fill=None, outline=color_class)
  
  if len(seg_class_ids) > 0:
    for i in range(seg_bbox.shape[0]):
      color_class = trainId2label[ seg_class_ids[i] ].color
      draw.rectangle([seg_bbox[i,1], seg_bbox[i,0]+input_image.shape[0], seg_bbox[i,3], seg_bbox[i,2]+input_image.shape[0]], fill=None, outline=color_class)
  
  save_path = CONFIG.IMG_TIME_SERIES_DIR + vid + "/"
  
  if not os.path.exists( save_path ):
    os.makedirs( save_path )
  
  image.save(save_path + "img_ts_seg" + str(n).zfill(6) + "_eps" + str(epsilon) + "_num_reg" + str(num_reg)  + ".png")
  
  print("plot image", n, ": time needed ", time.time()-t)  

  
def add_scatterplot_vs_iou(ious, sizes, dataset, name, setylim=True):
  
  rho = pearsonr(ious,dataset)
  plt.title(r"$\rho = {:.05f}$".format(rho[0]), fontsize=35)
  plt.scatter( ious, dataset, s = sizes/50, marker='.', c='cornflowerblue', alpha=0.1 ) 
  plt.xlabel('$\mathit{IoU}$', fontsize=35, labelpad=-15)
  plt.ylabel(name, fontsize=35, labelpad=-45)
  plt.xticks((0,1),fontsize=35)
  plt.yticks((np.amin(dataset),np.amax(dataset)),fontsize=35)
  plt.subplots_adjust(left=0.15)
  

def plot_scatter_metric_iou(list_metrics):
  
  print('plot scatter metric vs iou')

  if len(list_metrics) == 4:
    num_x = 2
    num_y = 2
  elif len(list_metrics) == 15:
    num_x = 3
    num_y = 5
  elif len(list_metrics) == 28:
    num_x = 7
    num_y = 4
  elif len(list_metrics) == 3:
    num_x = 3
    num_y = 1
  
  size_x = 11.0 
  size_y = 9.0 
  
  if not os.path.exists( CONFIG.IMG_METRICS_DIR + "scatter/" ):
    os.makedirs( CONFIG.IMG_METRICS_DIR + "scatter/" )
  
  epsilon = CONFIG.EPS_MATCHING
  num_reg = CONFIG.NUM_REG_MATCHING
  runs = CONFIG.NUM_RESAMPLING
  
  #metrics = time_series_metrics_load( "all", "_2d", epsilon, num_reg )
  tvs = np.load(CONFIG.HELPER_DIR + 'tvs_runs' + str(runs ) + '.npy')  
  metrics = time_series_metrics_load( tvs[0], 0, epsilon, num_reg, 1 ) 
  df_full = pd.DataFrame( data=metrics ) 
  df_full = df_full.copy().loc[df_full['S'].nonzero()[0]]

  plt.rc('axes', titlesize=30)
  plt.rc('figure', titlesize=25)
  
  fig = plt.figure(frameon=False)
  fig.set_size_inches(size_x*num_x,size_y*num_y)

  result_path = CONFIG.IMG_METRICS_DIR + "scatter/scatter" + str(len(list_metrics)) + ".txt"
  with open(result_path, 'wt') as fi:
    
    for i in range(len(list_metrics)):
      fig.add_subplot(num_y,num_x,i+1)
      if 'S' in list_metrics[i]:
        add_scatterplot_vs_iou(df_full['iou'], 5000, df_full[list_metrics[i]]/df_full[list_metrics[i]].max(), name_to_latex_scatter_plot(list_metrics[i]))
      elif 'rel' in list_metrics[i]:
        add_scatterplot_vs_iou(df_full['iou'], df_full['S'], df_full[list_metrics[i]]/df_full[list_metrics[i]].max(), name_to_latex_scatter_plot(list_metrics[i]))
      else:
        add_scatterplot_vs_iou(df_full['iou'], df_full['S'], df_full[list_metrics[i]], name_to_latex_scatter_plot(list_metrics[i]))
        
      print(list_metrics[i], "{:.5f}".format(pearsonr(df_full['iou'], df_full[list_metrics[i]])[0]), file=fi)

  plt.savefig(CONFIG.IMG_METRICS_DIR + "scatter/scatter" + str(len(list_metrics)) + ".png", bbox_inches='tight')
  plt.close()
  

def visualize_instances( comp, metric ):
  
  # split metric into R, G and B
  R = np.asarray( metric )
  R = 1-0.5*R
  G = np.asarray( metric )
  B = 0.3+0.35*np.asarray( metric )
  
  R = np.concatenate( (R, np.asarray([0,1])) )
  G = np.concatenate( (G, np.asarray([0,1])) )
  B = np.concatenate( (B, np.asarray([0,1])) )
  
  components = np.asarray(comp.copy(), dtype='int16')
  # because of this we concatente [0,1]
  components[components  < 0] = len(R)-1  # boundaries in black -> get 0
  components[components == 0] = len(R)    # unlabeled in white -> get 1
  
  img = np.zeros( components.shape+(3,) )
  
  for x in range(img.shape[0]):
    for y in range(img.shape[1]):
      # components -1 because of indexshifting and takes the RGB values of associated metrics value
      img[x,y,0] = R[components[x,y]-1]
      img[x,y,1] = G[components[x,y]-1]
      img[x,y,2] = B[components[x,y]-1]
  
  img = np.asarray( 255*img ).astype('uint8')
  
  return img
  

def plot_metrics_per_component( save_path, list_img_num, list0, list1, list2, list_iou, metrics_list, vid, counter, max_id ):
  
  t = time.time()
  
  metrics_names = []
  for c in range(len(metrics_list)):
    metrics_names.append( name_to_latex(metrics_list[c]) )
  
  save_path = save_path + "/img" + str(list_img_num[counter]) + ".png"
  
  dpi_val = 200
  epsilon = CONFIG.EPS_MATCHING
  num_reg = CONFIG.NUM_REG_MATCHING
  
  input_image = np.asarray( Image.open(get_save_path_image_i(vid, list_img_num[counter])) )
  
  colors_t = input_image.copy()
  
  instances = time_series_instances_load( vid, list_img_num[counter], epsilon, num_reg )
  instances_id = instances.copy()
  instances_id[instances<0] *= -1
  instances_id = instances_id  % 10000
  instances_id[instances<0] *= -1
  
  instances_2d = np.zeros((instances_id.shape[1], instances_id.shape[2]))

  for m in range(instances_id.shape[0]):
    if instances_id[m,:,:].min() == -max_id:
      instances_2d = instances_id[m]
      iou = np.zeros(( int(-instances_id.min()) ))
      iou[max_id-1] = list_iou[counter]
      img_iou = visualize_instances( instances_2d, iou )
      colors_t[instances_2d==max_id,:] = img_iou[instances_2d==max_id,:]
      colors_t[instances_2d==-max_id,:] = img_iou[instances_2d==-max_id,:]
      break
  
  I2 = colors_t * 0.8 + input_image * 0.2
  
  f, (ax, ax1, ax2) = plt.subplots(3, 1, sharex=True)
  f.set_size_inches(input_image.shape[1]/dpi_val,input_image.shape[0]/dpi_val*2)
  ax.plot(list_img_num, list0, color='deepskyblue')
  ax.plot(list_img_num[counter], list0[counter], 'o', color='deepskyblue')
  ax.set_ylabel(metrics_names[0])
  ax1.plot(list_img_num, list1, color='violet')
  ax1.plot(list_img_num[counter], list1[counter], 'o', color='violet')
  ax1.set_ylabel(metrics_names[1])
  ax2.plot(list_img_num, list2, color='darkturquoise')
  ax2.plot(list_img_num[counter], list2[counter], 'o', color='darkturquoise')
  ax2.set_ylabel(metrics_names[2])
  plt.xlabel('Image')
  
  f.savefig(save_path, dpi=dpi_val)
  plt.close()
  
  plot_image = np.asarray( Image.open(save_path).convert("RGB") )
  
  if input_image.shape[0]%2==1:
    plot_image = plot_image[1:,:]
  
  img = np.concatenate( (plot_image,I2), axis=0 )
  img1 = Image.fromarray(img.astype('uint8'), 'RGB')
  img1.save(save_path)
  
  print("plot image", list_img_num[counter], ": time needed ", time.time()-t)  

  
def plot_metrics_per_class( vid, comp_class_string, single_metric, max_inst ):
  
  t = time.time()
  
  epsilon = CONFIG.EPS_MATCHING
  num_reg = CONFIG.NUM_REG_MATCHING
  
  comp_class = named2label[ comp_class_string ].trainId
  
  images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + "/" ))
  
  # 3: metrics_class[0], S, iou
  metrics_class = np.zeros((len(images_all), max_inst, 3))
  
  for j in range(len(images_all)):
  
    metrics = time_series_metrics_load( vid, j, epsilon, num_reg )
    
    for k in range(max_inst):
      
      if metrics['S'][k] > 0 and metrics['class'][k] == comp_class:
        
        metrics_class[j, k, 0] = metrics[single_metric][k]
        metrics_class[j, k, 1] = metrics['S'][k]
        metrics_class[j, k, 2] = metrics['iou'][k]
   
  fig = plt.figure(1,frameon=False) 
  plt.clf()
  
  for k in range(max_inst):
    
    if np.count_nonzero(metrics_class[:, k, 1]) > 10:
      
      print(k)
      
      color_k = np.sum(metrics_class[:, k, 2]) / np.count_nonzero(metrics_class[:, k, 1])
      R = np.asarray( color_k )
      R = 1-0.5*R
      G = np.asarray( color_k )
      B = 0.3+0.35*np.asarray( color_k )
      
      for inx in range(len(images_all)):
        if metrics_class[inx, k, 1] > 0:
          break
        
      list_metrics = metrics_class[inx:len(images_all), k, 0]
      list_size = metrics_class[inx:len(images_all), k, 1]
      
      # for plotting nan values are obmitted
      list_metrics[list_size==0] = np.nan 
      
      for inx in range(len(list_size)):
        if list_size[len(list_size)-1-inx] > 0:
          break
        
      list_metrics = list_metrics[0:len(list_metrics)-inx]
      
      plt.plot(np.arange(0, len(list_metrics), dtype='int'), list_metrics, color=(R,G,B), marker='', alpha=0.7) 
  
  plt.xlabel('time series')
  plt.ylabel(name_to_latex(single_metric))  
  plt.close()
  
  print("plot video", vid, ": time needed ", time.time()-t)  
  
  return metrics_class


def plot_metrics_per_class_all( metrics_class_all, comp_class_string, single_metric ):
  
  t = time.time()
   
  fig = plt.figure(1,frameon=False) 
  plt.clf()
  
  for k in range(metrics_class_all.shape[1]):
    
    if np.count_nonzero(metrics_class_all[:, k, 1]) > 50:
      
      print(k)
      
      color_k = np.sum(metrics_class_all[:, k, 2]) / np.count_nonzero(metrics_class_all[:, k, 1])
      R = np.asarray( color_k )
      R = 1-0.5*R
      G = np.asarray( color_k )
      B = 0.3+0.35*np.asarray( color_k )
      
      for inx in range(metrics_class_all.shape[0]):
        if metrics_class_all[inx, k, 1] > 0:
          break
        
      list_metrics = metrics_class_all[inx:metrics_class_all.shape[0], k, 0]
      list_size = metrics_class_all[inx:metrics_class_all.shape[0], k, 1]
      
      list_metrics[list_size==0] = np.nan 
      
      for inx in range(len(list_size)):
        if list_size[len(list_size)-1-inx] > 0:
          break
        
      list_metrics = list_metrics[0:len(list_metrics)-inx]
      
      plt.plot(np.arange(0, len(list_metrics), dtype='int'), list_metrics, color=(R,G,B), marker='', alpha=0.4) 
  
  plt.xlabel('time series')
  plt.ylabel(name_to_latex(single_metric))  
  
  fig.savefig(CONFIG.IMG_METRICS_DIR + "per_class/img_class_" + comp_class_string + "_b50" + "_metric" + str(single_metric), dpi=300)
  plt.close()
  
  fig = plt.figure(1,frameon=False) 
  plt.clf()
  
  for k in range(metrics_class_all.shape[1]):
    
    if np.count_nonzero(metrics_class_all[:, k, 1]) <= 50 and np.count_nonzero(metrics_class_all[:, k, 1]) > 10:
      
      print(k)
      
      color_k = np.sum(metrics_class_all[:, k, 2]) / np.count_nonzero(metrics_class_all[:, k, 1])
      R = np.asarray( color_k )
      R = 1-0.5*R
      G = np.asarray( color_k )
      B = 0.3+0.35*np.asarray( color_k )
      
      for inx in range(metrics_class_all.shape[0]):
        if metrics_class_all[inx, k, 1] > 0:
          break
        
      list_metrics = metrics_class_all[inx:metrics_class_all.shape[0], k, 0]
      list_size = metrics_class_all[inx:metrics_class_all.shape[0], k, 1]
      
      list_metrics[list_size==0] = np.nan 
      
      for inx in range(len(list_size)):
        if list_size[len(list_size)-1-inx] > 0:
          break
        
      list_metrics = list_metrics[0:len(list_metrics)-inx]
      
      plt.plot(np.arange(0, len(list_metrics), dtype='int'), list_metrics, color=(R,G,B), marker='', alpha=0.4) 
  
  plt.xlabel('time series')
  plt.ylabel(name_to_latex(single_metric))  
  
  fig.savefig(CONFIG.IMG_METRICS_DIR + "per_class/img_class_" + comp_class_string + "_s50" + "_metric" + str(single_metric), dpi=300)
  plt.close()
  
  fig = plt.figure(1,frameon=False) 
  plt.clf()
  
  for k in range(metrics_class_all.shape[1]):
    
    if np.count_nonzero(metrics_class_all[:, k, 1]) > 0:
      
      print(k)
      
      color_k = np.sum(metrics_class_all[:, k, 2]) / np.count_nonzero(metrics_class_all[:, k, 1])
      R = np.asarray( color_k )
      R = 1-0.5*R
      G = np.asarray( color_k )
      B = 0.3+0.35*np.asarray( color_k )
      
      for inx in range(metrics_class_all.shape[0]):
        if metrics_class_all[inx, k, 1] > 0:
          break
        
      list_metrics = metrics_class_all[inx:metrics_class_all.shape[0], k, 0]
      list_size = metrics_class_all[inx:metrics_class_all.shape[0], k, 1]
      
      list_metrics[list_size==0] = np.nan 
      
      for inx in range(len(list_size)):
        if list_size[len(list_size)-1-inx] > 0:
          break
        
      list_metrics = list_metrics[0:len(list_metrics)-inx]
      
      plt.plot(np.arange(0, len(list_metrics), dtype='int'), list_metrics, color=(R,G,B), marker='', alpha=0.4) 
  
  plt.xlabel('time series')
  plt.ylabel(name_to_latex(single_metric))  
  
  fig.savefig(CONFIG.IMG_METRICS_DIR + "per_class/img_class_" + comp_class_string + "_metric" + str(single_metric), dpi=300)
  plt.close()
  
  print("plot all : time needed ", time.time()-t)  
  
  
def plot_instances_shapes( vid, max_id, comp_class_string ):
  
  t = time.time()

  epsilon = CONFIG.EPS_MATCHING
  num_reg = CONFIG.NUM_REG_MATCHING
  
  # saves min and max values for frames, horizontal and vertical coordinates
  min_max_axis = np.zeros((2,3))
  min_max_axis[0,:] = 13000
  
  images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + "/" ))
  for j in range(len(images_all)):

    time_series_instances = time_series_instances_load( vid, j, epsilon, num_reg )
    instances_id = time_series_instances.copy()
    instances_id[time_series_instances<0] *= -1
    instances_id = instances_id  % 10000
    instances_id[time_series_instances<0] *= -1
    
    index = -1
    for m in range(instances_id.shape[0]):
      if instances_id[m,:,:].min() == -max_id:
        index = m
        break
    if index > -1:

      x_y_indices = np.asarray( np.where( (instances_id[index] == -max_id) ) )   
      # change vertical axis because of image to numpy
      x_y_indices[0,:] = instances_id[index].shape[0] - x_y_indices[0,:]
      
      min_max_axis[0,0] = min(min_max_axis[0,0], j)
      min_max_axis[1,0] = max(min_max_axis[1,0], j)
      
      min_max_axis[0,1] = min(min_max_axis[0,1], x_y_indices[1,:].min())
      min_max_axis[1,1] = max(min_max_axis[1,1], x_y_indices[1,:].max())
      
      min_max_axis[0,2] = min(min_max_axis[0,2], x_y_indices[0,:].min())
      min_max_axis[1,2] = max(min_max_axis[1,2], x_y_indices[0,:].max())
      
  print(min_max_axis)
  
  save_path  = CONFIG.IMG_METRICS_DIR + 'shapes/' + vid + '_class_' + comp_class_string + "_start" + str(int(min_max_axis[0,0])) + "_end" + str(int(min_max_axis[1,0])) + "/"
  if not os.path.exists( save_path ):
    os.makedirs( save_path )
  
  for j in range(int(min_max_axis[0,0]), int(min_max_axis[1,0]+1)):
    plot_instances_shapes_i(vid, j, min_max_axis, max_id, save_path)
  
  print("plot video", vid, ": time needed ", time.time()-t)  
  
  
def plot_instances_shapes_i( vid, img_num, min_max_axis, max_id, save_path ):
  
  print("image:", img_num)
  
  save_path = save_path + "img_shape_" + str(img_num) + ".png"
  
  epsilon = CONFIG.EPS_MATCHING
  num_reg = CONFIG.NUM_REG_MATCHING
  
  dpi_val = 200
  c1 = 'lightskyblue' 
  c2 = 'purple'
  n = 20
  counter = 0
  
  input_image = np.asarray( Image.open(get_save_path_image_i(vid, img_num)) )
  
  fig = plt.figure()
  plt.clf()
  ax = fig.add_subplot(111, projection='3d')
  fig.set_size_inches(input_image.shape[1]/dpi_val,input_image.shape[0]/dpi_val*2)
  
  for j in range(int(min_max_axis[0,0]), img_num+1):

    time_series_instances = time_series_instances_load( vid, j, epsilon, num_reg )
    instances_id = time_series_instances.copy()
    instances_id[time_series_instances<0] *= -1
    instances_id = instances_id  % 10000
    instances_id[time_series_instances<0] *= -1
    
    index = -1
    for m in range(instances_id.shape[0]):
      if instances_id[m,:,:].min() == -max_id:
        index = m
        break
    if index > -1:
      
      # (2--vertical/horizontal coord, #points)
      x_y_indices = np.asarray( np.where( (instances_id[index] == -max_id) ) )   
      # change vertical axis because of image to numpy
      x_y_indices[0,:] = instances_id[index].shape[0] - x_y_indices[0,:]
  
      if (x_y_indices.shape[1] > 2) and (len(np.unique(x_y_indices[0,:])) > 1) and (len(np.unique(x_y_indices[1,:])) > 1):
        x_y_indices = x_y_indices.transpose()   # shape (#points, 2)
        hull = ConvexHull(x_y_indices)
        
        num_hull_p = x_y_indices[hull.vertices,0].shape[0]
        
        points = np.zeros((num_hull_p+1, 3))
        points[:,0] = j
        points[0:num_hull_p, 1:3] = x_y_indices[hull.vertices,:]
        points[num_hull_p, 1:3] = x_y_indices[hull.vertices[0],:]
        
        plt.plot( points[:,0], points[:,2], points[:,1], c=colorFader(c1,c2,(counter%(n*2))/n), ls='-', lw=2, alpha=0.4)
      
      else:
        
        plt.plot(np.zeros(x_y_indices.shape[1])+j, x_y_indices[1], x_y_indices[0], c=colorFader(c1,c2,(counter%(n*2))/n), ls='-', lw=2, alpha=0.4)   
    
    if j == img_num:
      
      colors_t = input_image.copy()
      if index > -1:
        colors_t[instances_id[index]==max_id,:] = hex_to_rgb( colorFader(c1,c2,(counter%(n*2))/n) )
        colors_t[instances_id[index]==-max_id,:] = (0,0,0)

      inp_img = colors_t * 0.8 + input_image * 0.2
  
    counter += 1
    
  ax.set_xlim3d(int(min_max_axis[0,0]),int(min_max_axis[1,0]))
  ax.set_ylim3d(int(min_max_axis[0,1]),int(min_max_axis[1,1]))
  ax.set_zlim3d(int(min_max_axis[0,2]),int(min_max_axis[1,2]))
  
  ax.set_xlabel('frame')
  ax.set_ylabel('horizontal coordinate')
  ax.set_zlabel('vertical coordinate')
  
  fig.savefig(save_path, dpi=dpi_val)
  plt.close()
  
  plot_image = np.asarray( Image.open(save_path).convert("RGB") )
  
  if input_image.shape[0]%2==1:
    plot_image = plot_image[1:,:]
  
  img = np.concatenate( (plot_image,inp_img), axis=0 )
  img1 = Image.fromarray(img.astype('uint8'), 'RGB')
  img1.save(save_path)
  
  
def plot_instances_shapes_flexible( vid, max_id, comp_class_string ):
  
  t = time.time()

  epsilon = CONFIG.EPS_MATCHING
  num_reg = CONFIG.NUM_REG_MATCHING
 
  points = np.zeros((1,3))
  
  images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + "/" ))
  for j in range(len(images_all)):

    time_series_instances = time_series_instances_load( vid, j, epsilon, num_reg )
    instances_id = time_series_instances.copy()
    instances_id[time_series_instances<0] *= -1
    instances_id = instances_id  % 10000
    instances_id[time_series_instances<0] *= -1
    
    index = -1
    for m in range(instances_id.shape[0]):
      if instances_id[m,:,:].min() == -max_id:
        index = m
        break
    if index > -1:
    
      x_y_indices = np.asarray( np.where( (instances_id[index] == -max_id) ) )
      x_y_indices[0,:] = instances_id[index].shape[0] - x_y_indices[0,:]
      
      if (x_y_indices.shape[1] > 2) and (len(np.unique(x_y_indices[0,:])) > 1) and (len(np.unique(x_y_indices[1,:])) > 1):
        x_y_indices = x_y_indices.transpose()   # shape (#points, 2)
        hull = ConvexHull(x_y_indices)
        
        num_hull_p = x_y_indices[hull.vertices,0].shape[0]
        points_j = np.zeros((num_hull_p, 3))
        points_j[:,0] = j
        points_j[:, 1] = x_y_indices[hull.vertices,1]
        points_j[:, 2] = x_y_indices[hull.vertices,0]

      else:
        
        x_y_indices = x_y_indices.transpose() 
        
        points_j = np.zeros((x_y_indices.shape[0], 3))
        points_j[:,0] = j
        points_j[:, 1] = x_y_indices[:,1]
        points_j[:, 2] = x_y_indices[:,0]
        
      points = np.append(points, points_j, axis=0)
  
  points = points[1:points.shape[0],:]
  
  fig = go.Figure()
  time_steps = np.unique(points[:,0])
  
  for k in range(1,len(time_steps)):
    
    if time_steps[k-1]+1 == time_steps[k]:
    
      points_plot = np.concatenate((points[points[:,0]==int(time_steps[k-1])], points[points[:,0]==int(time_steps[k])]), axis=0 )
      
      fig.add_traces(data=[go.Mesh3d(x=points_plot[:,0], y=points_plot[:,1], z=points_plot[:,2],
                    alphahull=0,
                    opacity=0.2,
                    color='plum')])
  
  fig.update_layout(scene = dict(
                  xaxis_title='frame',
                  yaxis_title='horizontal coordinate',
                  zaxis_title='vertical coordinate'))
  
  save_path = CONFIG.IMG_METRICS_DIR + 'shapes/' + vid + '_class_' + comp_class_string + "_start" + str(int(time_steps[0])) + "_end" + str(int(time_steps[-1])) + "/img_flex.html"
  
  fig.write_html(save_path, auto_open=False)
  
  print("plot video", vid, ": time needed ", time.time()-t)
  
  
def plot_scatter_lifetime( mean_lifetime, lifetime_mean_size_del, size_cut, lifetime_mean_size_del_cut ):
  
  save_path = CONFIG.IMG_METRICS_DIR + 'lifetime/'
  if not os.path.exists( save_path ):
    os.makedirs( save_path )
                           
  x = np.arange(2)
  f2, ax = plt.subplots()
  plt.bar(x, mean_lifetime, color='teal')
  plt.xticks(x, ('mean lifetime', 'mean lifetime for instances with mean $S >$' + str(size_cut))) 
  plt.ylabel('frames')
  plt.title("Mean lifetime for instances")
  f2.savefig(save_path+"mean_lifetime.png", dpi=300, bbox_inches='tight')
  plt.close()
  
  plt.rc('axes', titlesize=10)
  plt.rc('figure', titlesize=10)
  figsize=(3.0,13.0/5.0)
  
  plt.figure(figsize=figsize, dpi=300)
  plt.scatter( lifetime_mean_size_del[:,0], lifetime_mean_size_del[:,1], s = 10, c='palevioletred', alpha=0.05 )
  plt.yscale('log')
  plt.xscale('log')
  plt.xlabel("lifetime")
  plt.ylabel("mean $S_{in}$")
  plt.savefig(save_path+'lifetime_size_loglog.png', bbox_inches='tight')
  plt.close()
  
  plt.figure(figsize=figsize, dpi=300)
  plt.scatter( lifetime_mean_size_del[:,0], lifetime_mean_size_del[:,1], s = 10, c='palevioletred', alpha=0.05 )
  plt.xlabel("lifetime")
  plt.ylabel("mean $S_{in}$")
  plt.savefig(save_path+'lifetime_size.png', bbox_inches='tight')
  plt.close()
  
  
def plot_map(save_path, map_fp_fn_classic, map_fp_fn_meta):
  
  size_text = 22
  
  f1 = plt.figure(1,frameon=False) 
  plt.clf()
  plt.plot(map_fp_fn_classic[:,1]/1000, map_fp_fn_classic[:,2]/1000, color='olivedrab', marker='o', linewidth=2, markersize=10, label="score", alpha=0.7)  
  plt.plot(map_fp_fn_meta[:,1]/1000, map_fp_fn_meta[:,2]/1000, color='tab:pink', marker='o', linewidth=2, markersize=10, label="meta classification", alpha=0.7)
  plt.xlabel('$\#$ false positives ($\\times 10^3$)', fontsize=size_text) #labelpad=-15)
  plt.ylabel('$\#$ false negatives ($\\times 10^3$)', fontsize=size_text)
  plt.xticks(fontsize=size_text)
  plt.yticks(fontsize=size_text)
  plt.legend(fontsize=size_text)
  f1.savefig(save_path + 'fp_fn.png', dpi=300, bbox_inches='tight')
  plt.close()
  
  f1 = plt.figure(1,frameon=False) 
  plt.clf()
  x = np.arange(0, len(map_fp_fn_classic[:,0]))
  plt.plot(x, map_fp_fn_classic[:,0], color='cornflowerblue', marker='o', label="score", alpha=0.7)  
  plt.plot(x, map_fp_fn_meta[:,0], color='palevioletred', marker='o', label="meta classification", alpha=0.7)
  plt.xticks([])
  plt.ylabel('mean average precision', fontsize=size_text)
  plt.xticks(fontsize=size_text)
  plt.yticks(fontsize=size_text)
  plt.legend(fontsize=size_text)
  f1.savefig(save_path + 'map.png', dpi=300)
  plt.close()
  
  f1 = plt.figure(1,frameon=False) 
  plt.clf()
  plt.plot(map_fp_fn_classic[:,3]/(map_fp_fn_classic[:,3]+map_fp_fn_classic[:,2]), map_fp_fn_classic[:,3]/(map_fp_fn_classic[:,3]+map_fp_fn_classic[:,1]), color='tab:cyan', marker='o', label="score", alpha=0.7)  
  plt.plot(map_fp_fn_meta[:,3]/(map_fp_fn_meta[:,3]+map_fp_fn_meta[:,2]), map_fp_fn_meta[:,3]/(map_fp_fn_meta[:,3]+map_fp_fn_meta[:,1]), color='tab:purple', marker='o', label="meta classification", alpha=0.7)
  plt.xlabel('recall', fontsize=size_text)
  plt.ylabel('precision', fontsize=size_text)
  plt.xticks(fontsize=size_text)
  plt.yticks(fontsize=size_text)
  plt.legend(fontsize=size_text)
  f1.savefig(save_path + 'pre_rec.png', dpi=300)
  plt.close()
  

def plot_map_models(save_path, map_fp_fn_classic, map_fp_fn_meta, map_fp_fn_tp_classic_other, map_fp_fn_tp_meta_other):
  
  size_text = 22
  line_size = 2
  marker_size = 8
  
  f1 = plt.figure(figsize=(11,4)) # 6,9.5
  a = f1.add_subplot(111)    
  ax = f1.add_subplot(121)
  ax1 = f1.add_subplot(122)
  ax.plot(map_fp_fn_classic[:,1]/1000, map_fp_fn_classic[:,2]/1000, color='olivedrab', marker='o', linewidth=line_size, markersize=marker_size, label='score', alpha=0.7)  
  ax.plot(map_fp_fn_meta[:,1]/1000, map_fp_fn_meta[:,2]/1000, color='tab:pink', marker='o', linewidth=line_size, markersize=marker_size, label='meta classification', alpha=0.7)
  ax1.plot(map_fp_fn_tp_classic_other[:,1]/1000, map_fp_fn_tp_classic_other[:,2]/1000, color='olivedrab', marker='o', linewidth=line_size, markersize=marker_size, label='score', alpha=0.7)  
  ax1.plot(map_fp_fn_tp_meta_other[:,1]/1000, map_fp_fn_tp_meta_other[:,2]/1000, color='tab:pink', marker='o', linewidth=line_size, markersize=marker_size, label='meta classification', alpha=0.7)
  ax.tick_params(axis ='x', labelsize=size_text)
  ax1.tick_params(axis ='x', labelsize=size_text)
  ax.tick_params(axis ='y', labelsize=size_text)
  ax1.tick_params(axis ='y', labelsize=size_text)
  ax.legend(fontsize=size_text)
  ax1.legend(fontsize=size_text)
  
  # same x and y label
  a.spines['top'].set_color('none')
  a.spines['bottom'].set_color('none')
  a.spines['left'].set_color('none')
  a.spines['right'].set_color('none')
  a.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
  a.set_ylabel('$\#$ false negatives ($\\times 10^3$)', fontsize=size_text, labelpad=10)
  a.set_xlabel('$\#$ false positives ($\\times 10^3$)', fontsize=size_text, labelpad=10)

  f1.savefig(save_path + 'fp_fn_models.png', dpi=300, bbox_inches='tight')
  plt.close()
  
  
def visualize_regr_classif_i( iou, iou_pred, vid, n, counter, colors_list, flag_regr_classif ):
  
  t = time.time()
  
  epsilon = CONFIG.EPS_MATCHING
  num_reg = CONFIG.NUM_REG_MATCHING
  input_image = np.asarray( Image.open(get_save_path_image_i(vid, n)) )
  gt_image = ground_truth_load(vid, n)
  inst_image  = time_series_instances_load( vid, n, epsilon, num_reg )
  instances_id = inst_image.copy()
  instances_id[inst_image<0] *= -1
  instances_id = instances_id  % 10000

  inx_field_counter = np.zeros((inst_image.shape[0]))
  
  # sort instance predictions by size
  for m in range( 0, inst_image.shape[0] ):
    inx_field_counter[m] = np.count_nonzero(instances_id[m]!=0)
    
  inx_field_counter_copy = inx_field_counter.copy()
  instances = np.zeros((input_image.shape[0], input_image.shape[1]))
  
  for m in range( 0, inst_image.shape[0] ):
    max_index = int(np.argmax(inx_field_counter))
    instances[instances_id[max_index]>0] = instances_id[max_index].max()
    inx_field_counter[max_index] = -1
  
  I1 = input_image.copy()
  img_iou = visualize_instances( instances, iou )
  
  I2 = input_image.copy()
  img_pred = visualize_instances( instances, iou_pred )

  instance_masks = []
  gt_class_ids = []
  obj_ids = np.unique(gt_image)
  for i in range(len(obj_ids)):
    if obj_ids[i] != 0 and obj_ids[i] != 10000:
        m_i = np.zeros(np.shape(gt_image))
        m_i[gt_image==obj_ids[i]] = 1
        instance_masks.append(m_i)
        gt_class_ids.append(obj_ids[i] // 1000)
  # Pack instance masks into an array
  if len(instance_masks) > 0:
    mask_tmp = np.stack(instance_masks, axis=2).astype(np.bool)
    # [num_instances, (y1, x1, y2, x2)]
    gt_bbox = create_bb(mask_tmp)
  
  instance_masks = []
  seg_class_ids = []
  for i in range(inst_image.shape[0]):
    m_i = np.zeros(( inst_image.shape[1], inst_image.shape[2] ))
    m_i[inst_image[i,:,:]!=0] = 1
    instance_masks.append(m_i)
    seg_class_ids.append(inst_image[i].max() // 10000)        
  if len(instance_masks) > 0:
    mask_tmp = np.stack(instance_masks, axis=2).astype(np.bool)
    seg_bbox = create_bb(mask_tmp)
  
  I3 = input_image.copy()
  tmp = np.zeros((3))
  for i in range(input_image.shape[0]):
    for j in range(input_image.shape[1]):
      if gt_image[i,j] > 0 and gt_image[i,j] < 10000:   
        tmp = np.asarray( hex_to_rgb( colors_list[ (gt_image[i,j]-1) % len(colors_list) ] ) )
        I3[i,j] = tmp * 0.6 + input_image[i,j] * 0.4
      elif gt_image[i,j] == 10000:
        tmp = np.asarray((255,255,255))
        I3[i,j] = tmp * 0.6 + input_image[i,j] * 0.4
        
      if instances[i,j] != 0:
        I1[i,j] = img_iou[i,j] * 0.6 + I1[i,j] * 0.4
        I2[i,j] = img_pred[i,j] * 0.6 + I2[i,j] * 0.4
  
  I4 = input_image.copy()
  for k in range(inst_image.shape[0]):
    max_index = int(np.argmax(inx_field_counter_copy))
    for i in range(input_image.shape[0]):
      for j in range(input_image.shape[1]):
        tmp = np.zeros((3))
        counter = 0
        if inst_image[max_index,i,j] > 0: 
          tmp = np.asarray( hex_to_rgb( colors_list[ int(inst_image[max_index,i,j]-1) % len(colors_list) ] ) )
          I4[i,j] = tmp * 0.6 + input_image[i,j] * 0.4
    inx_field_counter_copy[max_index] = -1
        
  img12 = np.concatenate( (I1,I2), axis=1 )
  img34 = np.concatenate( (I3,I4), axis=1 )
  img = np.concatenate( (img12,img34), axis=0 )
  image = Image.fromarray(img.astype('uint8'), 'RGB')
  
  draw = ImageDraw.Draw(image)
  if len(gt_class_ids) > 0:
    for i in range(gt_bbox.shape[0]):
      color_class = trainId2label[ gt_class_ids[i] ].color
      draw.rectangle([gt_bbox[i,1], gt_bbox[i,0]+input_image.shape[0], gt_bbox[i,3], gt_bbox[i,2]+input_image.shape[0]], fill=None, outline=color_class)
  
  if len(seg_class_ids) > 0:
    for i in range(seg_bbox.shape[0]):
      color_class = trainId2label[ seg_class_ids[i] ].color
      draw.rectangle([seg_bbox[i,1]+input_image.shape[1], seg_bbox[i,0]+input_image.shape[0], seg_bbox[i,3]+input_image.shape[1], seg_bbox[i,2]+input_image.shape[0]], fill=None, outline=color_class)
  
  if flag_regr_classif == 0:
    save_path = CONFIG.IMG_IOU_INST_DIR + vid + "/"
  else:
    save_path = CONFIG.IMG_IOU0_INST_DIR + vid + "/"  
  
  if not os.path.exists( save_path ):
    os.makedirs( save_path )
  
  image.save(save_path + "img_iou" + str(n).zfill(6) + "_eps" + str(epsilon) + "_num_reg" + str(num_reg)  + ".png")
  
  print("plot image", vid, n, ": time needed ", time.time()-t)  
  
  
def plot_regression_scatter( Xa_test, ya_test, ya_test_pred, X_names, num_frames ):
  
  plt.rc('axes', titlesize=10)
  plt.rc('figure', titlesize=10)
  cmap=plt.get_cmap('tab20')

  S_ind = 0
  for S_ind in range(len(X_names)):
    if X_names[S_ind] == "S":
      break
  
  figsize=(3.0,13.0/5.0)
  plt.figure(figsize=figsize, dpi=300)
  plt.clf()
  
  sizes = np.squeeze(Xa_test[:,S_ind]*np.std(Xa_test[:,S_ind]))
  sizes = sizes - np.min(sizes)
  sizes = sizes / np.max(sizes) * 50 #+ 1.5      
  x = np.arange(0., 1, .01)
  plt.plot( x, x, color='black' , alpha=0.5, linestyle='dashed')
  plt.scatter( ya_test, np.clip(ya_test_pred,0,1), s=sizes, linewidth=.5, c=cmap(0), edgecolors=cmap(1), alpha=0.25 )
  plt.xlabel('$\mathit{IoU}$')
  plt.ylabel('predicted $\mathit{IoU}$')
  plt.savefig(CONFIG.ANALYZE_DIR + 'scatter/' + CONFIG.REGRESSION_MODEL + '_scatter_test_npf' + str(num_frames) + '.png', bbox_inches='tight')
  plt.close()
  
  
def plot_coef_timeline( mean_stats, X_names ):
  
  num_prev_frames = CONFIG.NUM_PREV_FRAMES

  switcher = {
          "E"           : 'lightpink',          
          "E_bd"        : 'palevioletred', 
          "E_in"        : 'deeppink',
          "E_rel"       : 'mediumvioletred',
          "E_rel_in"    : 'purple',
          "M"           : 'lightskyblue',          
          "M_bd"        : 'deepskyblue', 
          "M_in"        : 'steelblue',
          "M_rel"       : 'mediumblue',
          "M_rel_in"    : 'midnightblue',
          "S"           : 'plum',          
          "S_bd"        : 'mediumorchid', 
          "S_in"        : 'mediumpurple',
          "S_rel"       : 'blueviolet',
          "S_rel_in"    : 'indigo',
          "V"           : 'palegreen',          
          "V_bd"        : 'limegreen', 
          "V_in"        : 'olivedrab',
          "V_rel"       : 'mediumseagreen',
          "V_rel_in"    : 'darkgreen',
          "mean_x"      : 'darkturquoise',
          "mean_y"      : 'teal',
          "score"       : 'firebrick',
          "cprob0"      : 'gold',
          "cprob1"      : 'orange',
          "cprob2"      : 'peru',
          "survival"    : 'cornflowerblue',
          "ratio"       : 'lightcoral',
          "deformation" : 'magenta',
          "diff_mean"   : 'aquamarine',
          "diff_size"   : 'mediumaquamarine'
  }

  size_font = 20
  
  for i in range(num_prev_frames+1):
    
    coefs = np.asarray(mean_stats['coef'][i])
    num_timeseries = np.arange(0, i+1)
    x_ticks = []
    
    f1 = plt.figure(figsize=(10,6.2))
    plt.clf()
    
    # 1: smaller, shift to the left, 2: smaller, shift to the bottom
    ax = f1.add_axes([0.11, 0.12, 0.6, 0.75])
    
    for c in range(len(X_names)):
      if c == 0:
        x_ticks.append( "$t$" )
      else:
        x_ticks.append( "$t-$" + str(c) )

      index = 0
      for index in range(len(X_names)):
        if X_names[index] == X_names[c]:
          break
        
      coef_c = np.zeros((i+1))
      for k in range(i+1):
        coef_c[k] = coefs[k*len(X_names)+index]
        
      plt.plot(num_timeseries, coef_c, color=switcher.get(X_names[c], "red"), marker='o', label=name_to_latex(X_names[c]), alpha=0.8)  
    
    plt.xticks(fontsize = size_font)
    plt.yticks(fontsize = size_font)
    plt.xlabel('frame', fontsize=size_font)
    plt.xticks(num_timeseries, (x_ticks))
    if "LR" in CONFIG.CLASSIFICATION_MODEL:
      plt.ylabel('coefficients', fontsize=size_font)
    elif "GB" in CONFIG.CLASSIFICATION_MODEL:
      plt.ylabel('feature importance', fontsize=size_font)
    
    plt.legend(bbox_to_anchor=(1., 0.5), loc=6, ncol=2, borderaxespad=0.6, prop={'size': 16})
    
    save_path = CONFIG.ANALYZE_DIR +'feature_importance/' + CONFIG.CLASSIFICATION_MODEL + '_coef' + str(i)
    f1.savefig(save_path, dpi=300)
    plt.close()
  
  
def plot_train_val_test_timeline( num_timeseries, train, train_std, val, val_std, test_R, test_R_std, analyze_type ):
  
  f1 = plt.figure(1,frameon=False) 
  plt.clf()
  plt.plot(num_timeseries, train, color='violet', marker='o', label="train")  
  plt.fill_between(num_timeseries, train-train_std, train+train_std, color='violet', alpha=0.05 )
  plt.plot(num_timeseries, val, color='midnightblue', marker='o', label="val")
  plt.fill_between(num_timeseries, val-val_std, val+val_std, color='midnightblue', alpha=0.05 )
  plt.plot(num_timeseries, test_R, color='deepskyblue', marker='o', label="test")  
  plt.fill_between(num_timeseries, test_R-test_R_std, test_R+test_R_std, color='deepskyblue', alpha=0.05 )
  plt.xlabel('Frames')
  if analyze_type == 'r2':
    plt.ylabel('$R^2$')
    name = "r2_timeline_" + CONFIG.REGRESSION_MODEL + ".png"
  elif analyze_type == 'auc':
    plt.ylabel('$\mathit{AUROC}$')
    name = "auc_timeline_" + CONFIG.CLASSIFICATION_MODEL + ".png"
  elif analyze_type == 'acc':
    plt.ylabel('$ACC$')
    name = "acc_timeline_" + CONFIG.CLASSIFICATION_MODEL + ".png"
  plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
  save_path = CONFIG.ANALYZE_DIR +'train_val_test_timeline/' + name
  f1.savefig(save_path, dpi=300)
  plt.close()
  
  
def plot_r2_auc_timeline_data( names, mean_list_data, std_list_data, r2_or_auc=''):
  
  num_prev_frames = CONFIG.NUM_PREV_FRAMES
  num_curves = len(names)
  size_font = 26
  num_timeseries = np.arange(1, num_prev_frames+2)
  if num_curves == 6:
    color_map = ['red', 'darkgoldenrod', 'red', 'mediumvioletred', 'red', 'mediumseagreen']
  else:
    color_map = ['darkgoldenrod', 'mediumvioletred', 'mediumseagreen']
    
  f1 = plt.figure(1)
  plt.clf()
  
  for i in range(num_curves):
    if 'LR_L1' in names[i] or 'GB' in names[i] or 'NN_L2' in names[i]:
      name_tmp = names[i]
      name_label = name_tmp.replace("_", " ")
      r_min = (num_prev_frames+1) * i
      r_max = (num_prev_frames+1) * (i+1)
      plt.plot(num_timeseries, mean_list_data[r_min:r_max], color=color_map[i], marker='o', label=name_label)  
      plt.fill_between(num_timeseries, mean_list_data[r_min:r_max]-std_list_data[r_min:r_max], mean_list_data[r_min:r_max]+std_list_data[r_min:r_max], color=color_map[i], alpha=0.05 ) 
  
  matplotlib.rcParams['legend.numpoints'] = 1
  matplotlib.rcParams['legend.handlelength'] = 0
  plt.xticks(fontsize = size_font)
  plt.yticks(fontsize = size_font)
  plt.xlabel('number of considered frames', fontsize=size_font)#, labelpad=-5)
  if r2_or_auc == 'auc':
    plt.ylabel('$\mathit{AUROC}$', fontsize=size_font, labelpad=-1)
  elif r2_or_auc == 'r2':
    plt.ylabel('$R^2$', fontsize=size_font, labelpad=-1) 
  plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=6, mode="expand", borderaxespad=0., prop={'size': 16}, numpoints=1)
  save_path = CONFIG.IMG_ANALYZE_DIR + str(r2_or_auc) + "_timeline.png"#.pdf"
  f1.savefig(save_path, bbox_inches='tight', dpi=400)
  plt.close()
  
  
def plot_r2_auc_metrics( name, mean_list_data, std_list_data, r2_or_auc=''):
  
  num_prev_frames = CONFIG.NUM_PREV_FRAMES
  num_curves = len(mean_list_data[0])
  size_font = 26
  num_timeseries = np.arange(1, num_prev_frames+2)
  color_map = ['teal', 'purple', 'steelblue']
  metrics_com = ['$U^i$', '$U^i$ $\\cup$ $\{s,r\}$', '$V^i$']

  f1 = plt.figure(1)
  plt.clf()
  
  for i in range(num_curves):
    plt.plot(num_timeseries, mean_list_data[:,i], color=color_map[i], marker='o', label= metrics_com[i])  
    plt.fill_between(num_timeseries, mean_list_data[:,i]-std_list_data[:,i], mean_list_data[:,i]+std_list_data[:,i], color=color_map[i], alpha=0.05 ) 
  
  matplotlib.rcParams['legend.numpoints'] = 1
  matplotlib.rcParams['legend.handlelength'] = 0
  plt.xticks(fontsize = size_font)
  plt.yticks(fontsize = size_font)
  plt.xlabel('number of considered frames', fontsize=size_font)
  if r2_or_auc == 'auc':
    plt.ylabel('$\mathit{AUROC}$', fontsize=size_font, labelpad=-1)
  elif r2_or_auc == 'r2':
    plt.ylabel('$R^2$', fontsize=size_font, labelpad=-1) 
  plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=6, mode="expand", borderaxespad=0., prop={'size': 16}, numpoints=1)
  save_path = CONFIG.IMG_ANALYZE_DIR.split('_nm')[0] + '/'
  if not os.path.exists( save_path ):
    os.makedirs( save_path )
  f1.savefig(save_path + str(r2_or_auc) + '_' + str(name) + "_timeline.png", bbox_inches='tight', dpi=400)
  plt.close()


def plot_baselines_vs_ours(model1, baselines1, model2, baselines2):
  
  print('Plot baselines')
  size_font = 26
  num_approaches = np.arange(baselines1.shape[0])   
  dist = 0.05
  color_map = ['tab:blue', 'tab:olive', 'tab:cyan', 'tab:pink']
  x_ticks = ['entropy', 'score', 'ms', 'time ms', 'ours']
  
  switcher = {
          "mask_rcnn" : 'Mask R-CNN',          
          "yolact"    : 'YOLACT'
  }

  f1 = f1 = plt.figure(figsize=(7.2,5.8))
  plt.clf()
  
  print(baselines1[:,:,0])
  print(baselines2[:,:,0])
  
  label_tmp = '$\mathit{AUROC}$ ' + switcher.get(model1)
  plt.errorbar(num_approaches-dist, baselines1[:,1,0], baselines1[:,1,1], color=color_map[0], linestyle='', marker='o', capsize=3, label=label_tmp, alpha=1)
  label_tmp = '$\mathit{AUROC}$ ' + switcher.get(model2)
  plt.errorbar(num_approaches+dist, baselines2[:,1,0], baselines2[:,1,1], color=color_map[1], linestyle='', marker='o', capsize=3, label=label_tmp, alpha=1)
  
  label_tmp = '$R^2$ ' + switcher.get(model1)
  plt.errorbar(num_approaches-dist, baselines1[:,0,0], baselines1[:,0,1], color=color_map[2], linestyle='', marker='o', capsize=3, label=label_tmp, alpha=1)
  label_tmp = '$R^2$ ' + switcher.get(model2)
  plt.errorbar(num_approaches+dist, baselines2[:,0,0], baselines2[:,0,1], color=color_map[3], linestyle='', marker='o', capsize=3, label=label_tmp, alpha=1)
  
  matplotlib.rcParams['legend.numpoints'] = 1
  matplotlib.rcParams['legend.handlelength'] = 0
  plt.xticks(fontsize = size_font)
  plt.yticks(fontsize = size_font)
  plt.xticks(num_approaches, (x_ticks))
  plt.legend(prop={'size': 16})
  save_path = CONFIG.IMG_ANALYZE_DIR.split('_nm')[0] + '/'
  print(save_path)
  f1.savefig(save_path + 'baselines.png', bbox_inches='tight', dpi=400)
  plt.close()
  
  


