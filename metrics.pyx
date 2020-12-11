import numpy as np
cimport numpy as np
import os
import pickle
from scipy.stats import linregress

from global_defs import CONFIG
from in_out      import instances_small_load, get_save_path_time_series_instances_i,\
                        time_series_instances_load, time_series_instances_dump, ground_truth_load,\
                        score_small_load



def shift_overlap_distance(int i, int imx, int imy, np.ndarray flag_comp_num_n_in, np.ndarray flag_comp_num_n_1_in, np.ndarray ts_instances_in, np.ndarray instances_in, np.ndarray mean_n_in, float percentage, int epsilon):
  
  cdef int x, y, class_n_1, class_n, counter, x_shift, y_shift, x_t, y_t, i_n_1, value_i, max_index, min_index
  
  cdef float mean_x_n, mean_y_n, mean_x_n_1, mean_y_n_1, mean_x_n_2, mean_y_n_2, intersection, union, max_iou, min_distance, dist, dir_n_1_n_x, dir_n_1_n_y, dir_n_2_n_1_x, dir_n_2_n_1_y
  
  cdef char[:,:] shifted_comp
  
  flag_comp_num_n = flag_comp_num_n_in
  flag_comp_num_n_1 = flag_comp_num_n_1_in
  ts_instances = ts_instances_in
  instances = instances_in
  mean_n = mean_n_in
  
  # compute only for instance number i, if the number is existent
  if flag_comp_num_n_1[i] == 0:
    # class instance i (n-1)
    class_n_1 = int(instances_in[1,i,:,:].max() // 1000)
    
    # compute geometric center instance i
    mean_x_n_1 = 0
    mean_y_n_1 = 0
    counter = 0
    for x in range(imx):
      for y in range(imy):
        if ts_instances[1,i,x,y] > 0:
          mean_x_n_1 += x
          mean_y_n_1 += y
          counter = counter + 1
    mean_x_n_1 /= counter
    mean_y_n_1 /= counter
    
    #### overlap
    # compute geometric center n-2 with number i (same like compnent i (n-1))
    mean_x_n_2 = 0
    mean_y_n_2 = 0
    counter = 0
    value_i = ts_instances_in[1,i,:,:].max()
    
    for i_n_1 in range(len(flag_comp_num_n_1_in)):
      if ts_instances_in[2,i_n_1,:,:].max() == value_i:
        break
    for x in range(imx):
      for y in range(imy):
        if ts_instances[2,i_n_1,x,y] == value_i:
          mean_x_n_2 += x
          mean_y_n_2 += y
          counter = counter + 1
    mean_x_n_2 /= counter
    mean_y_n_2 /= counter
  
    ## if existent, the instance i (n-1) is shifted by the prediction and then the overlapping area is computed
    x_shift = int(mean_x_n_1 - mean_x_n_2)
    y_shift = int(mean_y_n_1 - mean_y_n_2)
    
    shifted_comp_in = np.zeros((imx, imy), dtype="uint8")
    shifted_comp = shifted_comp_in
    
    # compute the shifted instance i (n-1) to n
    for x in range(imx):
      for y in range(imy):
        if ts_instances[1,i,x,y] > 0:
          x_t = x + x_shift
          y_t = y + y_shift
          if x_t>=0 and x_t<imx and y_t>=0 and y_t<imy:
            shifted_comp[x_t,y_t] = 1
    
    # compute the overlapping areas
    max_iou = 0
    max_index = 0
    
    # compute the overlapping areas
    for j in range(len(flag_comp_num_n_in)):
      
      # j has not a number
      if flag_comp_num_n[j] == 0:
      
        # class instance j (n)
        class_n = int(instances_in[0,j,:,:].max() // 1000)
      
        if class_n_1 == class_n:

          intersection = 0
          union = 0
        
          for x in range(imx):
            for y in range(imy):
              if shifted_comp[x,y] == 1 and instances[0,j,x,y] > 0:
                intersection = intersection + 1
              if shifted_comp[x,y] == 1 or instances[0,j,x,y] > 0:
                union = union + 1
          if union > 0:     
            if (intersection / union ) > max_iou:
              max_iou = (intersection / union )
              max_index = j
      
    # it is a match if the number of overlapping pixel is 35 percent of instance j (n-1) and max overlapping
    if max_iou >= percentage:
      ts_instances_in[ 0,max_index,instances_in[0,max_index]>0 ] = ts_instances_in[1,i,:,:].max()
      flag_comp_num_n_1[i] = 1
      flag_comp_num_n[max_index] = 1
      print("geometric center and overlapping match with percentage: number of pixel, index i and j, class", max_iou, i, max_index, class_n_1)
      
    else:
      
      #### distance
      # tmp for instance with min distance and direction
      min_distance = 3000
      min_index = -1
      
      dir_n_2_n_1_x =  mean_x_n_1 - mean_x_n_2
      dir_n_2_n_1_y =  mean_y_n_1 - mean_y_n_2
            
      for j in range(len(flag_comp_num_n_in)):
              
        # j has not a number
        if flag_comp_num_n[j] == 0:
        
          # class instance j (n)
          class_n = int(instances_in[0,j,:,:].max() // 1000)
        
          # compute only, if instance i (n-1) and instance j (n) have the same class
          if class_n_1 == class_n:
            
            dir_n_1_n_x =  mean_n[j,0] - mean_x_n_1
            dir_n_1_n_y =  mean_n[j,1] - mean_y_n_1
            
            dist = ( dir_n_1_n_x**2 + dir_n_1_n_y**2 ) **0.5 + ( (dir_n_2_n_1_x - dir_n_1_n_x)**2 + (dir_n_2_n_1_y - dir_n_1_n_y)**2 )**0.5
            
            if dist < min_distance:
              min_distance = dist
              min_index = j
                
      if min_index > -1 and min_distance <= epsilon:
        ts_instances_in[ 0,min_index,instances_in[0,min_index]>0 ] = ts_instances_in[1,i,:,:].max()
        flag_comp_num_n_1[i] = 1
        flag_comp_num_n[min_index] = 1
        print("geometric center match: distance with direction, index i and j, class", min_distance, i, min_index, class_n_1) 
      
  return ts_instances_in, flag_comp_num_n_in, flag_comp_num_n_1_in
  
  
def shift_distance_simplified(int i, int imx, int imy, np.ndarray flag_comp_num_n_in, np.ndarray flag_comp_num_n_1_in, np.ndarray ts_instances_in, np.ndarray instances_in, np.ndarray mean_n_in, int epsilon):
  
  cdef int x, y, class_n_1, class_n, counter, min_index
  
  cdef float mean_x_n, mean_y_n, mean_x_n_1, mean_y_n_1, min_distance, dist
  
  flag_comp_num_n = flag_comp_num_n_in
  flag_comp_num_n_1 = flag_comp_num_n_1_in
  ts_instances = ts_instances_in
  instances = instances_in
  mean_n = mean_n_in
  
  if flag_comp_num_n_1[i] == 0:
    
    # class instance i (n-1)
    class_n_1 = int(instances_in[1,i,:,:].max() // 1000)
    
    # compute geometric center instance i
    mean_x_n_1 = 0
    mean_y_n_1 = 0
    counter = 0
    for x in range(imx):
      for y in range(imy):
        if ts_instances[1,i,x,y] > 0:
          mean_x_n_1 += x
          mean_y_n_1 += y
          counter = counter + 1
    mean_x_n_1 /= counter
    mean_y_n_1 /= counter
    
    # tmp for instance with min distance
    min_distance = 3000
    min_index = -1

    for j in range(len(flag_comp_num_n_in)):
      
      # j has not a number
      if flag_comp_num_n[j] == 0:
      
        # class instance j (n)
        class_n = int(instances_in[0,j,:,:].max() // 1000)
      
        if class_n_1 == class_n:
          
          dist = ( (mean_n[j,0] - mean_x_n_1)**2 + (mean_n[j,1] - mean_y_n_1)**2 )**0.5
          
          if dist < min_distance:
            min_distance = dist
            min_index = j
          
    if min_index > -1 and min_distance <= epsilon:
      ts_instances_in[ 0,min_index,instances_in[0,min_index]>0 ] = ts_instances_in[1,i,:,:].max()
      flag_comp_num_n_1[i] = 1
      flag_comp_num_n[min_index] = 1
      print("geometric center match: distance, index i and j, class", min_distance, i, min_index, class_n_1)   
    
  return ts_instances_in, flag_comp_num_n_in, flag_comp_num_n_1_in
  
   
def overlap(int i, int imx, int imy, np.ndarray flag_comp_num_n_in, np.ndarray flag_comp_num_n_1_in, np.ndarray ts_instances_in, np.ndarray instances_in, float percentage):
  
  cdef int x, y, class_n_1, class_n, max_index
  
  cdef float intersection, union, max_iou
  
  flag_comp_num_n = flag_comp_num_n_in
  flag_comp_num_n_1 = flag_comp_num_n_1_in
  ts_instances = ts_instances_in
  instances = instances_in
  
  # compute only for instance number i, if the number is existent
  if flag_comp_num_n_1[i] == 0:
    
    # class instance i (n-1)
    class_n_1 = int(instances_in[1,i,:,:].max() // 1000)
    
    # compute the overlapping areas
    max_iou = 0
    max_index = 0
    
    for j in range(len(flag_comp_num_n_in)):
      
      # j has no number
      if flag_comp_num_n[j] == 0:
      
        # class instance j (n)
        class_n = int(instances_in[0,j,:,:].max() // 1000)
      
        # compute only, if instance i (n-1) and instance j (n) have the same class
        if class_n_1 == class_n:
        
          intersection = 0
          union = 0
          
          for x in range(imx):
            for y in range(imy):
              if ts_instances[1,i,x,y] > 0 and instances[0,j,x,y] > 0:
                intersection = intersection + 1
              if ts_instances[1,i,x,y] > 0 or instances[0,j,x,y] > 0:
                union = union + 1
          
          if union > 0:
            if (intersection / union ) > max_iou:
              max_iou = (intersection / union )
              max_index = j
                    
    # it is a match if the number of overlapping pixel is 35 percent of instance j (n-1) and max overlapping
    if max_iou >= percentage:
      ts_instances_in[ 0,max_index,instances_in[0,max_index]>0 ] = ts_instances_in[1,i,:,:].max()
      flag_comp_num_n_1[i] = 1
      flag_comp_num_n[max_index] = 1
      print("overlapping match with percentage: number of pixel, index i and j, class", max_iou, i, max_index, class_n_1)
      
  return ts_instances_in, flag_comp_num_n_in, flag_comp_num_n_1_in
  

def regression_distance_overlap(int i, int imx, int imy, np.ndarray flag_comp_num_n_in, np.ndarray flag_comp_num_n_1_in, np.ndarray ts_instances_in, np.ndarray instances_in, np.ndarray mean_n_in, int eps_time, float percentage, int reg_steps):
  
  cdef int x, y, c, m, class_n_1, class_n, counter, min_index, max_index_timestep, x_shift, y_shift, x_t, y_t, max_index
  
  cdef float mean_x_n, mean_y_n, min_distance, dist, a_x, a_y, b_x, b_y, pred_x, pred_y, intersection, union, max_iou
  
  cdef short int[:] index_i
  cdef char[:,:] shifted_comp
  cdef float[:,:] mean_field
  cdef float[:] mean_counter_field
  
  flag_comp_num_n = flag_comp_num_n_in
  flag_comp_num_n_1 = flag_comp_num_n_1_in
  ts_instances = ts_instances_in
  instances = instances_in
  mean_n = mean_n_in
  
  # compute only for instances i, if the number is unused in ts_instances_in[0,:,:] and maybe i is not existent in (n-1)
  if np.count_nonzero(ts_instances_in[0,:,:,:]==i) == 0:
    
    # class of instance i (n-1), if the instance exist in the last five images
    class_n_1 = -2
    
    index_i_in = np.zeros( (reg_steps), dtype="int16" ) -2
    index_i = index_i_in
    
    for c in range(1,reg_steps+1):
      
      for m in range(len(flag_comp_num_n_1_in)):
        if ts_instances_in[c,m,:,:].max() == i:
          index_i[c-1] = m 
          break
      
      if index_i[c-1] > -2:
        class_n_1 = int(instances_in[c,index_i[c-1],:,:].max() // 1000)
      
    # instance i is exsitent in at least one of the last five images
    if class_n_1 > -2:
        
      mean_field_in = np.zeros( (reg_steps,2), dtype="float32" )
      mean_field = mean_field_in
      mean_counter_field_in = np.zeros( (reg_steps), dtype="float32" )
      mean_counter_field = mean_counter_field_in
  
      # compute geometric center of instance i in reg_steps times before
      for x in range(imx):
        for y in range(imy):
          for c in range(1,reg_steps+1):
            if ts_instances[c,index_i[c-1],x,y] == i:
              mean_field[c-1,0] += x
              mean_field[c-1,1] += y
              mean_counter_field[c-1] += 1
      
      # lists for geometric centers
      mean_x_list = []
      mean_y_list = []
      mean_t_list = []
    
      for c in range(reg_steps):
        if mean_counter_field[c] > 0:
          mean_field[c,0] /= mean_counter_field[c]
          mean_field[c,1] /= mean_counter_field[c]
          mean_x_list.append(mean_field[c,0])
          mean_y_list.append(mean_field[c,1])
          mean_t_list.append(c)
      
      if len(mean_t_list) >= 2:
        # linear regression of geometric centers
        b_x, a_x, _, _, _ = linregress(mean_t_list, mean_x_list)
        b_y, a_y, _, _, _ = linregress(mean_t_list, mean_y_list)
        pred_x = a_x + b_x * reg_steps
        pred_y = a_y + b_y * reg_steps
        
        #### distance
        ## shift the geometric center and search geometric centers that are near
        # tmp for instance with min distance
        min_distance = 3000
        min_index = -1
        
        for j in range(len(flag_comp_num_n_in)):
          
          # j has not a number
          if flag_comp_num_n[j] == 0:
          
            # class instance j (n)
            class_n = int(instances_in[0,j,:,:].max() // 1000)
            
            if class_n_1 == class_n:
              
              dist = ( (mean_n[j,0] - pred_x)**2 + (mean_n[j,1] - pred_y)**2 )**0.5
              
              if dist < min_distance:
                min_distance = dist
                min_index = j
                
        if min_index > -1 and min_distance <= eps_time:
          ts_instances_in[ 0,min_index,instances_in[0,min_index]>0 ] = i
          if index_i[0] > -2:
            flag_comp_num_n_1[index_i[0]] = 1
          flag_comp_num_n[min_index] = 1
          print("timeseries match: distance, index i and j, class", min_distance, index_i[0], min_index, class_n_1)  
        
        else:
          
          #### overlap
          ## the instance i (timestep with the maximum size) is shifted by the prediction and then the overlapping area is computed
          max_index_timestep = int(np.argmax(mean_counter_field))
          
          x_shift = int(pred_x - mean_field[max_index_timestep,0])
          y_shift = int(pred_y - mean_field[max_index_timestep,1])
          
          shifted_comp_in = np.zeros((imx, imy), dtype="uint8")
          shifted_comp = shifted_comp_in
          
          # compute the shifted instance i (timestep with the maximum size) to n
          for x in range(imx):
            for y in range(imy):
              if ts_instances[max_index_timestep+1,index_i[max_index_timestep],x,y] == i:
                x_t = x + int(x_shift)
                y_t = y + int(y_shift)
                if x_t>=0 and x_t<imx and y_t>=0 and y_t<imy:
                  shifted_comp[x_t,y_t] = 1
          
          # compute the overlapping areas
          max_iou = 0
          max_index = 0
      
          # compute the overlapping areas
          for j in range(len(flag_comp_num_n_in)):
            
            # j has to be existent and j has not a number
            if flag_comp_num_n[j] == 0:
            
              # class instance j (n)
              class_n = int(instances_in[0,j,:,:].max() // 1000)
            
              # compute only, if instance i (n-1) and instance j (n) have the same class
              if class_n_1 == class_n:
              
                intersection = 0
                union = 0
              
                for x in range(imx):
                  for y in range(imy):
                    if shifted_comp[x,y] == 1 and instances[0,j,x,y] > 0:
                      intersection = intersection + 1
                    if shifted_comp[x,y] == 1 or instances[0,j,x,y] > 0:
                      union = union + 1
                
                if union > 0:
                  if (intersection / union ) > max_iou:
                    max_iou = (intersection / union )
                    max_index = j
              
          # it is a match if the number of overlapping pixel is 35 percent of instance j (n-1) and max overlapping
          if max_iou >= percentage:
            ts_instances_in[ 0,max_index,instances_in[0,max_index]>0 ] = i
            if index_i[0] > -2:
              flag_comp_num_n_1[index_i[0]] = 1
            flag_comp_num_n[max_index] = 1
            print("timeseries and overlapping match with percentage: number of pixel, index i and j, class", max_iou, i, max_index, class_n_1)
          
  return ts_instances_in, flag_comp_num_n_in, flag_comp_num_n_1_in
      

"""
algorithm for instances that can overlap (dim: num instances, height, weight)
"""
def compute_ts_instances(vid, num_img): 
  
  cdef int epsilon, num_reg, reg_steps, max_inst_per_image, num_inst_n, imx, imy, n, m, i, j, counter, max_index, counter_doubled
  
  cdef float percentage, counter_new, max_instance_number
  
  cdef np.ndarray flag_comp_num_n_1_in
  cdef np.ndarray ts_instances_in
  cdef np.ndarray instances_in
  cdef np.ndarray obj_id_n
  cdef short int[:] flag_comp_num_n_1
  cdef float[:] inx_field_counter

  epsilon = CONFIG.EPS_MATCHING
  num_reg = CONFIG.NUM_REG_MATCHING 
  eps_time = epsilon/2
  percentage = 0.35
  
  max_inst_per_image = 200
  max_instance_number = 0
  
  # [no instances, imx, imy]
  seg_0  = instances_small_load(vid,0)
  imx = seg_0.shape[1]
  imy = seg_0.shape[2]
  
  for n in range(num_img):
    
    if os.path.isfile( get_save_path_time_series_instances_i( vid, n, epsilon, num_reg ) ):  
      print("skip image", n)
      
      instances_tmp = np.array( time_series_instances_load( vid, n, epsilon, num_reg) , dtype="int16" )
      instances_tmp[instances_tmp<0] *= -1
      instances_tmp = instances_tmp % 10000
      if len(np.unique(instances_tmp)) != 0:
        max_instance_number = max(max_instance_number, instances_tmp.max())
      
    else:
      
      # background: obj_ids=0, (gt: ignore regions: obj_ids=10,000)
      # load instances and instances (n-1 to n-5)
      instances_in = np.zeros((num_reg+1, max_inst_per_image, imx, imy), dtype="uint16")
      instances = instances_in
      instances_tmp = np.array( instances_small_load( vid, n ), dtype="uint32" )
      instances_in[0,:instances_tmp.shape[0],:,:] = instances_tmp
      num_inst_n = instances_tmp.shape[0] 
      
      ts_instances_in = np.zeros((num_reg+1, max_inst_per_image, imx, imy), dtype="uint32")
      ts_instances = ts_instances_in
      for m in range(1,num_reg+1):
        if n >= m: 
          instances_tmp = np.array( instances_small_load( vid, n-m ), dtype="uint16" )
          instances_in[m,:instances_tmp.shape[0],:,:] = instances_tmp
          instances_tmp = np.array( time_series_instances_load( vid, n-m, epsilon, num_reg), dtype="int16" )
          instances_tmp[instances_tmp<0] *= -1
          ts_instances_in[m,:instances_tmp.shape[0],:,:] = instances_tmp
          ts_instances_in[m] = ts_instances_in[m] % 10000

      # specialcase image 0
      if n == 0:
        
        # ts_instances : background 0, instances 1,2,...
        counter = 1
        for j in range(num_inst_n):
          ts_instances_in[ 0, j, instances_in[0,j]>0 ] = counter
          counter += 1
        
      else:  
        
        # 0 : instance j is unused, 1 : used or nonexistent  -> index 0 = instance matrix 0
        flag_comp_num_n_in = np.zeros( (max_inst_per_image), dtype="int16" )
        flag_comp_num_n = flag_comp_num_n_in
        
        for j in range(max_inst_per_image):
          if instances_in[0,j,:,:].max() == 0:  
            flag_comp_num_n[j] = 1
            
        # calculate geometric centers of instances in n
        mean_n_in = np.zeros((num_inst_n, 2), dtype="float32" )
        mean_n = mean_n_in
        
        for j in range(num_inst_n):
          if flag_comp_num_n_in[j] == 0:
            counter = 0
            for x in range(imx):
              for y in range(imy):
                if instances[0,j,x,y] > 0:
                  mean_n[j,0] += x
                  mean_n[j,1] += y
                  counter = counter + 1
            mean_n[j,0] /= counter
            mean_n[j,1] /= counter 
      
        # 0 : instance i is unused, 1 : used or nonexistent  -> index 0 = instance matrix 0
        flag_comp_num_n_1_in = np.zeros( (max_inst_per_image), dtype="int16" )
        flag_comp_num_n_1 = flag_comp_num_n_1_in
      
        # compute inx_field: start with the biggist instance i and get smaller
        inx_field_counter = np.zeros((max_inst_per_image), dtype="float32" )
        inx_field = np.zeros((max_inst_per_image), dtype="int16" )
        for m in range( 0, max_inst_per_image ):
          inx_field_counter[m] = np.count_nonzero(ts_instances_in[1,m,:,:]>0)
      
        for m in range(max_inst_per_image):
          max_index = int(np.argmax(inx_field_counter))
          inx_field[m] = max_index
          inx_field_counter[max_index] = -1
        
        # from here on i describes the index from num_instances (not the id)
        
        for i in inx_field:
          if ts_instances_in[1,i,:,:].max() == 0:
            flag_comp_num_n_1[i] = 1
        
        #### geometric center matching
        print("start geometric center matching")   
        for i in inx_field:
          
          ## min three images and the instance i (n-2) has to be existent
          if n >= 2 and np.count_nonzero(ts_instances_in[2,:,:,:]==ts_instances_in[1,i,:,:].max()) > 0:
            
            ts_instances_in, flag_comp_num_n_in, flag_comp_num_n_1_in = shift_overlap_distance(i, imx, imy, flag_comp_num_n_in, flag_comp_num_n_1_in, ts_instances_in, instances_in, mean_n_in, percentage, epsilon)
                      
          else:

            ts_instances_in, flag_comp_num_n_in, flag_comp_num_n_1_in = shift_distance_simplified(i, imx, imy, flag_comp_num_n_in, flag_comp_num_n_1_in, ts_instances_in, instances_in, mean_n_in, epsilon)
                    
        #### overlapping matching:
        print("start overlapping matching")
        for i in inx_field:
          
          ts_instances_in, flag_comp_num_n_in, flag_comp_num_n_1_in = overlap(i, imx, imy, flag_comp_num_n_in, flag_comp_num_n_1_in, ts_instances_in, instances_in, percentage)
        
        if n >=  3:
          #### timeseries matching
          print("start timeseries matching")
          if n < num_reg:
            reg_steps = n 
          else:
            reg_steps = num_reg
          
          # all ids of instances in frames n-1 to n-num_reg, save in list inst_n_i_list
          tmp_inst_n_i_list = []
          for c in range(1,reg_steps+1):
            tmp_inst_n_i_list.extend( np.unique(ts_instances_in[c])[1:] )
          inst_n_i_list = np.unique(tmp_inst_n_i_list)
          
          for i in inst_n_i_list:
            
            ts_instances_in, flag_comp_num_n_in, flag_comp_num_n_1_in = regression_distance_overlap(i, imx, imy, flag_comp_num_n_in, flag_comp_num_n_1_in, ts_instances_in, instances_in, mean_n_in, eps_time, percentage, reg_steps)
              
        #### remaining numbers 
        #last number which is given to a instance
        print("start remaining numbers")    
        max_instance_number = max(max_instance_number, ts_instances_in[1].max())
        counter_new = max_instance_number
        for j in range(num_inst_n):
          
          if flag_comp_num_n_in[j] == 0:
            counter_new = counter_new + 1
            ts_instances_in[ 0,j,instances_in[0,j]>0 ] = counter_new
            
      for j in range(num_inst_n):
        ts_instances_in[ 0, j, instances_in[0,j]>=1000 ] += 10000 
        ts_instances_in[ 0, j, instances_in[0,j]>=2000 ] += 10000
      
      # interior and boundary
      ts_instances_finish = ts_instances_in[0, 0:num_inst_n]
      
      ts_instances_finish = np.asarray( ts_instances_finish, dtype="int16" )
      
      for j in range(ts_instances_finish.shape[0]):
        
        inst_j = ts_instances_finish[j].copy()
        inst_j[inst_j>0] = 1
        # addition of 8 neigbours for every pixel (not included image boundary)
        inst_j_small = inst_j[1:ts_instances_finish.shape[1]-1,1:ts_instances_finish.shape[2]-1] \
                     + inst_j[0:ts_instances_finish.shape[1]-2,1:ts_instances_finish.shape[2]-1] \
                     + inst_j[2:ts_instances_finish.shape[1],1:ts_instances_finish.shape[2]-1] \
                     + inst_j[1:ts_instances_finish.shape[1]-1,2:ts_instances_finish.shape[2]] \
                     + inst_j[1:ts_instances_finish.shape[1]-1,0:ts_instances_finish.shape[2]-2] \
                     + inst_j[0:ts_instances_finish.shape[1]-2,0:ts_instances_finish.shape[2]-2] \
                     + inst_j[2:ts_instances_finish.shape[1],0:ts_instances_finish.shape[2]-2] \
                     + inst_j[0:ts_instances_finish.shape[1]-2,2:ts_instances_finish.shape[2]] \
                     + inst_j[2:ts_instances_finish.shape[1],2:ts_instances_finish.shape[2]]
        # 1 for interior, 0 for boundary and background
        inst_j_small = (inst_j_small == 9 ) 
        # 0 background, 1 boundary, 2 interior
        inst_j[1:ts_instances_finish.shape[1]-1,1:ts_instances_finish.shape[2]-1] += inst_j_small 
        
        ts_instances_finish[j,inst_j==1] *= -1
        
      time_series_instances_dump( ts_instances_finish, vid, n, epsilon, num_reg )  
      print("finished image", n)


def compute_matches_gt_pred( np.ndarray gt_in, np.ndarray instances_in, np.ndarray score, float iou_threshold = 0.5, int flag_double = 0 ):
  
  cdef int m, k, max_index, x, y, I, U, idx
  cdef float tmp_max_iou
  
  cdef short int[:,:] gt
  cdef short int[:,:] gt_class
  cdef short int[:,:,:] instances_id
  
  gt_in_copy = gt_in.copy()
  gt = gt_in_copy
  gt_class_in = gt_in_copy // 1000
  
  instances_in[instances_in<0] *= -1
  instances_id_in = instances_in  % 10000
  instances_id = instances_id_in
  
  gt_id_list = []
  for k in np.unique(gt_in_copy):
    if (k != 0) and (k != 10000):
      gt_id_list.append(k)
  
  # gt_match: for each GT instance (sorted) it has the id (1,2,.) of the matched predicted instance
  # pred_match: for each predicted instance (index-wise), it has the id (1000,.) of the matched ground truth
  gt_match = np.zeros((len(gt_id_list)), dtype="int16" ) -1
  pred_match = np.zeros((instances_in.shape[0]), dtype="int16" ) -1
  
  ind_field_counter = np.zeros((instances_in.shape[0]), dtype="float32" )
  
  # sort instance predictions by size
  for m in range( 0, instances_in.shape[0] ):
    ind_field_counter[m] = score[m] #np.count_nonzero(instances_id_in[m]>0)
    
  for m in range( 0, instances_in.shape[0] ):
    
    max_index = int(np.argmax(ind_field_counter))
    class_inst = int(instances_in[max_index].max() // 10000)
    
    tmp_max_gt_id = -1
    tmp_max_iou = 0
    
    for k, idx in zip(gt_id_list, range(len(gt_id_list))):
      
      if np.sum(gt_in_copy[gt_in_copy==k]) > 0:
      
        if class_inst == gt_class_in[gt_in_copy==k].max():
          #print(max_index, k, tmp_max_iou)
          I = 0
          U = 0
          
          for x in range(instances_in.shape[1]):
            for y in range(instances_in.shape[2]):
              
              if instances_id[max_index,x,y] > 0 and gt[x,y] == k:
                I += 1
                U += 1
              
              elif instances_id[max_index,x,y] > 0 or gt[x,y] == k: 
                U += 1
                
          if U > 0:
            if (float(I) / float(U)) > tmp_max_iou:
              tmp_max_iou = (float(I) / float(U))
              tmp_max_gt_id = idx
            
    if tmp_max_gt_id > -1 and tmp_max_iou >= iou_threshold:
      if flag_double == 0:
        gt_in_copy[gt_in_copy==gt_id_list[tmp_max_gt_id]] = 0
      pred_match[max_index] = gt_id_list[tmp_max_gt_id]
      gt_match[tmp_max_gt_id] = instances_id_in[max_index].max()
    ind_field_counter[max_index] = -1
  
  return pred_match, gt_match
  
  
def analyze_tracking_vid( vid, num_img, list_gt_ids, c ):

  cdef int epsilon, num_reg, imx, imy, x, y, k, i, idx_instance, idx, counter_instance, counter_gt, min_x_instance, max_x_instance, min_y_instance, max_y_instance, min_x_gt, max_x_gt, min_y_gt, max_y_gt
  
  cdef float mean_x_instance, mean_y_instance, mean_x_gt, mean_y_gt
  
  cdef short int[:,:,:] instances_id
  cdef short int[:,:] gt
  
  epsilon = CONFIG.EPS_MATCHING
  num_reg = CONFIG.NUM_REG_MATCHING 
  
  seg_0 = time_series_instances_load( vid, 0, epsilon, num_reg)
  imx = seg_0.shape[1]
  imy = seg_0.shape[2]
    
  tracking_metrics = { "num_frames": np.zeros((1)), "gt_obj": np.zeros((1)), "fp": np.zeros((1)), "misses": np.zeros((1)), "mot_a": np.zeros((1)), "dist_bb": np.zeros((1)), "dist_geo": np.zeros((1)), "matches": np.zeros((1)), "mot_p_bb": np.zeros((1)), "mot_p_geo": np.zeros((1)), "far": np.zeros((1)), "f_measure": np.zeros((1)), "precision": np.zeros((1)), "recall": np.zeros((1)), "switch_id": np.zeros((1)), "num_gt_ids": np.zeros((1)), "mostly_tracked": np.zeros((1)), "partially_tracked": np.zeros((1)), "mostly_lost": np.zeros((1)), "switch_tracked": np.zeros((1)) }

  tracking_unique_gt = { "last_match_id": np.ones((list_gt_ids.shape[0]))*-1, "num_tracked": np.zeros((list_gt_ids.shape[0])), "lifespan_gt_id": np.zeros((list_gt_ids.shape[0])), "last_tracked": np.ones((list_gt_ids.shape[0]))*-1 }
  
  for n in range(num_img):
    
    instances_in = np.array( time_series_instances_load( vid, n, epsilon, num_reg), dtype="int16" )
    instances_in[instances_in<0] *= -1
    if c == 1:
      instances_in[instances_in>=20000] = 0
    elif c == 2:
      instances_in[instances_in<20000] = 0
    if np.sum(instances_in) == 0:
      size_new_instances = 0
    else:
      size_new_instances = len(np.unique(instances_in))-1
    instances_c_in = np.zeros((size_new_instances, imx, imy), dtype="int16")
    counter = 0
    for i in range(instances_in.shape[0]):
      if np.sum(instances_in[i]) > 0:
        instances_c_in[counter] = instances_in[i]
        counter += 1
    instances_id_in = instances_c_in % 10000
    instances_id = instances_id_in
    
    gt_in = np.array( ground_truth_load(vid, n), dtype="int16")
    if c == 1:
      gt_in[gt_in>=2000] = 0
    elif c == 2:
      gt_in[gt_in<2000] = 0
    gt = gt_in
    
    score = score_small_load( vid, n )
    pred_match, gt_match = compute_matches_gt_pred(gt_in, instances_c_in, score)
    
    tracking_metrics['gt_obj'] += len(gt_match)
    tracking_metrics['fp'] += np.count_nonzero(pred_match==-1)
    tracking_metrics['misses'] += np.count_nonzero(gt_match==-1)
    tracking_metrics['matches'] += np.count_nonzero(gt_match>-1)
    
    gt_id_list = []
    for k in np.unique(gt_in):
      if (k != 0) and (k != 10000):
        gt_id_list.append(k)
        
    for k, idx in zip(gt_id_list, range(len(gt_id_list))):
      
      if gt_match[idx] > -1:
      
        idx_instance = -1
        for i in range(instances_c_in.shape[0]):
          if instances_id_in[i].max() == gt_match[idx]:
            idx_instance = i
            break
        
        mean_x_gt = 0
        mean_y_gt = 0
        counter_gt = 0
        min_x_gt = imx
        max_x_gt = 0
        min_y_gt = imy
        max_y_gt = 0
        
        mean_x_instance = 0
        mean_y_instance = 0
        counter_instance = 0
        min_x_instance = imx
        max_x_instance = 0
        min_y_instance = imy
        max_y_instance = 0
        
        for x in range(imx):
          for y in range(imy):
            
            if gt[x,y] == k:
              mean_x_gt += x
              mean_y_gt += y
              counter_gt += 1
              min_x_gt = min(min_x_gt, x)
              max_x_gt = max(max_x_gt, x)
              min_y_gt = min(min_y_gt, y)
              max_y_gt = max(max_y_gt, y)
              
            if instances_id[idx_instance,x,y] > 0:
              mean_x_instance += x
              mean_y_instance += y
              counter_instance += 1
              min_x_instance = min(min_x_instance, x)
              max_x_instance = max(max_x_instance, x)
              min_y_instance = min(min_y_instance, y)
              max_y_instance = max(max_y_instance, y)
              
        mean_x_gt /= counter_gt
        mean_y_gt /= counter_gt
        mean_x_instance /= counter_instance
        mean_y_instance /= counter_instance
        
        tracking_metrics['dist_bb'] += ( (float(min_x_gt+max_x_gt)/2 - float(min_x_instance+max_x_instance)/2)**2 + (float(min_y_gt+max_y_gt)/2 - float(min_y_instance+max_y_instance)/2)**2 )**0.5
        
        tracking_metrics['dist_geo'] += ( (mean_x_gt - mean_x_instance)**2 + (mean_y_gt - mean_y_instance)**2 )**0.5
         
    for k in range(len(list_gt_ids)):  
      
      if list_gt_ids[k] in gt_id_list:
        tracking_unique_gt['lifespan_gt_id'][k] += 1
        
        if tracking_unique_gt['last_tracked'][k] == -1 and list_gt_ids[k] in pred_match:
          tracking_unique_gt['last_tracked'][k] = 1
          
        elif tracking_unique_gt['last_tracked'][k] == 1 and list_gt_ids[k] not in pred_match:
          tracking_unique_gt['last_tracked'][k] = 0
        
        elif tracking_unique_gt['last_tracked'][k] == 0 and list_gt_ids[k] in pred_match:
          tracking_unique_gt['last_tracked'][k] = 1
          tracking_metrics['switch_tracked'] += 1
           
      if list_gt_ids[k] in pred_match:
        tracking_unique_gt['num_tracked'][k] += 1
        
        if tracking_unique_gt['last_match_id'][k] == -1:
          tracking_unique_gt['last_match_id'][k] = gt_match[gt_id_list.index(list_gt_ids[k])]
        
        elif tracking_unique_gt['last_match_id'][k] != gt_match[gt_id_list.index(list_gt_ids[k])]:
          
          tracking_unique_gt['last_match_id'][k] = gt_match[gt_id_list.index(list_gt_ids[k])]
          tracking_metrics['switch_id'] += 1
        
  tracking_metrics['num_frames'] += num_img
  
  tracking_metrics['mot_a'] = 1 - ((tracking_metrics['misses'] + tracking_metrics['fp'] + tracking_metrics['switch_id'])/tracking_metrics['gt_obj'])
  
  tracking_metrics['mot_p_bb'] = tracking_metrics['dist_bb'] / tracking_metrics['matches']
  
  tracking_metrics['mot_p_geo'] = tracking_metrics['dist_geo'] / tracking_metrics['matches']
  
  tracking_metrics['far'] = tracking_metrics['fp'] / tracking_metrics['num_frames'] * 100
  
  tracking_metrics['f_measure'] = (2 * tracking_metrics['matches']) / (2 * tracking_metrics['matches'] + tracking_metrics['misses'] + tracking_metrics['fp'])
  
  tracking_metrics['precision'] = tracking_metrics['matches'] / ( tracking_metrics['matches'] + tracking_metrics['fp'])
  
  tracking_metrics['recall'] = tracking_metrics['matches'] / ( tracking_metrics['matches'] + tracking_metrics['misses'])
  
  tracking_metrics['num_gt_ids'] += len(list_gt_ids)
  
  for k in range(len(list_gt_ids)):
    
    quotient = tracking_unique_gt['num_tracked'][k] / tracking_unique_gt['lifespan_gt_id'][k]
    
    if quotient >= 0.8:
      tracking_metrics['mostly_tracked'] += 1
    elif quotient >= 0.2:
      tracking_metrics['partially_tracked'] += 1
    else:
      tracking_metrics['mostly_lost'] += 1
      
  pickle.dump( tracking_metrics, open( CONFIG.ANALYZE_TRACKING_DIR + 'tracking_metrics_' + vid + '_class' + str(c) + ".p", "wb" ) )  
  
  
def entropy( probs ):
  E = np.sum( np.multiply( probs, np.log(probs+np.finfo(np.float32).eps) ) , axis=-1) / np.log(1.0/probs.shape[-1])
  return np.asarray( E, dtype="float32" )


def probdist( probs ):
  cdef int i, j
  arrayA = np.asarray(np.argsort(probs,axis=-1), dtype="uint8")
  arrayD = np.ones( probs.shape[:-1], dtype="float32" )
  cdef float[:,:,:] P = probs
  cdef float[:,:]   D = arrayD
  cdef char[:,:,:]  A = arrayA
  for i in range( arrayD.shape[0] ):
    for j in range( arrayD.shape[1] ):
      D[i,j] = ( 1 - P[ i, j, A[i,j,-1] ] + P[ i, j, A[i,j,-2] ] )
  return arrayD


def variation_ratio( probs ):
  cdef int i, j
  arrayA = np.asarray(np.argsort(probs,axis=-1), dtype="uint8")
  arrayD = np.ones( probs.shape[:-1], dtype="float32" )
  cdef float[:,:,:] P = probs
  cdef float[:,:]   D = arrayD
  cdef char[:,:,:]  A = arrayA
  for i in range( arrayD.shape[0] ):
    for j in range( arrayD.shape[1] ):
      D[i,j] = ( 1 - P[ i, j, A[i,j,-1] ] )
  return arrayD


def comp_time_series_metrics( instances_in, probs_in, score, gt_in, max_instances ): 
  
  cdef int imx, imy, imc, counter, x, y, i, ic, n_in, n_bd, I, U, c, class_j, class_k, index
  
  cdef short int[:,:,:] instances
  cdef short int[:,:,:] instances_id
  cdef short int[:,:] gt
  cdef short int[:,:] gt_class
  
  imx = gt_in.shape[0]
  imy = gt_in.shape[1]
  imc = probs_in.shape[-1]
  
  instances_in = np.asarray( instances_in, dtype="int16" )
  instances = instances_in
  instances_in_tmp = instances_in.copy()
  instances_in_tmp[instances_in_tmp<0] *= -1
  instances_id_in = instances_in_tmp  % 10000
  instances_id_in[instances_in<0] *= -1
  instances_id = instances_id_in
  instances_class_in = instances_in_tmp // 10000

  probs_in = np.asarray( probs_in, dtype="float32" )
  probs  = probs_in
  
  gt_in = np.asarray( gt_in, dtype="int16" )
  gt = gt_in
  gt_class_in = gt_in // 1000
  gt_class = gt_class_in
  
  if CONFIG.FLAG_OBJ_SEG == 0:
    pred_match, _ = compute_matches_gt_pred(gt_in, instances_in, score)
  elif CONFIG.FLAG_OBJ_SEG == 1:
    pred_match, _ = compute_matches_gt_pred(gt_in, instances_in, score, 0, 1)
  
  if CONFIG.MODEL_NAME == 'mask_rcnn':
    entropy_instances = np.zeros(instances_in.shape)
    probdist_instances = np.zeros(instances_in.shape)
    variation_ratio_instances = np.zeros(instances_in.shape)
  
    for m in range(instances_in.shape[0]):
      entropy_instances[m] = entropy(probs_in[m])
      probdist_instances[m] = probdist(probs_in[m])
      variation_ratio_instances[m] = variation_ratio(probs_in[m])
    
    heatmaps = { "E": entropy_instances, "M": probdist_instances, "V": variation_ratio_instances } 
  
    timeseries_metrics = { "iou": list([]), "iou0": list([]), "class": list([]), "mean_x": list([]), "mean_y": list([]), "score": list([]) } 
  
  elif CONFIG.MODEL_NAME == 'yolact':
    
    heatmaps = { }
    
    timeseries_metrics = { "iou": list([]), "iou0": list([]), "class": list([]), "mean_x": list([]), "mean_y": list([]), "score": list([]), "E": list([]), "E_rel": list([]) } 
    
  for m in list(heatmaps)+["S"]:
    timeseries_metrics[m          ] = list([])
    timeseries_metrics[m+"_in"    ] = list([])
    timeseries_metrics[m+"_bd"    ] = list([])
    timeseries_metrics[m+"_rel"   ] = list([])
    timeseries_metrics[m+"_rel_in"] = list([])
    
  for i in range(imc):
    timeseries_metrics['cprob'+str(i)] = list([])
    
  # all arrays have the same lenght and empty instances get the vaulue 0 forall metrics
  for i in range( 1, max_instances+1 ):
    
    for m in timeseries_metrics:
      timeseries_metrics[m].append( 0 )
      
    index = -1
    for m in range(instances_in.shape[0]):
      if instances_id_in[m,:,:].min() == -i:
        index = m
        break
    
    if index > -1:
      
      n_in = 0
      n_bd = 0
      I = 0
      U = 0
      
      # class component i (n-1)
      c = int(instances_class_in[index,:,:].max())
      
      for x in range(imx):
        for y in range(imy):
          if instances_id[index,x,y] != 0 and gt[x,y] != 10000:
            if instances_id[index,x,y] == i:
              for h in heatmaps:
                timeseries_metrics[h+"_in"][-1] += heatmaps[h][index,x,y]
              n_in += 1
            elif instances_id[index,x,y] == -i:
              for h in heatmaps:
                timeseries_metrics[h+"_bd"][-1] += heatmaps[h][index,x,y]
              n_bd += 1
            if CONFIG.MODEL_NAME == 'mask_rcnn':
              for ic in range(imc):
                timeseries_metrics["cprob"+str(ic)][-1] += probs[index,x,y,ic]
            timeseries_metrics["mean_x"][-1] += x
            timeseries_metrics["mean_y"][-1] += y
            
          if instances_id[index,x,y] != 0 and gt[x,y] == pred_match[index]:
            I += 1
            U += 1
          elif instances_id[index,x,y] != 0 or gt[x,y] == pred_match[index]:
            U += 1

      # compute all timeseries_metrics
      timeseries_metrics["class"   ][-1] = c
      timeseries_metrics["score"   ][-1] = score[index]   
      if U > 0:
        # IoU
        timeseries_metrics["iou"     ][-1] = float(I) / float(U)
        timeseries_metrics["iou0"    ][-1] = 0 if (float(I) / float(U))>=0.5 else 1
      else:
        timeseries_metrics["iou"     ][-1] = 0
        timeseries_metrics["iou0"    ][-1] = 1
        
      if n_bd > 0:
        timeseries_metrics["S"       ][-1] = n_in + n_bd
        timeseries_metrics["S_in"    ][-1] = n_in
        timeseries_metrics["S_bd"    ][-1] = n_bd
        timeseries_metrics["S_rel"   ][-1] = float( n_in + n_bd ) / float(n_bd)
        timeseries_metrics["S_rel_in"][-1] = float( n_in ) / float(n_bd)
        timeseries_metrics["mean_x"][-1] /= ( n_in + n_bd )
        timeseries_metrics["mean_y"][-1] /= ( n_in + n_bd )
        
        if CONFIG.MODEL_NAME == 'mask_rcnn':
          for nc in range(imc):
            timeseries_metrics["cprob"+str(nc)][-1] /= ( n_in + n_bd )
        elif CONFIG.MODEL_NAME == 'yolact':
          for nc in range(imc):
            timeseries_metrics["cprob"+str(nc)][-1] = probs[index, nc]
          timeseries_metrics["E"    ][-1] = entropy(probs[index])
          timeseries_metrics["E_rel"][-1] = timeseries_metrics["E"][-1] * timeseries_metrics["S_rel"][-1]
        
        for h in heatmaps:
          timeseries_metrics[h][-1] = (timeseries_metrics[h+"_in"][-1] + timeseries_metrics[h+"_bd"][-1]) / float( n_in + n_bd )
          if ( n_in > 0 ):
            timeseries_metrics[h+"_in"][-1] /= float(n_in)
          timeseries_metrics[h+"_bd"][-1] /= float(n_bd)
            
          timeseries_metrics[h+"_rel"   ][-1] = timeseries_metrics[h      ][-1] * timeseries_metrics["S_rel"   ][-1]
          timeseries_metrics[h+"_rel_in"][-1] = timeseries_metrics[h+"_in"][-1] * timeseries_metrics["S_rel_in"][-1]
    
    else:
      # all metrics are 0; only the gt is -1
      timeseries_metrics["iou"     ][-1] = -1
      timeseries_metrics["iou0"    ][-1] = -1
      
  return timeseries_metrics


def shifted_iou( inst1_in, inst2_in):
  
  cdef int imx, imy, x, y, counter, x_shift, y_shift, x_t, y_t 
  cdef float mean_x_1, mean_y_1, mean_x_2, mean_y_2, intersection, union, iou
  
  cdef char[:,:] inst1
  cdef char[:,:] inst2
  cdef char[:,:] shifted_comp
  
  inst1_in = np.asarray( inst1_in, dtype="uint8" )
  inst1 = inst1_in
  inst2_in = np.asarray( inst2_in, dtype="uint8" )
  inst2 = inst2_in
  
  imx = inst1_in.shape[0]
  imy = inst1_in.shape[1]
  
  mean_x_1 = 0
  mean_y_1 = 0
  counter = 0
  for x in range(imx):
    for y in range(imy):
      if inst1[x,y] == 1:
        mean_x_1 += x
        mean_y_1 += y
        counter = counter + 1
  mean_x_1 /= counter
  mean_y_1 /= counter
  
  mean_x_2 = 0
  mean_y_2 = 0
  counter = 0
  for x in range(imx):
    for y in range(imy):
      if inst2[x,y] == 1:
        mean_x_2 += x
        mean_y_2 += y
        counter = counter + 1
  mean_x_2 /= counter
  mean_y_2 /= counter
  
  x_shift = int(mean_x_1 - mean_x_2)
  y_shift = int(mean_y_1 - mean_y_2)
  
  shifted_comp_in = np.zeros((imx, imy), dtype="uint8")
  shifted_comp = shifted_comp_in
  
  # compute the shifted instance
  for x in range(imx):
    for y in range(imy):
      if inst2[x,y] == 1:
        x_t = x + x_shift
        y_t = y + y_shift
        if x_t>=0 and x_t<imx and y_t>=0 and y_t<imy:
          shifted_comp[x_t,y_t] = 1
  
  intersection = 0
  union = 0
  for x in range(imx):
    for y in range(imy):
      if shifted_comp[x,y] == 1 and inst1[x,y] > 0:
        intersection = intersection + 1
      if shifted_comp[x,y] == 1 or inst1[x,y] > 0:
        union = union + 1
        
  if union > 0:     
    iou = intersection / union 
  return iou
  
  


