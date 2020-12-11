#!/usr/bin/env python3
"""
main script executing tasks defined in global settings file
"""

from global_defs    import CONFIG
from main_functions import compute_time_series_instances, plot_time_series_instances, compute_mean_ap,\
                           analyze_tracking_algo, compute_time_series_metrics,\
                           visualize_time_series_metrics, compute_mean_ap_metrics,\
                           plot_mean_average_precision, visualize_meta_prediction,\
                           visualize_IoU_prediction, analyze_metrics, visualize_analyzed_metrics
 
 
def main():
  

  """ COMMENT:
  Compute time series instances on the basis of the tracking algorithm and save in TIME_SERIES_INST_DIR as pickle (*.p) files.
  """  
  if CONFIG.COMPUTE_TIME_SERIES_INSTANCES:
    run = compute_time_series_instances()   
    run.compute_time_series_instances_per_image()  
    

  """ COMMENT:
  For visualizing the tracking algorithm, the time series instances are used and in IMG_TIME_SERIES_DIR the resulting visualization images (*.png) are stored. 
  """  
  if CONFIG.PLOT_TIME_SERIES_INSTANCES:
    run = plot_time_series_instances()   
    run.plot_time_series_instances_per_image() 
    
  
  """ COMMENT:
  Analyze the tracking algorithm and save results in ANALYZE_TRACKING_DIR.
  """
  if CONFIG.ANALYZE_TRACKING:
    run = analyze_tracking_algo()  
    run.analyze_tracking()
    
  
  """ COMMENT:
  Compute time series metrics and save in METRICS_DIR as pickle (*.p) files.
  """    
  if CONFIG.COMPUTE_TIME_SERIES_METRICS:
    run = compute_time_series_metrics()
    run.compute_time_series_metrics_per_image()
    
  
  """ COMMENT:
  For visualizing the metrics over time, the time series metrics are used and saved in IMG_METRICS_DIR the resulting visualization images (*.png) are stored. 
  """ 
  if CONFIG.VISUALIZE_METRICS:
    run = visualize_time_series_metrics()
    run.visualize_metrics_vs_iou()                       
    run.visualize_time_series_metrics_per_component()     
    run.visualize_time_series_metrics_per_class()        
    run.plot_time_series_instances_shapes()               
    run.visualize_lifetime_mean()
    

  """ COMMENT:
  Calculate the mean average precision to evaluate the prediction of the network. 
  """
  if CONFIG.COMPUTE_MEAN_AP:
    run = compute_mean_ap()  
    run.compute_map()
    
    
  """ COMMENT:
  Calculate the mean average precision to evaluate the prediction of the network after application of meta classification.
  """
  if CONFIG.COMPUTE_MEAN_AP_METRICS:
    run = compute_mean_ap_metrics()  
    run.compute_map_metrics()
    
  
  """ COMMENT:
  Visualization of the mean average precision.
  """
  if CONFIG.PLOT_MEAN_AP:
    run = plot_mean_average_precision()  
    run.plot_mean_ap()
    
    
  """ COMMENT:
  For visualizing the rating by meta regression, the underlying metrics for the meta model need to be computed and saved in METRICS_DIR defined in "global_defs.py". In IMG_IOU_INST_DIR the resulting visualization images (*.png) are stored. Refer to paper for interpretation.
  """
  if CONFIG.VISUALIZE_REGRESSION:
    run = visualize_meta_prediction() 
    run.visualize_regression_per_image() 
    

  """ COMMENT:
  For visualizing the IoU<0.5/>=0.5 prediction by meta classification, the underlying metrics for the meta model need to be computed and saved in METRICS_DIR defined in "global_defs.py". In IMG_IOU0_INST_DIR the resulting visualization images (*.png) are stored. Refer to paper for interpretation.
  """
  if CONFIG.VISUALIZE_CLASSIFICATION:
    run = visualize_IoU_prediction() 
    run.visualize_classification_per_image() 
    
  
  """ COMMENT:
  For analyzing meta tasks performance based on the derived metrics, the underlying metrics for the meta model need to be computed and saved in METRICS_DIR defined in "global_defs.py". Results for viewing are saved in ANALYZE_DIR. The calculation results file is saved in ANALYZE_DIR/stats. 
  """
  if CONFIG.ANALYZE_METRICS:
    run = analyze_metrics() 
    run.analyze_time_series_metrics()
    
    
  """ COMMENT:
  For plotting the analyzed meta tasks ANALYZE_METRICS must have been executed for all regressions and classifications models, as well as all different training data.
  """
  if CONFIG.PLOT_ANALYZE_METRICS:
    run = visualize_analyzed_metrics() 
    run.visualize_analyzed_time_series_metrics()
    
    
    
if __name__ == '__main__':
  
  print( "===== START =====" )
  main()
  print( "===== DONE! =====" )
  
  
