#!/usr/bin/python
#
# KITTI and MOT labels
#

from collections import namedtuple


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# label and all information

Label = namedtuple('Label',['name','Id','trainId','color'])


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

kitti_labels = [
    #       name                     id    trainId   color
    Label(  'unlabeled'            ,  0 ,      10 ,  (255, 255, 255) ),
    Label(  'background'           ,  0 ,       0 ,  (255, 255, 255) ),
    Label(  'person'               , 24 ,       2 ,  (220,  20,  60) ),
    Label(  'car'                  , 26 ,       1 ,  (  0,   0, 142) ),
]



