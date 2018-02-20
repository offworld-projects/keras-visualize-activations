import rospy, pdb
import numpy as np
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt

from read_activations import *


def actvToImgMsg(actv, bridge):
    actv = actv[:,:,0:3] # drop the alpha layer
    #if lossType == 'categorical':
    #    actv = np.exp(actv) / (np.exp(actv) + 1.0) # sigmoid output for visualization purposes
    actv *= 255. # scale first 3 channels
    stacked = actv.astype('uint8') # HxWx3 -- includes alpha channel
    new_image = cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR) # switch order from plt (rgb) to cv2 (bgr)
    img_msg = bridge.cv2_to_imgmsg(new_image, encoding='bgr8') # ros image msg expects order: h, w
    return img_msg

class KerasRos(object):
    def __init__(self, numLayers):
        # reminder: keras is not thread-safe, so we can't do inference in a callback
        self.pubs = [rospy.Publisher("ML/visualize/activations_"+str(i), Image) for i in range(numLayers)]
        self.pub = rospy.Publisher("ML/visualize/activations/all", Image)
        self.numLayers = numLayers
        self.bridge = CvBridge()

    def pubActvs(self, fullArr, actvs): 
        # actvs should be the output of stack_activations
        # order: h, w
        img_msg = actvToImgMsg(fullArr, self.bridge, '')
        self.pub.publish(img_msg)
        for idx, actv in enumerate(actvs):
            img_msg = actvToImgMsg(actv, self.bridge)
            self.pubs[idx].publish(img_msg)