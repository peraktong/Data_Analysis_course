import sys, platform, os
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
import pickle
print('Using CAMB installed at %s'%(os.path.realpath(os.path.join(os.getcwd(),'..'))))
#uncomment this if you are running remotely and want to keep in synch with repo changes
#if platform.system()!='Windows':
#    !cd $HOME/git/camb; git pull github master; git log -1
sys.path.insert(0,os.path.realpath(os.path.join(os.getcwd(),'..')))
import camb
from camb import model, initialpower

####
# save path:


# Let's read and plot Behroozi
plot_path = "/Users/caojunzhi/Desktop/NYU/2018 Spring/Data_analysis/HW1/plots/"


