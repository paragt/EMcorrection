import time
import glob
#import mahotas
import numpy as np
import scipy
import scipy.misc
import random
from keras.models import Model, Sequential, model_from_json
from PIL import Image
import h5py
from scipy.ndimage import gaussian_filter, label, find_objects, distance_transform_edt
import pickle
from rotate3d import *
import itertools
import pdb
from scipy.spatial import KDTree
import os

def relabel_from_one(a):
    labels = np.unique(a)
    labels0 = labels[labels!=0]
    m = labels.max()
    if m == len(labels0): # nothing to do, already 1...n labels
        return a, labels, labels
    forward_map = np.zeros(m+1, int)
    forward_map[labels0] = np.arange(1, len(labels0)+1)
    if not (labels == 0).any():
        labels = np.concatenate(([0], labels))
    inverse_map = labels
    return forward_map[a], forward_map, inverse_map




class GenerateData:
    
    def __init__(self,train_trial,train_imagedir, train_segname, train_skfile, patchSize, patchZ):
        #pdb.set_trace() 
        fid = h5py.File(train_segname)
        if 'stack' in fid.keys() : self.seg = np.array(fid['stack'])
        elif 'main' in fid.keys(): self.seg = np.array(fid['main'])   
        fid.close()
        print 'seg dtype: ', self.seg.dtype

        
        self.candidates = pickle.load(open(os.path.join(train_skfile),'rb'))        
        
        allfiles = sorted(glob.glob(os.path.join(train_imagedir+'/*.png')))
        
        img = np.array(Image.open(allfiles[0])) #read the first image to get imformation about the shape
        
        self.grayImages    = np.zeros((len(allfiles), img.shape[0], img.shape[1])).astype(np.float32)

        for ii,filename in enumerate(allfiles):
            print filename
            img = np.array(Image.open(filename)).astype(np.float32)
            
            img_normalize_toufiq = img *1.0/255

            self.grayImages[ii,:,:] = img_normalize_toufiq
            
            
        self.patchSize = patchSize
        self.patchZ = patchZ
        
        self.separate_class()

    def inside_margin2(self, pentry ):

        depth, height, width = self.seg.shape

        flag = (pentry['z'] >= (self.patchZ/2)) * (pentry['z'] < (depth-(self.patchZ/2))) * (pentry['y'] >= (self.patchSize/2)) * (pentry['y'] < (height-(self.patchSize/2))) * (pentry['x'] >= (self.patchSize/2)) * (pentry['x'] < (width-(self.patchSize/2)))

        return flag



    def separate_class(self):
        
        #pdb.set_trace()
        
        self.positive_examples=[]
        self.negative_examples=[]
        for ii in range(len(self.candidates)):
            if not (self.inside_margin2(self.candidates[ii])):
                continue  

            if self.candidates[ii]['label']==1:
                self.positive_examples.append(self.candidates[ii])
            else:
                self.negative_examples.append(self.candidates[ii])

        print 'num positive examples: ',len(self.positive_examples)
        print 'num negative examples: ',len(self.negative_examples)
             
    def compute_box(self, pentry, translate_endpt=0):# pre, post, ex,ey,ez):
        
        pre = pentry['seg']
        post = pentry['nbr']
        ex = pentry['x']
        ey = pentry['y']
        ez = pentry['z']
        seg = self.seg
        szw = self.patchSize/2
        szd = self.patchZ/2
        
        depth, height, width = seg.shape
        if ex < szw or ex > (width-szw):
            return np.array([])
        if ey < szw or ey > (height-szw):
            return np.array([])
        if ez < szd or ez > (depth-szd):
            return np.array([])

        if translate_endpt==1:
            #pdb.set_trace()
            z_disp = random.sample(range(-1,1), 1)[0]
            y_disp = random.sample(range(-10,10), 1)[0]
            x_disp = random.sample(range(-10,10), 1)[0]

            if (ez+z_disp)>szd and (ez+z_disp) < (depth-szd):
                ez = ez+z_disp
            if (ey+y_disp)>szw and (ey+y_disp) < (height-szw):
                ey = ey+y_disp
            if (ex+x_disp)>szw and (ex+x_disp) < (width-szw):
                ex = ex+x_disp
        

        box = seg[ez-szd:ez+szd, ey-szw:ey+szw, ex-szw:ex+szw]
        box_img = self.grayImages[ez-szd:ez+szd, ey-szw:ey+szw, ex-szw:ex+szw]
        #box_membrane = self.membrane[:,ez-szd:ez+szd, ey-szw:ey+szw, ex-szw:ex+szw]
        
        #sz_diff = szw - self.patchSize/2
        #box = box[:, sz_diff:-sz_diff, sz_diff:-sz_diff]
        #pdb.set_trace()
        box1 = (box==pre).astype(np.float32)
        box2 = (box==post).astype(np.float32)
        return box_img, box1, box2
            
            
        
    

    def get_3d_sample2(self, nsamples_batch=5, translate_endpt=0 ):

        #pdb.set_trace()
        nsamples=nsamples_batch    
        npositive = 5
        nnegative = nsamples-npositive
        
        
        input_set = np.zeros((nsamples, 3, self.patchZ, self.patchSize, self.patchSize))
        label_set = np.zeros(nsamples).astype(np.int32)
        
        random_pidx = random.sample(range(len(self.positive_examples)), npositive)
        random_nidx = random.sample(range(len(self.negative_examples)), nnegative)
        #random_pidx = range(npositive)
        #random_nidx = range(nnegative) 

        count=0
        for pidx in random_pidx:
            pentry = self.positive_examples[pidx]
            combined = self.compute_box(pentry,translate_endpt)
            
            combined = self.reflect_swap(combined)
            
            #imgvol1 = self.scale_intensity(imgvol1)
            
            combined = self.rotate3d(combined)
            
            input_set[count,:,:,:,:] = combined
            label_set[count] = pentry['label']
            count=count+1
            
        for nidx in random_nidx:
            pentry = self.negative_examples[nidx]
            combined = self.compute_box(pentry,translate_endpt)
            
            combined = self.reflect_swap(combined)
            
            #imgvol1 = self.scale_intensity(imgvol1)
            
            combined = self.rotate3d(combined)
            
            input_set[count,:,:,:,:] = combined
            label_set[count] = pentry['label']
            count=count+1
            
            
    
        #start_time = time.time()
        
        

        print "number of positive labels: "+str(np.sum(label_set>0))


        data_set = (input_set, label_set)
        

        #end_time = time.time()
        #total_time = (end_time - start_time)


        return data_set

    def rotate3d(self, input_vol):

        #pdb.set_trace()
        nrotate = random.randint(0,3)
        axis = 0 #random.randint(0,2)
        
        output_vol=[]
        for i in range(len(input_vol)):
            grayImg_set = input_vol[i]
            grayImg_set = axial_rotations(grayImg_set, rot=nrotate, ax=axis)
        
            output_vol.append([])
            output_vol[-1] = grayImg_set
            
        return output_vol
    
    def reflect_swap(self, input_vol):

        
        reflectz=random.randint(0,1)
        reflecty=random.randint(0,1)
        reflectx=random.randint(0,1)
        swapxy=random.randint(0,1)

        output_vol=[]
        for i in range(len(input_vol)):
            grayImg_set = input_vol[i]
            if reflectz:
                grayImg_set = grayImg_set[::-1,:,:]

            if reflecty:
                grayImg_set = grayImg_set[:,::-1,:]

            if reflectx:
                grayImg_set = grayImg_set[:,:,::-1]

            if swapxy:
                grayImg_set = grayImg_set.transpose((0,2,1))
            
            output_vol.append([])
            output_vol[-1] = grayImg_set
            
        return output_vol
    
    
        
    def scale_intensity(self, input_vol):
            
        grayImg_set = input_vol
            
        self.scale_low = 0.8
        self.scale_high = 1.2
        self.shift_low = -0.2
        self.shift_high = 0.2


        grayImg_set_mean = grayImg_set.mean()
        grayImg_set = grayImg_set_mean + (grayImg_set -grayImg_set_mean)*np.random.uniform(low=self.scale_low,high=self.scale_high)
        grayImg_set = grayImg_set + np.random.uniform(low=self.shift_low,high=self.shift_high)
        grayImg_set = np.clip(grayImg_set, 0.05, 0.95)
        
        return grayImg_set
    

