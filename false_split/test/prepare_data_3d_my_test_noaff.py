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
    

    def __init__(self,train_imagedir, train_segname, train_skfile, patchSize, patchZ):
        #pdb.set_trace() 
        fid = h5py.File(train_segname)
        if 'stack' in fid.keys(): self.seg = np.array(fid['stack'])
        elif 'main' in fid.keys() : self.seg=np.array(fid['main']) 
        fid.close()



        self.candidates0 = pickle.load(open(os.path.join(train_skfile),'rb'))

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
            z_disp = random.sample(range(-2,2), 1)[0]
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

        #sz_diff = szw - self.patchSize/2
        #box = box[:, sz_diff:-sz_diff, sz_diff:-sz_diff]
        #pdb.set_trace()
        box1 = (box==pre).astype(np.float32)
        box2 = (box==post).astype(np.float32)
        return box_img, box1, box2




    def get_next_test_sample(self, sample_id):

        combined0 = self.compute_box(self.candidates[sample_id])
        #pdb.set_trace()     
        total_aug=16

        transformed = np.zeros((total_aug, len(combined0), combined0[0].shape[0],combined0[0].shape[1],combined0[0].shape[2]),dtype=np.float32)
        transformed[0,...] = combined0
        #reflectz, reflecty, reflectx, swapxy
        ch = 1
        if total_aug>7:
            transformed[ch,...] = self.reflect_swap(combined0, 0, 0, 0, 1)
            ch = ch+1
            transformed[ch,...] = self.reflect_swap(combined0, 0, 1, 0, 0)
            ch = ch+1
            transformed[ch,...] = self.reflect_swap(combined0, 0, 1, 0, 1)
            ch = ch+1
            transformed[ch,...] = self.reflect_swap(combined0, 1, 0, 0, 0)
            ch = ch+1
            transformed[ch,...] = self.reflect_swap(combined0, 1, 0, 0, 1)
            ch = ch+1
            transformed[ch,...] = self.reflect_swap(combined0, 1, 1, 0, 0)
            ch = ch+1
            transformed[ch,...] = self.reflect_swap(combined0, 1, 1, 0, 1)
            ch = ch+1

        if total_aug>15:
            transformed[ch,...] = self.reflect_swap(combined0, 0, 1, 1, 0)
            ch = ch+1
            transformed[ch,...] = self.reflect_swap(combined0, 0, 1, 1, 1)
            ch = ch+1
            transformed[ch,...] = self.reflect_swap(combined0, 0, 0, 1, 0)
            ch = ch+1
            transformed[ch,...] = self.reflect_swap(combined0, 0, 0, 1, 1)
            ch = ch+1
            transformed[ch,...] = self.reflect_swap(combined0, 1, 0, 1, 0)
            ch = ch+1
            transformed[ch,...] = self.reflect_swap(combined0, 1, 0, 1, 1)
            ch = ch+1
            transformed[ch,...] = self.reflect_swap(combined0, 1, 1, 1, 0)
            ch = ch+1
            transformed[ch,...] = self.reflect_swap(combined0, 1, 1, 1, 1)
        '''
        transformed = []
        transformed.append([])
        transformed[-1] = combined0
        for reflectz in range(2):
            for reflecty in range(2):
                for reflectx in range(1):
                    for swapxy in range(2):
                        #for nrotate in range(4):
                        combined = self.reflect_swap(combined0, reflectz, reflecty, reflectx, swapxy)
                        #combined = self.rotate3d(combined, nrotate)
                        transformed.append([])
                        transformed[-1]=combined
        '''   
        label = self.candidates[sample_id]['label']

        seg = self.candidates[sample_id]['seg']
        nbr = self.candidates[sample_id]['nbr']
        seg_gt = self.candidates[sample_id]['seg_gt']
        nbr_gt = self.candidates[sample_id]['nbr_gt']
	
        data_set = (transformed, label, seg, nbr)
        


        return data_set
    
    def set_prediction(self,sample_id, pred):
	    self.candidates[sample_id]['pred']=pred

    def save_predictions(self,savename):
        pickle.dump(self.candidates, open(savename,'wb'))

    def inside_margin(self, contact_loc):
        
        depth, height, width = self.segvol.shape
        
        zedge=0
        if contact_loc[0] < self.patchZ/2 or contact_loc[0]> (depth-(self.patchZ/2)):
            zedge=1
        yedge=0    
        if contact_loc[1] < self.patchSize/2 or contact_loc[1]> (height-(self.patchSize/2)):
            yedge = 1
        xedge = 0
        if contact_loc[2] < self.patchSize/2 or contact_loc[2]> (width-(self.patchSize/2)):
            xedge = 1
        
        sum_edge = yedge +xedge
        
        if zedge>0 or sum_edge>0:
            return False
        else:
            return True
        
    def inside_margin2(self, pentry ):
        
        depth, height, width = self.seg.shape
        
        flag = (pentry['z'] >= (self.patchZ/2)) * (pentry['z'] < (depth-(self.patchZ/2))) * (pentry['y'] >= (self.patchSize/2)) * (pentry['y'] < (height-(self.patchSize/2))) * (pentry['x'] >= (self.patchSize/2)) * (pentry['x'] < (width-(self.patchSize/2)))
        
        return flag 
        #flag= np.array(flag).astype(int)
        
        #if np.sum(flag)>0:
        #    found=True
        #    idx=np.where(flag==1)
        #else:
        #    found=False
        #    idx=-1
        #return found, idx
        
    def ntrue_positives_inside_margin(self, only_cleft=False):
	#pdb.set_trace()
        '''
        gtvol = self.gtvol[self.patchZ/2:-(self.patchZ/2),self.patchSize/2:-self.patchSize/2,self.patchSize/2:-self.patchSize/2]
        
	    if only_cleft==True:
	       uids_common=np.setdiff1d(np.unique(gtvol),[0])
	
	    else:
	        pre_mask= ((gtvol%2)==1)
            post_mask= ((gtvol%2)==0)
            uids_pre = np.setdiff1d(np.unique(gtvol[pre_mask]),[0])
            uids_post = np.setdiff1d(np.unique(gtvol[post_mask]),[0])
            uids_common = set(uids_pre+1).intersection(uids_post)
	    return len(uids_common)
		'''

    def get_num_candidates(self):
        #pdb.set_trace()    
        print 'initial candidates = ',len(self.candidates0)
        self.candidates=[]
        for ii in range(len(self.candidates0)):
            if self.inside_margin2(self.candidates0[ii]) == True:
                self.candidates.append(self.candidates0[ii])      
            
        print 'remaining candidates = ',len(self.candidates)
 
        return len(self.candidates)
    
    def rotate3d(self, input_vol, nrotate):

        #pdb.set_trace()
        #nrotate = random.randint(0,3)
        axis = 0 #random.randint(0,2)
        
        output_vol=[]
        for i in range(len(input_vol)):
            grayImg_set = input_vol[i]
            grayImg_set = axial_rotations(grayImg_set, rot=nrotate, ax=axis)
        
            output_vol.append([])
            output_vol[-1] = grayImg_set
            
        return output_vol
    
    def reflect_swap(self, input_vol, reflectz, reflecty, reflectx, swapxy):

        
        #reflectz=random.randint(0,1)
        #reflecty=random.randint(0,1)
        #reflectx=random.randint(0,1)
        #swapxy=random.randint(0,1)

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

                    
