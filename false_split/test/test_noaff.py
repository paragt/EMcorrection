import time
import glob


import os, sys
os.environ['KERAS_BACKEND']='theano'

from keras import backend as K
K.set_image_dim_ordering('th')

import numpy as np
import scipy.misc
import scipy.ndimage
from prepare_data_3d_my_test_noaff import *
import pdb
import argparse

from keras.models import Model, Sequential, model_from_json

import keras.activations
import theano



parser = argparse.ArgumentParser(description='Generate syn partner candidates...')
parser.add_argument('--imagedir', dest='imagedir', action='store', default='./grayscale_maps', help='image subfolder')
parser.add_argument('--segname', dest='segname', action='store', required=True, help='segmentation file')
parser.add_argument('--sk_endpoints', dest='sk_endpoints', action='store', required=True, help='pickle file with skeleton endpoints')
parser.add_argument('--inputSize_xy', dest='patchSize', action='store', default=None, help='segmentation GT')
parser.add_argument('--inputSize_z', dest='patchZ', action='store', default=None, help='segmentation GT')
parser.add_argument('--modelname', dest='modelname', action='store', default=None, help='deep net model name')
parser.add_argument('--weightname', dest='weightname', action='store', default=None, help='deep net weight name')
#parser.add_argument('--threshold', dest='threshold', action='store', default=None, help='deep net weight name')




if __name__=='__main__':

    args = parser.parse_args()
    #pdb.set_trace()

    train_imagedir = args.imagedir
    train_segname = args.segname
    train_skfile = args.sk_endpoints


    doFineTune = False

    #patchSize = 192#428#316#204
    patchSize = int(args.patchSize)#428#316#204
    patchSize_out = 116#340#228#116
    #patchZ = 22
    patchZ = int(args.patchZ)
    patchZ_out = 4
    print 'input patch xy size: ',patchSize
    print 'input patch z size: ',patchZ


    gen_data = GenerateData(train_imagedir,train_segname, train_skfile, patchSize, patchZ)



    modelname = args.modelname
    weightname = args.weightname
    print weightname   
    model = model_from_json(open(modelname).read())
    
    model.load_weights(weightname)
    model.compile(loss='mse', optimizer='Adam')

    #pdb.set_trace()
    ndata = gen_data.get_num_candidates()

    t0 = time.time()    
    
    #results = np.zeros((data_x.shape[0],patchZ_out,patchSize_out,patchSize_out))
    #print "Data_y shape: ", data_y.shape
    #gen_data.create_result_vol()
    
    #actual_gt=gen_data.ntrue_positives_inside_margin(only_cleft=only_cleft)
    #pdb.set_trace()
    #pred_thd=-0.05 
    #pred_thd=-0.175
    #pred_thd=-0.175
    
    syn_gt_detected=[]
    syn_gt_missed=[]
    syn_false_alarm=[]
    input_positive_detected=[]
    true_positive=0
    true_negative=0
    false_negative=0
    false_positive=0
    
    res=np.zeros((ndata,2)).astype(np.float32)
    for k in range(ndata):
        
        data = gen_data.get_next_test_sample(k)
        data_x = np.array(data[0])
        #pdb.set_trace()
        #data_x = np.reshape(data_x, [1,-1, patchZ, patchSize, patchSize])
        #print "{0}th data_x shape: {1}".format(k, data_x.shape)
	     
        im_pred_array = model.predict(x=data_x, batch_size = data_x.shape[0])
        im_pred = np.mean(im_pred_array)
        res[k,:] = [data[1], im_pred]
        gen_data.set_prediction(k, im_pred)

    #pdb.set_trace()
    pred_range=np.array(range(0,10))/10.
    for pred_thd in pred_range:	

        bin_pred = res[:,1]>pred_thd

        label = res[:,0]>0

        true_positive = np.sum((label==1)*(bin_pred==1))
        true_negative = np.sum((label==0)*(bin_pred==0))
        false_negative = np.sum((label==1)*(bin_pred<1))
        false_positive = np.sum((label==0)*(bin_pred==1))

    
    	#print 'false negative: {0}'.format(len(np.setdiff1d(syn_gt_missed,syn_gt_detected)))
	#pdb.set_trace()
        print 'threshold: ', pred_thd
        print 'total gt: {0}'.format(true_positive+false_negative)
        print 'true positive: {0}'.format(true_positive)
        print 'true negative: {0}'.format(true_negative)
        print 'false ngative: {0}  ({1})'.format(false_negative, false_negative*1.0/(true_positive+false_negative))
        print 'false positive: {0} ({1})'.format(false_positive, false_positive*1.0/(false_positive+true_positive))
    
    gen_data.save_predictions(train_skfile)	
