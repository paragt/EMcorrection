import os, sys, glob
import h5py
import numpy as np
import pdb
import scipy.sparse
import pickle
import argparse

parser = argparse.ArgumentParser(description='Generate syn partner candidates...')
parser.add_argument('--dataset', dest='dataset', action='store', required=True, help='')
parser.add_argument('--segmentation', dest='seg', action='store', required=True, help='')
parser.add_argument('--groundtruth', dest='gt', action='store', required=True, help='')

MIN_OVERLAP_WITH_GT = 0.7

if __name__=='__main__':

    #pdb.set_trace()

    args = parser.parse_args()
    
    
    dtst = args.dataset
    segmentation_file = args.seg
    groundtruth_file = args.gt

    savedir = os.path.dirname(segmentation_file) 
     
    fid = h5py.File(segmentation_file)
    if 'stack' in fid.keys(): seg = np.array(fid['stack'])
    elif 'main' in fid.keys(): seg = np.array(fid['main'])
    fid.close()
    
    
    fid = h5py.File(groundtruth_file)
    if 'stack' in fid.keys(): gt = np.array(fid['stack'])
    elif 'main' in fid.keys(): gt = np.array(fid['main'])
    fid.close()

    seg_uid= np.setdiff1d(np.unique(seg),[0])
    seg_szs = np.bincount(seg.ravel())
    gt_szs = np.bincount(gt.ravel())
    seg_gt_overlap = scipy.sparse.csc_matrix((np.ones_like(seg.ravel()), (seg.ravel(), gt.ravel())))
    
    seg_gt_map = np.argmax(seg_gt_overlap,axis=1)
    
    seg_gt_map_array={}
    seg_ignore_sizes={}
    for uid in seg_uid:
        seg_gt_id_ = seg_gt_map[uid][0,0]
        overlap_pct = seg_gt_overlap[uid,seg_gt_id_]*1.0/(seg_szs[uid])
        if (seg_gt_id_!=0) and (overlap_pct>MIN_OVERLAP_WITH_GT):
            seg_gt_map_array[uid] = seg_gt_id_
        else:
            array1 = [seg_szs[uid], gt_szs[seg_gt_id_], np.uint32(100*overlap_pct), seg_gt_id_]
            seg_ignore_sizes[uid] = array1
    
    savename = dtst+'_gt_map.pkl'
    pickle.dump(seg_gt_map_array, open(os.path.join(savedir,savename),'wb'))
    savename2 = dtst+'_ignore_sz.pkl'
    pickle.dump(seg_ignore_sizes, open(os.path.join(savedir,savename2),'wb'))     
