import os, sys, glob
import h5py
import numpy as np
import pdb
import scipy.sparse
import pickle
import argparse
import Queue as Q

szw = 25#25 for kasthuri
szd = 5 #5 for kasthuri
szw_ignore = 0
szd_ignore = 0

MIN_SEG_SZ = 2000

MIN_NBR_SZ = MIN_SEG_SZ

def compute_box(seg, ex,ey,ez):
    
    depth, height, width = seg.shape
    if ex < szw or ex > (width-szw):
        return np.array([])
    if ey < szw or ey > (height-szw):
        return np.array([])
    if ez < szd or ez > (depth-szd):
        return np.array([])

    box = seg[ez-szd:ez+szd, ey-szw:ey+szw, ex-szw:ex+szw].copy()
    return box


#def prune_small_segs(seg):
    
    #useg_ids = np.setdiff1d(np.unique(seg),0)
    #seg_szs = np.bincount(seg.ravel())
    
    #for id1 in useg_ids:
        #if seg_szs[id1]<20000:
            #seg[seg==id1]=0

from scipy.ndimage import grey_dilation, grey_erosion
'''A 3d structuring element connecting only left/right top/bottom up/down''' 
SIX_CONNECTED = np.array([[[False, False, False],                           
                           [False, True, False],                           
                           [False, False, False]],
                           [[False, True, False],                           
                            [True, True, True],                           
                            [False, True, False]],                          
                           [[False, False, False],                           
                            [False, True, False],                           
                            [False, False, False]]]) 
def erode_segmentation(segmentation, strel, in_place=False):    
    '''Erode a segmentation using a structuring element        :param segmentation: a labeling of a volume    :param strel: the structuring element for the erosion. This should be                  a boolean 3-d array with the voxels to erode marked as True    :param in_place: True to erode the segmentation volume in-place, False                     to return a new volume.    '''    
    if not in_place:        
        segmentation = segmentation.copy()    
    mask = grey_dilation(segmentation, footprint=strel) != grey_erosion(segmentation, footprint=strel)    
    
    #segmentation[mask] = 0    
    #return segmentation
    
    return (mask).astype(np.uint32)

parser = argparse.ArgumentParser(description='Generate syn partner candidates...')
parser.add_argument('--dataset', dest='dataset', action='store', required=True, help='')
parser.add_argument('--segmentation', dest='seg', action='store', required=True, help='')
parser.add_argument('--seg_gt_map', dest='seg_gt_map', action='store', required=True, help='seg to gt map computed by compute_seg_gt_map.py')
parser.add_argument('--skeleton_folder', dest='skfolder', action='store', required=True, help='')


if __name__=='__main__':

    #pdb.set_trace()

    args = parser.parse_args()
    
    #train_trial = sys.argv[1]
    #train_datadir = sys.argv[2]
    #train_imagedir = sys.argv[3]
    #train_predname = sys.argv[4]
    #train_syn_gtname = sys.argv[5]
    #train_segname = sys.argv[6]
    #pdb.set_trace()
    
    dtst = args.dataset
    segmentation_file = args.seg
    seg_gt_map_name = args.seg_gt_map
    skeleton_folder = args.skfolder

    savedir = os.path.dirname(segmentation_file)
 
    savename = dtst+'_candidates_'+str(2*szw)+'_'+str(2*szd)+'.pkl'
        
    #seg = dataIO.ReadSegmentationData(dtst)
     
    fid = h5py.File(segmentation_file)
    if 'stack' in fid.keys(): seg = np.array(fid['stack'])
    elif 'main' in fid.keys(): seg = np.array(fid['main'])
    fid.close()
    
    seg_gt_map = pickle.load(open(seg_gt_map_name,'rb'))
    all_seg_ids = np.unique(seg) 
    #fid = h5py.File(groundtruth_file)
    #gt = np.array(fid['stack'])
    #fid.close()
    #gt = dataIO.ReadGoldData(dtst)
    
    #seg=seg[:160,:,:]
    #gt=gt[:160,:,:]
    
    candidates = []
    allpairs={}
    npos=0
    nneg=0
    nnbr=0
    all_skeleton_files=sorted(glob.glob(os.path.join(skeleton_folder,'*.txt')))
    
    #pdb.set_trace()
    #for seg_id in skeletons.keys():
    nsingleton=0
    for ifile,skfile in enumerate(all_skeleton_files):
        #print skfile
        seg_id = np.uint32(os.path.splitext(os.path.basename(skfile))[0])
        #print 'seg id: ',seg_id
        if seg_id not in all_seg_ids:
            print 'seg id not in corrected seg ', seg_id
            continue
            #pdb.set_trace()
        #pdb.set_trace()
        #endpoints = skeletons[seg_id]
        
        skeleton_data = np.loadtxt(skfile,dtype=np.uint32, delimiter=' ',ndmin=2)
        
        if skeleton_data.shape[1]>3:
            ep_idx=(skeleton_data[:,3]==1).nonzero()[0]
            endpoints = skeleton_data[ep_idx,:3]
        elif skeleton_data.shape[1]==3:
            endpoints = skeleton_data  

        if endpoints.shape[0]==1:
            print 'seg id ', seg_id
            nsingleton +=1
        #print 'endpt shape: ',endpoints.shape 
        for ip in range(endpoints.shape[0]):
            #pdb.set_trace()
            endpt1 = endpoints[ip]
            ex = endpt1[2]
            ey = endpt1[1]
            ez = endpt1[0]
            #pdb.set_trace()
            seg_box = compute_box(seg, ex,ey,ez)
            #pdb.set_trace()
            #if (seg_box.shape[0]<1) or (np.sum(seg_box==seg_id)< MIN_SEG_SZ) or (seg_gt_map.has_key(seg_id)==False):
                #continue
            if (seg_box.shape[0]<1) or (seg_gt_map.has_key(seg_id)==False):
                continue
            seg_gt = seg_gt_map[seg_id]
            #seg_box = seg_box[szd_ignore:seg_box.shape[0]-szd_ignore, szw_ignore:seg_box.shape[1]-szw_ignore, szw_ignore:seg_box.shape[2]-szw_ignore]
            uid_nbr_seg = np.setdiff1d(np.unique(seg_box),[0,seg_id]) #np.unique(seg_box)
            nbr_seg_szs = np.bincount(seg_box.ravel())

            dist_trans = scipy.ndimage.morphology.distance_transform_edt(1-seg_box,sampling=[7,1,1])
            q1 = Q.PriorityQueue()
            for uid in uid_nbr_seg:
                mindist = np.amin(dist_trans[(seg_box==uid)])
                q1.put((mindist,uid))


            for ii in range(10):
            #for uid in uid_nbr_seg:
                if q1.empty():
                    break
                
                qelem = q1.get()
                mindist = qelem[0]
                uid = qelem[1]
                if mindist>100:
                    break
                #print tid
                #if (uid == seg_id) or (nbr_seg_szs[uid]<MIN_NBR_SZ):
                    #continue
                skfilename = str(uid)+'.txt'
                skfilepath = os.path.join(skeleton_folder,skfilename)
                if os.path.exists(skfilepath)==False:
                    continue;
                if (uid == seg_id):
                    pdb.set_trace()
                    continue
                
                
                
                
                if allpairs.has_key(seg_id):
                    stored_nbr = allpairs[seg_id]
                    if uid in stored_nbr:
                        #pdb.set_trace()
                        continue
                elif allpairs.has_key(uid):
                    stored_nbr = allpairs[uid]
                    if seg_id in stored_nbr:
                        #pdb.set_trace()
                        continue
                
                
                nnbr=nnbr+1
                if(seg_gt_map.has_key(uid)==False): #same id
                    continue
                nbr_gt = seg_gt_map[uid]
                if (seg_gt  == nbr_gt) :
                    label1 = 1
                    npos = npos +1
                elif seg_gt != nbr_gt:
                    label1 = -1
                    nneg = nneg+1
                
                candidate1 = {}
                #seg_gt_match[seg_id]
                candidate1['x'] = ex
                candidate1['y'] = ey
                candidate1['z'] = ez
                candidate1['label'] = label1
                candidate1['seg'] = seg_id
                candidate1['nbr'] = uid
                                            
                candidate1['seg_gt'] = seg_gt
                candidate1['nbr_gt'] = nbr_gt
                #candidate1['bdry1'] = np.sum(bdry1)
                #candidate1['bdry2'] = np.sum(bdry2)
                #candidate1['bdry3'] = np.sum(bdry3)
                #candidate1['seg1dist'] = seg1q
                #candidate1['seg2dist'] = seg2q
                
                candidates.append(candidate1)
                if allpairs.has_key(seg_id):
                    allpairs[seg_id].append(uid)
                else:
                    allpairs[seg_id]=[uid]
                
            #endpoints.append(skeleton.EndPoint(ip))
    print 'total single ep = ',nsingleton
    print 'npos: {0}, nneg: {1}, nnbr={2}'.format(npos,nneg,nnbr)
    ntotal = npos+nneg
    print 'npos: {0}, nneg: {1}, nnbr={2}'.format(npos*100./ntotal,nneg*100./ntotal,nnbr*100./(npos+nnbr))
        
    #pdb.set_trace()
    #savename = dtst+str(MIN_SZ_IN_INPUT_SEG)+'_candidates_gt'+str(MAX_ZERORATIO_IN_GT)++'_gtovrlp'+str(MIN_OVERLAP_WITH_GT)+'_szthd'+str(MIN_NBR_SZ)+'_'+str(szw)+'_'+str(szd)+'.pkl'
    pickle.dump(candidates, open(os.path.join(savedir,savename),'wb'))    
