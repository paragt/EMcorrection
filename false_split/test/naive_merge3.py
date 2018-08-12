
import os,sys
import h5py
import numpy as np
import pdb
import pickle
import argparse
import networkx
import sets

def prune_small_segs(seg):
    
    useg_ids = np.setdiff1d(np.unique(seg),0)
    seg_szs = np.bincount(seg.ravel())
    
    for id1 in useg_ids:
        print id1
        if seg_szs[id1]<5000:
            seg[seg==id1]=0
    return seg

def find_latest_id(merge_map, nid):
    ret_id = nid
    
    while merge_map.has_key(ret_id):
        #pdb.set_trace()
        ret_id = merge_map[ret_id]
        
    return ret_id

parser = argparse.ArgumentParser(description='Generate syn partner candidates...')
parser.add_argument('--segmentation', dest='segname', action='store', required=True, help='segmentation file')
parser.add_argument('--sk_endpoints', dest='sk_endpoints', action='store', required=True, help='pickle file with skeleton endpoints')
parser.add_argument('--oracle', action='store_true', default=False, dest='oracle', help = 'use labels to merge')
parser.add_argument('--threshold', dest='threshold', action='store', default=None, help='deep net weight name')

if __name__=='__main__':
    args = parser.parse_args()

    segname = args.segname
    skeleton_name=args.sk_endpoints
    if args.oracle==False:
        pred_thd = float(args.threshold)

    savedir = os.path.dirname(segname)
    #pdb.set_trace()
    fid=h5py.File(segname)
    if 'stack' in fid.keys(): seg=np.array(fid['stack'])
    elif 'main' in fid.keys(): seg=np.array(fid['main'])
    fid.close()
    
    if args.oracle==True:
        savename = os.path.basename(segname)[:-3] +'_'+os.path.basename(skeleton_name)[:-4]+'_oracle.h5'
    else:
        savename = os.path.basename(segname)[:-3] +'_'+os.path.basename(skeleton_name)[:-4]+'_merged_thd'+str(pred_thd)+'.h5'
    
    unique_segids = np.unique(seg)
    unique_segids0 = unique_segids[unique_segids!=0]
    print 'unique ids: {0}'.format(len(unique_segids0))
    
    seg_szs = np.bincount(seg.ravel())
    
    candidates = pickle.load(open(os.path.join(skeleton_name),'rb'))
    
    merge_map = {}
    count = 0
    merge_graph = networkx.Graph()
    ids_to_check=sets.Set()
    for ii in range(len(candidates)):
        decision = False

        if args.oracle == True:
            decision = (candidates[ii]['label']>0)
        else: 
            if not candidates[ii].has_key('pred'):
                continue
            decision = (candidates[ii]['pred']>pred_thd)
        
        if decision==True:
        #if candidates[ii]['pred'][0][0]>0.6:
        #if candidates[ii]['label']>0:
            if seg_szs[candidates[ii]['seg']] > seg_szs[candidates[ii]['nbr']]:
                merged_id = candidates[ii]['nbr']
                new_id = candidates[ii]['seg']
            elif seg_szs[candidates[ii]['seg']] < seg_szs[candidates[ii]['nbr']]:
                merged_id = candidates[ii]['seg']
                new_id = candidates[ii]['nbr']
    
            merge_graph.add_edge(merged_id, new_id)
            ids_to_check.add(merged_id)
            ids_to_check.add(new_id)
            #new_id2 = find_latest_id(merge_map,new_id)
            #seg[seg==merged_id] = new_id2
            #merge_map[merged_id] = new_id2
            
            count=count+1
    
    max_id1 = unique_segids.max()
    forward_map = np.zeros(max_id1+1, np.uint32)
    forward_map[unique_segids0] = unique_segids0
    
    while len(ids_to_check)>0:
        uid = ids_to_check.pop()
        nodes_reachable=list(networkx.dfs_preorder_nodes(merge_graph, source=uid))
        nodes_reachable = np.array(nodes_reachable)
        nodes_reachable = np.append(nodes_reachable,[uid])
        new_id = np.amin(np.unique(nodes_reachable))
            
        forward_map[uid] = new_id
    
    stitched_seg = forward_map[seg]

    
    #pdb.set_trace()
    print 'remaining unique ids: {0}'.format(len(np.unique(forward_map))-1)
    fidw=h5py.File(os.path.join(savedir, savename),'w')
    fidw.create_dataset('stack',data=stitched_seg)
    fidw.close()
    
