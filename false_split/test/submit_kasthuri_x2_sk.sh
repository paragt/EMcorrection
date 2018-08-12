#!/bin/bash
# add all other SBATCH directives here...

#SBATCH -p seas_dgx1

#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine                        
#SBATCH --mem=80000
#SBATCH -t 3-0:00:00
#SBATCH -o skeleton_k2_%j.log

#module load cuda/9.0-fasrc01
#module load cudnn/7.0.3-fasrc02

module load Anaconda

source /n/home05/paragt/anaconda2/bin/activate skeleton_env

SKDIR=skeletons_konstantin_04_08
DTST=kasthuri_x2_minsz500_full
SEGNAME=combined_pred_x2_malaseg_0.5.h5
PREFIX=`basename $SEGNAME .h5`
GTNAME=/n/coxfs01/paragt/test_submit/skeleton/data/gold/microns-test-gold.h5

if [ ! -d "$SKDIR" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
    mkdir $SKDIR
fi


#/n/home05/paragt/NeuronComplete/build/get_skeleton -segmentation $SEGNAME stack -output_folder $SKDIR -min_region_sz 500 -type 1 -zbuffer 15  -xybuffer 100

#chmod -R 777 $SKDIR

source deactivate

source /n/home05/paragt/anaconda2/bin/activate eval_env

echo compute gt map
#python /n/coxfs01/paragt/test_submit/skeleton/compute_seg_gt_map.py --dataset $DTST --segmentation $SEGNAME --groundtruth $GTNAME

echo generate candidates
#python /n/coxfs01/paragt/test_submit/skeleton/generate_candidates_noedgeprob.py --dataset $DTST --segmentation $SEGNAME --seg_gt_map ${DTST}_gt_map.pkl --skeleton_folder  $SKDIR

echo naive merge
#python /n/coxfs01/paragt/test_submit/skeleton/naive_merge3.py  --segmentation $SEGNAME --sk_endpoints ${DTST}_candidates_50_10.pkl --oracle

echo evaluation
/n/home05/paragt/evaluation/PixelPred2Seg/comparestacks --stack1 $SEGNAME --stackbase $GTNAME --relabel1 --relabelbase --dilate1 1 --dilatebase 1 --filtersize 5000 --anisotropic

/n/home05/paragt/evaluation/PixelPred2Seg/comparestacks --stack1 ${PREFIX}_${DTST}_candidates_50_10_oracle.h5 --stackbase $GTNAME --relabel1 --relabelbase --dilate1 1 --dilatebase 1 --filtersize 5000 --anisotropic

source deactivate
# end of program
exit 0;
