#!/bin/bash
# add all other SBATCH directives here...

#SBATCH -p seas_dgx1
#SBATCH --gres=gpu:1
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine                        
#SBATCH --mem=60000
#SBATCH -t 3-0:00:00
#SBATCH -o skeleton_test_x2_%j.log

module load cuda/9.0-fasrc01
module load cudnn/7.0.3-fasrc02

module load Anaconda

source ~/anaconda2/bin/activate kears_theano

MODELDIR=/n/coxfs01/paragt/test_submit/skeleton
SEGNAME=combined_pred_x2_malaseg_0.5.h5
PREFIX=`basename $SEGNAME .h5`
SK_CANDIDATES=kasthuri_x2_minsz2000_candidates_50_10.pkl
PREFIX2=`basename $SK_CANDIDATES .pkl`
THD=0.6

#for iter in 40000 45000 50000 55000 60000 65000 70000 75000 
for iter in 200000 #200000 
do
    echo $iter
    
#    THEANO_FLAGS=device=cuda,floatX=float32,dnn.enabled=True python -u  ${MODELDIR}/test.py --imagedir ${MODELDIR}/data/combined_images_x2/ --predname ${MODELDIR}/data/combined_pred_x2.h5 --segname /n/coxfs01/kdmitriev/em/kasthuri_old/konstantin_corrected_03_22/corrected_labels_test.h5  --sk_endpoints /n/coxfs01/kdmitriev/em/kasthuri_old/konstantin_corrected_03_22/kasthuri_x2_thd0.25_minsz2000_candidates_50_10.pkl  --inputSize_xy 192 --inputSize_z 22 --modelname ${MODELDIR}/kasthuri_x1_minsz2000_leaky_f24_316_32_gt_x1/syn_prune_kasthuri_x1_minsz2000_leaky_f24_316_32_gt_x1_${iter}.json --weightname  ${MODELDIR}/kasthuri_x1_minsz2000_leaky_f24_316_32_gt_x1/sys_prune_kasthuri_x1_minsz2000_leaky_f24_316_32_gt_x1_${iter}_weights.h5


    ###THEANO_FLAGS=device=cuda,floatX=float32,dnn.enabled=True python -u  ${MODELDIR}/test_noaff.py --imagedir ${MODELDIR}/data/combined_images_x2/  --segname $SEGNAME  --sk_endpoints $SK_CANDIDATES  --inputSize_xy 192 --inputSize_z 22 --modelname ${MODELDIR}/kasthuri_x1_minsz2000_leaky_f24_316_32_gt_x1_noaff/syn_prune_kasthuri_x1_minsz2000_leaky_f24_316_32_gt_x1_noaff_${iter}.json --weightname  ${MODELDIR}/kasthuri_x1_minsz2000_leaky_f24_316_32_gt_x1_noaff/sys_prune_kasthuri_x1_minsz2000_leaky_f24_316_32_gt_x1_noaff_${iter}_weights.h5

done

source deactivate

source ~/anaconda2/bin/activate eval_env

python ${MODELDIR}/naive_merge3.py  --segmentation $SEGNAME --sk_endpoints $SK_CANDIDATES --threshold ${THD}


/n/home05/paragt/evaluation/PixelPred2Seg/comparestacks --stack1 $SEGNAME --stackbase /n/coxfs01/paragt/test_submit/skeleton/data/gold/microns-test-gold.h5 --relabel1 --relabelbase --dilate1 1 --dilatebase 1 --filtersize 5000 --anisotropic

echo $THD

/n/home05/paragt/evaluation/PixelPred2Seg/comparestacks --stack1 ${PREFIX}_${PREFIX2}_merged_thd${THD}.h5 --stackbase /n/coxfs01/paragt/test_submit/skeleton/data/gold/microns-test-gold.h5 --relabel1 --relabelbase --dilate1 1 --dilatebase 1 --filtersize 5000 --anisotropic

source deactivate
# end of program
exit 0;
