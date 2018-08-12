#!/bin/bash
# add all other SBATCH directives here...

#SBATCH -p seas_dgx1
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine                        
#SBATCH --mem=60000
#SBATCH -t 3-0:00:00
#SBATCH -o skeleton_test_x2_%j.log

module load Anaconda
source ~/anaconda2/bin/activate eval_env


for thd in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
do
    echo $thd
    /n/home05/paragt/evaluation/PixelPred2Seg/comparestacks --stack1 /n/coxfs01/paragt/test_submit/skeleton/cremi/seg/combined_pred_x2_malaseg_${thd}.h5 --stackbase /n/coxfs01/paragt/test_submit/skeleton/data/gold/microns-test-gold.h5 --relabel1 --relabelbase --dilate1 1 --dilatebase 1 --filtersize 5000 --anisotropic

done

source deactivate
# end of program
exit 0;
