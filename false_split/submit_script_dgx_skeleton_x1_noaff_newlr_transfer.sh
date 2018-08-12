#!/bin/bash
# add all other SBATCH directives here...

#SBATCH -p seas_dgx1 
#SBATCH --gres=gpu:1
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine                        
#SBATCH --mem=60000
#SBATCH -t 3-0:00:00
#SBATCH -o skeleton1_316_32_%j.log

module load cuda/9.0-fasrc01
module load cudnn/7.0.3-fasrc02

module load Anaconda

source ~/anaconda2/bin/activate kears_theano

mkdir -p train_samples

THEANO_FLAGS=device=cuda,floatX=float32,dnn.enabled=True python -u  /n/coxfs01/paragt/test_submit/skeleton/train_prune_cnn_leaky_noaff_newlr.py --trial kasthuri_x1_minsz2000_leaky_f24_316_32_noaff_ub_transfer --imagedir /n/coxfs01/paragt/test_submit/skeleton/data/combined_images_x1/  --segname corrected.h5  --sk_endpoints  kasthuri_x1_minsz2000_candidates_50_10.pkl --inputSize_xy 192 --inputSize_z 22 --fine_tune

# end of program
exit 0;
