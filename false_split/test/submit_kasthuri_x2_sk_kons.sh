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

SKDIR=skeletons_konstantin_04_02
DTST=kasthuri_x2_minsz2000
SEGNAME=combined_pred_x2_malaseg_0.5.h5
PREFIX=`basename $SEGNAME .h5`
GTNAME=/n/coxfs01/paragt/test_submit/skeleton/data/gold/microns-test-gold.h5

if [ ! -d "$SKDIR" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
    mkdir $SKDIR
fi


/n/home05/paragt/NeuronComplete/build/get_skeleton -segmentation $SEGNAME stack -output_folder $SKDIR -min_region_sz 2000 -type 1

source deactivate

# end of program
exit 0;
