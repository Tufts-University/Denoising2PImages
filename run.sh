#!/bin/bash
#SBATCH -J FS_080822_Denoising   #job name
#SBATCH --time=05-00:0:00  #requested time
#SBATCH -p preempt     #running on "preempt" partition/queue
#SBATCH -N 1    #1 nodes
#SBATCH	-n 10   #10 tasks total
#SBATCH	-c 1   #using 1 cpu core/task
#SBATCH --gres=gpu:a100:1
##SBATCH --nodelist=p1cmp072
#SBATCH --exclude=cc1gpu004,cc1gpu002
#SBATCH --mem=20g  #requesting 2GB of RAM total
#SBATCH --output=../FAD_model_0713_cervix_SSIML1.%j.out  #saving standard output to file -- %j jobID -- %N nodename
#SBATCH --error=../FAD_model_0713_cervix_SSIML1.%j.err  #saving standard error to file -- %j jobID -- %N nodename
#SBATCH --mail-type=ALL    #email options
#SBATCH --mail-user=20193005@student.anatolia.edu.gr

module load anaconda/2021.05
source activate Denoising

git pull
echo "Starting python script..."
echo "=========================================================="
echo "" # empty line

# When changing an important parameter, change the name both here and in the output/error files (above SBATCH arguments).

# NADH_CAREmodel_0713_cervix_SSIMR2_Wavelet:
# python -u main.py train care "NADH_CAREmodel_0713_cervix_SSIMR2_Wavelet" cwd=.. nadh_data=NV_713_NADH_healthy.npz loss=ssimr2_loss wavelet=True

python -u main.py eval rcan "FAD_model_0713_cervix_SSIML1" cwd=.. fad_data=NV_713_FAD_healthy.npz