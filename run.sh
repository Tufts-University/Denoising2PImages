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
#SBATCH --output=../NADH_model_0713_cervix_SSIMR2_ap5_Wavelet_gaus1.%j.out  #saving standard output to file -- %j jobID -- %N nodename
#SBATCH --error=../NADH_model_0713_cervix_SSIMR2_ap5_Wavelet_gaus1.%j.err  #saving standard error to file -- %j jobID -- %N nodename
#SBATCH --mail-type=ALL    #email options
#SBATCH --mail-user=20193005@student.anatolia.edu.gr

module load anaconda/2021.05
source activate Denoising

git pull
echo "Starting python script..."
echo "=========================================================="
echo "" # empty line #

# When changing an important parameter, change the name both here and in the output/error files (above SBATCH arguments).

### MARK: FAD Model — NADH Eval #################################

# FAD RCAN SSIM   ✅
# python -u main.py eval rcan "FAD_model_0629_cervix_SSIM" cwd=.. nadh_data=NV_713_NADH_healthy.npz

# FAD RCAN SSIML1 ✅
# python -u main.py eval rcan "FAD_model_0713_cervix_SSIML1" cwd=.. nadh_data=NV_713_NADH_healthy.npz

# FAD CARE SSIML1 ✅
# python -u main.py eval care "FAD_CAREmodel_0713_cervix_SSIML1_BS50" cwd=.. nadh_data=NV_713_NADH_healthy.npz unet_n_depth=2
#                                                                                                              ^~~~~~~~~~~~~~ Optional

##################################################################

### MARK: NADH CARE + SSIMR2 ap5 ⏰ ###################################

# python -u main.py train care "NADH_CAREmodel_0713_cervix_SSIMR2_ap5" cwd=.. nadh_data=NV_713_NADH_healthy.npz loss=ssimr2_loss 

#################################################################

### MARK: Different Wavelet Functions ###########################

# NADH RCAN SSIMR2 ap5 Wavelet Haar ⏰
# python -u main.py train rcan "NADH_model_0713_cervix_SSIMR2_ap5_Wavelet_haar" cwd=.. nadh_data=NV_713_NADH_healthy.npz loss=ssimr2_loss wavelet_function=haar

# NADH RCAN SSIMR2 ap5 Wavelet Morl ⏰
# python -u main.py train rcan "NADH_model_0713_cervix_SSIMR2_ap5_Wavelet_morl" cwd=.. nadh_data=NV_713_NADH_healthy.npz loss=ssimr2_loss wavelet_function=morl

# NADH RCAN SSIMR2 ap5 Wavelet Gaus1 ⏰
python -u main.py train rcan "NADH_model_0713_cervix_SSIMR2_ap5_Wavelet_gaus1" cwd=.. nadh_data=NV_713_NADH_healthy.npz loss=ssimr2_loss wavelet_function=gaus1

#################################################################

### CARE + Wavelet Denoising #####################################

# Train NADH CARE + SSIMR2 ap5 Wavelet bior4.4 ⏰
# python -u main.py train care "NADH_CAREmodel_0713_cervix_SSIMR2_ap5_Wavelet_bior4p4" cwd=.. nadh_data=NV_713_NADH_healthy.npz loss=ssimr2_loss wavelet_function=bior4.4

### Archives #####################################################
# NADH_CAREmodel_0713_cervix_SSIMR2_Wavelet
# TODO: Change to include wavelet family
# python -u main.py train care "NADH_CAREmodel_0713_cervix_SSIMR2_Wavelet" cwd=.. nadh_data=NV_713_NADH_healthy.npz loss=ssimr2_loss wavelet=True

# FAD_model_0713_cervix_SSIML1
# python -u main.py eval care "FAD_model_0713_cervix_SSIML1" cwd=.. nadh_data=NV_713_FAD_healthy.npz

# NADH_model_0713_cervix_SSIMR2_Wavelet
# python -u main.py eval rcan "NADH_model_0713_cervix_SSIMR2_Wavelet" cwd=.. nadh_data=NV_713_NADH_healthy.npz fad_data=NV_713_FAD_healthy.npz wavelet=True

# NADH_CAREmodel_0713_cervix_SSIMR2_ap5
# python -u main.py eval care "NADH_CAREmodel_0713_cervix_SSIMR2_ap5" cwd=.. fad_data=NV_713_FAD_healthy.npz loss=ssimr2_loss