#!/bin/bash
#SBATCH -J FS_080822_Denoising   #job name
#SBATCH --time=05-00:0:00  #requested time
#SBATCH -p preempt     #running on "preempt" partition/queue
#SBATCH -N 1    #1 nodes
#SBATCH	-n 10   #10 tasks total
#SBATCH	-c 1   #using 1 cpu core/task
#SBATCH --gres=gpu:a100:1
##SBATCH --nodelist=p1cmp110
#SBATCH --exclude=cc1gpu005
#SBATCH --mem=20g  #requesting 2GB of RAM total
#SBATCH --output=../NADH_SRGAN_0823_cervix_SSIMR2_seed0_new.%j.out  #saving standard output to file -- %j jobID -- %N nodename
#SBATCH --error=../NADH_SRGAN_0823_cervix_SSIMR2_seed0_new.%j.err  #saving standard error to file -- %j jobID -- %N nodename
#SBATCH --mail-type=ALL    #email options
#SBATCH --mail-user=nvora01@tufts.edu

module load anaconda/2021.05
source activate Denoising

git pull
echo "Starting python script..." 
echo "==========================================================" 
echo "" # empty line #

# When changing an important parameter, change the name both here and in the output/error files (above SBATCH arguments).

### MARK: FAD Model — NADH Eval ##################################

# FAD RCAN SSIM ✅ ✅
# python -u main.py eval rcan "FAD_model_0629_cervix_SSIM" cwd=.. nadh_data=NV_713_NADH_healthy.npz

# FAD RCAN SSIML1 ✅ ✅
# python -u main.py eval rcan "FAD_model_0713_cervix_SSIML1" cwd=.. nadh_data=NV_713_NADH_healthy.npz

# FAD CARE SSIML1 ✅ ✅
# python -u main.py eval care "FAD_CAREmodel_0713_cervix_SSIML1_BS50" cwd=.. nadh_data=NV_713_NADH_healthy.npz unet_n_depth=2
#                                                                                                              ^~~~~~~~~~~~~~ Optional

##################################################################

### MARK: NADH CARE + SSIMR2 ap5 #################################

# ✅ ✅
# python -u main.py eval care "NADH_CAREmodel_0713_cervix_SSIMR2_ap5" cwd=.. fad_data=NV_713_FAD_healthy.npz loss=ssimr2_loss 

##################################################################

### MARK: Different Wavelet Functions ############################

# NADH RCAN SSIMR2 ap5 Wavelet Haar ✅ ✅ 
# python -u main.py eval rcan "NADH_model_0713_cervix_SSIMR2_ap5_Wavelet_haar" cwd=.. nadh_data=NV_713_NADH_healthy.npz fad_data=NV_713_FAD_healthy.npz loss=ssimr2_loss wavelet_function=haar

# NADH RCAN SSIMR2 ap5 Wavelet Morl ❌ continuous wavelet issue (no cwt2 func)
# python -u main.py train rcan "NADH_model_0713_cervix_SSIMR2_ap5_Wavelet_morl" cwd=.. nadh_data=NV_713_NADH_healthy.npz loss=ssimr2_loss wavelet_function=morl

# NADH RCAN SSIMR2 ap5 Wavelet Gaus1 ❌ continuous wavelet issue (no cwt2 func)
# python -u main.py train rcan "NADH_model_0713_cervix_SSIMR2_ap5_Wavelet_gaus1" cwd=.. nadh_data=NV_713_NADH_healthy.npz loss=ssimr2_loss wavelet_function=gaus1

# NADH RCAN + SSIMR2 ap5 Wavelet bior1.1 SSIMR2 (new_data loader) ✅
# python -u main.py eval rcan "NADH_model_0713_cervix_SSIMR2_Wavelet_bior" cwd=.. fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz wavelet_function=bior1.1 val_seed=0 val_split=4 test_split=8 test_flag=0

# NADH RCAN + SSIMR2 ap5 seed 1 ✅ ✅ ✅ ✅
# python -u main.py train rcan "NADH_RCAN_0823_cervix_SSIMR2_seed1" cwd=.. nadh_data=NV_823_NADH_healthy.npz  loss_alpha=0.5 val_seed=1 val_split=4 test_split=8 test_flag=1 
# python -u main.py eval rcan "NADH_RCAN_0823_cervix_SSIMR2_seed1" cwd=.. fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz val_seed=1 val_split=4 test_split=8 test_flag=1 
# python -u main.py eval rcan "NADH_RCAN_0823_cervix_SSIMR2_seed1" cwd=.. fad_data=NV_907_FAD_Colpo.npz nadh_data=NV_907_NADH_Colpo.npz val_seed=1 val_split=4 test_split=8 test_flag=1 train_mode=0
# python -u main.py eval rcan "NADH_RCAN_0823_cervix_SSIMR2_seed1" cwd=.. fad_data=NV_907_FAD_Leep.npz nadh_data=NV_907_NADH_Leep.npz val_seed=1 val_split=4 test_split=8 test_flag=1 train_mode=0 

# NADH RCAN + SSIMR2 ap5 seed 0 ✅ ✅ ✅ ✅ 39729478    
# python -u main.py train rcan "NADH_RCAN_0823_cervix_SSIMR2_seed0" cwd=.. nadh_data=NV_823_NADH_healthy.npz  loss_alpha=0.5 val_seed=0 val_split=4 test_split=8 test_flag=1 
# python -u main.py eval rcan "NADH_RCAN_0823_cervix_SSIMR2_seed0" cwd=.. fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz val_seed=0 val_split=4 test_split=8 test_flag=1 
# python -u main.py eval rcan "NADH_RCAN_0823_cervix_SSIMR2_seed0" cwd=.. fad_data=NV_907_FAD_Colpo.npz nadh_data=NV_907_NADH_Colpo.npz val_seed=0 val_split=4 test_split=8 test_flag=1 train_mode=0
# python -u main.py eval rcan "NADH_RCAN_0823_cervix_SSIMR2_seed0" cwd=.. fad_data=NV_907_FAD_Leep.npz nadh_data=NV_907_NADH_Leep.npz val_seed=0 val_split=4 test_split=8 test_flag=1 train_mode=0

# NADH RCAN + SSIMR2 ap5 seed 2 ✅ ✅ ✅ ✅ 39729208 
# python -u main.py train rcan "NADH_RCAN_0823_cervix_SSIMR2_seed2" cwd=.. nadh_data=NV_823_NADH_healthy.npz  loss_alpha=0.5 val_seed=2 val_split=4 test_split=8 test_flag=1 
# python -u main.py eval rcan "NADH_RCAN_0823_cervix_SSIMR2_seed2" cwd=.. fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz val_seed=2 val_split=4 test_split=8 test_flag=1 
# python -u main.py eval rcan "NADH_RCAN_0823_cervix_SSIMR2_seed2" cwd=.. fad_data=NV_907_FAD_Colpo.npz nadh_data=NV_907_NADH_Colpo.npz val_seed=2 val_split=4 test_split=8 test_flag=1 train_mode=0
# python -u main.py eval rcan "NADH_RCAN_0823_cervix_SSIMR2_seed2" cwd=.. fad_data=NV_907_FAD_Leep.npz nadh_data=NV_907_NADH_Leep.npz val_seed=2 val_split=4 test_split=8 test_flag=1 train_mode=0

# NADH RCAN + SSIMR2 ap5 seed 3 ✅ ✅ ✅ ✅  39740904   
# python -u main.py train rcan "NADH_RCAN_0823_cervix_SSIMR2_seed3" cwd=.. nadh_data=NV_823_NADH_healthy.npz  loss_alpha=0.5 val_seed=3 val_split=4 test_split=8 test_flag=1 
# python -u main.py eval rcan "NADH_RCAN_0823_cervix_SSIMR2_seed3" cwd=.. fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz val_seed=3 val_split=4 test_split=8 test_flag=1 
# python -u main.py eval rcan "NADH_RCAN_0823_cervix_SSIMR2_seed3" cwd=.. fad_data=NV_907_FAD_Colpo.npz nadh_data=NV_907_NADH_Colpo.npz val_seed=3 val_split=4 test_split=8 test_flag=1 train_mode=0
# python -u main.py eval rcan "NADH_RCAN_0823_cervix_SSIMR2_seed3" cwd=.. fad_data=NV_907_FAD_Leep.npz nadh_data=NV_907_NADH_Leep.npz val_seed=3 val_split=4 test_split=8 test_flag=1 train_mode=0

# NADH RCAN + SSIMR2 ap5 seed 4 ✅ ✅ ✅ ✅  39740989
# python -u main.py train rcan "NADH_RCAN_0823_cervix_SSIMR2_seed4" cwd=.. nadh_data=NV_823_NADH_healthy.npz  loss_alpha=0.5 val_seed=4 val_split=4 test_split=8 test_flag=1 
# python -u main.py eval rcan "NADH_RCAN_0823_cervix_SSIMR2_seed4" cwd=.. fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz val_seed=4 val_split=4 test_split=8 test_flag=1 
# python -u main.py eval rcan "NADH_RCAN_0823_cervix_SSIMR2_seed4" cwd=.. fad_data=NV_907_FAD_Colpo.npz nadh_data=NV_907_NADH_Colpo.npz val_seed=4 val_split=4 test_split=8 test_flag=1 train_mode=0
# python -u main.py eval rcan "NADH_RCAN_0823_cervix_SSIMR2_seed4" cwd=.. fad_data=NV_907_FAD_Leep.npz nadh_data=NV_907_NADH_Leep.npz val_seed=4 val_split=4 test_split=8 test_flag=1 train_mode=0
##################################################################

### CARE + Wavelet Denoising #####################################

# NADH CARE + SSIMR2 ap5 Wavelet bior4.4 ✅ ✅
# python -u main.py eval care "NADH_CAREmodel_0713_cervix_SSIMR2_ap5_Wavelet_bior4p4" cwd=.. fad_data=NV_713_FAD_healthy.npz loss=ssimr2_loss wavelet_function=bior4.4

# NADH CARE + SSIMR2 ap5 Wavelet bior1.1 ✅ ✅
# python -u main.py train care "NADH_CAREmodel_0713_cervix_SSIMR2_ap5_Wavelet_bior1p1" cwd=.. nadh_data=NV_713_NADH_healthy.npz loss=ssimr2_loss wavelet_function=bior1.1 loss_alpha=0.5
# python -u main.py eval care "NADH_CAREmodel_0713_cervix_SSIMR2_ap5_Wavelet_bior1p1" cwd=.. fad_data=NV_713_FAD_healthy.npz nadh_data=NV_713_NADH_healthy.npz wavelet_function=bior1.1

# NADH CARE + SSIMR2 ap5 Wavelet bior1.1 SSIM filter_size=3 ✅ ✅
# python -u main.py train care "NADH_CAREmodel_0713_cervix_SSIMR2_ap5_Wavelet_bior1p1_f3" cwd=.. nadh_data=NV_713_NADH_healthy.npz loss=ssimr2_loss wavelet_function=bior1.1 loss_alpha=0.5
# python -u main.py eval care "NADH_CAREmodel_0713_cervix_SSIMR2_ap5_Wavelet_bior1p1_f3" cwd=.. fad_data=NV_713_FAD_healthy.npz nadh_data=NV_713_NADH_healthy.npz wavelet_function=bior1.1

# NADH CARE + SSIMPCC ap5 Wavelet bior1.1 SSIM ✅ ✅
# python -u main.py train care "NADH_CAREmodel_0713_cervix_SSIMPCC_ap5_Wavelet_bior1p1" cwd=.. nadh_data=NV_713_NADH_healthy.npz loss=ssimpcc_loss wavelet_function=bior1.1 loss_alpha=0.5
# python -u main.py eval care "NADH_CAREmodel_0713_cervix_SSIMPCC_ap5_Wavelet_bior1p1" cwd=.. fad_data=NV_713_FAD_healthy.npz nadh_data=NV_713_NADH_healthy.npz wavelet_function=bior1.1

# NADH CARE + SSIMR2 ap5 Wavelet bior1.1 SSIM deep ✅ ✅
# python -u main.py train care "NADH_CAREmodel_0713_cervix_SSIMR2_ap5_Wavelet_bior1p1_deep" cwd=.. nadh_data=NV_713_NADH_healthy.npz loss=ssimr2_loss wavelet_function=bior1.1 loss_alpha=0.5
# python -u main.py eval care "NADH_CAREmodel_0713_cervix_SSIMR2_ap5_Wavelet_bior1p1_deep" cwd=.. fad_data=NV_713_FAD_healthy.npz nadh_data=NV_713_NADH_healthy.npz wavelet_function=bior1.1

# NADH CARE + SSIMR2 ap5 Wavelet bior1.1 SSIM deep filter_size=3✅ ✅
# python -u main.py train care "NADH_CAREmodel_0713_cervix_SSIMR2_ap5_Wavelet_bior1p1_deep_fs3" cwd=.. nadh_data=NV_713_NADH_healthy.npz loss=ssimr2_loss wavelet_function=bior1.1 loss_alpha=0.5
# python -u main.py eval care "NADH_CAREmodel_0713_cervix_SSIMR2_ap5_Wavelet_bior1p1_deep_fs3" cwd=.. fad_data=NV_713_FAD_healthy.npz nadh_data=NV_713_NADH_healthy.npz wavelet_function=bior1.1

# NADH CARE + SSIMR2 ap5 Wavelet bior1.1 SSIM deep (new_data) ✅ ✅
# python -u main.py train care "NADH_CAREmodel_0713_cervix_SSIMR2_ap5_Wavelet_bior1p1_deep" cwd=.. nadh_data=NV_823_NADH_healthy.npz loss=ssimr2_loss wavelet_function=bior1.1 loss_alpha=0.5
# python -u main.py eval care "NADH_CAREmodel_0713_cervix_SSIMR2_ap5_Wavelet_bior1p1_deep" cwd=.. fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz wavelet_function=bior1.1 val_seed=0

# NADH CARE + SSIML2 ap84 SSIM deep (new_data)  ✅
#python -u main.py eval care "NADH_CAREmodel_0713_cervix_SSIML2_BS50_Deep_fs3" cwd=.. fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz val_seed=0

# NADH CARE + SSIMR2 ap5 Wavelet bior1.1 SSIMR2 deep (new_data loader) ✅ ✅
# python -u main.py train care "NADH_CAREmodel_823_cervix_SSIMR2_ap5_Wavelet_bior1p1_deep_seed0" cwd=.. nadh_data=NV_823_NADH_healthy.npz loss=ssimr2_loss wavelet_function=bior1.1 loss_alpha=0.5 val_seed=0 val_split=4 test_split=8 test_flag=1
# python -u main.py eval care "NADH_CAREmodel_823_cervix_SSIMR2_ap5_Wavelet_bior1p1_deep_seed0" cwd=.. fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz wavelet_function=bior1.1 val_seed=0 val_split=4 test_split=8 test_flag=1

# NADH CARE + SSIMR2 ap5 Wavelet bior1.1 SSIMR2 deep (new_data loader) ✅ ✅
# python -u main.py eval care "NADH_CAREmodel_0713_cervix_SSIMR2_ap5_Wavelet_bior1p1_deep_fs3" cwd=.. fad_data=NV_907_FAD_Colpo.npz nadh_data=NV_907_NADH_Colpo.npz wavelet_function=bior1.1 val_seed=0 val_split=4 test_split=8 test_flag=1 train_mode=0
# python -u main.py eval care "NADH_CAREmodel_0713_cervix_SSIMR2_ap5_Wavelet_bior1p1_deep_fs3" cwd=.. fad_data=NV_907_FAD_Leep.npz nadh_data=NV_907_NADH_Leep.npz wavelet_function=bior1.1 val_seed=0 val_split=4 test_split=8 test_flag=1 train_mode=0

# NADH CARE + SSIMR2 ap5 deep (new_data loader) ✅ ✅ 
# python -u main.py eval care "NADH_CAREmodel_0713_cervix_SSIML2_BS50_Deep_fs3" cwd=.. fad_data=NV_907_FAD_Colpo.npz nadh_data=NV_907_NADH_Colpo.npz val_seed=0 val_split=4 test_split=8 test_flag=1 train_mode=0
# python -u main.py eval care "NADH_CAREmodel_0713_cervix_SSIML2_BS50_Deep_fs3" cwd=.. fad_data=NV_907_FAD_Leep.npz nadh_data=NV_907_NADH_Leep.npz val_seed=0 val_split=4 test_split=8 test_flag=1 train_mode=0

# NADH CARE + SSIML2 ap84 SSIM deep seed 0 ✅ ✅ ✅ ✅ 39729778       
# python -u main.py train care "NADH_CARE_0823_cervix_SSIML2_seed0" cwd=.. nadh_data=NV_823_NADH_healthy.npz  loss_alpha=0.84 val_seed=0 val_split=4 test_split=8 test_flag=1 ssim_FSize=3 ssim_FSig=0.5
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIML2_seed0" cwd=.. fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz val_seed=0 val_split=4 test_split=8 test_flag=1 
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIML2_seed0" cwd=.. fad_data=NV_907_FAD_Colpo.npz nadh_data=NV_907_NADH_Colpo.npz val_seed=0 val_split=4 test_split=8 test_flag=1 train_mode=0
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIML2_seed0" cwd=.. fad_data=NV_907_FAD_Leep.npz nadh_data=NV_907_NADH_Leep.npz val_seed=0 val_split=4 test_split=8 test_flag=1 train_mode=0

# NADH CARE + SSIMR2 ap5 Wavelet Bior1.1 seed 0 ✅ ✅ ✅ ✅ 40039233          
# python -u main.py train care "NADH_CARE_0823_cervix_SSIMR2_ap5_Wavelet_bior1p1_seed0" cwd=.. nadh_data=NV_823_NADH_healthy.npz  loss_alpha=0.5 val_seed=0 val_split=4 test_split=8 test_flag=1 ssim_FSize=11 ssim_FSig=1.5 wavelet_function=bior1.1
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIMR2_ap5_Wavelet_bior1p1_seed0" cwd=.. fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz val_seed=0 val_split=4 test_split=8 test_flag=1 wavelet_function=bior1.1
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIMR2_ap5_Wavelet_bior1p1_seed0" cwd=.. fad_data=NV_907_FAD_Colpo.npz nadh_data=NV_907_NADH_Colpo.npz val_seed=0 val_split=4 test_split=8 test_flag=1 train_mode=0 wavelet_function=bior1.1
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIMR2_ap5_Wavelet_bior1p1_seed0" cwd=.. fad_data=NV_907_FAD_Leep.npz nadh_data=NV_907_NADH_Leep.npz val_seed=0 val_split=4 test_split=8 test_flag=1 train_mode=0 wavelet_function=bior1.1

# NADH CARE + SSIMR2 ap5 Wavelet Bior1.1 seed 1 ✅ ✅ ✅ ✅ 40039596           
# python -u main.py train care "NADH_CARE_0823_cervix_SSIMR2_ap5_Wavelet_bior1p1_seed1" cwd=.. nadh_data=NV_823_NADH_healthy.npz  loss_alpha=0.5 val_seed=1 val_split=4 test_split=8 test_flag=1 ssim_FSize=11 ssim_FSig=1.5 wavelet_function=bior1.1
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIMR2_ap5_Wavelet_bior1p1_seed1" cwd=.. fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz val_seed=1 val_split=4 test_split=8 test_flag=1 wavelet_function=bior1.1
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIMR2_ap5_Wavelet_bior1p1_seed1" cwd=.. fad_data=NV_907_FAD_Colpo.npz nadh_data=NV_907_NADH_Colpo.npz val_seed=1 val_split=4 test_split=8 test_flag=1 train_mode=0 wavelet_function=bior1.1
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIMR2_ap5_Wavelet_bior1p1_seed1" cwd=.. fad_data=NV_907_FAD_Leep.npz nadh_data=NV_907_NADH_Leep.npz val_seed=1 val_split=4 test_split=8 test_flag=1 train_mode=0 wavelet_function=bior1.1

# NADH CARE + SSIMR2 ap5 Wavelet Bior1.1 seed 2 ✅ ✅ ✅ ✅ 40039655          
# python -u main.py train care "NADH_CARE_0823_cervix_SSIMR2_ap5_Wavelet_bior1p1_seed2" cwd=.. nadh_data=NV_823_NADH_healthy.npz  loss_alpha=0.5 val_seed=2 val_split=4 test_split=8 test_flag=1 ssim_FSize=11 ssim_FSig=1.5 wavelet_function=bior1.1
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIMR2_ap5_Wavelet_bior1p1_seed2" cwd=.. fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz val_seed=2 val_split=4 test_split=8 test_flag=1 wavelet_function=bior1.1
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIMR2_ap5_Wavelet_bior1p1_seed2" cwd=.. fad_data=NV_907_FAD_Colpo.npz nadh_data=NV_907_NADH_Colpo.npz val_seed=2 val_split=4 test_split=8 test_flag=1 train_mode=0 wavelet_function=bior1.1
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIMR2_ap5_Wavelet_bior1p1_seed2" cwd=.. fad_data=NV_907_FAD_Leep.npz nadh_data=NV_907_NADH_Leep.npz val_seed=2 val_split=4 test_split=8 test_flag=1 train_mode=0 wavelet_function=bior1.1

# NADH CARE + SSIMR2 ap5 Wavelet Bior1.1 seed 3 ✅ ✅ ✅ ✅ 40039712           
# python -u main.py train care "NADH_CARE_0823_cervix_SSIMR2_ap5_Wavelet_bior1p1_seed3" cwd=.. nadh_data=NV_823_NADH_healthy.npz  loss_alpha=0.5 val_seed=3 val_split=4 test_split=8 test_flag=1 ssim_FSize=11 ssim_FSig=1.5 wavelet_function=bior1.1
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIMR2_ap5_Wavelet_bior1p1_seed3" cwd=.. fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz val_seed=3 val_split=4 test_split=8 test_flag=1 wavelet_function=bior1.1
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIMR2_ap5_Wavelet_bior1p1_seed3" cwd=.. fad_data=NV_907_FAD_Colpo.npz nadh_data=NV_907_NADH_Colpo.npz val_seed=3 val_split=4 test_split=8 test_flag=1 train_mode=0 wavelet_function=bior1.1
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIMR2_ap5_Wavelet_bior1p1_seed3" cwd=.. fad_data=NV_907_FAD_Leep.npz nadh_data=NV_907_NADH_Leep.npz val_seed=3 val_split=4 test_split=8 test_flag=1 train_mode=0 wavelet_function=bior1.1

# NADH CARE + SSIMR2 ap5 Wavelet Bior1.1 seed 4 ✅ ✅ ✅ ✅ 40039771
# python -u main.py train care "NADH_CARE_0823_cervix_SSIMR2_ap5_Wavelet_bior1p1_seed4" cwd=.. nadh_data=NV_823_NADH_healthy.npz  loss_alpha=0.5 val_seed=4 val_split=4 test_split=8 test_flag=1 ssim_FSize=11 ssim_FSig=1.5 wavelet_function=bior1.1
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIMR2_ap5_Wavelet_bior1p1_seed4" cwd=.. fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz val_seed=4 val_split=4 test_split=8 test_flag=1 wavelet_function=bior1.1
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIMR2_ap5_Wavelet_bior1p1_seed4" cwd=.. fad_data=NV_907_FAD_Colpo.npz nadh_data=NV_907_NADH_Colpo.npz val_seed=4 val_split=4 test_split=8 test_flag=1 train_mode=0 wavelet_function=bior1.1
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIMR2_ap5_Wavelet_bior1p1_seed4" cwd=.. fad_data=NV_907_FAD_Leep.npz nadh_data=NV_907_NADH_Leep.npz val_seed=4 val_split=4 test_split=8 test_flag=1 train_mode=0 wavelet_function=bior1.1

# FAD CARE + SSIML2 ap84 SSIM deep seed 0 ✅ ✅ ✅ ✅  40037557   
# python -u main.py train care "FAD_CARE_0823_cervix_SSIML2_seed0" cwd=.. fad_data=NV_823_FAD_healthy.npz  loss_alpha=0.84 val_seed=0 val_split=4 test_split=8 test_flag=1 ssim_FSize=3 ssim_FSig=0.5
# python -u main.py eval care "FAD_CARE_0823_cervix_SSIML2_seed0" cwd=.. fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz val_seed=0 val_split=4 test_split=8 test_flag=1 
# python -u main.py eval care "FAD_CARE_0823_cervix_SSIML2_seed0" cwd=.. fad_data=NV_907_FAD_Colpo.npz nadh_data=NV_907_NADH_Colpo.npz val_seed=0 val_split=4 test_split=8 test_flag=1 train_mode=0
# python -u main.py eval care "FAD_CARE_0823_cervix_SSIML2_seed0" cwd=.. fad_data=NV_907_FAD_Leep.npz nadh_data=NV_907_NADH_Leep.npz val_seed=0 val_split=4 test_split=8 test_flag=1 train_mode=0

# CARE + SSIML2 ap84 SSIM deep seed 0 2 Frame ✅ ✅ ✅ ✅  40037557   
# python -u main.py train care "NADH_CARE_0823_cervix_SSIML2_2frame_seed0" cwd=.. nadh_data=NV_907_NADH_healthy.npz  loss_alpha=0.84 val_seed=0 val_split=4 test_split=8 test_flag=1 ssim_FSize=3 ssim_FSig=0.5
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIML2_2frame_seed0" cwd=.. fad_data=NV_907_FAD_healthy.npz nadh_data=NV_907_NADH_healthy.npz val_seed=0 val_split=4 test_split=8 test_flag=1 
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIML2_2frame_seed0" cwd=.. fad_data=NV_907_FAD_Colpo.npz nadh_data=NV_907_NADH_Colpo.npz val_seed=0 val_split=4 test_split=8 test_flag=1 train_mode=0
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIML2_2frame_seed0" cwd=.. fad_data=NV_907_FAD_Leep.npz nadh_data=NV_907_NADH_Leep.npz val_seed=0 val_split=4 test_split=8 test_flag=1 train_mode=0

# NADH CARE + SSIML2 ap84 SSIM deep seed 1 ✅ ✅ ✅ ✅ cc1gpu001 39713146   
# python -u main.py train care "NADH_CARE_0823_cervix_SSIML2_seed1" cwd=.. nadh_data=NV_823_NADH_healthy.npz  loss_alpha=0.84 val_seed=1 val_split=4 test_split=8 test_flag=1 ssim_FSize=3 ssim_FSig=0.5
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIML2_seed1" cwd=.. fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz val_seed=1 val_split=4 test_split=8 test_flag=1 
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIML2_seed1" cwd=.. fad_data=NV_907_FAD_Colpo.npz nadh_data=NV_907_NADH_Colpo.npz val_seed=1 val_split=4 test_split=8 test_flag=1 train_mode=0
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIML2_seed1" cwd=.. fad_data=NV_907_FAD_Leep.npz nadh_data=NV_907_NADH_Leep.npz val_seed=1 val_split=4 test_split=8 test_flag=1 train_mode=0

# NADH CARE + SSIML2 ap84 SSIM deep seed 2 ✅ ✅ ✅ ✅ cc1gpu004 39713941   
# python -u main.py train care "NADH_CARE_0823_cervix_SSIML2_seed2" cwd=.. nadh_data=NV_823_NADH_healthy.npz  loss_alpha=0.84 val_seed=2 val_split=4 test_split=8 test_flag=1 ssim_FSize=3 ssim_FSig=0.5
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIML2_seed2" cwd=.. fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz val_seed=2 val_split=4 test_split=8 test_flag=1 
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIML2_seed2" cwd=.. fad_data=NV_907_FAD_Colpo.npz nadh_data=NV_907_NADH_Colpo.npz val_seed=2 val_split=4 test_split=8 test_flag=1 train_mode=0
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIML2_seed2" cwd=.. fad_data=NV_907_FAD_Leep.npz nadh_data=NV_907_NADH_Leep.npz val_seed=2 val_split=4 test_split=8 test_flag=1 train_mode=0

# NADH CARE + SSIML2 ap84 SSIM deep seed 3 ✅ ✅ ✅ ✅  39730258      
# python -u main.py train care "NADH_CARE_0823_cervix_SSIML2_seed3" cwd=.. nadh_data=NV_823_NADH_healthy.npz  loss_alpha=0.84 val_seed=3 val_split=4 test_split=8 test_flag=1 ssim_FSize=3 ssim_FSig=0.5
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIML2_seed3" cwd=.. fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz val_seed=3 val_split=4 test_split=8 test_flag=1 
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIML2_seed3" cwd=.. fad_data=NV_907_FAD_Colpo.npz nadh_data=NV_907_NADH_Colpo.npz val_seed=3 val_split=4 test_split=8 test_flag=1 train_mode=0
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIML2_seed3" cwd=.. fad_data=NV_907_FAD_Leep.npz nadh_data=NV_907_NADH_Leep.npz val_seed=3 val_split=4 test_split=8 test_flag=1 train_mode=0

# NADH CARE + SSIML2 ap84 SSIM deep seed 4 ✅ ✅ ✅ ✅  39730108         
# python -u main.py train care "NADH_CARE_0823_cervix_SSIML2_seed4" cwd=.. nadh_data=NV_823_NADH_healthy.npz  loss_alpha=0.84 val_seed=4 val_split=4 test_split=8 test_flag=1 ssim_FSize=3 ssim_FSig=0.5
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIML2_seed4" cwd=.. fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz val_seed=4 val_split=4 test_split=8 test_flag=1 
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIML2_seed4" cwd=.. fad_data=NV_907_FAD_Colpo.npz nadh_data=NV_907_NADH_Colpo.npz val_seed=4 val_split=4 test_split=8 test_flag=1 train_mode=0
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIML2_seed4" cwd=.. fad_data=NV_907_FAD_Leep.npz nadh_data=NV_907_NADH_Leep.npz val_seed=4 val_split=4 test_split=8 test_flag=1 train_mode=0

# NADH CARE + SSIMFFL ap84 SSIM deep seed 0 ✅ ✅ ✅ ✅ 39729778       
# python -u main.py train care "NADH_CARE_0823_cervix_SSIMFFL_seed0" cwd=.. nadh_data=NV_823_NADH_healthy.npz  loss_alpha=0.84 val_seed=0 val_split=4 test_split=8 test_flag=1 ssim_FSize=3 ssim_FSig=0.5 loss=SSIMFFL
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIMFFL_seed0" cwd=.. fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz val_seed=0 val_split=4 test_split=8 test_flag=1 
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIMFFL_seed0" cwd=.. fad_data=NV_907_FAD_Colpo.npz nadh_data=NV_907_NADH_Colpo.npz val_seed=0 val_split=4 test_split=8 test_flag=1 train_mode=0
# python -u main.py eval care "NADH_CARE_0823_cervix_SSIMFFL_seed0" cwd=.. fad_data=NV_907_FAD_Leep.npz nadh_data=NV_907_NADH_Leep.npz val_seed=0 val_split=4 test_split=8 test_flag=1 train_mode=0

# NADH CARE + SSIMFFL ap84 SSIM deep seed 0 ✅ ✅ ✅ ✅ 39729778       
# python -u main.py config.json
# python -u main.py config.json mode=eval fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz
# python -u main.py config.json mode=eval fad_data=NV_907_FAD_Colpo.npz nadh_data=NV_907_NADH_Colpo.npz train_mode=0
# python -u main.py config.json mode=eval fad_data=NV_907_FAD_Leep.npz nadh_data=NV_907_NADH_Leep.npz train_mode=0

### PCC ##########################################################

# NADH RCAN SSIMPCC (ap84) ✅ (Though Premptied) ⏰
# python -u main.py eval rcan "NADH_model_0713_cervix_SSIMPCC" cwd=.. nadh_data=NV_713_NADH_healthy.npz fad_data=NV_713_FAD_healthy.npz loss=ssimpcc_loss loss_alpha=0.84
# python -u main.py eval rcan "NADH_model_0713_cervix_SSIMPCC_dup" cwd=.. nadh_data=NV_713_NADH_healthy.npz fad_data=NV_713_FAD_healthy.npz loss=ssimpcc_loss loss_alpha=0.84 

# NADH RCAN SSIMPCC ap5 ✅ ⏰ 
# python -u main.py eval rcan "NADH_model_0713_cervix_SSIMPCC_ap5" cwd=.. nadh_data=NV_713_NADH_healthy.npz fad_data=NV_713_FAD_healthy.npz loss=ssimpcc_loss loss_alpha=0.5

##################################################################

### SRGAN ########################################################

# SRGAN Trial 1
# python -u srgan.py

# SRGAN Trial 2 run only ✅ ✅ 
# python -u main.py train srgan "NADH_SRGAN_0823_cervix_standard" cwd=.. nadh_data=NV_823_NADH_healthy.npz loss_alpha=0
# python -u main.py eval srgan "NADH_SRGAN_0823_cervix_standard" cwd=.. nadh_data=NV_823_NADH_healthy.npz fad_data=NV_823_FAD_healthy.npz loss_alpha=0

# NADH SRGAN + MSE ✅ ✅
# python -u main.py train srgan "NADH_SRGAN_0823_cervix_mse" cwd=.. nadh_data=NV_823_NADH_healthy.npz num_residual_blocks=6 loss_alpha=0 
# python -u main.py eval srgan "NADH_SRGAN_0823_cervix_mse" cwd=.. fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz num_residual_blocks=6

# NADH SRGAN + 8 residual blocks ✅ ✅
# python -u main.py train srgan "NADH_SRGAN_0823_cervix_mse_rb8" cwd=.. nadh_data=NV_823_NADH_healthy.npz num_residual_blocks=8 loss_alpha=0 
# python -u main.py eval srgan "NADH_SRGAN_0823_cervix_mse_rb8" cwd=.. fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz num_residual_blocks=8

# NADH SRGAN 5 residual blocks seed 1✅ ✅ ✅ ✅ 39729970     
# python -u main.py train srgan "NADH_SRGAN_0823_cervix_SSIMR2_seed1" cwd=.. nadh_data=NV_823_NADH_healthy.npz loss_alpha=0 val_seed=1 val_split=4 test_split=8 test_flag=1 epochs=500
# python -u main.py eval srgan "NADH_SRGAN_0823_cervix_SSIMR2_seed1" cwd=.. fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz val_seed=1 val_split=4 test_split=8 test_flag=1
# python -u main.py eval srgan "NADH_SRGAN_0823_cervix_SSIMR2_seed1" cwd=.. fad_data=NV_907_FAD_Colpo.npz nadh_data=NV_907_NADH_Colpo.npz val_seed=1 val_split=4 test_split=8 test_flag=1 train_mode=0
# python -u main.py eval srgan "NADH_SRGAN_0823_cervix_SSIMR2_seed1" cwd=.. fad_data=NV_907_FAD_Leep.npz nadh_data=NV_907_NADH_Leep.npz val_seed=1 val_split=4 test_split=8 test_flag=1 train_mode=0

# NADH SRGAN 5 residual blocks seed 0✅ ✅ ✅ ✅ 39730537 
# python -u main.py train srgan "NADH_SRGAN_0823_cervix_SSIMR2_seed0" cwd=.. nadh_data=NV_823_NADH_healthy.npz loss_alpha=0 val_seed=0 val_split=4 test_split=8 test_flag=1 epochs=500
# python -u main.py eval srgan "NADH_SRGAN_0823_cervix_SSIMR2_seed0" cwd=.. fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz val_seed=0 val_split=4 test_split=8 test_flag=1
# python -u main.py eval srgan "NADH_SRGAN_0823_cervix_SSIMR2_seed0" cwd=.. fad_data=NV_907_FAD_Colpo.npz nadh_data=NV_907_NADH_Colpo.npz val_seed=0 val_split=4 test_split=8 test_flag=1 train_mode=0
# python -u main.py eval srgan "NADH_SRGAN_0823_cervix_SSIMR2_seed0" cwd=.. fad_data=NV_907_FAD_Leep.npz nadh_data=NV_907_NADH_Leep.npz val_seed=0 val_split=4 test_split=8 test_flag=1 train_mode=0

# NADH SRGAN 5 residual blocks seed 2✅ ✅ ✅ ✅ 39741169 
# python -u main.py train srgan "NADH_SRGAN_0823_cervix_SSIMR2_seed2" cwd=.. nadh_data=NV_823_NADH_healthy.npz loss_alpha=0 val_seed=2 val_split=4 test_split=8 test_flag=1 epochs=500
# python -u main.py eval srgan "NADH_SRGAN_0823_cervix_SSIMR2_seed2" cwd=.. fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz val_seed=2 val_split=4 test_split=8 test_flag=1
# python -u main.py eval srgan "NADH_SRGAN_0823_cervix_SSIMR2_seed2" cwd=.. fad_data=NV_907_FAD_Colpo.npz nadh_data=NV_907_NADH_Colpo.npz val_seed=2 val_split=4 test_split=8 test_flag=1 train_mode=0
# python -u main.py eval srgan "NADH_SRGAN_0823_cervix_SSIMR2_seed2" cwd=.. fad_data=NV_907_FAD_Leep.npz nadh_data=NV_907_NADH_Leep.npz val_seed=2 val_split=4 test_split=8 test_flag=1 train_mode=0

# NADH SRGAN 5 residual blocks seed 3✅ ✅ ✅ ✅ 39741257 
# python -u main.py train srgan "NADH_SRGAN_0823_cervix_SSIMR2_seed3" cwd=.. nadh_data=NV_823_NADH_healthy.npz loss_alpha=0 val_seed=3 val_split=4 test_split=8 test_flag=1 epochs=500
# python -u main.py eval srgan "NADH_SRGAN_0823_cervix_SSIMR2_seed3" cwd=.. fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz val_seed=3 val_split=4 test_split=8 test_flag=1
# python -u main.py eval srgan "NADH_SRGAN_0823_cervix_SSIMR2_seed3" cwd=.. fad_data=NV_907_FAD_Colpo.npz nadh_data=NV_907_NADH_Colpo.npz val_seed=3 val_split=4 test_split=8 test_flag=1 train_mode=0
# python -u main.py eval srgan "NADH_SRGAN_0823_cervix_SSIMR2_seed3" cwd=.. fad_data=NV_907_FAD_Leep.npz nadh_data=NV_907_NADH_Leep.npz val_seed=3 val_split=4 test_split=8 test_flag=1 train_mode=0

# NADH SRGAN 5 residual blocks seed 4✅ ✅ ✅ ✅ 39741376 
# python -u main.py train srgan "NADH_SRGAN_0823_cervix_SSIMR2_seed4" cwd=.. nadh_data=NV_823_NADH_healthy.npz loss_alpha=0 val_seed=4 val_split=4 test_split=8 test_flag=1 epochs=500
# python -u main.py eval srgan "NADH_SRGAN_0823_cervix_SSIMR2_seed4" cwd=.. fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz val_seed=4 val_split=4 test_split=8 test_flag=1
# python -u main.py eval srgan "NADH_SRGAN_0823_cervix_SSIMR2_seed4" cwd=.. fad_data=NV_907_FAD_Colpo.npz nadh_data=NV_907_NADH_Colpo.npz val_seed=4 val_split=4 test_split=8 test_flag=1 train_mode=0
# python -u main.py eval srgan "NADH_SRGAN_0823_cervix_SSIMR2_seed4" cwd=.. fad_data=NV_907_FAD_Leep.npz nadh_data=NV_907_NADH_Leep.npz val_seed=4 val_split=4 test_split=8 test_flag=1 train_mode=0

# NADH SRGAN  residual blocks seed 0
python -u main.py train srgan "NADH_SRGAN_0823_cervix_SSIMR2_seed0_new" cwd=.. nadh_data=NV_823_NADH_healthy.npz loss_alpha=0 val_seed=0 val_split=4 test_split=8 test_flag=1 epochs=1000
python -u main.py eval srgan "NADH_SRGAN_0823_cervix_SSIMR2_seed0_new" cwd=.. fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz val_seed=0 val_split=4 test_split=8 test_flag=1
python -u main.py eval srgan "NADH_SRGAN_0823_cervix_SSIMR2_seed0_new" cwd=.. fad_data=NV_907_FAD_Colpo.npz nadh_data=NV_907_NADH_Colpo.npz val_seed=0 val_split=4 test_split=8 test_flag=1 train_mode=0
python -u main.py eval srgan "NADH_SRGAN_0823_cervix_SSIMR2_seed0_new" cwd=.. fad_data=NV_907_FAD_Leep.npz nadh_data=NV_907_NADH_Leep.npz val_seed=0 val_split=4 test_split=8 test_flag=1 train_mode=0

##################################################################
### Resnet ########################################################

# NADH resnet + SSIMR2 ✅ ✅
# python -u main.py train resnet "NADH_Resnet_0823_cervix_SSIMR2_ap5" cwd=.. nadh_data=NV_823_NADH_healthy.npz loss=ssimr2_loss loss_alpha=0.5 num_residual_blocks=6
# python -u main.py eval resnet "NADH_Resnet_0823_cervix_SSIMR2_ap5" cwd=.. fad_data=NV_823_FAD_healthy.npz nadh_data=NV_823_NADH_healthy.npz num_residual_blocks=6

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