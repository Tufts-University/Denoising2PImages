#!/bin/bash
#SBATCH -J FS_080822_Denoising   #job name
#SBATCH --time=05-00:0:00  #requested time
#SBATCH -p preempt     #running on "preempt" partition/queue
#SBATCH -N 1    #1 nodes
#SBATCH	-n 10   #10 tasks total
#SBATCH	-c 1   #using 1 cpu core/task
#SBATCH --gres=gpu:a100:1
##SBATCH --nodelist=p1cmp072
#SBATCH --exclude=cc1gpu004
#SBATCH --mem=20g  #requesting 2GB of RAM total
#SBATCH --output=Denoising_monorepo_test.%j.out  #saving standard output to file -- %j jobID -- %N nodename
#SBATCH --error=Denoising_monorepo_test.%j.err  #saving standard error to file -- %j jobID -- %N nodename
#SBATCH --mail-type=ALL    #email options
#SBATCH --mail-user=20193005@student.anatolia.edu.gr

module load anaconda/2021.05
source activate Denoising

cd Cervical-Project 
git pull
echo "Starting python script..."
echo "==========================================================\n"
python -u main.py train 