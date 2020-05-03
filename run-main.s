#!/bin/bash
# This line tells the shell how to execute this script, and is unrelated
# to SLURM.

# at the beginning of the script, lines beginning with "#SBATCH" are read by
# SLURM and used to set queueing options. You can comment out a SBATCH
# directive with a second leading #, eg:
##SBATCH --nodes=1

# we need 1 node, will launch a maximum of one task. The task uses 1 CPU cores
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12

# we expect the job to finish within 10 hours. If it takes longer than 1
# hours, SLURM can kill it:
#SBATCH --time=24:00:00

# we expect the job to use no more than 10GB of memory:
#SBATCH --mem=80GB

# we want the job to be named "xxxx" rather than something generated
# from the script name. s will affect the name of the job as reported
# by squeue:
#SBATCH --job-name=doorkey_GPU1

# when the job ends, send me an email at this email address.
# replace with your email address, and uncomment that line if you really need to receive an email.

#SBATCH --mail-type=END
#SBATCH --mail-user=wy547@nyu.edu

# both standard output and standard error are directed to the same file.
# It will be placed in the directory I submitted the job from and will
# have a name like slurm_12345.out

#SBATCH --output=slurm_%j.out

# require GPU
#SBATCH --gres=gpu:1


# once the first non-comment, non-SBATCH-directive line is encountered, SLURM
# stops looking for SBATCH directives. The remainder of the script is  executed
# as a normal Unix shell script

# first we ensure a clean running environment:
module purge
# and load the module for the software we are using:
module load  python3/intel/3.7.3
cd ~/virtual_RL
source RL01/bin/activate
# the script will have started running in $HOME, so we need to move into the
# directory we just created earlier

cd /scratch/wy547/CCM/rl-minigrid


# now start the job:

python3 train.py --algo ppo --env MiniGrid-DoorKey-8x8-v0 --model Doorkey --save-interval 10 --exp-name seed_01 --seed 1 --comet-project-name doorkey-m

# Leave a few empty lines in the end to avoid occasional EOF trouble.

