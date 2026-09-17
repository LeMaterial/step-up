#!/bin/bash

#SBATCH --job-name=step-up-bostmc
#SBATCH --output=train-bostmc-%j.out
#SBATCH --gres=gpu:volta:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

bash scripts/train.sh configs/bostmc.yaml

