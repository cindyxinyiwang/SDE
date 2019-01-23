#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=1000:00:00
#SBATCH --nodes=1
#SBATCH --mem=12g
#SBATCH --job-name="multlin"
##SBATCH --mail-user=gneubig@cs.cmu.edu
##SBATCH --mail-type=ALL
##SBATCH --requeue
#Specifies that the job will be requeued after a node failure.
#The default is that the job will not be requeued.
set -e
export PYTHONPATH="$(pwd)"                                                       
export CUDA_VISIBLE_DEVICES="3" 
version=v7_abl_s1
mkdir -p outputs_"$version"
for f in `ls scripts/cfg_"$version"/ | grep -v trans.sh$`; do
  f1=`basename $f .sh`
  if [[ ! -e outputs_"$version"/$f1.started ]]; then
    echo "running $f1"
    touch outputs_"$version"/$f1.started
    hostname
    nvidia-smi
    ./scripts/cfg_"$version"/$f
  else
    echo "already started $f1"
  fi
done

