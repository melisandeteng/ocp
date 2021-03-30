#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=/miniscratch/%u/ocp_jobs/sample-%j.out
#SBATCH --error=/miniscratch/%u/ocp_jobs/sample-%j.err

## partition name
#SBATCH --partition=long
#SBATCH --mail-user=assouelr@mila.quebec
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH --exclude=rtx7,kepler4
#SBATCH --time=0:10:0
#SBATCH --job-name=ocp_pred
#SBATCH --ntasks-per-node=1

module load anaconda/3
source activate ocp-models

slurm_env_list=$(printenv | grep "SLURM_NODEID\|SLURMD_NODENAME")
master_node=$(echo $slurm_env_list | egrep -o 'leto[0-9]{1,4}|eos[0-9]{1,4}|bart[0-9]{1,4}|kepler[0-9]{1,4}|mila0[0-9]{1,4}|power9[1-2]{1,4}|apollov0[1-5]|apollor0[1-9]|apollor1[0-9]|rtx[0-9]{1,4}')
echo $slurm_env_list
echo $master_node

printenv | grep SLURM

dist_url="tcp://"
dist_url+=$master_node
dist_url+=":40000"
echo $dist_url

srun python main.py --mode predict --config-yml /home/mila/a/assouelr/ocp/configs/is2re/all/dimenet_plus_plus/dpp.yml --checkpoint /home/mila/a/assouelr/ocp/checkpoints/dimenetpp_all.pt --nonddp $true

#srun python main.py --mode predict --config-yml /home/mila/a/assouelr/ocp/configs/is2re/all/dimenet_plus_plus/dpp.yml --checkpoint /home/mila/a/assouelr/ocp/checkpoints/2021-03-26-01-43-28/checkpoint.pt --nonddp $true




