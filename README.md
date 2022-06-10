# DAFE

#!/bin/bash
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu_p4
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=50
#SBATCH --output=multi_domain_1365.log.out
#SBATCH --error=multi_domain_1365.log.err
#SBATCH -A sfz@gpu
module purge
source /gpfsdswork/projects/rech/sfz/utt84zy/anaconda3/etc/profile.d/conda.sh
conda activate mdmt
module load cuda/11.2
module load cudnn/8.1.1.33-cuda
module load nccl/2.4.2-1+cuda10.1
module load gcc/10.1.0-cuda-openacc
module load cmake/3.18.0
which python
python -c 'import tensorflow; print("tensorflow OK"); import transformers; print("transformers OK")'
python -u practice.py train_elbo_topK_sparse_layer_multi_layer --config configs/config_1365.yml

