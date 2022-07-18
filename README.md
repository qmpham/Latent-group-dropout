# Latent-group-dropout

This is code for my paper "Latent Group Dropout for Multilingual and Multidomain Machine Translation" (Minh-Quang PHAM, François Yvon, Josep Crego) accepted to the Findings of NAACL 2022. 

# Requirements
* TensorFlow == 2.3
* Python == 3.7
* TensorFlow Probability == 0.13
* TensorFlow addons == 0.13
* OpenNMT-tf == 2.1.1

# Format of slurm bash
#!/bin/bash <br>
#SBATCH --gres=gpu:8 <br>
#SBATCH --partition=gpu_p4 <br>
#SBATCH --nodes=1 <br>
#SBATCH --time=20:00:00 <br>
#SBATCH --cpus-per-task=50 <br>
#SBATCH --output=multi_domain_1365.log.out <br>
#SBATCH --error=multi_domain_1365.log.err <br>
#SBATCH -A sfz@gpu <br>
module purge <br>
source /gpfsdswork/projects/rech/sfz/utt84zy/anaconda3/etc/profile.d/conda.sh <br>
conda activate mdmt <br>
module load cuda/11.2 <br>
module load cudnn/8.1.1.33-cuda <br>
module load nccl/2.4.2-1+cuda10.1 <br>
module load gcc/10.1.0-cuda-openacc <br>
module load cmake/3.18.0 <br>
which python <br>
python -c 'import tensorflow; print("tensorflow OK")' <br>
python -u practice.py train_elbo_topK_sparse_layer_multi_layer --config configs/config_1365.yml <br>

# Config file format (one example already provided in configs)

src: 

tgt: 

domain:

eval_src:

eval_ref:

eval_domain:

src_vocab: 

tgt_vocab: 

model_dir: 

batch_train_size: 3072

batch_type: tokens

experiment: TopK_Sparse_Layers_multi_layer

num_units: 512

num_heads: 8

ffn_inner_dim: 2048

dropout: 0.1

ffn_dropout: 0.1

attention_dropout: 0.1

version: 2

accumulation_step: 1

train_steps: 300000

num_domain_unit_group: 16

num_shared_units: 0

unit_group_size: 32

domain_group_allocation_num: 12

gumbel_temperature_decay: 1000

r_coeff: 0.00005

kl_coeff: 0.0001

latent_logit_lr: 0.001

min_temperature: 0.2

start_temperature: 0.2

step_duration: 8

save_every: 5000

eval_every: 10000

num_domains: 20

num_inspected_domains: 6

picking_prob: Natural

temperature: 1.0

learning_rate: 1.0

# Attention!!! Please do not change these following options

batch_type: tokens

experiment: TopK_Sparse_Layers_multi_layer

version: 2

num_domain_unit_group: 16

num_shared_units: 0

unit_group_size: 32

domain_group_allocation_num: 12

picking_prob: Natural

temperature: 1.0

# For the inference, please use this command

python -u practice.py translate_topK_sparse_layer_multi_layer --config configs/config_1365.yml --src configs/translate_file.yml

The ouput files will be located in model_dir/eval (or in this case is /gpfsdsstore/projects/rech/sfz/utt84zy/models/config_1365/eval)
