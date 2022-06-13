import sys

import argparse
import logging
import yaml
import tensorflow as tf
import tensorflow_addons as tfa
import opennmt as onmt
import io
import os
import utils
from opennmt import START_OF_SENTENCE_ID
from opennmt import END_OF_SENTENCE_ID
from opennmt.utils.misc import print_bytes
from opennmt.data import dataset as dataset_util
from opennmt.optimizers import utils as optimizer_util
from utils.utils_ import average_checkpoints, average_checkpoints_tf2_3
tf.get_logger().setLevel(logging.INFO)
from utils.my_inputter import My_inputter, LDR_inputter, DC_inputter, ProbInputter, ProbInputter_v1
from opennmt.models.sequence_to_sequence import SequenceToSequence
from model import Multi_domain_SequenceToSequence_Instace_Aware_TopK_sparse_multi_layer, Multi_domain_SequenceToSequence_TopK_sparse, Multi_domain_SequenceToSequence_TopK_sparse_multi_layer_v0, Multi_domain_SequenceToSequence_TopK_sparse_multi_layer_v1, Multi_domain_SequenceToSequence_TopK_sparse_multi_layer, Multi_domain_SequenceToSequence_sparse, Priming_SequenceToSequence, Priming_SequenceToSequence_v1, Multi_domain_SequenceToSequence, LDR_SequenceToSequence, SequenceToSequence_WDC, LDR_SequenceToSequence_v1, SequenceToSequence_with_dprob, Multi_domain_SequenceToSequence_DRO
from encoders.self_attention_encoder import *
from decoders.self_attention_decoder import *
import numpy as np
from utils.dataprocess import merge_map_fn, create_meta_trainining_dataset, create_trainining_dataset, create_multi_domain_meta_trainining_dataset
from opennmt.utils import BLEUScorer
from opennmt.inputters.text_inputter import WordEmbedder, TextInputter
from utils.utils_ import variance_scaling_initialier, MultiBLEUScorer, create_slurm_strategy
import task
from optimizer import schedules as my_schedules
from opennmt.encoders.self_attention_encoder import SelfAttentionEncoder
from layers.layers import Regulation_Gate, Multi_domain_FeedForwardNetwork_v7, Multi_domain_FeedForwardNetwork_v8, Multi_domain_FeedForwardNetwork_v6, Multi_domain_Gate_v1, Multi_domain_FeedForwardNetwork_v5, Multi_domain_FeedForwardNetwork, Multi_domain_FeedForwardNetwork_v2, DAFE, Multi_domain_FeedForwardNetwork_v1, Multi_domain_FeedForwardNetwork_v0
def main():
  seed = 1234
  tf.random.set_seed(seed)
  np.random.seed(seed) 
  #tf.random.set_seed(seed)
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("run", choices=["train","train_tf_25","translate_topK_sparse_layer_multi_layer_cluster","finetune_elbo_topK_sparse_layer_multi_layer","train_elbo_sparse_layer","train_elbo_topK_sparse_layer_multi_layer","fewshot_elbo_topK_sparse_layer_multi_layer","train_elbo_topK_sparse_layer_multi_layer_v1","translatev3_tf_25","train_elbo_topK_sparse_layer","translate_topK_sparse_layer","translate_topK_sparse_layer_cluster","translate_topK_sparse_layer_multi_layer","translate_sparse_layer","priming_translate","priming_train_chasing", "priming_translate_chasing", "priming_train","CL_marine","train_domain_mixing_residual","train_L2W","train_IW_v0","train_NGD_L2W_v1","train_L2W_v2","train_L2W_g","train_L2W_v3","debug_L2W_v1","debug_L2W_v2","debug_L2W_v3","train_L2W_v1","train_NGD_L2W","debug_NGD","train_NGD", "continue_NGD", "score", "EWC_stat", "EWC_res_stat", "translate_farajan", "translate_farajan_residual", "train_adv", "train_wada", "finetune_noisy_v1", "finetune_wada", "finetune_wada_v1", "proxy", "debug_slurm_train", "metatrainv16", "proxy1","translatev7","kmeans", "translatev5", "translatev6","sentence_encode", "train_wdc", "train_denny_britz", "train_ldr", "visualize", "experimental_translate", "trainv3", "dcote", "metatrainv12", "trainv13", "trainv2", "trainv12", "metatrainv15", "translatev1", "trainv8", "translate", "translatev2", "translatev3", "metatrainv9", "metatrainv11", "debug","metatrainv1", "metatrainv2", "metatrainv3", "inspect", "metatrainv5", "metatrainv6", "metatrainv7", "metatrainv8", "metatrainv10", "elastic_finetune", "finetune"], help="Run type.")
  parser.add_argument("--config", help="configuration file")
  parser.add_argument("--config_root")
  parser.add_argument("--src")
  parser.add_argument("--src_files", nargs="+")
  parser.add_argument("--context", nargs="+")
  parser.add_argument("--emb_files", nargs="+")
  parser.add_argument("--n_clusters", type=int, default=30)
  parser.add_argument("--kmeans_save_path")
  parser.add_argument("--ckpt", default=None)
  parser.add_argument("--output", default="trans")
  parser.add_argument("--domain", type=int, default=0)
  parser.add_argument("--encoder_domain", default=0)
  parser.add_argument("--decoder_domain", default=0)
  parser.add_argument("--ref", default=None)
  parser.add_argument("--maxcount", type=int, default=3)
  parser.add_argument("--translation_file",default=None)
  parser.add_argument("--gpu_id",type= int, default=0)
  parser.add_argument("--stderr",default=None)
  parser.add_argument("--stdout",default=None)
  args = parser.parse_args()
  print("Running mode: ", args.run)
  config_file = args.config
  with open(config_file, "r") as stream:
      config = yaml.load(stream)

  data_config = {
      "source_vocabulary": config["src_vocab"],
      "source_1_vocabulary": config["src_vocab"],
      "source_2_vocabulary": config["src_vocab"],
      "source_3_vocabulary": config["tgt_vocab"],
      "target_vocabulary": config["tgt_vocab"]
    }

  if config.get("cross_device",False):
    print("training over multi workers")
    import horovod.tensorflow as hvd  # pylint: disable=import-outside-toplevel
    gpus = tf.config.list_physical_devices(device_type="GPU")
    hvd.init()
    is_master = hvd.rank() == 0
    if gpus:
      local_gpu = gpus[hvd.local_rank()]
      tf.config.experimental.set_visible_devices(local_gpu, device_type="GPU")
      gpus = [local_gpu]
      if is_master and not os.path.exists(os.path.join(config["model_dir"],"eval")):
        os.makedirs(os.path.join(config["model_dir"],"eval"))

  else:
    if not "translate" in args.run:
      gpus = tf.config.list_physical_devices('GPU')
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
      devices = tf.config.experimental.list_logical_devices(device_type="GPU")
      strategy = tf.distribute.MirroredStrategy(devices=[d.name for d in devices])
      if not os.path.exists(os.path.join(config["model_dir"],"eval")):
        os.makedirs(os.path.join(config["model_dir"],"eval"))
    else:
      gpus = tf.config.list_physical_devices('GPU')
      tf.config.experimental.set_visible_devices(gpus[int(args.gpu_id)], 'GPU')

  experiment = config.get("experiment","residual")
  print("running experiment: ", experiment)
  ADAP_layer_stopping_gradient = config.get("ADAP_layer_stopping_gradient",False)
  ADAP_gate_stopping_gradient = config.get("ADAP_gate_stopping_gradient",False)
  d_classification_gate_stopping_gradient_enc = config.get("d_classification_gate_stopping_gradient_enc",False)
  d_classification_gate_stopping_gradient_dec = config.get("d_classification_gate_stopping_gradient_dec",False)
  print("ADAP_layer_stopping_gradient: ", ADAP_layer_stopping_gradient)
  print("ADAP_gate_stopping_gradient: ", ADAP_gate_stopping_gradient)
  print("d_classification_gate_stopping_gradient_enc: ", d_classification_gate_stopping_gradient_enc)
  print("d_classification_gate_stopping_gradient_dec: ", d_classification_gate_stopping_gradient_dec)
  num_domain_units = config.get("num_domain_units",128)
  num_domains = config.get("num_domains", 6)
  if experiment=="residual":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder(
        num_layers=6,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1),
    decoder=Multi_domain_SelfAttentionDecoder(
        num_layers=6,
        num_domains=6,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1))
  elif experiment=="residualv2":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v2(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1),
    decoder=Multi_domain_SelfAttentionDecoder_v2(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1))
  elif experiment=="residualv5":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v2(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        res_using_rate=config.get("res_using_rate",1.0),
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3),
    decoder=Multi_domain_SelfAttentionDecoder_v2(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        res_using_rate=config.get("res_using_rate",1.0),
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3))
  elif experiment=="residualv20":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v2(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v7),
    decoder=Multi_domain_SelfAttentionDecoder_v2(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v7))
  elif experiment=="residualv10":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v2(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=[1024,1024,1024,1024,1024,128],
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v5),
    decoder=Multi_domain_SelfAttentionDecoder_v2(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=[1024,1024,1024,1024,1024,128],
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v5))
  elif experiment=="residualv6":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v1(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=ADAP_gate_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3),
    decoder=Multi_domain_SelfAttentionDecoder_v6(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=ADAP_gate_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3))
  elif experiment=="residualv21":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v1(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=ADAP_gate_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v7),
    decoder=Multi_domain_SelfAttentionDecoder_v6(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=ADAP_gate_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v7))
  elif experiment=="residualv11":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v6(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=ADAP_gate_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3),
    decoder=Multi_domain_SelfAttentionDecoder_v9(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=ADAP_gate_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3))
  elif experiment=="residualv13":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v8(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=ADAP_gate_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
        input_gate_regularization=config.get("input_gate_regularizing", False)),
    decoder=Multi_domain_SelfAttentionDecoder_v10(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=ADAP_gate_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
        input_gate_regularization=config.get("input_gate_regularizing", False)))
  elif experiment=="residualv12":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v7(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=ADAP_gate_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3),
    decoder=Multi_domain_SelfAttentionDecoder_v9(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=ADAP_gate_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3))
  elif experiment=="residualv7":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v5(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3),
    decoder=Multi_domain_SelfAttentionDecoder_v7(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3))
  elif experiment=="residualv8":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v5(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
        multi_domain_adapter_gate_class=Multi_domain_Gate_v1),
    decoder=Multi_domain_SelfAttentionDecoder_v7(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
        multi_domain_adapter_gate_class=Multi_domain_Gate_v1))
  elif experiment=="residualv9":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v5(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v0,
        multi_domain_adapter_gate_class=Multi_domain_Gate),
    decoder=Multi_domain_SelfAttentionDecoder_v7(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v0,
        multi_domain_adapter_gate_class=Multi_domain_Gate))
  elif experiment=="residualv0":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v0(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v1),
    decoder=Multi_domain_SelfAttentionDecoder_v0(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v1))
  elif experiment=="residualv1":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v2(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1),
    decoder=Multi_domain_SelfAttentionDecoder_v1(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1))
  elif experiment=="residualv3":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v2(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1),
    decoder=Multi_domain_SelfAttentionDecoder_v5(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1))
  elif experiment=="residualv15":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v1(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=ADAP_gate_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_gate_class=Regulation_Gate,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3),
    decoder=Multi_domain_SelfAttentionDecoder_v6(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=ADAP_gate_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_gate_class=Regulation_Gate,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3))
  elif experiment=="residualv22":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v1(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=ADAP_gate_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_gate_class=Regulation_Gate,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v7),
    decoder=Multi_domain_SelfAttentionDecoder_v6(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=ADAP_gate_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_gate_class=Regulation_Gate,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v7))
  elif experiment=="residualv16":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v1(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=ADAP_gate_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_gate_class=Regulation_Gate,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v6,
        fake_domain_prob=config.get("fake_domain_prob", 0.1),
        noisy_prob=config.get("noisy_prob", None)),
    decoder=Multi_domain_SelfAttentionDecoder_v6(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=ADAP_gate_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_gate_class=Regulation_Gate,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v6,
        fake_domain_prob=config.get("fake_domain_prob", 0.1),
        noisy_prob=config.get("noisy_prob", None)))
  elif experiment=="residualv23":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v1(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=ADAP_gate_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_gate_class=Regulation_Gate,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v8,
        fake_domain_prob=config.get("fake_domain_prob", 0.1),
        noisy_prob=config.get("noisy_prob", None)),
    decoder=Multi_domain_SelfAttentionDecoder_v6(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=ADAP_gate_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_gate_class=Regulation_Gate,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v8,
        fake_domain_prob=config.get("fake_domain_prob", 0.1),
        noisy_prob=config.get("noisy_prob", None)))
  elif experiment=="residualv19":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v10(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=ADAP_gate_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_gate_class=Regulation_Gate,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
        fake_domain_prob=config.get("fake_domain_prob", 0.1),
        noisy_prob=config.get("noisy_prob", None)),
    decoder=Multi_domain_SelfAttentionDecoder_v12(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=ADAP_gate_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_gate_class=Regulation_Gate,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
        fake_domain_prob=config.get("fake_domain_prob", 0.1),
        noisy_prob=config.get("noisy_prob", None)))
  elif experiment=="residualv17":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v1(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=ADAP_gate_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v6,
        fake_domain_prob=config.get("fake_domain_prob", 0.1),
        noisy_prob=config.get("noisy_prob", None)),
    decoder=Multi_domain_SelfAttentionDecoder_v6(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=ADAP_gate_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v6,
        fake_domain_prob=config.get("fake_domain_prob", 0.1),
        noisy_prob=config.get("noisy_prob", None)))
  elif experiment=="residualv18":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v2(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v6,
        fake_domain_prob=config.get("fake_domain_prob", 0.1),
        noisy_prob=config.get("noisy_prob", None)),
    decoder=Multi_domain_SelfAttentionDecoder_v2(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v6,
        fake_domain_prob=config.get("fake_domain_prob", 0.1),
        noisy_prob=config.get("noisy_prob", None)))
  elif experiment=="ldr":
    model = LDR_SequenceToSequence(
    source_inputter=LDR_inputter(embedding_size=config.get("ldr_embedding_size",504), num_domains=config.get("num_domains", 6), num_domain_units=config.get("num_embedding_domain_units", 8)),
    target_inputter=LDR_inputter(embedding_size=config.get("ldr_embedding_size",504), num_domains=config.get("num_domains", 6), num_domain_units=config.get("num_embedding_domain_units", 8)),
    encoder=onmt.encoders.self_attention_encoder.SelfAttentionEncoder(
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1),
    decoder=onmt.decoders.self_attention_decoder.SelfAttentionDecoder(
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1))
  elif experiment=="DC":
    model = LDR_SequenceToSequence(
    source_inputter=DC_inputter(embedding_size=config.get("ldr_embedding_size",508), num_domains=config.get("num_domains", 6), num_domain_units=config.get("num_embedding_domain_units", 4)),
    target_inputter=DC_inputter(embedding_size=config.get("ldr_embedding_size",508), num_domains=config.get("num_domains", 6), num_domain_units=config.get("num_embedding_domain_units", 4)),
    encoder=onmt.encoders.self_attention_encoder.SelfAttentionEncoder(
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1),
    decoder=onmt.decoders.self_attention_decoder.SelfAttentionDecoder(
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1))
  elif experiment=="DC1":
    model = LDR_SequenceToSequence(
    source_inputter=DC_inputter(embedding_size=config.get("ldr_embedding_size",508), num_domains=config.get("num_domains", 6), num_domain_units=config.get("num_embedding_domain_units", 4)),
    target_inputter=WordEmbedder(embedding_size=512),
    encoder=onmt.encoders.self_attention_encoder.SelfAttentionEncoder(
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1),
    decoder=onmt.decoders.self_attention_decoder.SelfAttentionDecoder(
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1))
  elif experiment=="baseline":
    model = LDR_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=onmt.encoders.SelfAttentionEncoder(
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1),
    decoder=onmt.decoders.SelfAttentionDecoder(
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1),
    num_domains=num_domains,
    num_units=512)
  elif experiment=="baselinev3":
    model = onmt.models.Transformer(
      source_inputter=onmt.inputters.WordEmbedder(embedding_size=512),
      target_inputter=onmt.inputters.WordEmbedder(embedding_size=512),
      num_layers=6,
      num_units=512,
      num_heads=8,
      ffn_inner_dim=2048,
      dropout=0.1,
      attention_dropout=0.1,
      ffn_dropout=0.1,
      share_embeddings=onmt.models.EmbeddingsSharingLevel.TARGET)
  elif experiment=="small_transformer":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    num_domains=num_domains,
    num_units=512,
    encoder=Multi_domain_SelfAttentionEncoder_v15(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=d_classification_gate_stopping_gradient_enc,
        num_units=config.get("num_units",512),
        num_heads=config.get("num_heads",4),
        ffn_inner_dim=config.get("ffn_inner_dim",1024),
        dropout=config.get("dropout",0.3),
        training_res_using_rate=config.get("training_res_using_rate",1.0),
        testing_res_using_rate=config.get("testing_res_using_rate",1.0),
        attention_dropout=config.get("attention_dropout",0.3),
        ffn_dropout=config.get("ffn_dropout",0.3),
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
        version=config.get("version"),
        inner_layer_norm=None if not config.get("inner_layer_norm") else Multi_LayerNorm,
        stop_gradient_version=config.get("stop_gradient_version",1)),
    decoder=Multi_domain_SelfAttentionDecoder_v17(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=d_classification_gate_stopping_gradient_dec,
        num_units=config.get("num_units",512),
        num_heads=config.get("num_heads",4),
        ffn_inner_dim=config.get("ffn_inner_dim",1024),
        dropout=config.get("dropout",0.3),
        training_res_using_rate=config.get("training_res_using_rate",1.0),
        testing_res_using_rate=config.get("testing_res_using_rate",1.0),
        attention_dropout=config.get("attention_dropout",0.3),
        ffn_dropout=config.get("ffn_dropout",0.3),
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
        inner_layer_norm=None if not config.get("inner_layer_norm") else Multi_LayerNorm,
        version=config.get("version"),
        stop_gradient_version=config.get("stop_gradient_version",1)))
  elif experiment=="rnn":
    model = SequenceToSequence(
    source_inputter=WordEmbedder(embedding_size=512),
    target_inputter=WordEmbedder(embedding_size=512),
    encoder=onmt.encoders.rnn_encoder.LSTMEncoder(
      num_layers=1,
      num_units=1024,
      bidirectional=True,
      residual_connections=False,
      dropout=0.1),
    decoder=onmt.decoders.rnn_decoder.AttentionalRNNDecoder(
      num_layers=1,
      num_units=1024,
      attention_mechanism_class=tfa.seq2seq.BahdanauAttention,
      cell_class=tf.keras.layers.LSTMCell,
      bridge_class=onmt.layers.DenseBridge,
      residual_connections=False,
      dropout=0.1))
  elif experiment=="baselinev2":
    model = LDR_SequenceToSequence_v1(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=onmt.encoders.SelfAttentionEncoder(
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1),
    decoder=onmt.decoders.SelfAttentionDecoder(
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1),
    num_domains=num_domains,
    num_units=512)
  elif experiment=="baselinev1":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v2(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        ADAP_contribution=[0.0] * 6,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3),
    decoder=Multi_domain_SelfAttentionDecoder_v2(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        ADAP_contribution=[0.0] * 6,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3))
  elif experiment=="DRO":
    model = Multi_domain_SequenceToSequence_DRO(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    probs_inputter=ProbInputter_v1(),
    encoder=Multi_domain_SelfAttentionEncoder_v2(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        ADAP_contribution=[0.0] * 6,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3),
    decoder=Multi_domain_SelfAttentionDecoder_v2(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        ADAP_contribution=[0.0] * 6,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3))
  elif experiment=="residualv25":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v12(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
        inner_layer_norm=None),
    decoder=Multi_domain_SelfAttentionDecoder_v16(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
        inner_layer_norm=None))
  elif experiment=="residualv27":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v12(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
        inner_layer_norm=Multi_LayerNorm),
    decoder=Multi_domain_SelfAttentionDecoder_v16(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
        inner_layer_norm=Multi_LayerNorm))
  elif experiment=="residualv26":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=config.get("num_units",512)),
    target_inputter=My_inputter(embedding_size=config.get("num_units",512)),
    num_domains=num_domains,
    num_units=config.get("num_units",512),
    encoder=Multi_domain_SelfAttentionEncoder_v15(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        domain_region_sizes = config.get("domain_region_sizes",None),
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=d_classification_gate_stopping_gradient_enc,
        num_units=config.get("num_units",512),
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        training_res_using_rate=config.get("training_res_using_rate",1.0),
        testing_res_using_rate=config.get("testing_res_using_rate",1.0),
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
        version=config.get("version"),
        inner_layer_norm=None if not config.get("inner_layer_norm") else Multi_LayerNorm,
        stop_gradient_version=config.get("stop_gradient_version",1)),
    decoder=Multi_domain_SelfAttentionDecoder_v17(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        domain_region_sizes = config.get("domain_region_sizes",None),
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=d_classification_gate_stopping_gradient_dec,
        num_units=config.get("num_units",512),
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        training_res_using_rate=config.get("training_res_using_rate",1.0),
        testing_res_using_rate=config.get("testing_res_using_rate",1.0),
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
        inner_layer_norm=None if not config.get("inner_layer_norm") else Multi_LayerNorm,
        version=config.get("version"),
        stop_gradient_version=config.get("stop_gradient_version",1)))
  
  elif experiment=="Sparse_Layers":
    model = Multi_domain_SequenceToSequence_sparse(
    source_inputter=My_inputter(embedding_size=config.get("num_units",512)),
    target_inputter=My_inputter(embedding_size=config.get("num_units",512)),
    num_domains=num_domains,
    num_units=config.get("num_units",512),
    dropout_rate=config.get("dropout_rate",0.5),
    num_domain_unit_group=config.get("num_domain_unit_group",16),
    unit_group_size=config.get("unit_group_size",12),
    num_shared_units=config.get("num_shared_units",480),
    version = config.get("version",1),
    encoder=Multi_domain_SelfAttentionEncoder_sparse(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        domain_region_sizes = config.get("domain_region_sizes",None),
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=d_classification_gate_stopping_gradient_enc,
        num_units=config.get("num_units",512),
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        training_res_using_rate=config.get("training_res_using_rate",1.0),
        testing_res_using_rate=config.get("testing_res_using_rate",1.0),
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
        version=config.get("encoder_version",1),
        inner_layer_norm=None if not config.get("inner_layer_norm") else Multi_LayerNorm,
        stop_gradient_version=config.get("stop_gradient_version",1)),
    decoder=Multi_domain_SelfAttentionDecoder_sparse(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        domain_region_sizes = config.get("domain_region_sizes",None),
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=d_classification_gate_stopping_gradient_dec,
        num_units=config.get("num_units",512),
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        training_res_using_rate=config.get("training_res_using_rate",1.0),
        testing_res_using_rate=config.get("testing_res_using_rate",1.0),
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
        inner_layer_norm=None if not config.get("inner_layer_norm") else Multi_LayerNorm,
        version=config.get("decoder_version",1),
        stop_gradient_version=config.get("stop_gradient_version",1)))

  elif experiment=="TopK_Sparse_Layers":
    model = Multi_domain_SequenceToSequence_TopK_sparse(
    source_inputter=My_inputter(embedding_size=config.get("num_units",512)),
    target_inputter=My_inputter(embedding_size=config.get("num_units",512)),
    num_domains=num_domains,
    num_units=config.get("num_units",512),
    dropout_rate=config.get("dropout_rate",0.5),
    num_domain_unit_group=config.get("num_domain_unit_group",16),
    unit_group_size=config.get("unit_group_size",12),
    num_shared_units=config.get("num_shared_units",480),
    version = config.get("version",1),
    encoder=Multi_domain_SelfAttentionEncoder_sparse(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        domain_region_sizes = config.get("domain_region_sizes",None),
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=d_classification_gate_stopping_gradient_enc,
        num_units=config.get("num_units",512),
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        training_res_using_rate=config.get("training_res_using_rate",1.0),
        testing_res_using_rate=config.get("testing_res_using_rate",1.0),
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
        version=config.get("encoder_version",2),
        inner_layer_norm=None if not config.get("inner_layer_norm") else Multi_LayerNorm,
        stop_gradient_version=config.get("stop_gradient_version",1)),
    decoder=Multi_domain_SelfAttentionDecoder_sparse(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        domain_region_sizes = config.get("domain_region_sizes",None),
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=d_classification_gate_stopping_gradient_dec,
        num_units=config.get("num_units",512),
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        training_res_using_rate=config.get("training_res_using_rate",1.0),
        testing_res_using_rate=config.get("testing_res_using_rate",1.0),
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
        inner_layer_norm=None if not config.get("inner_layer_norm") else Multi_LayerNorm,
        version=config.get("decoder_version",2),
        stop_gradient_version=config.get("stop_gradient_version",1)))
  
  elif experiment=="TopK_Sparse_Layers_multi_layer":
    model = Multi_domain_SequenceToSequence_TopK_sparse_multi_layer(
    source_inputter=My_inputter(embedding_size=config.get("num_units",512)),
    target_inputter=My_inputter(embedding_size=config.get("num_units",512)),
    num_domains=num_domains,
    num_units=config.get("num_units",512),
    dropout_rate=config.get("dropout_rate",0.5),
    num_domain_unit_group=config.get("num_domain_unit_group",16),
    unit_group_size=config.get("unit_group_size",12),
    num_shared_units=config.get("num_shared_units",480),
    version = config.get("version",1),
    encoder=Multi_domain_SelfAttentionEncoder_sparse_multi_layer(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        domain_region_sizes = config.get("domain_region_sizes",None),
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=d_classification_gate_stopping_gradient_enc,
        num_units=config.get("num_units",512),
        num_heads=config.get("num_heads",8),
        ffn_inner_dim=config.get("ffn_inner_dim",2048),
        dropout=config.get("dropout",0.1),
        training_res_using_rate=config.get("training_res_using_rate",1.0),
        testing_res_using_rate=config.get("testing_res_using_rate",1.0),
        attention_dropout=config.get("attention_dropout",0.1),
        ffn_dropout=config.get("ffn_dropout",0.1),
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
        version=config.get("encoder_version",2),
        inner_layer_norm=None if not config.get("inner_layer_norm") else Multi_LayerNorm,
        stop_gradient_version=config.get("stop_gradient_version",1)),
    decoder=Multi_domain_SelfAttentionDecoder_sparse_multi_layer(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        domain_region_sizes = config.get("domain_region_sizes",None),
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=d_classification_gate_stopping_gradient_dec,
        num_units=config.get("num_units",512),
        num_heads=config.get("num_heads",8),
        ffn_inner_dim=config.get("ffn_inner_dim",2048),
        dropout=config.get("dropout",0.1),
        training_res_using_rate=config.get("training_res_using_rate",1.0),
        testing_res_using_rate=config.get("testing_res_using_rate",1.0),
        attention_dropout=config.get("attention_dropout",0.1),
        ffn_dropout=config.get("ffn_dropout",0.1),
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
        inner_layer_norm=None if not config.get("inner_layer_norm") else Multi_LayerNorm,
        version=config.get("decoder_version",2),
        stop_gradient_version=config.get("stop_gradient_version",1)))
  
  elif experiment=="TopK_Sparse_Layers_multi_layer_v1":
    model = Multi_domain_SequenceToSequence_TopK_sparse_multi_layer_v1(
    source_inputter=My_inputter(embedding_size=config.get("num_units",512)),
    target_inputter=My_inputter(embedding_size=config.get("num_units",512)),
    num_domains=num_domains,
    num_units=config.get("num_units",512),
    dropout_rate=config.get("dropout_rate",0.5),
    num_domain_unit_group=config.get("num_domain_unit_group",16),
    unit_group_size=config.get("unit_group_size",12),
    num_shared_units=config.get("num_shared_units",480),
    version = config.get("version",1),
    encoder=Multi_domain_SelfAttentionEncoder_sparse_multi_layer(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        domain_region_sizes = config.get("domain_region_sizes",None),
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=d_classification_gate_stopping_gradient_enc,
        num_units=config.get("num_units",512),
        num_heads=config.get("num_heads",8),
        ffn_inner_dim=config.get("ffn_inner_dim",2048),
        dropout=config.get("dropout",0.1),
        training_res_using_rate=config.get("training_res_using_rate",1.0),
        testing_res_using_rate=config.get("testing_res_using_rate",1.0),
        attention_dropout=config.get("attention_dropout",0.1),
        ffn_dropout=config.get("ffn_dropout",0.1),
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
        version=config.get("encoder_version",2),
        inner_layer_norm=None if not config.get("inner_layer_norm") else Multi_LayerNorm,
        stop_gradient_version=config.get("stop_gradient_version",1)),
    decoder=Multi_domain_SelfAttentionDecoder_sparse_multi_layer_v1(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        domain_region_sizes = config.get("domain_region_sizes",None),
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=d_classification_gate_stopping_gradient_dec,
        num_units=config.get("num_units",512),
        num_heads=config.get("num_heads",8),
        ffn_inner_dim=config.get("ffn_inner_dim",2048),
        dropout=config.get("dropout",0.1),
        training_res_using_rate=config.get("training_res_using_rate",1.0),
        testing_res_using_rate=config.get("testing_res_using_rate",1.0),
        attention_dropout=config.get("attention_dropout",0.1),
        ffn_dropout=config.get("ffn_dropout",0.1),
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
        inner_layer_norm=None if not config.get("inner_layer_norm") else Multi_LayerNorm,
        version=config.get("decoder_version",2),
        stop_gradient_version=config.get("stop_gradient_version",1)))
  
  elif experiment=="Hierarchical_TopK_Sparse_Layers_multi_layer":
    model = Multi_domain_SequenceToSequence_TopK_sparse_multi_layer_v0(
    source_inputter=My_inputter(embedding_size=config.get("num_units",512)),
    target_inputter=My_inputter(embedding_size=config.get("num_units",512)),
    num_domains=num_domains,
    num_units=config.get("num_units",512),
    dropout_rate=config.get("dropout_rate",0.5),
    num_domain_unit_group=config.get("num_domain_unit_group",16),
    unit_group_size=config.get("unit_group_size",12),
    num_shared_units=config.get("num_shared_units",480),
    version = config.get("version",1),
    encoder=Multi_domain_SelfAttentionEncoder_sparse_multi_layer_v0(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        domain_region_sizes = config.get("domain_region_sizes",None),
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=d_classification_gate_stopping_gradient_enc,
        num_units=config.get("num_units",512),
        num_heads=config.get("num_heads",8),
        ffn_inner_dim=config.get("ffn_inner_dim",2048),
        dropout=config.get("dropout",0.1),
        training_res_using_rate=config.get("training_res_using_rate",1.0),
        testing_res_using_rate=config.get("testing_res_using_rate",1.0),
        attention_dropout=config.get("attention_dropout",0.1),
        ffn_dropout=config.get("ffn_dropout",0.1),
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
        version=config.get("encoder_version",2),
        inner_layer_norm=None if not config.get("inner_layer_norm") else Multi_LayerNorm,
        stop_gradient_version=config.get("stop_gradient_version",1)),
    decoder=Multi_domain_SelfAttentionDecoder_sparse_multi_layer_v0(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        domain_region_sizes = config.get("domain_region_sizes",None),
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=d_classification_gate_stopping_gradient_dec,
        num_units=config.get("num_units",512),
        num_heads=config.get("num_heads",8),
        ffn_inner_dim=config.get("ffn_inner_dim",2048),
        dropout=config.get("dropout",0.1),
        training_res_using_rate=config.get("training_res_using_rate",1.0),
        testing_res_using_rate=config.get("testing_res_using_rate",1.0),
        attention_dropout=config.get("attention_dropout",0.1),
        ffn_dropout=config.get("ffn_dropout",0.1),
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
        inner_layer_norm=None if not config.get("inner_layer_norm") else Multi_LayerNorm,
        version=config.get("decoder_version",2),
        stop_gradient_version=config.get("stop_gradient_version",1)))
  
  elif experiment=="Multi_domain_SequenceToSequence_Instace_Aware_TopK_sparse_multi_layer":

    model = Multi_domain_SequenceToSequence_Instace_Aware_TopK_sparse_multi_layer(
    source_inputter=My_inputter(embedding_size=config.get("num_units",512)),
    target_inputter=My_inputter(embedding_size=config.get("num_units",512)),
    num_domains=num_domains,
    num_units=config.get("num_units",512),
    dropout_rate=config.get("dropout_rate",0.5),
    num_domain_unit_group=config.get("num_domain_unit_group",16),
    unit_group_size=config.get("unit_group_size",12),
    num_shared_units=config.get("num_shared_units",480),
    version = config.get("version",1),
    meta_encoder = SelfAttentionEncoder(num_layers=2),
    encoder=Multi_domain_SelfAttentionEncoder_sparse_multi_layer_v1(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        domain_region_sizes = config.get("domain_region_sizes",None),
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=d_classification_gate_stopping_gradient_enc,
        num_units=config.get("num_units",512),
        num_heads=config.get("num_heads",8),
        ffn_inner_dim=config.get("ffn_inner_dim",2048),
        dropout=config.get("dropout",0.1),
        training_res_using_rate=config.get("training_res_using_rate",1.0),
        testing_res_using_rate=config.get("testing_res_using_rate",1.0),
        attention_dropout=config.get("attention_dropout",0.1),
        ffn_dropout=config.get("ffn_dropout",0.1),
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
        version=config.get("encoder_version",2),
        inner_layer_norm=None if not config.get("inner_layer_norm") else Multi_LayerNorm,
        stop_gradient_version=config.get("stop_gradient_version",1)),
    decoder=Multi_domain_SelfAttentionDecoder_sparse_multi_layer_v2(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        domain_region_sizes = config.get("domain_region_sizes",None),
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=d_classification_gate_stopping_gradient_dec,
        num_units=config.get("num_units",512),
        num_heads=config.get("num_heads",8),
        ffn_inner_dim=config.get("ffn_inner_dim",2048),
        dropout=config.get("dropout",0.1),
        training_res_using_rate=config.get("training_res_using_rate",1.0),
        testing_res_using_rate=config.get("testing_res_using_rate",1.0),
        attention_dropout=config.get("attention_dropout",0.1),
        ffn_dropout=config.get("ffn_dropout",0.1),
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
        inner_layer_norm=None if not config.get("inner_layer_norm") else Multi_LayerNorm,
        version=config.get("decoder_version",2),
        stop_gradient_version=config.get("stop_gradient_version",1)))

  elif experiment=="residualv28":
    model = SequenceToSequence_with_dprob(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    probs_inputter=ProbInputter(num_domains),
    encoder=Multi_domain_SelfAttentionEncoder_v18(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v9,
        ADAP_contribution=[1.0]*num_domains),
    decoder=Multi_domain_SelfAttentionDecoder_v19(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v9,
        ADAP_contribution=[1.0]*num_domains))
  elif experiment=="residualv29":
    model = SequenceToSequence_with_dprob(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    probs_inputter=ProbInputter(num_domains),
    encoder=Multi_domain_SelfAttentionEncoder_v18(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v9,
        ADAP_contribution=[0.0]*num_domains),
    decoder=Multi_domain_SelfAttentionDecoder_v19(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v9,
        ADAP_contribution=[0.0]*num_domains))
  
  elif experiment=="WDC":
    model = SequenceToSequence_WDC(
      source_inputter=My_inputter(embedding_size=512),
      target_inputter=My_inputter(embedding_size=512),
      encoder=onmt.encoders.SelfAttentionEncoder(
          num_layers=6,
          num_units=512,
          num_heads=8,
          ffn_inner_dim=2048,
          dropout=0.1,
          attention_dropout=0.1,
          ffn_dropout=0.1),
      decoder= Multi_domain_SelfAttentionDecoder_WDC(
          num_layers=6,
          num_units=512,
          num_heads=8,
          ffn_inner_dim=2048,
          dropout=0.1,
          attention_dropout=0.1,
          ffn_dropout=0.1),
      num_domains=num_domains,
      num_units=512)
  elif experiment=="gated_residual_v5":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=512),
    target_inputter=My_inputter(embedding_size=512),
    encoder=Multi_domain_SelfAttentionEncoder_v11(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=ADAP_gate_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3),
    decoder=Multi_domain_SelfAttentionDecoder_v15(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        ADAP_gate_stopping_gradient=ADAP_gate_stopping_gradient,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3))
  elif experiment=="residual_big_transformer":
    model = Multi_domain_SequenceToSequence(
    source_inputter=My_inputter(embedding_size=config.get("d_model",1024)),
    target_inputter=My_inputter(embedding_size=config.get("d_model",1024)),
    encoder=Multi_domain_SelfAttentionEncoder_v16(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=config.get("d_model",1024),
        num_heads=16,
        ffn_inner_dim=4096,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        ADAP_contribution=[config.get("ADAP_contribution",1.0)] * 6,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
        inner_layer_norm=None if not config.get("inner_layer_norm",True) else Multi_LayerNorm,
        version=config.get("version",1)),
    decoder=Multi_domain_SelfAttentionDecoder_v18(
        num_layers=6,
        num_domains=num_domains,
        num_domain_units=num_domain_units,
        ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient,
        num_units=config.get("d_model",1024),
        num_heads=16,
        ffn_inner_dim=4096,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        ADAP_contribution=[config.get("ADAP_contribution",1.0)] * 6,
        multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
        inner_layer_norm=None if not config.get("inner_layer_norm",True) else Multi_LayerNorm,
        version=config.get("version",1)))
  elif experiment=="priming_nmt_2":
    model = Priming_SequenceToSequence(
    source_inputter = onmt.inputters.ParallelInputter([onmt.inputters.WordEmbedder(embedding_size=512),  
                                                      onmt.inputters.WordEmbedder(embedding_size=512)], 
                                                      share_parameters=False,
                                                      combine_features=False),
    target_inputter = onmt.inputters.WordEmbedder(embedding_size=512),
    encoder = onmt.encoders.SelfAttentionEncoder(
              num_layers=6,
              num_units=512,
              num_heads=8,
              ffn_inner_dim=2048,
              dropout=0.1,
              attention_dropout=0.1,
              ffn_dropout=0.1),
    pre_encoder = onmt.encoders.SelfAttentionEncoder(
              num_layers=6,
              num_units=512,
              num_heads=8,
              ffn_inner_dim=2048,
              dropout=0.1,
              attention_dropout=0.1,
              ffn_dropout=0.1),
    decoder = onmt.decoders.SelfAttentionDecoder(
              num_layers=6,
              num_units=512,
              num_heads=8,
              ffn_inner_dim=2048,
              dropout=0.1,
              attention_dropout=0.1,
              ffn_dropout=0.1),
    version=config.get("version",1))
  elif experiment=="priming_nmt_3":
    model = Priming_SequenceToSequence_v1(
    source_inputter = onmt.inputters.ParallelInputter([onmt.inputters.WordEmbedder(embedding_size=512),  
                                                      onmt.inputters.WordEmbedder(embedding_size=512)], 
                                                      share_parameters=True,
                                                      combine_features=False),
    target_inputter = onmt.inputters.WordEmbedder(embedding_size=512),
    encoder = onmt.encoders.SelfAttentionEncoder(
              num_layers=6,
              num_units=512,
              num_heads=8,
              ffn_inner_dim=2048,
              dropout=0.1,
              attention_dropout=0.1,
              ffn_dropout=0.1),
    decoder = SelfAttentionDecoder_v1(
              num_layers=6,
              num_units=512,
              num_heads=8,
              ffn_inner_dim=2048,
              dropout=0.1,
              attention_dropout=0.1,
              ffn_dropout=0.1))
  elif experiment=="priming_nmt_adv":
    model = Priming_SequenceToSequence_v1(
    source_inputter = onmt.inputters.WordEmbedder(embedding_size=512),
    target_inputter = onmt.inputters.ParallelInputter([onmt.inputters.WordEmbedder(embedding_size=512),  
                                                      onmt.inputters.WordEmbedder(embedding_size=512)], 
                                                      share_parameters=True,
                                                      combine_features=False),
    encoder = onmt.encoders.SelfAttentionEncoder(
              num_layers=6,
              num_units=512,
              num_heads=8,
              ffn_inner_dim=2048,
              dropout=0.1,
              attention_dropout=0.1,
              ffn_dropout=0.1),
    decoder = SelfAttentionDecoder(
              num_layers=6,
              num_units=512,
              num_heads=8,
              ffn_inner_dim=2048,
              dropout=0.1,
              attention_dropout=0.1,
              ffn_dropout=0.1))
  elif experiment=="pretrain":
    return
  warmup_steps = config.get("warmup_steps",4000)
  print("warmup_steps: ", warmup_steps)
  print("step_duration: ", config.get("step_duration",16))
  print("d_model: ", config.get("d_model",512))
  #learning_rate = onmt.schedules.ScheduleWrapper(schedule=my_schedules.NGDDecay(scale=config.get("learning_rate",1.0), model_dim=config.get("d_model",512), warmup_steps=warmup_steps), step_duration= config.get("step_duration",16))
  learning_rate = onmt.schedules.ScheduleWrapper(schedule=onmt.schedules.NoamDecay(scale=config.get("learning_rate",1.0), model_dim=config.get("d_model",512), warmup_steps=warmup_steps), step_duration= config.get("step_duration",16))

  print("learning_rate: ", learning_rate)
  meta_train_optimizer = tf.keras.optimizers.SGD(config.get("meta_train_lr"))
  meta_test_optimizer = tfa.optimizers.LazyAdam(learning_rate, beta_1=config.get("adam_beta_1",0.9), beta_2=config.get("adam_beta_2",0.999),epsilon=config.get("adam_epsilon",1e-8))
  
  adapter_optimizer = tfa.optimizers.LazyAdam(learning_rate)
  model.initialize(data_config)
  checkpoint = tf.train.Checkpoint(model=model, optimizer=meta_test_optimizer)
  checkpoint_manager = tf.train.CheckpointManager(checkpoint, config["model_dir"], max_to_keep=config.get("max_to_keep",10))
  ######
  model.params.update({"label_smoothing": 0.1})
  model.params.update({"average_loss_in_time": config.get("average_loss_in_time",True)})
  model.params.update({"beam_width": 5})

  if args.run == "inspect":
    task.model_inspect(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, checkpoint_path=args.ckpt, experiment=experiment)
  if args.run == "metatrainv7":
    task.meta_train_v7(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run == "metatrainv8":
    task.meta_train_v8(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000), 
                      meta_train_picking_prob=config.get("meta_train_picking_prob",None), meta_test_picking_prob=config.get("meta_test_picking_prob",None))
  elif args.run == "metatrainv15":
    task.meta_train_v15(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment, picking_prob=config.get("picking_prob",None))
  elif args.run == "metatrainv13":
    task.meta_train_v13(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment, picking_prob=config.get("picking_prob",None))
  elif args.run == "trainv8":
    task.train_v8(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment, picking_prob=config.get("picking_prob",None))
  elif args.run == "metatrainv10":
    task.meta_train_v10(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run == "metatrainv11":
    task.meta_train_v11(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run == "metatrainv12":
    task.meta_train_v12(config, meta_train_optimizer, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run == "trainv12":
    task.train_v12(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run == "metatrainv9":
    task.meta_train_v9(config, meta_test_optimizer, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run == "metatrainv6":
    task.meta_train_v6(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run == "metatrainv5":
    task.meta_train_v5(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run == "metatrainv2":
    task.meta_train_v2(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run == "metatrainv3":
    task.meta_train_v3(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run == "metatrainv1":
    task.meta_train_v1(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run == "train":
    task.train(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint,adapter_optimizer=adapter_optimizer, checkpoint_path=config.get("checkpoint_path",None), maximum_length=config.get("maximum_length",80), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "train_tf_25":
    task.train_tf_25(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint,adapter_optimizer=adapter_optimizer, checkpoint_path=config.get("checkpoint_path",None), maximum_length=config.get("maximum_length",80), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "train_elbo_topK_sparse_layer":
    task.train_elbo_topK_sparse_layer(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint,adapter_optimizer=adapter_optimizer, checkpoint_path=config.get("checkpoint_path",None), maximum_length=config.get("maximum_length",80), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "train_elbo_topK_sparse_layer_multi_layer":
    print("num_units",config.get("num_units",512)),
    print("num_heads",config.get("num_heads",8)),
    print("ffn_inner_dim",config.get("ffn_inner_dim",2048)),
    print("dropout",config.get("dropout",0.1)),
    print("attention_dropout",config.get("attention_dropout",0.1)),
    print("ffn_dropout",config.get("ffn_dropout",0.1)),
    task.train_elbo_topK_sparse_layer_multi_layer(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint,adapter_optimizer=adapter_optimizer, checkpoint_path=config.get("checkpoint_path",None), maximum_length=config.get("maximum_length",80), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "finetune_elbo_topK_sparse_layer_multi_layer":
    print("num_units",config.get("num_units",512)),
    print("num_heads",config.get("num_heads",8)),
    print("ffn_inner_dim",config.get("ffn_inner_dim",2048)),
    print("dropout",config.get("dropout",0.1)),
    print("attention_dropout",config.get("attention_dropout",0.1)),
    print("ffn_dropout",config.get("ffn_dropout",0.1)),
    task.finetune_elbo_topK_sparse_layer_multi_layer(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint,adapter_optimizer=adapter_optimizer, checkpoint_path=config.get("checkpoint_path",None), maximum_length=config.get("maximum_length",80), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "fewshot_elbo_topK_sparse_layer_multi_layer":
    print("num_units",config.get("num_units",512)),
    print("num_heads",config.get("num_heads",8)),
    print("ffn_inner_dim",config.get("ffn_inner_dim",2048)),
    print("dropout",config.get("dropout",0.1)),
    print("attention_dropout",config.get("attention_dropout",0.1)),
    print("ffn_dropout",config.get("ffn_dropout",0.1)),
    task.fewshot_elbo_topK_sparse_layer_multi_layer(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint,adapter_optimizer=adapter_optimizer, checkpoint_path=config.get("checkpoint_path",None), maximum_length=config.get("maximum_length",80), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))

  elif args.run == "train_elbo_topK_sparse_layer_multi_layer_v1":
    print("num_units",config.get("num_units",512)),
    print("num_heads",config.get("num_heads",8)),
    print("ffn_inner_dim",config.get("ffn_inner_dim",2048)),
    print("dropout",config.get("dropout",0.1)),
    print("attention_dropout",config.get("attention_dropout",0.1)),
    print("ffn_dropout",config.get("ffn_dropout",0.1)),
    task.train_elbo_topK_sparse_layer_multi_layer_v1(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint,adapter_optimizer=adapter_optimizer, checkpoint_path=config.get("checkpoint_path",None), maximum_length=config.get("maximum_length",80), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))

  elif args.run == "train_elbo_sparse_layer":
    task.train_elbo_sparse_layer(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint,adapter_optimizer=adapter_optimizer, checkpoint_path=config.get("checkpoint_path",None), maximum_length=config.get("maximum_length",80), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "priming_train":
    task.priming_train(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint,adapter_optimizer=adapter_optimizer, checkpoint_path=config.get("checkpoint_path",None), maximum_length=config.get("maximum_length",80), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "priming_train_chasing":
    task.priming_train_chasing(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint,adapter_optimizer=adapter_optimizer, checkpoint_path=config.get("checkpoint_path",None), maximum_length=config.get("maximum_length",80), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "CL_marine":
    task.CL_marine(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint,adapter_optimizer=adapter_optimizer, checkpoint_path=config.get("checkpoint_path",None), maximum_length=config.get("maximum_length",80), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "train_L2W":
    task.train_L2W(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint,adapter_optimizer=adapter_optimizer, checkpoint_path=config.get("checkpoint_path",None), maximum_length=config.get("maximum_length",80), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "train_L2W_v1":
    task.train_L2W_v1(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint,adapter_optimizer=adapter_optimizer, checkpoint_path=config.get("checkpoint_path",None), maximum_length=config.get("maximum_length",80), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "train_L2W_v2":
    task.train_L2W_v2(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, adapter_optimizer=adapter_optimizer, checkpoint_path=config.get("checkpoint_path",None), maximum_length=config.get("maximum_length",80), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "train_L2W_g":
    task.train_L2W_g(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, adapter_optimizer=adapter_optimizer, checkpoint_path=config.get("checkpoint_path",None), maximum_length=config.get("maximum_length",80), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "train_L2W_v3":
    task.train_L2W_v3(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, checkpoint_path=config.get("checkpoint_path",None), maximum_length=config.get("maximum_length",80), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "train_IW_v0":
    task.train_IW_v0(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint,adapter_optimizer=adapter_optimizer, checkpoint_path=config.get("checkpoint_path",None), maximum_length=config.get("maximum_length",80), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "debug_L2W_v1":
    task.debug_L2W_v1(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint,adapter_optimizer=adapter_optimizer, checkpoint_path=config.get("checkpoint_path",None), maximum_length=config.get("maximum_length",80), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "debug_L2W_v2":
     task.debug_L2W_v2(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint,adapter_optimizer=adapter_optimizer, checkpoint_path=config.get("checkpoint_path",None), maximum_length=config.get("maximum_length",80), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  # elif args.run == "debug_L2W_v3":
  #   task.debug_L2W_v3(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint,adapter_optimizer=adapter_optimizer, checkpoint_path=config.get("checkpoint_path",None), maximum_length=config.get("maximum_length",80), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "train_NGD":
    task.train_NGD(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, checkpoint_path=config.get("checkpoint_path",None), maximum_length=config.get("maximum_length",80), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "train_NGD_L2W":
    task.train_NGD_L2W(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, checkpoint_path=config.get("checkpoint_path",None), maximum_length=config.get("maximum_length",80), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "train_NGD_L2W_v1":
    task.train_NGD_L2W_v1(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, checkpoint_path=config.get("checkpoint_path",None), maximum_length=config.get("maximum_length",80), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "continue_NGD":
    task.train_NGD(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, checkpoint_path=config.get("checkpoint_path",None), maximum_length=config.get("maximum_length",80), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "debug_NGD":
    task.debug_NGD(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, checkpoint_path=config.get("checkpoint_path",None), maximum_length=config.get("maximum_length",80), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "train_wada":
    task.train_wada(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, checkpoint_path=config.get("checkpoint_path",None), maximum_length=config.get("maximum_length",80), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "finetune_wada":
    task.finetune_wada(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, checkpoint_path=config.get("checkpoint_path",None), maximum_length=config.get("maximum_length",80), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "finetune_wada_v1":
    task.finetune_wada_v1(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, checkpoint_path=config.get("checkpoint_path",None), maximum_length=config.get("maximum_length",80), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "finetune_noisy_v1":
    task.finetune_noisy_v1(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, checkpoint_path=config.get("checkpoint_path",None), maximum_length=config.get("maximum_length",80), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "metatrainv16":
    task.meta_train_v16(config, meta_test_optimizer, meta_train_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, checkpoint_path=config.get("checkpoint_path",None), maximum_length=config.get("maximum_length",80), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000),report_every=config.get("report_every",100))
  elif args.run == "train_ldr":
    task.train_ldr(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "dcote":
    task.domain_classification_on_top_encoder(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment, save_every=config.get("save_every",1000), eval_every=config.get("eval_every",2000))
  elif args.run == "trainv2":
    task.train_v2(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run == "trainv3":
    task.train_v3(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment)
  elif args.run == "visualize":
    task.visualize(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "translate":
    model.create_variables()
    print("translate in domain %d"%(int(args.domain)))
    task.translate(args.src, args.ref, model, checkpoint_manager,
              checkpoint, int(args.domain), args.output, length_penalty=0.6, checkpoint_path=args.ckpt, experiment=experiment)    
  elif args.run == "translatev1":
    model.build(None)
    translate_config_file = args.src
    with open(translate_config_file, "r") as stream:
      translate_config = yaml.load(stream)
    for src_file, domain in zip(translate_config["src"], translate_config["domain"]):
      output_file = "%s.trans"%src_file.strip().split("/")[-1]
      print("translating %s in domain %d"%(src_file, domain))
      print("output_file: ", output_file)
      task.translate(src_file, None, model, checkpoint_manager,
              checkpoint, int(domain), output_file, length_penalty=0.6, checkpoint_path=args.ckpt, experiment=experiment)
  elif args.run == "translatev2":
    model.create_variables()
    print("translate in domain %d"%(int(args.domain)))
    task.averaged_checkpoint_translate(config, args.src, args.ref, model, checkpoint_manager,
              checkpoint, int(args.domain), args.output, length_penalty=0.6, experiment=experiment, max_count=int(args.maxcount))
  elif args.run == "translatev5":
    model.create_variables()
    root = args.src
    for i in range(30):
      task.averaged_checkpoint_translate(config, "%s.cluster.%d.tagged"%(root,i), None, model, checkpoint_manager,
              checkpoint, int(i), os.path.join(config["model_dir"],"eval","%s.cluster.%d.tagged.trans"%(os.path.basename(root),i)), length_penalty=0.6, experiment=experiment, max_count=int(args.maxcount))
  elif args.run == "translatev6":
    model.create_variables()
    root = args.src
    for i in range(30):
      task.averaged_checkpoint_translate(config, "%s.cluster.%d"%(root,i), None, model, checkpoint_manager,
              checkpoint, int(i), os.path.join(config["model_dir"],"eval","%s.cluster.%d.trans"%(os.path.basename(root),i)), length_penalty=0.6, experiment=experiment, max_count=int(args.maxcount))
  elif args.run == "translatev7":
    model.create_variables()
    root = args.src
    for i in range(30):
      config_file_root = args.config_root
      with open("%s_%d.yml"%(config_file_root,i), "r") as stream:
        config_ = yaml.load(stream)
      task.averaged_checkpoint_translate(config_, "%s.cluster.%d"%(root,i), None, model, checkpoint_manager,
              checkpoint, int(0), os.path.join(config["model_dir"],"eval","%s.cluster.%d.trans"%(os.path.basename(root),i)), length_penalty=0.6, experiment=experiment, max_count=int(args.maxcount))              
  elif args.run == "translatev3":
    model.create_variables()
    translate_config_file = args.src
    with open(translate_config_file, "r") as stream:
      translate_config = yaml.load(stream)
    new_checkpoint_manager = average_checkpoints(config["model_dir"], output_dir="%s/averaged_checkpoint"%config["model_dir"], trackables={"model":model},
                        max_count=translate_config.get("max_count",3),
                        model_key="model")
    for src_file, domain in zip(translate_config["src"], translate_config["domain"]):      
      output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src_file) + ".trans")
      print("translating %s in domain %d"%(src_file, domain))
      print("output_file: ", output_file)
      """ task.averaged_checkpoint_translate(config, src_file, None, model, checkpoint_manager,
              checkpoint, int(domain), output_file, length_penalty=0.6, experiment=experiment, max_count=translate_config.get("max_count",3)) """
      task.translate(src_file, None, model, new_checkpoint_manager, checkpoint, int(domain), output_file, length_penalty=0.6, experiment=experiment)
  elif args.run == "translatev3_tf_25":
    model.create_variables()
    translate_config_file = args.src
    with open(translate_config_file, "r") as stream:
      translate_config = yaml.load(stream)
    new_checkpoint_manager = average_checkpoints_tf2_3(config["model_dir"], output_dir="%s/averaged_checkpoint"%config["model_dir"], trackables={"model":model},
                        max_count=translate_config.get("max_count",3),
                        model_key="model")
    for src_file, domain in zip(translate_config["src"], translate_config["domain"]):      
      output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src_file) + ".trans")
      print("translating %s in domain %d"%(src_file, domain))
      print("output_file: ", output_file)
      task.translate(src_file, None, model, new_checkpoint_manager, checkpoint, int(domain), output_file, length_penalty=0.6, experiment=experiment)

  elif args.run == "translate_sparse_layer":
    model.create_variables()
    translate_config_file = args.src
    with open(translate_config_file, "r") as stream:
      translate_config = yaml.load(stream)
    new_checkpoint_manager = average_checkpoints_tf2_3(config["model_dir"], output_dir="%s/averaged_checkpoint"%config["model_dir"], trackables={"model":model},
                        max_count=translate_config.get("max_count",3),
                        model_key="model")
    for src_file, domain in zip(translate_config["src"], translate_config["domain"]):      
      output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src_file) + ".trans")
      print("translating %s in domain %d"%(src_file, domain))
      print("output_file: ", output_file)
      task.translate_sparse_layer(src_file, None, model, new_checkpoint_manager, checkpoint, int(domain), output_file, length_penalty=0.6, experiment=experiment)
  elif args.run == "translate_topK_sparse_layer":
    topK = config.get("domain_group_allocation_num",int( (1-config.get("dropout_rate",0.5)) * config.get("num_domain_unit_group",16)))
    model.create_variables()
    translate_config_file = args.src
    with open(translate_config_file, "r") as stream:
      translate_config = yaml.load(stream)
    new_checkpoint_manager = average_checkpoints_tf2_3(config["model_dir"], output_dir="%s/averaged_checkpoint"%config["model_dir"], trackables={"model":model},
                        max_count=translate_config.get("max_count",3),
                        model_key="model")
    
    for src_file, domain in zip(translate_config["src"], translate_config["domain"]):      
      output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src_file) + ".trans")
      print("translating %s in domain %d"%(src_file, domain))
      print("output_file: ", output_file)
      task.translate_topK_sparse_layer(src_file, None, model, new_checkpoint_manager, checkpoint, int(domain), output_file, topK=topK, length_penalty=0.6, experiment=experiment)
  elif args.run == "translate_topK_sparse_layer_multi_layer":
    topK = config.get("domain_group_allocation_num",int( (1-config.get("dropout_rate",0.3)) * config.get("num_domain_unit_group",12)))
    model.create_variables()
    print("topK:",topK)
    translate_config_file = args.src
    with open(translate_config_file, "r") as stream:
      translate_config = yaml.load(stream)
    new_checkpoint_manager = average_checkpoints_tf2_3(config["model_dir"], output_dir="%s/averaged_checkpoint"%config["model_dir"], trackables={"model":model},
                        max_count=translate_config.get("max_count",3),
                        model_key="model")
    
    for i, logit in enumerate(model.latent_group_allocation_logit_per_layer):
      print("unit_allocation_logit_layer_%d"%i)
      tf.print(logit,summarize=-1)
      
    for src_file, domain in zip(translate_config["src"], translate_config["domain"]):      
      output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src_file) + ".trans")
      print("translating %s in domain %d"%(src_file, domain))
      print("output_file: ", output_file)
      task.translate_topK_sparse_layer_multi_layer(src_file, None, model, new_checkpoint_manager, checkpoint, int(domain), output_file, topK=topK, length_penalty=0.6, experiment=experiment)
  
  elif args.run == "translate_topK_sparse_layer_cluster":
    topK = config.get("domain_group_allocation_num",int( (1-config.get("dropout_rate")) * config.get("num_domain_unit_group")))
    model.create_variables()
    new_checkpoint_manager = average_checkpoints_tf2_3(config["model_dir"], output_dir="%s/averaged_checkpoint"%config["model_dir"], trackables={"model":model},
                        max_count=args.maxcount,
                        model_key="model")
    src_list = args.src_files
    for root in src_list:
      for i in range(int(args.n_clusters)):
        src_file = "%s.cluster.%d"%(root,i)
        output_file = os.path.join(config["model_dir"],"eval","%s.cluster.%d.trans"%(os.path.basename(root),i))
        domain = i
        print("translating %s"%src_file)
        task.translate_topK_sparse_layer(src_file, None, model, new_checkpoint_manager, checkpoint, int(domain), output_file, topK=topK, length_penalty=0.6, experiment=experiment)
  
  elif args.run == "translate_topK_sparse_layer_multi_layer_cluster":
    topK = config.get("domain_group_allocation_num",int((1-config.get("dropout_rate",0.25)) * config.get("num_domain_unit_group")))
    model.create_variables()
    print("topK: ",topK)
    new_checkpoint_manager = average_checkpoints_tf2_3(config["model_dir"], output_dir="%s/averaged_checkpoint"%config["model_dir"], trackables={"model":model},
                        max_count=args.maxcount,
                        model_key="model")
    checkpoint_path = new_checkpoint_manager.latest_checkpoint
    src_list = args.src_files
    
    for root in src_list:
      output_files = []
      for i in range(int(args.n_clusters)):
        src_file = "%s.cluster.%d"%(root,i)
        output_file = os.path.join(config["model_dir"],"eval","%s.cluster.%d.trans"%(os.path.basename(root),i))
        domain = i
        print("translating %s"%src_file)
        output_files.append(output_file)
        task.translate_topK_sparse_layer_multi_layer(src_file, None, model, new_checkpoint_manager, checkpoint, int(domain), output_file, topK=topK, length_penalty=0.6, experiment=experiment)
      output_file_concat = task.file_concatenate(output_files,"%s.%s"%(os.path.basename(root),os.path.basename(checkpoint_path)))
  elif args.run == "priming_translate":
    model.create_variables()
    translate_config_file = args.src
    with open(translate_config_file, "r") as stream:
      translate_config = yaml.load(stream)
    for src_file, pre_file in zip(translate_config["src"],translate_config["pre"]):      
      output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src_file) + ".trans")
      print("translating %s"%(src_file))
      print("output_file: ", output_file)
      task.priming_avg_ckpt_translate(config, [src_file, pre_file], None, model, checkpoint_manager,
              checkpoint, int(0), output_file, length_penalty=0.6, experiment=experiment, max_count=translate_config.get("max_count",3))
  elif args.run == "priming_translate_chasing":
    model.create_variables()
    translate_config_file = args.src
    with open(translate_config_file, "r") as stream:
      translate_config = yaml.load(stream)
    new_checkpoint_manager = average_checkpoints(config["model_dir"], output_dir="%s/averaged_checkpoint"%config["model_dir"], trackables={"model":model},
                        max_count=translate_config.get("max_count",3),
                        model_key="model")
    for src_pre_file, source_hide_file in zip(translate_config["src_pre"],translate_config["src_hide"]):      
      output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src_pre_file) + ".%s.trans"%("translate_with_hide" if translate_config.get("translate_with_hide",True) else "translate_with_pre"))
      print("translating %s"%(src_pre_file))
      print("output_file: ", output_file)
      task.priming_avg_ckpt_translate_v1(config, [src_pre_file, source_hide_file], None, model, new_checkpoint_manager,
              checkpoint, int(0), output_file, translate_with_hide= translate_config.get("translate_with_hide",True), length_penalty=0.6, experiment=experiment, max_count=translate_config.get("max_count",3))
  elif args.run == "translate_farajan":
    source_file = args.src
    reference = args.ref
    context_src_file, context_tgt_file, context_score = args.context
    checkpoint_path = args.ckpt
    output_file = args.output
    domain = args.domain
    task.translate_farajan(source_file, context_src_file, context_tgt_file, context_score, reference, model, config, meta_test_optimizer, checkpoint_manager, checkpoint, int(domain), output_file, length_penalty=0.6, 
                  checkpoint_path=checkpoint_path, experiment=experiment)
  elif args.run == "translate_farajan_residual":
    source_file = args.src
    reference = args.ref
    context_src_file, context_tgt_file = args.context
    checkpoint_path = args.ckpt
    output_file = args.output
    domain = args.domain
    task.translate_farajan_residual(source_file, context_src_file, context_tgt_file, reference, model, config, strategy, meta_test_optimizer, checkpoint_manager, checkpoint, int(domain), output_file, length_penalty=0.6, 
                  checkpoint_path=checkpoint_path, experiment=experiment)
  elif args.run == "finetune":
    task.finetuning(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "elastic_finetune":
    task.elastic_finetuning(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, elastic_type=config.get("elastic_type","Uniform"), EWC_path=config.get("EWC_path",None), checkpoint_path=config.get("checkpoint_path",None), experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "debug":
    task.debug(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment, picking_prob=config.get("picking_prob",None))
  elif args.run == "train_wdc":
    task.train_wdc(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "sentence_encode":
    output_file = args.output
    source_file = args.src
    domain = int(args.domain) if args.domain else 0
    task.sentence_encode(source_file, model, checkpoint_manager, checkpoint, domain, output_file, experiment=experiment, batch_size=1)
  elif args.run == "train_denny_britz":
    task.train_denny_britz(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "train_domain_mixing_residual":
    task.train_domain_mixing_residual(config, meta_test_optimizer, learning_rate, model, strategy, checkpoint_manager, checkpoint, experiment=experiment, save_every=config.get("save_every",5000), eval_every=config.get("eval_every",10000))
  elif args.run == "experimental_translate":
    model.create_variables()
    print("translate with encoder_domain %d and decoder_domain %d"%(int(args.encoder_domain), int(args.decoder_domain)))
    task.experimental_translate(args.src, args.ref, model, checkpoint_manager,
              checkpoint, int(args.encoder_domain), int(args.decoder_domain), args.output, length_penalty=0.6, checkpoint_path=args.ckpt, experiment=experiment)
  elif args.run == "score":
    source_file = args.src
    translation_file = args.translation_file
    domain = args.domain
    output_file = args.output
    checkpoint_path = args.ckpt
    task.score(source_file, translation_file, model, config, strategy, meta_test_optimizer, checkpoint_manager, checkpoint, int(domain), output_file, length_penalty=0.6, 
                  checkpoint_path=checkpoint_path, experiment=experiment)
  elif args.run == "EWC_stat":
    source_file = args.src
    reference = args.translation_file
    checkpoint_path = args.ckpt
    task.EWC_stat(source_file, reference, model, config, strategy, meta_test_optimizer, checkpoint_manager, checkpoint, checkpoint_path=checkpoint_path)
  elif args.run == "EWC_res_stat":
    source_file = args.src
    reference = args.translation_file
    checkpoint_path = args.ckpt
    task.EWC_res_stat(source_file, reference, model, config, strategy, meta_test_optimizer, checkpoint_manager, checkpoint, checkpoint_path=checkpoint_path)

if __name__ == "__main__":
  main()
