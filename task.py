import sys
sys.path.append("/gpfsdswork/projects/rech/sfz/utt84zy/anaconda3/envs/huggingface/lib/python3.7/site-packages")
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
from utils import dataprocess
if tf.__version__ in ['2.3.0','2.6.0','2.5.0','2.7.0','2.8.0','2.9.0']:
  from optimizer import utils_23 as optimizer_util
else:
  from optimizer import utils as optimizer_util
tf.get_logger().setLevel(logging.INFO)
from utils.my_inputter import My_inputter, LDR_inputter
from opennmt.models.sequence_to_sequence import SequenceToSequence
from model import Multi_domain_SequenceToSequence, LDR_SequenceToSequence, SequenceToSequence_with_dprob
from encoders.self_attention_encoder import Multi_domain_SelfAttentionEncoder
from decoders.self_attention_decoder import Multi_domain_SelfAttentionDecoder
import numpy as np
from utils.dataprocess import create_priming_training_dataset, create_training_dataset_robustness, create_training_dataset_DRO, create_training_dataset_with_dprob, create_training_dataset_hvd, merge_map_fn, create_training_dataset_v1, create_multi_domain_meta_training_dataset_v2, create_meta_training_dataset, create_training_dataset, create_multi_domain_meta_training_dataset, create_training_dataset_v2, create_multi_domain_meta_training_dataset_v1
from opennmt.utils import BLEUScorer
from opennmt.inputters.text_inputter import WordEmbedder
from utils.utils_ import variance_scaling_initialier, MultiBLEUScorer, var_spec
from layers.layers import Multi_domain_FeedForwardNetwork, Multi_domain_FeedForwardNetwork_v2, DAFE
from utils.utils_ import average_checkpoints, load_and_update_if_needed_from_ckpt, average_checkpoints_tf2_3
from utils.dataprocess import count_lines
from opennmt.utils import misc


import seaborn as sns; sns.set_theme()
import numpy as np
import matplotlib.pyplot as plt

def map_prod(x_labels,y_labels, matrix,name):
    plt.figure()
    ax = sns.heatmap(np.array(matrix),xticklabels=x_labels, yticklabels=y_labels)
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(length=0)
    plt.xticks(rotation=90)
    fig = ax.get_figure()
    fig.savefig(name,dpi=1000)


def reward_rescale(rewards):
  abs_max = max([abs(r) for r in rewards])
  return [r/abs_max for r in rewards]

def file_concatenate(files,name,dir_name=None):
  lines = []
  for f in files:
    with open(f,"r") as f_:
      for l in f_.readlines():
        lines.append(l.strip())
  
  parent_dir = os.path.dirname(files[0])
  if not dir_name:
    dir_name = parent_dir
  with open(os.path.join(dir_name,name),"w") as f_w:
    for l in lines:
      print(l,file=f_w)
  return os.path.join(dir_name,name)

def _assert_loss_is_finite(loss):
  if tf.math.is_nan(loss):
    raise RuntimeError("Model diverged with loss = NaN.")

def update(v,g,lr=1.0):
  if isinstance(g, tf.IndexedSlices):
    return tf.tensor_scatter_nd_sub(v/lr,tf.expand_dims(g.indices,1),g.values)*lr
  else:
    return v-lr*g

def translate(source_file,
              reference,
              model,
              checkpoint_manager,
              checkpoint,              
              domain,
              output_file,
              length_penalty,
              is_noisy=1,
              checkpoint_path=None,
              probs_file=None,
              experiment="ldr",
              score_type="MultiBLEU",
              batch_size=5,
              beam_size=5):
  
  # Create the inference dataset.
  if checkpoint_path == None:
    checkpoint_path = checkpoint_manager.latest_checkpoint
  tf.get_logger().info("Evaluating model %s", checkpoint_path)
  print("In domain %d"%domain)
  checkpoint.restore(checkpoint_path)
  if isinstance(model, SequenceToSequence_with_dprob):
    dataset = model.examples_inputter.make_inference_dataset(source_file, probs_file, batch_size)
  elif isinstance(model, onmt.models.Transformer):
    dataset = model.examples_inputter.make_inference_dataset(source_file, batch_size)
  else:
    dataset = model.examples_inputter.make_inference_dataset(source_file, batch_size, domain, is_noisy=is_noisy)
  iterator = iter(dataset)

  # Create the mapping for target ids to tokens.
  ids_to_tokens = model.labels_inputter.ids_to_tokens

  @tf.function
  def predict_next():    
    source = next(iterator)
    source_length = source["length"]
    batch_size = tf.shape(source_length)[0]
    source_inputs = model.features_inputter(source)
    if experiment in ["residual","small_transformer","residualv15","DRO","residualv25","residualv27","residualv28","residualv29","residual_big_transformer","residualv26","gated_residual_v5","residualv16","residualv19","residualv20","residualv21","residualv22","residualv23","residualv17","residualv18","residualv2","residualv1","residualv3","residualv5","residualv13","residualv12","residualv6","residualv7","residualv11","residualv8","residualv9","baselinev1"]:
      encoder_outputs, _, _ = model.encoder([source_inputs, source["domain"], source["is_noisy"]], source_length, training=False, internal_node_printing=True)
    else:
      encoder_outputs, _, _ = model.encoder(source_inputs, source_length, training=False)

    # Prepare the decoding strategy.
    if beam_size > 1:
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      source_length = tfa.seq2seq.tile_batch(source_length, beam_size)
      decoding_strategy = onmt.utils.BeamSearch(beam_size, length_penalty=length_penalty)
    else:
      decoding_strategy = onmt.utils.GreedySearch()

    # Run dynamic decoding.
    decoder_state = model.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=source_length)
    if experiment in ["residual","residualv2","small_transformer","DRO","residualv15","residualv25","residualv27","residual_big_transformer","residualv26","gated_residual_v5","residualv16","residualv19","residualv20","residualv21","residualv22","residualv23","residualv17","residualv18","residualv1","residualv3","residualv5","residualv6","residualv7","residualv13","residualv12","residualv11","residualv8","residualv9","baselinev1"]:
      map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}, training=False), tf.dtypes.cast(tf.fill(tf.expand_dims(tf.shape(ids)[0],0), domain), tf.int64)]
    elif experiment in ["DC","ldr"]:
      map_input_fn = lambda ids: model.labels_inputter({"ids": ids}, domain=domain, training=False)
    elif experiment in ["WDC"]:
      e_r, _ = model.classification_layer(encoder_outputs, source_length, training=False)
      e_s, _ = model.adv_classification_layer(encoder_outputs, source_length, training=False)
      g_s = model.share_gate(tf.concat([tf.tile(tf.expand_dims(e_s,1),[1,tf.shape(encoder_outputs)[1],1]),encoder_outputs],-1))
      g_r = model.specific_gate(tf.concat([tf.tile(tf.expand_dims(e_r,1),[1,tf.shape(encoder_outputs)[1],1]),encoder_outputs],-1))
      h_r = g_r * encoder_outputs
      h_s = g_s * encoder_outputs
      encoder_mask = model.encoder.build_mask(source_inputs, sequence_length=source_length)
      map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}, training=False), h_r, h_s, encoder_mask]
    elif experiment in ["residualv28","residualv29"]:
      map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}, training=False), source["domain"]]
    else:
      map_input_fn = lambda ids: model.labels_inputter({"ids": ids}, training=False)
    decoded = model.decoder.dynamic_decode(
        map_input_fn,
        tf.fill([batch_size], START_OF_SENTENCE_ID),
        end_id=END_OF_SENTENCE_ID,
        initial_state=decoder_state,
        decoding_strategy=decoding_strategy,
        maximum_iterations=250)
    target_lengths = decoded.lengths
    target_tokens = ids_to_tokens.lookup(tf.cast(decoded.ids, tf.int64))
    return target_tokens, target_lengths

  # Iterates on the dataset.
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  print("output file: ", output_file)
  with open(output_file, "w") as output_:
    while True:    
      try:
        batch_tokens, batch_length = predict_next()
        for tokens, length in zip(batch_tokens.numpy(), batch_length.numpy()):
          sentence = b" ".join(tokens[0][:length[0]])
          print_bytes(sentence, output_)
          #print_bytes(sentence)
      except tf.errors.OutOfRangeError:
        break
  if reference!=None:
    print("score of model %s on test set %s: "%(checkpoint_manager.latest_checkpoint, source_file), scorer(reference, output_file))
    score = scorer(reference, output_file)
    if score is None:
      return 0.0
    else:
      return score

def priming_translate(source_files,
              reference,
              model,
              checkpoint_manager,
              checkpoint,              
              domain,
              output_file,
              length_penalty,
              is_noisy=1,
              checkpoint_path=None,
              probs_file=None,
              experiment="ldr",
              score_type="MultiBLEU",
              batch_size=5,
              beam_size=5):
  
  # Create the inference dataset.
  if checkpoint_path == None:
    checkpoint_path = checkpoint_manager.latest_checkpoint
    
  tf.get_logger().info("Evaluating model %s", checkpoint_path)
  checkpoint.restore(checkpoint_path)
  dataset = model.examples_inputter.make_inference_dataset(source_files, batch_size)
  iterator = iter(dataset)

  # Create the mapping for target ids to tokens.
  ids_to_tokens = model.labels_inputter.ids_to_tokens

  @tf.function
  def predict_next():    
    source = next(iterator)
    source_length = model.features_inputter.get_length(source) #source["length"]
    batch_size = tf.shape(source_length[0])[0]
    source_inputs = model.features_inputter(source)

    source_length, pre_length = source_length
    source_inputs, pre_inputs = source_inputs
    
    encoder_outputs, encoder_state, encoder_sequence_length = model.encoder(
      source_inputs, sequence_length=source_length, training=False)
    pre_encoder_outputs, pre_encoder_state, pre_encoder_sequence_length = model.pre_encoder(
      pre_inputs, sequence_length=pre_length, training=False)
    
    # Prepare the decoding strategy.
    if beam_size > 1:

      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      source_length = tfa.seq2seq.tile_batch(source_length, beam_size)

      pre_encoder_outputs = tfa.seq2seq.tile_batch(pre_encoder_outputs, beam_size)
      pre_length = tfa.seq2seq.tile_batch(pre_length, beam_size)

      # encoder_outputs = [encoder_outputs, pre_encoder_outputs]
      # source_length = [source_length, pre_length]
      decoding_strategy = onmt.utils.BeamSearch(beam_size, length_penalty=length_penalty)
    else:
      decoding_strategy = onmt.utils.GreedySearch()

    # Run dynamic decoding.

    if model.version in [1,5]:
      decoder_state = model.decoder.initial_state(
        memory=tf.concat([encoder_outputs, pre_encoder_outputs], axis=1),
        memory_sequence_length= [encoder_sequence_length, pre_encoder_sequence_length],
        initial_state= None)
    
    map_input_fn = lambda ids: model.labels_inputter({"ids": ids}, training=False)
    decoded = model.decoder.dynamic_decode(
        map_input_fn,
        tf.fill([batch_size], START_OF_SENTENCE_ID),
        end_id=END_OF_SENTENCE_ID,
        initial_state=decoder_state,
        decoding_strategy=decoding_strategy,
        maximum_iterations=250)
    target_lengths = decoded.lengths
    target_tokens = ids_to_tokens.lookup(tf.cast(decoded.ids, tf.int64))
    return target_tokens, target_lengths

  # Iterates on the dataset.
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  print("output file: ", output_file)
  with open(output_file, "w") as output_:
    while True:    
      try:
        batch_tokens, batch_length = predict_next()
        for tokens, length in zip(batch_tokens.numpy(), batch_length.numpy()):
          sentence = b" ".join(tokens[0][:length[0]])
          print_bytes(sentence, output_)
          #print_bytes(sentence)
      except tf.errors.OutOfRangeError:
        break
  if reference!=None:
    print("score of model %s on test set %s: "%(checkpoint_manager.latest_checkpoint, source_files[0]), scorer(reference, output_file))
    score = scorer(reference, output_file)
    if score is None:
      return 0.0
    else:
      return score

def priming_translate_v1(source_files,
              reference,
              model,
              checkpoint_manager,
              checkpoint,              
              domain,
              output_file,
              length_penalty,
              is_noisy=1,
              checkpoint_path=None,
              probs_file=None,
              experiment="ldr",
              score_type="MultiBLEU",
              batch_size=5,
              beam_size=5):
  
  # Create the inference dataset.
  if checkpoint_path == None:
    checkpoint_path = checkpoint_manager.latest_checkpoint
    
  tf.get_logger().info("Evaluating model %s", checkpoint_path)
  checkpoint.restore(checkpoint_path)
  dataset = model.examples_inputter.make_inference_dataset(source_files, batch_size)
  iterator = iter(dataset)

  # Create the mapping for target ids to tokens.
  ids_to_tokens = model.labels_inputter.ids_to_tokens

  @tf.function
  def predict_next():    
    source = next(iterator)
    source_lengths = model.features_inputter.get_length(source) #source["length"]
    batch_size = tf.shape(source_lengths[0])[0]
    source_inputs = model.features_inputter(source)

    source_pre_length, source_hide_length = source_lengths
    source_pre_inputs, source_hide_inputs = source_inputs

    encoder_hide_outputs, encoder_hide_state, encoder_hide_sequence_length = model.encoder(
        source_hide_inputs, sequence_length=source_hide_length, training=False)
    
    encoder_outputs = encoder_hide_outputs
    source_length = encoder_hide_sequence_length

    # Prepare the decoding strategy.
    if beam_size > 1:
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      source_length = tfa.seq2seq.tile_batch(source_length, beam_size)
      decoding_strategy = onmt.utils.BeamSearch(beam_size, length_penalty=length_penalty)
    else:
      decoding_strategy = onmt.utils.GreedySearch()

    # Run dynamic decoding.
    
    decoder_state = model.decoder.initial_state(
      memory= encoder_outputs ,
      memory_sequence_length= source_length,
      initial_state= None)
    
    map_input_fn = lambda ids: model.labels_inputter({"ids": ids}, training=False)
    decoded = model.decoder.dynamic_decode(
        map_input_fn,
        tf.fill([batch_size], START_OF_SENTENCE_ID),
        end_id=END_OF_SENTENCE_ID,
        initial_state=decoder_state,
        decoding_strategy=decoding_strategy,
        maximum_iterations=250)
    target_lengths = decoded.lengths
    target_tokens = ids_to_tokens.lookup(tf.cast(decoded.ids, tf.int64))
    return target_tokens, target_lengths

  # Iterates on the dataset.
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  print("output file: ", output_file)
  with open(output_file, "w") as output_:
    while True:    
      try:
        batch_tokens, batch_length = predict_next()
        for tokens, length in zip(batch_tokens.numpy(), batch_length.numpy()):
          sentence = b" ".join(tokens[0][:length[0]])
          print_bytes(sentence, output_)
          #print_bytes(sentence)
      except tf.errors.OutOfRangeError:
        break
  if reference!=None:
    print("score of model %s on test set %s: "%(checkpoint_manager.latest_checkpoint, source_files[0]), scorer(reference, output_file))
    score = scorer(reference, output_file)
    if score is None:
      return 0.0
    else:
      return score

def priming_avg_ckpt_translate(config, source_files,
              reference,
              model,
              checkpoint_manager,
              checkpoint,
              domain,
              output_file,
              length_penalty,
              is_noisy=1,
              experiment="ldr",
              score_type="MultiBLEU",
              batch_size=10,
              beam_size=10,
              max_count=3):
  
  # Create the inference dataset.
  new_checkpoint_manager = average_checkpoints(config["model_dir"], output_dir="%s/averaged_checkpoint"%config["model_dir"], trackables={"model":model},
                        max_count=max_count,
                        model_key="model")
  checkpoint.restore(new_checkpoint_manager.latest_checkpoint)
  tf.get_logger().info("Evaluating model %s", new_checkpoint_manager.latest_checkpoint)
  dataset = model.examples_inputter.make_inference_dataset(source_files, batch_size)
  iterator = iter(dataset)

  # Create the mapping for target ids to tokens.
  ids_to_tokens = model.labels_inputter.ids_to_tokens

  @tf.function
  def predict_next():    
    source = next(iterator)
    source_length = model.features_inputter.get_length(source) #source["length"]
    batch_size = tf.shape(source_length[0])[0]
    source_inputs = model.features_inputter(source)

    source_length, pre_length = source_length
    source_inputs, pre_inputs = source_inputs

    encoder_outputs, encoder_state, encoder_sequence_length = model.encoder(
      source_inputs, sequence_length=source_length, training=False)
    pre_encoder_outputs, pre_encoder_state, pre_encoder_sequence_length = model.pre_encoder(
      pre_inputs, sequence_length=pre_length, training=False)
    
    # Prepare the decoding strategy.
    if beam_size > 1:
      
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      source_length = tfa.seq2seq.tile_batch(source_length, beam_size)

      pre_encoder_outputs = tfa.seq2seq.tile_batch(pre_encoder_outputs, beam_size)
      pre_length = tfa.seq2seq.tile_batch(pre_length, beam_size)

      decoding_strategy = onmt.utils.BeamSearch(beam_size, length_penalty=length_penalty)
    else:
      decoding_strategy = onmt.utils.GreedySearch()

    # Run dynamic decoding.
    if model.version in [1,5]:
      decoder_state = model.decoder.initial_state(
        memory=tf.concat([encoder_outputs, pre_encoder_outputs], axis=1),
        memory_sequence_length= [encoder_sequence_length,pre_encoder_sequence_length],
        initial_state = None)
        
    map_input_fn = lambda ids: model.labels_inputter({"ids": ids}, training=False)
    decoded = model.decoder.dynamic_decode(
        map_input_fn,
        tf.fill([batch_size], START_OF_SENTENCE_ID),
        end_id=END_OF_SENTENCE_ID,
        initial_state=decoder_state,
        decoding_strategy=decoding_strategy,
        maximum_iterations=250)

    target_lengths = decoded.lengths
    target_tokens = ids_to_tokens.lookup(tf.cast(decoded.ids, tf.int64))
    return target_tokens, target_lengths

  # Iterates on the dataset.
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  print("output file: ", output_file)
  with open(output_file, "w") as output_:
    while True:    
      try:
        batch_tokens, batch_length = predict_next()
        for tokens, length in zip(batch_tokens.numpy(), batch_length.numpy()):
          sentence = b" ".join(tokens[0][:length[0]])
          print_bytes(sentence, output_)
          #print_bytes(sentence)
      except tf.errors.OutOfRangeError:
        break
  if reference!=None:
    print("score of model %s on test set %s: "%(checkpoint_manager.latest_checkpoint, source_files[0]), scorer(reference, output_file))
    score = scorer(reference, output_file)
    if score is None:
      return 0.0
    else:
      return score

def priming_avg_ckpt_translate_v1(config, source_files,
              reference,
              model,
              checkpoint_manager,
              checkpoint,
              domain,
              output_file,
              length_penalty,
              translate_with_hide=True,
              is_noisy=1,
              experiment="ldr",
              score_type="MultiBLEU",
              batch_size=10,
              beam_size=10,
              max_count=3):
  
  # Create the inference dataset.
  """ new_checkpoint_manager = average_checkpoints(config["model_dir"], output_dir="%s/averaged_checkpoint"%config["model_dir"], trackables={"model":model},
                        max_count=max_count,
                        model_key="model") """
  checkpoint.restore(checkpoint_manager.latest_checkpoint)
  tf.get_logger().info("Evaluating model %s", checkpoint_manager.latest_checkpoint)
  dataset = model.examples_inputter.make_inference_dataset(source_files, batch_size)
  iterator = iter(dataset)

  # Create the mapping for target ids to tokens.
  ids_to_tokens = model.labels_inputter.ids_to_tokens

  @tf.function
  def predict_next():    
    source = next(iterator)
    source_lengths = model.features_inputter.get_length(source) #source["length"]
    batch_size = tf.shape(source_lengths[0])[0]
    source_inputs = model.features_inputter(source)

    source_pre_length, source_hide_length = source_lengths
    source_pre_inputs, source_hide_inputs = source_inputs

    if translate_with_hide:
      encoder_hide_outputs, encoder_hide_state, encoder_hide_sequence_length = model.encoder(
        source_hide_inputs, sequence_length=source_hide_length, training=False)
      encoder_outputs = encoder_hide_outputs
      source_length = encoder_hide_sequence_length
    else:
      encoder_pre_outputs, encoder_pre_state, encoder_pre_sequence_length = model.encoder(
        source_pre_inputs, sequence_length=source_pre_length, training=False)
      encoder_outputs = encoder_pre_outputs
      source_length = encoder_pre_sequence_length
    
    # Prepare the decoding strategy.
    if beam_size > 1:
      
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      source_length = tfa.seq2seq.tile_batch(source_length, beam_size)
      decoding_strategy = onmt.utils.BeamSearch(beam_size, length_penalty=length_penalty)
    else:
      decoding_strategy = onmt.utils.GreedySearch()

    # Run dynamic decoding.
    
    decoder_state = model.decoder.initial_state(
      memory= encoder_outputs,
      memory_sequence_length= source_length,
      initial_state = None)
        
    map_input_fn = lambda ids: model.labels_inputter({"ids": ids}, training=False)
    decoded = model.decoder.dynamic_decode(
        map_input_fn,
        tf.fill([batch_size], START_OF_SENTENCE_ID),
        end_id=END_OF_SENTENCE_ID,
        initial_state=decoder_state,
        decoding_strategy=decoding_strategy,
        maximum_iterations=250)

    target_lengths = decoded.lengths
    target_tokens = ids_to_tokens.lookup(tf.cast(decoded.ids, tf.int64))
    return target_tokens, target_lengths

  # Iterates on the dataset.
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  print("output file: ", output_file)
  with open(output_file, "w") as output_:
    while True:    
      try:
        batch_tokens, batch_length = predict_next()
        for tokens, length in zip(batch_tokens.numpy(), batch_length.numpy()):
          sentence = b" ".join(tokens[0][:length[0]])
          print_bytes(sentence, output_)
          #print_bytes(sentence)
      except tf.errors.OutOfRangeError:
        break
  if reference!=None:
    print("score of model %s on test set %s: "%(checkpoint_manager.latest_checkpoint, source_files[0]), scorer(reference, output_file))
    score = scorer(reference, output_file)
    if score is None:
      return 0.0
    else:
      return score

def debug(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100,
          picking_prob=None): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  #####
  batch_train_size = config["batch_train_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  prob_file = config["prob"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))
  train_dataset = create_training_dataset_DRO(strategy, model, source_file, target_file, prob_file, domain, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True),picking_prob=config.get("picking_prob",None))
  
  def _accumulate_gradients(source, target):
    tf.print(source)
    return 0, 0
 
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples

  @dataset_util.function_on_next(train_dataset)
  def _train_iteration(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      return per_replica_source, per_replica_target
  
  
  # Runs the training loop.
  train_data_flow = iter(_train_forward())
  _, _ = next(train_data_flow)

  while True:
    #####Training batch
    for _ in range(int(config.get("accumulation_step",1))):
      _, _ = next(train_data_flow)    

def meta_train_v1(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_meta_training_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length)
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    meta_train_gradient_accumulator = optimizer_util.GradientAccumulator()  
    meta_test_gradient_accumulator = optimizer_util.GradientAccumulator()

  def _accumulate_meta_train_gradients(source, target):
    print("source: ", source)
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    variables = [] #model.trainable_variables
    for variable in model.trainable_variables:
      if "ADAP_" in variable.name or "ldr_embedding" in variable.name or "ldr_inputter" in variable.name:
        variables.append(variable)
    print("var numb: ", len(variables))
    training_loss = model.regularize_loss(training_loss, variables=variables)
    gradients = optimizer.get_gradients(training_loss, variables)
    meta_train_gradient_accumulator(gradients)
    num_examples = tf.shape(source["length"])[0]
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

  def _accumulate_meta_test_gradients(source, target):
    print("source: ", source)
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    variables = [] #model.trainable_variables
    for variable in model.trainable_variables:
      if not("ADAP_" in variable.name or "ldr_embedding" in variable.name or "ldr_inputter" in variable.name):
        variables.append(variable)
    print("var numb: ", len(variables))
    training_loss = model.regularize_loss(training_loss, variables=variables)
    gradients = optimizer.get_gradients(training_loss, variables)
    meta_test_gradient_accumulator(gradients)
    num_examples = tf.shape(source["length"])[0]
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

  def _apply_meta_train_gradients():
    variables = [] #model.trainable_variables
    for variable in model.trainable_variables:
      if "ADAP_" in variable.name or "ldr_embedding" in variable.name or "ldr_inputter" in variable.name:
        variables.append(variable)
    print("var numb: ", len(variables))
    grads_and_vars = []
    
    for gradient, variable in zip(meta_train_gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      #if "ADAP_" in variable.name or "ldr_embedding" in variable.name or "ldr_inputter" in variable.name:
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(meta_train_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    meta_train_gradient_accumulator.reset()

  def _apply_meta_test_gradients():
    variables = [] #model.trainable_variables
    for variable in model.trainable_variables:
      if not("ADAP_" in variable.name or "ldr_embedding" in variable.name or "ldr_inputter" in variable.name):
        variables.append(variable)
    print("var numb: ", len(variables))
    grads_and_vars = []
    
    for gradient, variable in zip(meta_test_gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      #if not("ADAP_" in variable.name or "ldr_embedding" in variable.name or "ldr_inputter" in variable.name):
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(meta_test_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    meta_test_gradient_accumulator.reset()
 
  @dataset_util.function_on_next(meta_train_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_meta_train_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_examples = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_num_examples, None)    
    return loss, num_examples

  @dataset_util.function_on_next(meta_test_dataset)
  def _meta_test_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, _ = strategy.experimental_run_v2(
          _accumulate_meta_test_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
    return loss

  @dataset_util.function_on_next(meta_train_dataset)
  def _meta_train_iteration(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      return per_replica_source, per_replica_target
  
  @dataset_util.function_on_next(meta_test_dataset)
  def _meta_test_iteration(next_fn):    
    with strategy.scope():
      return next_fn()
  
  @tf.function
  def _meta_train_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_meta_train_gradients)

  @tf.function
  def _meta_test_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_meta_test_gradients)

  def _set_weight(v, w):
    v.assign(w)

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  meta_train_data_flow = iter(_meta_train_forward())
  meta_test_data_flow = iter(_meta_test_forward())
  _loss = []  
  with _summary_writer.as_default():
    while True:
      #####Training batch
      loss, _ = next(meta_train_data_flow)  
      #print("number_examples_per_replica: ", num_examples)
      _loss.append(loss)  
      #snapshots = [v.value() for v in model.trainable_variables]
      _meta_train_step()
      #####Testing batch
      loss = next(meta_test_data_flow)
      #weight_reset(snapshots)
      _meta_test_step()
      ####      
      step = optimizer.iterations.numpy()//2
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; after %f seconds",
            step, learning_rate(step), np.mean(_loss), elapsed)
        _loss = []
        start = time.time()
      if step % save_every == 0 and optimizer.iterations.numpy()%2==0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0 and optimizer.iterations.numpy()%2==0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def meta_train_v2(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  with strategy.scope():
    model.create_variables()
    gradient_accumulator = optimizer_util.GradientAccumulator()  
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
      checkpoint_path = checkpoint_manager.latest_checkpoint
      output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
      score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_multi_domain_meta_training_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length)
  #####
  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target):
    tf.print("meta_train_source: ",meta_train_source, output_stream=sys.stderr)
    tf.print("meta_train_target: ",meta_train_target, output_stream=sys.stderr)
    tf.print("meta_test_source: ",meta_test_source, output_stream=sys.stderr)
    tf.print("meta_test_target: ",meta_test_target, output_stream=sys.stderr)
    outputs, _ = model(
        meta_train_source,
        labels=meta_train_target,
        training=True,
        step=optimizer.iterations)    
    loss = model.compute_loss(outputs, meta_train_target, training=True)
    training_loss = loss[0] / loss[1]
    reported_loss = loss[0] / loss[2]
    variables = model.trainable_variables       
    args_dict = dict()
    for v in variables:
      args_dict.update({v.name:v})
    training_loss = model.regularize_loss(training_loss, variables=variables)
    gradients = tf.gradients(training_loss, variables)
    ##### Inner adaptation
    def update(v,g,lr=1.0):
      if "embedding" in v.name:
        # print("embedding gradient's values: __________", g.values)
        # print("embedding gradient's indices: _________", g.indices)
        print(v)
        print(g)
        return tf.tensor_scatter_nd_sub(v/lr,tf.expand_dims(g.indices,1),g.values)*lr
      else:
        return v-lr*g
    if config.get("stopping_gradient",True):
      print("apply stopping_gradient")
      for g, v in zip(gradients, variables):      
        args_dict.update({v.name: v-g})
    else:
      print("passing gradient")
      for g, v in zip(gradients, variables):
        args_dict.update({v.name: update(v,g)})
    #### Meta_loss:
    outputs, _ = model.forward_fn(meta_test_source,
        args_dict,
        labels=meta_test_target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, meta_test_target, training=True)
    meta_training_loss = loss[0] / loss[1]
    meta_reported_loss = loss[0] / loss[2]
    meta_training_loss = model.regularize_loss(meta_training_loss, variables=variables)
    gradients = optimizer.get_gradients(meta_training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.shape(meta_test_target["length"])[0]
    return reported_loss, num_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync)
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_examples = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_num_examples, None)    
    return loss, num_examples
    
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  
  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  meta_train_data_flow = iter(_meta_train_forward())
  _loss = []  
  with _summary_writer.as_default():
    while True:
      #####Training batch
      loss, _ = next(meta_train_data_flow)  
      _loss.append(loss)
      _step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; after %f seconds",
            step, learning_rate(step), np.mean(_loss), elapsed)
        _loss = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def elastic_finetuning(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          elastic_type="Uniform", # "Uniform", "EWC"
          EWC_path=None,
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  else:
    if checkpoint_path is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
      checkpoint.restore(checkpoint_path)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=True,picking_prob=config.get("picking_prob",None))
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
  star_vars = []

  def build_model(source, target):
    _, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
  
  @dataset_util.function_on_next(train_dataset)
  def _build_model(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      strategy.experimental_run_v2(
          build_model, args=(per_replica_source, per_replica_target))

  @tf.function
  def star_vars_init():
    variables = model.trainable_variables
    with tf.init_scope():
      for var in variables:
        value=var.numpy()
        star_vars.append(tf.constant(value))

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    variables = model.trainable_variables
    lambda_ = config.get("lambda", 0.001)
    print("elastic weights: ", lambda_)
    print("elastic_type: ", elastic_type)
    if elastic_type =="Uniform":
      for i in range(len(variables)):
        training_loss += tf.reduce_sum(tf.square(variables[i] - star_vars[i])) * lambda_
    elif elastic_type=="EWC":
      assert EWC_path !=None
      EWC_weights = np.load(EWC_path)
      if not config.get("EWC_global"):
        for i in range(len(variables)):
          training_loss += tf.reduce_sum(tf.square(variables[i] - star_vars[i]) * EWC_weights[variables[i].name]) / np.average(EWC_weights[variables[i].name]) * lambda_
      else:
        sum = 0
        count = 0
        for w in EWC_weights:
          sum += np.sum(EWC_weights[w])
          count += np.sum(EWC_weights[w])/np.average(EWC_weights[w])
        
        EWC_global_avg = sum / count
        print("EWC_global_avg: ", EWC_global_avg)
        print("params numb: ", count)

        for i in range(len(variables)):
          training_loss += tf.reduce_sum(tf.square(variables[i] - star_vars[i]) * EWC_weights[variables[i].name]) / EWC_global_avg * lambda_
    print("var numb: ", len(variables))
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  # Runs the training loop.
  import time
  start = time.time()  
  first_run = iter(_build_model())
  next(first_run)
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  _loss = []  
  _number_examples = []
  step = optimizer.iterations.numpy()     
  ###
  if config.get("continual_learning", False):
    print("Continual Learning needs to load from old model")
    assert config.get("checkpoint_path") != None
    checkpoint_path = config.get("checkpoint_path")
    load_and_update_if_needed_from_ckpt(config["model_dir"],   
                        checkpoint_path,                        
                        trackables={"model":model},
                        model_key="model")

  ### assign value to star_vars
  star_vars_init()
  step = optimizer.iterations.numpy()
  if config.get("reset_step",None):
    print("start from %d-th step"%config.get("reset_step",150000))
    optimizer.iterations.assign(config.get("reset_step",150000))
    step = optimizer.iterations.numpy()
  ###
  train_data_flow = iter(_train_forward())
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        loss, num_examples = next(train_data_flow)    
        _loss.append(loss)
        _number_examples.append(num_examples)
      _step()  
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      tf.summary.flush()
      if step > train_steps:
        break

def finetuning(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,          
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  checkpoint_path = config.get("checkpoint_path",None)
  
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=(config["experiment"]!="baseline"),picking_prob=config.get("picking_prob",None))
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss

    if config.get("ADAP_activity_regularizing",False):
        layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
        print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
        
        if isinstance(layer_activity_regularization_loss_scale, list):
          domain = source["domain"][0]
          layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
          layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        regularization_losses = model.losses
        print("model_name_scope", model.name_scope())
        print(regularization_losses)
        layer_activity_regularization_losses = []
        for loss_ in regularization_losses:
          if "multi_adap__dense" in loss_.name:
            continue
          elif "ADAP_gate" in loss_.name: #and "ActivityRegularizer" not in loss_.name and "Regularizer" not in loss_.name
            continue
          elif "ADAP_" in loss_.name:
            layer_activity_regularization_losses.append(loss_)

        print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
        if (len(layer_activity_regularization_losses)>0):
          training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)

    if config.get("ADAP_weight_decay",False):
      lambda_ = config.get("lambda",0.00001)
      print("ADAP_weight_decay: ", lambda_)
      for v in model.trainable_variables:
        if "ADAP_" in v.name and "layer_norm" in v.name:
          training_loss += tf.reduce_sum(tf.square(v)) * lambda_
    variables = []
    for v in model.trainable_variables:
      if "ADAP_" in v.name or "ldr_embedding" in v.name or "ldr_inputter" in v.name or "lhuc" in v.name:
        print(v.name)
        variables.append(v)
    print("var numb: ", len(variables))
    training_loss = model.regularize_loss(training_loss, variables=variables)
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.shape(source["length"])[0]
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

  def _apply_gradients():
    variables = []
    for v in model.trainable_variables:
      if "ADAP_" in v.name or "ldr_embedding" in v.name or "ldr_inputter" in v.name or "lhuc" in v.name:
        variables.append(v)
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync)
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @dataset_util.function_on_next(train_dataset)
  def _finetuning_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_examples = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_num_examples, None)    
    return loss, num_examples

  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  finetuning_data_flow = iter(_finetuning_forward())
  
  _loss = [] 
  #####
  if config.get("using_tf_restore",True):
    if checkpoint_manager.latest_checkpoint is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
      checkpoint.restore(checkpoint_manager.latest_checkpoint)
    else:
      if checkpoint_path is not None:
        tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
        checkpoint.restore(checkpoint_path)
  else:
    if checkpoint_path is None:
      if checkpoint_manager.latest_checkpoint is not None:
        tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
        checkpoint_path = checkpoint_manager.latest_checkpoint
        load_and_update_if_needed_from_ckpt(config["model_dir"],   
                        checkpoint_path,                        
                        trackables={"model":model},
                        vocab_update=False,
                        model_key="model") 
        #checkpoint.restore(checkpoint_manager.latest_checkpoint)
    else:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
      #checkpoint.restore(checkpoint_path)
      load_and_update_if_needed_from_ckpt(config["model_dir"],   
                        checkpoint_path,                        
                        trackables={"model":model},
                        vocab_update=False,
                        model_key="model") 
  #####
  
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        loss, _ = next(finetuning_data_flow) 
        _loss.append(loss)
      _step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; after %f seconds",
            step, learning_rate(step), np.mean(_loss), elapsed)
        _loss = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def meta_train_v7(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_multi_domain_meta_training_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length)
  #####
  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target): 
     
    with tf.GradientTape(persistent=True) as tape:
      ##### Inner adaptation
      outputs, _ = model(
          meta_train_source,
          labels=meta_train_target,
          training=True,
          step=optimizer.iterations)    
      loss = model.compute_loss(outputs, meta_train_target, training=True)
      training_loss = loss[0] / loss[1]
      if config.get("ADAP_activity_regularizing",False):
        layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
        output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
        print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
        print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
        if isinstance(layer_activity_regularization_loss_scale, list):
          domain = meta_train_source["domain"][0]
          layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
          layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        if isinstance(output_activity_regularization_loss_scale, list):
          domain = meta_train_source["domain"][0]
          output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
          output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
        regularization_losses = model.losses
        print("model_name_scope", model.name_scope())
        print(regularization_losses)
        layer_activity_regularization_losses = []
        output_activity_regularization_losses = []
        for loss_ in regularization_losses:
          if "multi_adap__dense" in loss_.name:
            output_activity_regularization_losses.append(loss_)
          else:
            layer_activity_regularization_losses.append(loss_)
        print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
        print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
        if len(layer_activity_regularization_losses)>0:
          training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)
        if len(output_activity_regularization_losses)>0:
          training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)
      variables = model.trainable_variables       
      args_dict = dict()
      for v in variables:
        args_dict.update({v.name:v})
      shared_gradients = []
      adap_gradients = []
      adap_variables = []
      shared_variables = []
      for v in variables:
        if "ADAP_" in v.name or "ldr_embedding" in v.name or "ldr_inputter" in v.name:
          adap_variables.append(v)
        else:
          shared_variables.append(v)
      variables = adap_variables + shared_variables
      adap_variables_name = [v.name for v in adap_variables]
      shared_variables_name = [v.name for v in shared_variables]
      gradients = tape.gradient(training_loss, variables)  
      gradient_accumulator(gradients) 
      var_spec(variables)
      var_spec(shared_variables)
      var_spec(adap_variables)
      for g,v in zip(gradients, variables):
        if v.name in shared_variables_name:
          shared_gradients.append(g)
        elif v.name in adap_variables_name:
          adap_gradients.append(g)

      meta_train_lr = config.get("meta_train_lr",1.0)
      print("meta_train_lr: ", meta_train_lr)
      
      if config.get("stopping_gradient",True):
        print("apply stopping_gradient")
        for g, v in zip(shared_gradients, shared_variables):      
          args_dict.update({v.name: v-meta_train_lr*tf.stop_gradient(g)})
      else:
        print("passing gradient")
        for g, v in zip(shared_gradients, shared_variables):
          args_dict.update({v.name: update(v,g,lr=meta_train_lr)})
      
      #### Meta_loss:
        #### update adap parameters first
      outputs, _ = model(
          meta_test_source,
          labels=meta_test_target,
          training=True,
          step=optimizer.iterations)    
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      training_loss = loss[0] / loss[1]
      if config.get("ADAP_activity_regularizing",False):
        layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
        output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
        print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
        print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
        if isinstance(layer_activity_regularization_loss_scale, list):
          domain = meta_test_source["domain"][0]
          layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
          layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        if isinstance(output_activity_regularization_loss_scale, list):
          domain = meta_test_source["domain"][0]
          output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
          output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
        regularization_losses = model.losses
        print("model_name_scope", model.name_scope())
        print(regularization_losses)
        layer_activity_regularization_losses = []
        output_activity_regularization_losses = []
        for loss_ in regularization_losses:
          if "multi_adap__dense" in loss_.name:
            output_activity_regularization_losses.append(loss_)
          else:
            layer_activity_regularization_losses.append(loss_)
        print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
        print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
        if len(layer_activity_regularization_losses)>0:
          training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)
        if len(output_activity_regularization_losses)>0:
          training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)
      adap_gradients = tape.gradient(training_loss, adap_variables)
        #### meta gradient for shared parameters
      outputs, _ = model.forward_fn(meta_test_source,
          args_dict,
          labels=meta_test_target,
          training=True,
          step=optimizer.iterations)
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      meta_training_loss = loss[0] / loss[1]
      shared_gradients = tape.gradient(meta_training_loss, shared_variables)
      gradients = adap_gradients + shared_gradients
      gradient_accumulator(gradients)
      num_word_examples = tf.reduce_sum(meta_test_target["length"]) + tf.reduce_sum(meta_train_target["length"])
    
    return meta_training_loss, training_loss, num_word_examples

  def _apply_gradients():
    variables = model.trainable_variables
    adap_variables = []
    shared_variables = []
    for v in variables:
      if "ADAP_" in v.name or "ldr_embedding" in v.name or "ldr_inputter" in v.name:
        adap_variables.append(v)
      else:
        shared_variables.append(v)
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, adap_variables+shared_variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_meta_loss, per_replica_loss, per_replica_num_word_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      meta_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_meta_loss, None)
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_word_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_word_examples, None)    
    return meta_loss, loss, num_word_examples
    
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  
  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  meta_train_data_flow = iter(_meta_train_forward())
  _loss = []
  _meta_loss = []  
  _num_word_examples = []
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        meta_loss, loss, num_word_examples = next(meta_train_data_flow)  
        _loss.append(loss)
        _meta_loss.append(meta_loss)
        _num_word_examples.append(num_word_examples)
      _step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Meta_loss = %f; num_word_examples = %d; after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_meta_loss), np.sum(_num_word_examples), elapsed)
        _loss = []
        _meta_loss = []
        _num_word_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def meta_train_v3(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    """
    for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
      checkpoint_path = checkpoint_manager.latest_checkpoint
      output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
      score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
    """
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_multi_domain_meta_training_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length)
  #####
  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target): 
     
    with tf.GradientTape(persistent=True) as tape:
      outputs, _ = model(
          meta_train_source,
          labels=meta_train_target,
          training=True,
          step=optimizer.iterations)    
      loss = model.compute_loss(outputs, meta_train_target, training=True)
      training_loss = loss[0] / loss[1]
      variables = model.trainable_variables       
      args_dict = dict()
      for v in variables:
        args_dict.update({v.name:v})
      adap_variables = []
      shared_variables = []
      for v in variables:
        if "ADAP_" in v.name or "ldr_embedding" in v.name or "ldr_inputter" in v.name:
          adap_variables.append(v)
        else:
          shared_variables.append(v)
      ##### Inner adaptation
      training_loss = model.regularize_loss(training_loss, variables=adap_variables)
      gradients = tape.gradient(training_loss, adap_variables)    
      meta_train_lr = config.get("meta_train_lr", 0.1)
      print("meta_train_lr: ", meta_train_lr)

      if config.get("stopping_gradient",True):
        print("apply stopping_gradient")
        for g, v in zip(gradients, adap_variables):      
          args_dict.update({v.name: v-meta_train_lr*tf.stop_gradient(g)})
      else:
        print("passing gradient")
        for g, v in zip(gradients, adap_variables):
          args_dict.update({v.name: update(v,g,lr=meta_train_lr)})
      #### Meta_loss:
      outputs, _ = model.forward_fn(meta_test_source,
          args_dict,
          labels=meta_test_target,
          training=True,
          step=optimizer.iterations)
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      meta_training_loss = loss[0] / loss[1]
      meta_training_loss = model.regularize_loss(meta_training_loss, variables=variables)
      gradients = tape.gradient(meta_training_loss, variables)
      gradient_accumulator(gradients)
      num_word_examples = tf.reduce_sum(meta_test_target["length"])
    
    return meta_training_loss, training_loss, num_word_examples

  def _apply_gradients():
    variables = model.trainable_variables      
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_meta_loss, per_replica_loss, per_replica_num_word_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      meta_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_meta_loss, None)
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_word_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_word_examples, None)    
    return meta_loss, loss, num_word_examples
    
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  
  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  meta_train_data_flow = iter(_meta_train_forward())
  _loss = []
  _meta_loss = []  
  _num_word_examples = []
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        meta_loss, loss, num_word_examples = next(meta_train_data_flow)  
        _loss.append(loss)
        _meta_loss.append(meta_loss)
        _num_word_examples.append(num_word_examples)
      _step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Meta_loss = %f; num_word_examples = %d; after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_meta_loss), np.sum(_num_word_examples), elapsed)
        _loss = []
        _meta_loss = []
        _num_word_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def meta_train_v5(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    """
    for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
      checkpoint_path = checkpoint_manager.latest_checkpoint
      output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
      score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
    """
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_multi_domain_meta_training_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length)
  #####
  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target): 
     
    with tf.GradientTape(persistent=True) as tape:
      outputs, _ = model(
          meta_train_source,
          labels=meta_train_target,
          training=True,
          step=optimizer.iterations)    
      loss = model.compute_loss(outputs, meta_train_target, training=True)
      training_loss = loss[0] / loss[1]
      variables = model.trainable_variables       
      args_dict = dict()
      for v in variables:
        args_dict.update({v.name:v})
      adap_variables = []
      shared_variables = []
      for v in variables:
        if "ADAP_" in v.name or "ldr_embedding" in v.name or "ldr_inputter" in v.name:
          adap_variables.append(v)
        else:
          shared_variables.append(v)
      ##### Inner adaptation
      training_loss = model.regularize_loss(training_loss, variables=shared_variables)
      gradients = tape.gradient(training_loss, shared_variables)    
      meta_train_lr = config.get("meta_train_lr",1.0)
      print("meta_train_lr: ", meta_train_lr)
      def update(v,g,lr=1.0):
        if isinstance(g, tf.IndexedSlices):
          return tf.tensor_scatter_nd_sub(v/lr,tf.expand_dims(g.indices,1),g.values)*lr
        else:
          return v-lr*g
      if config.get("stopping_gradient",True):
        print("apply stopping_gradient")
        for g, v in zip(gradients, shared_variables):      
          args_dict.update({v.name: v-meta_train_lr*tf.stop_gradient(g)})
      else:
        print("passing gradient")
        for g, v in zip(gradients, shared_variables):
          args_dict.update({v.name: update(v,g,lr=meta_train_lr)})
      #### Meta_loss:
      outputs, _ = model.forward_fn(meta_test_source,
          args_dict,
          labels=meta_test_target,
          training=True,
          step=optimizer.iterations)
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      meta_training_loss = loss[0] / loss[1]
      meta_training_loss = model.regularize_loss(meta_training_loss, variables=variables)
      gradients = tape.gradient(meta_training_loss, variables)
      gradient_accumulator(gradients)
      num_word_examples = tf.reduce_sum(meta_test_target["length"])
    
    return meta_training_loss, training_loss, num_word_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_meta_loss, per_replica_loss, per_replica_num_word_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      meta_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_meta_loss, None)
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_word_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_word_examples, None)    
    return meta_loss, loss, num_word_examples
    
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  
  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  meta_train_data_flow = iter(_meta_train_forward())
  _loss = []
  _meta_loss = []  
  _num_word_examples = []
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        meta_loss, loss, num_word_examples = next(meta_train_data_flow)  
        _loss.append(loss)
        _meta_loss.append(meta_loss)
        _num_word_examples.append(num_word_examples)
      _step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Meta_loss = %f; num_word_examples = %d; after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_meta_loss), np.sum(_num_word_examples), elapsed)
        _loss = []
        _meta_loss = []
        _num_word_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def meta_train_v6(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_multi_domain_meta_training_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length)
  #####
  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target): 
     
    with tf.GradientTape(persistent=True) as tape:
      ##### Inner adaptation
      outputs, _ = model(
          meta_train_source,
          labels=meta_train_target,
          training=True,
          step=optimizer.iterations)    
      loss = model.compute_loss(outputs, meta_train_target, training=True)
      training_loss = loss[0] / loss[1]
      variables = model.trainable_variables       
      args_dict = dict()
      for v in variables:
        args_dict.update({v.name:v})
      adap_variables = []
      shared_variables = []
      for v in variables:
        if "ADAP_" in v.name or "ldr_embedding" in v.name or "ldr_inputter" in v.name:
          adap_variables.append(v)
        else:
          shared_variables.append(v)
      training_loss = model.regularize_loss(training_loss, variables=shared_variables)
      gradients = tape.gradient(training_loss, shared_variables)    
      meta_train_lr = config.get("meta_train_lr",1.0)
      print("meta_train_lr: ", meta_train_lr)
      def update(v,g,lr=1.0):
        if isinstance(g, tf.IndexedSlices):
          return tf.tensor_scatter_nd_sub(v/lr,tf.expand_dims(g.indices,1),g.values)*lr
        else:
          return v-lr*g
      if config.get("stopping_gradient",True):
        print("apply stopping_gradient")
        for g, v in zip(gradients, shared_variables):      
          args_dict.update({v.name: v-meta_train_lr*tf.stop_gradient(g)})
      else:
        print("passing gradient")
        for g, v in zip(gradients, shared_variables):
          args_dict.update({v.name: update(v,g,lr=meta_train_lr)})
      #### Meta_loss:
        #### update adap parameters first
      outputs, _ = model(
          meta_test_source,
          labels=meta_test_target,
          training=True,
          step=optimizer.iterations)    
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      training_loss = loss[0] / loss[1]
      gradients = tape.gradient(training_loss, adap_variables)
        #### meta gradient for shared parameters
      outputs, _ = model.forward_fn(meta_test_source,
          args_dict,
          labels=meta_test_target,
          training=True,
          step=optimizer.iterations)
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      meta_training_loss = loss[0] / loss[1]
      meta_training_loss = model.regularize_loss(meta_training_loss, variables=shared_variables)
      gradients.extend(tape.gradient(meta_training_loss, shared_variables))
      gradient_accumulator(gradients)
      num_word_examples = tf.reduce_sum(meta_test_target["length"])
    
    return meta_training_loss, training_loss, num_word_examples

  def _apply_gradients():
    variables = model.trainable_variables
    adap_variables = []
    shared_variables = []
    for v in variables:
      if "ADAP_" in v.name or "ldr_embedding" in v.name or "ldr_inputter" in v.name:
        adap_variables.append(v)
      else:
        shared_variables.append(v)
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, adap_variables+shared_variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_meta_loss, per_replica_loss, per_replica_num_word_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      meta_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_meta_loss, None)
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_word_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_word_examples, None)    
    return meta_loss, loss, num_word_examples
    
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  
  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  meta_train_data_flow = iter(_meta_train_forward())
  _loss = []
  _meta_loss = []  
  _num_word_examples = []
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        meta_loss, loss, num_word_examples = next(meta_train_data_flow)  
        _loss.append(loss)
        _meta_loss.append(meta_loss)
        _num_word_examples.append(num_word_examples)
      _step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Meta_loss = %f; num_word_examples = %d; after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_meta_loss), np.sum(_num_word_examples), elapsed)
        _loss = []
        _meta_loss = []
        _num_word_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def train(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          adapter_optimizer=None,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=1): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  else:
    if checkpoint_path is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
      checkpoint.restore(checkpoint_path)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  ###### early stopping criterion
  current_max_eval_bleu = 0.0
  descending_streak = 0
  ######
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  
  print("There are %d in-domain corpora"%len(source_file))
  classification_loss_sign = tf.Variable(0.0,trainable=False)
  
  if experiment=="residualv28":
    prob_file = config["prob"]
    train_dataset = create_training_dataset_with_dprob(strategy, model, source_file, target_file, prob_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True),picking_prob=config.get("picking_prob",None))
  else:
    train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))
  from utils.dataprocess import count_lines
  datasets_size = [count_lines(src) for src in source_file]
  importance_weights = [data_size/sum(datasets_size) for data_size in datasets_size]
  temperature=config.get("temperature",1.0)
  importance_weights = [w ** temperature for w in importance_weights]
  importance_weights = [w/sum(importance_weights) for w in importance_weights]
  importance_weights = tf.constant(importance_weights)
  tf.print("importance_weights: ", importance_weights)
  #####
  with strategy.scope():
    classifier_optimizer = tfa.optimizers.LazyAdam(0.001)
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
    model_gradient_accumulator = optimizer_util.GradientAccumulator()
    classifier_gradient_accumulator = optimizer_util.GradientAccumulator()

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    if config.get("multi_domain", True):
      _domain = source["domain"][0]
    else:
      _domain = 0
    if config.get("apply_importance_weight", False):
      print("apply_importance_weight")
      training_loss = training_loss * importance_weights[_domain]
    
    if config.get("ADAP_activity_regularizing",False):
      if experiment=="residualv28":
        layer_activity_regularization_losses = []
        output_activity_regularization_losses = []
        regularization_losses = model.losses
        for loss_ in regularization_losses:
          if "multi_adap__dense" in loss_.name:
            output_activity_regularization_losses.append(loss_)
          else:
            layer_activity_regularization_losses.append(loss_)
        layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
        if len(layer_activity_regularization_losses)>0:
          print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
          training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)
      else:
        layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
        output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
        d_classification_gate_loss_scale = config.get("d_classification_gate_loss_scale",0.01)
        d_classifier_activity_regularization_loss_scale = config.get("d_classifier_activity_regularization_loss_scale",1.0)
        d_classifier_weight_regularization_losses_scale = config.get("d_classifier_weight_regularization_losses_scale",1.0)
        print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
        print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
        print("d_classification_gate_loss_scale: ", d_classification_gate_loss_scale)
        print("d_classifier_weight_regularization_losses_scale: ", d_classifier_weight_regularization_losses_scale)
        if isinstance(layer_activity_regularization_loss_scale, list):
          domain = source["domain"][0]
          layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
          layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
          #tf.print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale, "domain: ", domain)
        if isinstance(output_activity_regularization_loss_scale, list):
          domain = source["domain"][0]
          output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
          output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
        regularization_losses = model.losses
        print("model_name_scope", model.name_scope())
        print(regularization_losses)
        layer_activity_regularization_losses = []
        output_activity_regularization_losses = []
        d_classification_gate_losses = []
        d_classifier_activity_regularization_losses = []
        d_classifier_weight_regularization_losses = []
        for loss_ in regularization_losses:
          if "multi_adap__dense" in loss_.name:
            output_activity_regularization_losses.append(loss_)
          elif "ADAP_gate" in loss_.name: #and "ActivityRegularizer" not in loss_.name and "Regularizer" not in loss_.name
            if "ActivityRegularizer" in loss_.name:
              d_classifier_activity_regularization_losses.append(loss_)
            elif "Regularizer" in loss_.name:
              d_classifier_weight_regularization_losses.append(loss_)
            else:
              d_classification_gate_losses.append(loss_)
          elif "ADAP_" in loss_.name:
            layer_activity_regularization_losses.append(loss_)

        print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
        print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
        print("There are %d adaptation regularization loss on domain classification gate_____"%len(d_classification_gate_losses))
        print("There are %d d_classifier_activity_regularization_losses"%len(d_classifier_activity_regularization_losses))
        print("There are %d d_classifier_weight_regularization_losses"%len(d_classifier_weight_regularization_losses))
        if (len(layer_activity_regularization_losses)>0) and layer_activity_regularization_loss_scale>0:
          training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)

        if len(output_activity_regularization_losses)>0 and output_activity_regularization_loss_scale>0:
          training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)

        if len(d_classification_gate_losses)>0 and d_classification_gate_loss_scale>0:
          training_loss += d_classification_gate_loss_scale * tf.add_n(d_classification_gate_losses) / importance_weights[domain]

        if len(d_classifier_activity_regularization_losses)>0 and d_classifier_activity_regularization_loss_scale>0:
          training_loss += d_classifier_activity_regularization_loss_scale * tf.add_n(d_classifier_activity_regularization_losses)

        if len(d_classifier_weight_regularization_losses)>0 and d_classifier_weight_regularization_losses_scale>0:
          training_loss += d_classifier_weight_regularization_losses_scale * tf.add_n(d_classifier_weight_regularization_losses)
        
    if config.get("ADAP_weight_decay",False):
      lambda_ = config.get("lambda",0.00001)
      print("ADAP_weight_decay: ", lambda_)
      for v in model.trainable_variables:
        if "ADAP_" in v.name and "layer_norm" in v.name:
          training_loss += tf.reduce_sum(tf.square(v)) * lambda_

    variables = model.trainable_variables
    print("var numb: ", len(variables))
    """ for var in variables:
      print(var.name) """
    model_vars = []
    classifier_vars = []
    for var in variables:
      if "ADAP_gate/dense" in var.name:
        classifier_vars.append(var)
      else:
        model_vars.append(var)
    variables = model_vars + classifier_vars
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples, _domain

  def _accumulate_model_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    domain = source["domain"][0]
    if config.get("apply_importance_weight", False):
      print("apply_importance_weight")
      training_loss = training_loss * importance_weights[domain]
    if config.get("ADAP_activity_regularizing",False):
        layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
        output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
        d_classification_gate_loss_scale = config.get("d_classification_gate_loss_scale",0.01)
        d_classifier_activity_regularization_loss_scale = config.get("d_classifier_activity_regularization_loss_scale",1.0)
        d_classifier_weight_regularization_losses_scale = config.get("d_classifier_weight_regularization_losses_scale",1.0)
        print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
        print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
        print("d_classification_gate_loss_scale: ", d_classification_gate_loss_scale)
        print("d_classifier_weight_regularization_losses_scale: ", d_classifier_weight_regularization_losses_scale)
        if isinstance(layer_activity_regularization_loss_scale, list):
          domain = source["domain"][0]
          layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
          layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        if isinstance(output_activity_regularization_loss_scale, list):
          domain = source["domain"][0]
          output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
          output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
        regularization_losses = model.losses
        print("model_name_scope", model.name_scope())
        print(regularization_losses)
        layer_activity_regularization_losses = []
        output_activity_regularization_losses = []
        d_classification_gate_losses = []
        d_classifier_activity_regularization_losses = []
        d_classifier_weight_regularization_losses = []
        for loss_ in regularization_losses:
          if "multi_adap__dense" in loss_.name:
            output_activity_regularization_losses.append(loss_)
          elif "ADAP_gate" in loss_.name: #and "ActivityRegularizer" not in loss_.name and "Regularizer" not in loss_.name
            if "ActivityRegularizer" in loss_.name:
              d_classifier_activity_regularization_losses.append(loss_)
            elif "Regularizer" in loss_.name:
              d_classifier_weight_regularization_losses.append(loss_)
            else:
              d_classification_gate_losses.append(loss_)
          elif "ADAP_" in loss_.name:
            layer_activity_regularization_losses.append(loss_)

        print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
        print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
        print("There are %d adaptation regularization loss on domain classification gate_____"%len(d_classification_gate_losses))
        print("There are %d d_classifier_activity_regularization_losses"%len(d_classifier_activity_regularization_losses))
        print("There are %d d_classifier_weight_regularization_losses"%len(d_classifier_weight_regularization_losses))
        if (len(layer_activity_regularization_losses)>0) and layer_activity_regularization_loss_scale>0:
          training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)

        if len(output_activity_regularization_losses)>0 and output_activity_regularization_loss_scale>0:
          training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)

        if len(d_classification_gate_losses)>0 and d_classification_gate_loss_scale>0:
          classification_loss = d_classification_gate_loss_scale * tf.add_n(d_classification_gate_losses) / importance_weights[domain]
          training_loss += classification_loss * classification_loss_sign

        if len(d_classifier_activity_regularization_losses)>0 and d_classifier_activity_regularization_loss_scale>0:
          training_loss += d_classifier_activity_regularization_loss_scale * tf.add_n(d_classifier_activity_regularization_losses)

        if len(d_classifier_weight_regularization_losses)>0 and d_classifier_weight_regularization_losses_scale>0:
          training_loss += d_classifier_weight_regularization_losses_scale * tf.add_n(d_classifier_weight_regularization_losses)    
        

    variables = model.trainable_variables
    print("var numb: ", len(variables))
    model_vars = []
    classifier_vars = []
    for var in variables:
      if "ADAP_gate/dense" in var.name:
        classifier_vars.append(var)
      else:
        model_vars.append(var)
    variables = model_vars + classifier_vars
    model_gradients = optimizer.get_gradients(training_loss, model_vars)
    model_gradient_accumulator(model_gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

  def _accumulate_classifier_gradients(source, target):
    _, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    domain = source["domain"][0]    
    regularization_losses = model.losses
    d_classification_gate_losses = []
    for loss_ in regularization_losses:
      if "multi_adap__dense" in loss_.name:
        continue
      elif "ADAP_gate" in loss_.name: #and "ActivityRegularizer" not in loss_.name and "Regularizer" not in loss_.name
        if "ActivityRegularizer" in loss_.name:
          continue
        elif "Regularizer" in loss_.name:
          continue
        else:
          d_classification_gate_losses.append(loss_)
    training_loss = tf.add_n(d_classification_gate_losses) / importance_weights[domain]
    reported_loss = training_loss
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    model_vars = []
    classifier_vars = []
    for var in variables:
      if "ADAP_gate/dense" in var.name:
        classifier_vars.append(var)
      else:
        model_vars.append(var)
    classifier_gradients = classifier_optimizer.get_gradients(training_loss, classifier_vars)
    classifier_gradient_accumulator(classifier_gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples
     
  def _apply_gradients():
    variables = model.trainable_variables
    model_vars = []
    classifier_vars = []
    for var in variables:
      if "ADAP_gate/dense" in var.name:
        classifier_vars.append(var)
      else:
        model_vars.append(var)
    variables = model_vars + classifier_vars
    grads_and_vars = []

    tf.print("gradient_accumulator.step: ",gradient_accumulator.step)
    tf.print("strategy.num_replicas_in_sync: ",strategy.num_replicas_in_sync)
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  def _apply_model_gradients():
    variables = model.trainable_variables
    model_vars = []
    classifier_vars = []
    for var in variables:
      if "ADAP_gate/dense" in var.name:
        classifier_vars.append(var)
      else:
        model_vars.append(var)
    variables = model_vars + classifier_vars
    grads_and_vars = []
    for gradient, variable in zip(model_gradient_accumulator.gradients, model_vars):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    model_gradient_accumulator.reset()

  def _apply_classifier_gradients():
    variables = model.trainable_variables
    model_vars = []
    classifier_vars = []
    for var in variables:
      if "ADAP_gate/dense" in var.name:
        classifier_vars.append(var)
      else:
        model_vars.append(var)
    variables = model_vars + classifier_vars
    grads_and_vars = []
    for gradient, variable in zip(classifier_gradient_accumulator.gradients, classifier_vars):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    classifier_optimizer.apply_gradients(grads_and_vars)
    classifier_gradient_accumulator.reset()

  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples, per_replica_domain = strategy.run(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
      _domain = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_domain, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, _domain, num_examples

  @dataset_util.function_on_next(train_dataset)
  def _train_model_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.run(
          _accumulate_model_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples

  @dataset_util.function_on_next(train_dataset)
  def _train_classifier_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.run(
          _accumulate_classifier_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples

  @dataset_util.function_on_next(train_dataset)
  def _train_iteration(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      return per_replica_source, per_replica_target
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.run(_apply_gradients)

  @tf.function
  def _model_step():
    with strategy.scope():
      strategy.run(_apply_model_gradients)

  @tf.function
  def _classifier_step():
    with strategy.scope():
      strategy.run(_apply_classifier_gradients)

  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  train_model_data_flow = iter(_train_model_forward())
  train_classifier_data_flow = iter(_train_classifier_forward())
  _, _, _ = next(train_data_flow)

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _per_domain_loss = []
  _per_domain_accum_loss = []
  for _ in domain:
    _per_domain_loss.append([])
    _per_domain_accum_loss.append([])

  _d_classfication_loss = []
  _number_examples = []
  step = optimizer.iterations.numpy()
  if config.get("reset_step",None):
    print("start from %d-th step"%config.get("reset_step",150000))
    optimizer.iterations.assign(config.get("reset_step",150000))
  if step <= 1:
    initializer = config.get("initializer","default")
    if initializer == "default":
      print("Initializing variables by tensorflow default")      
    elif initializer == "variance_scaling":
      print("Initializing variables by tf.variance_scaling")
      initial_value = []
      for v in model.trainable_variables:
        shape = tf.shape(v).numpy()
        initial_value.append(variance_scaling_initialier(shape, scale=1.0, mode="fan_avg", distribution="uniform"))
      weight_reset(initial_value)       

  if config.get("continual_learning", False):
    print("Continual Learning needs to load from old model")
    assert config.get("checkpoint_path") != None
    checkpoint_path = config.get("checkpoint_path")
    load_and_update_if_needed_from_ckpt(config["model_dir"],   
                        checkpoint_path,                        
                        trackables={"model":model},
                        vocab_update=True,
                        model_key="model")

  score_type = config.get("score_type","MultiBLEU")
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  ref_eval_concat = file_concatenate(config["eval_ref"],"ref_eval_concat",dir_name=os.path.join(config["model_dir"],"eval"))

  with _summary_writer.as_default():
    while True:
      #####Training batch
      
      if config.get("adv_step",None):          
        if step==config.get("adv_step",None):
          classification_loss_sign.assign(-1.0)
        for _ in range(2):
          d_classfication_loss, _ = next(train_classifier_data_flow)
          _d_classfication_loss.append(d_classfication_loss)
          _classifier_step()
        loss, num_examples = next(train_model_data_flow)    
        _loss.append(loss.numpy())
        _number_examples.append(num_examples)
        _model_step()
      else:
        for _ in range(int(config.get("accumulation_step",1))):
          loss, _domain, num_examples = next(train_data_flow)    
          _loss.append(loss.numpy())
          _number_examples.append(num_examples.numpy())
        _step()  
      step = optimizer.iterations.numpy()
      """ for i in range(len(domain)):
        if len(_per_domain_accum_loss[i])==report_every:
          #_per_domain_loss[i].append(np.mean(_per_domain_accum_loss[i]))
          tf.summary.experimental.set_step()
          tf.summary.scalar("loss_%d"%i, np.mean(_per_domain_accum_loss[i]), description="loss in domain %d"%i)
          _per_domain_accum_loss[i] = [] """

      if step % report_every == 0:
        elapsed = time.time() - start
        if config.get("adv_step",None):
          tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; classification_loss = %f, number_examples = %d, after %f seconds",
          step, learning_rate(step), np.mean(_loss), np.mean(_d_classfication_loss), np.sum(_number_examples), elapsed)
          _loss = []
          _d_classfication_loss = []
          _number_examples = []
          start = time.time()
        else:
          tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
          _loss = []
          _number_examples = []
          start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        output_files = []
        new_bleu = 0.0
        if experiment=="residualv28":
          for src, ref, prob, i in zip(config["eval_src"],config["eval_ref"],config["eval_prob"], config["eval_domain"]):
            output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
            score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), probs_file=prob, experiment=experiment)
            tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
        else:
          for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
            output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
            score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
            tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
            output_files.append(output_file)
        ##### BLEU on concat dev set.
        output_file_concat = file_concatenate(output_files,"output_file_concat.%s"%os.path.basename(checkpoint_path))
        score = scorer(ref_eval_concat, output_file_concat)
        print("score of model %s on concat dev set: "%checkpoint_manager.latest_checkpoint, score)
        new_bleu = score
        tf.summary.scalar("concat_eval_score", score, description="BLEU on concat dev set")
        #############################
        if new_bleu >= current_max_eval_bleu:
          current_max_eval_bleu = new_bleu
          descending_streak = 0
        else:
          descending_streak += 1
      if descending_streak >= 5:
        break
      tf.summary.flush()
      if step > train_steps:
        break
  
def train_v2(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=5000000,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  length_bucket_width = config.get("length_bucket_width",1)
  print("There are %d in-domain corpora"%len(source_file))

  train_dataset = create_training_dataset_v2(strategy, model, domain, source_file, target_file, 
                                              batch_train_size, batch_type, shuffle_buffer_size, maximum_length, 
                                              length_bucket_width, multi_domain=True)
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    
    if config.get("ADAP_activity_regularizing",False):
      layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
      output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
      print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
      print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
      if isinstance(layer_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
        layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
      if isinstance(output_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
        output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
      regularization_losses = model.losses
      print("model_name_scope", model.name_scope())
      print(regularization_losses)
      layer_activity_regularization_losses = []
      output_activity_regularization_losses = []
      for loss_ in regularization_losses:
        if "multi_adap__dense" in loss_.name:
          output_activity_regularization_losses.append(loss_)
        else:
          layer_activity_regularization_losses.append(loss_)
      print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
      print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
      if len(layer_activity_regularization_losses)>0:
        training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)
      if len(output_activity_regularization_losses)>0:
        training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.print("token_numb:____", num_examples, "domain:____", source["domain"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples

  @dataset_util.function_on_next(train_dataset)
  def _train_iteration(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      return per_replica_source, per_replica_target
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  _, _ = next(train_data_flow)
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  _loss = []  
  _number_examples = []
  step = optimizer.iterations.numpy()
  if step <= 1:
    initializer = config.get("initializer","default")
    if initializer == "default":
      print("Initializing variables by tensorflow default")      
    elif initializer == "variance_scaling":
      print("Initializing variables by tf.variance_scaling")
      initial_value = []
      for v in model.trainable_variables:
        shape = tf.shape(v).numpy()
        initial_value.append(variance_scaling_initialier(shape, scale=1.0, mode="fan_avg", distribution="uniform"))
      weight_reset(initial_value)       

  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        loss, num_examples = next(train_data_flow)    
        _loss.append(loss)
        _number_examples.append(num_examples)
      _step()  
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      tf.summary.flush()
      if step > train_steps:
        break
    
def meta_train_v8(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          meta_train_picking_prob=None,
          meta_test_picking_prob=None,
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_multi_domain_meta_training_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, 
                                                                        shuffle_buffer_size, maximum_length, meta_test_picking_prob=meta_test_picking_prob,
                                                                        meta_train_picking_prob=meta_train_picking_prob)
  #####
  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target): 
    #tf.print("meta_train_domain", meta_train_source["domain"][0], "meta_test_domain: ", meta_test_source["domain"][0], sep="|")
    with tf.GradientTape(persistent=True) as tape:
      ##### Inner adaptation
      outputs, _ = model(
          meta_train_source,
          labels=meta_train_target,
          training=True,
          step=optimizer.iterations)    
      loss = model.compute_loss(outputs, meta_train_target, training=True)
      training_loss = loss[0] / loss[1]
      if config.get("ADAP_activity_regularizing",False):
        layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
        output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
        print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
        print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
        if isinstance(layer_activity_regularization_loss_scale, list):
          domain = meta_train_source["domain"][0]
          layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
          layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        if isinstance(output_activity_regularization_loss_scale, list):
          domain = meta_train_source["domain"][0]
          output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
          output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
        regularization_losses = model.losses
        print("model_name_scope", model.name_scope())
        print(regularization_losses)
        layer_activity_regularization_losses = []
        output_activity_regularization_losses = []
        for loss_ in regularization_losses:
          if "multi_adap__dense" in loss_.name:
            output_activity_regularization_losses.append(loss_)
          else:
            layer_activity_regularization_losses.append(loss_)
        print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
        print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
        if len(layer_activity_regularization_losses)>0:
          training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)
        if len(output_activity_regularization_losses)>0:
          training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)
      variables = model.trainable_variables       
      args_dict = dict()
      for v in variables:
        args_dict.update({v.name:v})
      gradients = tape.gradient(training_loss, variables)  
      gradient_accumulator(gradients) 

      meta_train_lr = config.get("meta_train_lr",1.0)
      print("meta_train_lr: ", meta_train_lr)
      
      if config.get("stopping_gradient",True):
        print("apply stopping_gradient")
        for g, v in zip(gradients, variables):      
          args_dict.update({v.name: v-meta_train_lr*tf.stop_gradient(g)})
      else:
        print("passing gradient")
        for g, v in zip(gradients, variables):
          args_dict.update({v.name: update(v,g,lr=meta_train_lr)})
      
      #### Meta_loss:
        #### meta gradient for shared parameters
      outputs, _ = model.forward_fn(meta_test_source,
          args_dict,
          labels=meta_test_target,
          training=True,
          step=optimizer.iterations)
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      meta_training_loss = loss[0] / loss[1]
      gradients = tape.gradient(meta_training_loss, variables)
      gradient_accumulator(gradients)
      num_word_examples = tf.reduce_sum(meta_test_target["length"]) + tf.reduce_sum(meta_train_target["length"])
    
    return meta_training_loss, training_loss, num_word_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_meta_loss, per_replica_loss, per_replica_num_word_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      meta_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_meta_loss, None)
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_word_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_word_examples, None)    
    return meta_loss, loss, num_word_examples
    
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  
  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  meta_train_data_flow = iter(_meta_train_forward())
  _loss = []
  _meta_loss = []  
  _num_word_examples = []
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        meta_loss, loss, num_word_examples = next(meta_train_data_flow)  
        _loss.append(loss)
        _meta_loss.append(meta_loss)
        _num_word_examples.append(num_word_examples)
      _step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Meta_loss = %f; num_word_examples = %d; after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_meta_loss), np.sum(_num_word_examples), elapsed)
        _loss = []
        _meta_loss = []
        _num_word_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def meta_train_v15(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          picking_prob=None,
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_multi_domain_meta_training_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, 
                                                                        shuffle_buffer_size, maximum_length, picking_prob=picking_prob)
  #####
  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target): 
    #tf.print("meta_train_domain", meta_train_source["domain"][0], "meta_test_domain: ", meta_test_source["domain"][0], sep="|")
    with tf.GradientTape(persistent=True) as tape:
      ##### Inner adaptation
      outputs, _ = model(
          meta_train_source,
          labels=meta_train_target,
          training=True,
          step=optimizer.iterations)    
      loss = model.compute_loss(outputs, meta_train_target, training=True)
      training_loss = loss[0] / loss[1]
      if config.get("ADAP_activity_regularizing",False):
        layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
        output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
        print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
        print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
        if isinstance(layer_activity_regularization_loss_scale, list):
          domain = meta_train_source["domain"][0]
          layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
          layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        if isinstance(output_activity_regularization_loss_scale, list):
          domain = meta_train_source["domain"][0]
          output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
          output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
        regularization_losses = model.losses
        print("model_name_scope", model.name_scope())
        print(regularization_losses)
        layer_activity_regularization_losses = []
        output_activity_regularization_losses = []
        for loss_ in regularization_losses:
          if "multi_adap__dense" in loss_.name:
            output_activity_regularization_losses.append(loss_)
          else:
            layer_activity_regularization_losses.append(loss_)
        print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
        print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
        if len(layer_activity_regularization_losses)>0:
          training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)
        if len(output_activity_regularization_losses)>0:
          training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)
      variables = model.trainable_variables       
      args_dict = dict()
      for v in variables:
        args_dict.update({v.name:v})
      gradients = tape.gradient(training_loss, variables)  
      #gradient_accumulator(gradients) 

      meta_train_lr = config.get("meta_train_lr",1.0)
      print("meta_train_lr: ", meta_train_lr)
      
      if config.get("stopping_gradient",True):
        print("apply stopping_gradient")
        for g, v in zip(gradients, variables):      
          args_dict.update({v.name: v-meta_train_lr*tf.stop_gradient(g)})
      else:
        print("passing gradient")
        for g, v in zip(gradients, variables):
          args_dict.update({v.name: update(v,g,lr=meta_train_lr)})
      
      #### Meta_loss:
        #### meta gradient for shared parameters
      outputs, _ = model.forward_fn(meta_test_source,
          args_dict,
          labels=meta_test_target,
          training=True,
          step=optimizer.iterations)
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      meta_training_loss = loss[0] / loss[1]
      gradients = tape.gradient(meta_training_loss, variables)
      gradient_accumulator(gradients)
      num_word_examples = tf.reduce_sum(meta_test_target["length"]) + tf.reduce_sum(meta_train_target["length"])
    
    return meta_training_loss, training_loss, num_word_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_meta_loss, per_replica_loss, per_replica_num_word_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      meta_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_meta_loss, None)
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_word_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_word_examples, None)    
    return meta_loss, loss, num_word_examples
    
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  
  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  meta_train_data_flow = iter(_meta_train_forward())
  _loss = []
  _meta_loss = []  
  _num_word_examples = []
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        meta_loss, loss, num_word_examples = next(meta_train_data_flow)  
        _loss.append(loss)
        _meta_loss.append(meta_loss)
        _num_word_examples.append(num_word_examples)
      _step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Meta_loss = %f; num_word_examples = %d; after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_meta_loss), np.sum(_num_word_examples), elapsed)
        _loss = []
        _meta_loss = []
        _num_word_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def meta_train_v10(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_multi_domain_meta_training_dataset_v1(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length)
  #####
  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target): 
     
    with tf.GradientTape(persistent=True) as tape:
      ##### Inner adaptation
      outputs, _ = model(
          meta_train_source,
          labels=meta_train_target,
          training=True,
          step=optimizer.iterations)    
      loss = model.compute_loss(outputs, meta_train_target, training=True)
      training_loss = loss[0] / loss[1]
      if config.get("ADAP_activity_regularizing",False):
        layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
        output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
        print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
        print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
        if isinstance(layer_activity_regularization_loss_scale, list):
          domain = meta_train_source["domain"][0]
          layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
          layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        if isinstance(output_activity_regularization_loss_scale, list):
          domain = meta_train_source["domain"][0]
          output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
          output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
        regularization_losses = model.losses
        print("model_name_scope", model.name_scope())
        print(regularization_losses)
        layer_activity_regularization_losses = []
        output_activity_regularization_losses = []
        for loss_ in regularization_losses:
          if "multi_adap__dense" in loss_.name:
            output_activity_regularization_losses.append(loss_)
          else:
            layer_activity_regularization_losses.append(loss_)
        print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
        print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
        if len(layer_activity_regularization_losses)>0:
          training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)
        if len(output_activity_regularization_losses)>0:
          training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)
      variables = model.trainable_variables       
      args_dict = dict()
      for v in variables:
        args_dict.update({v.name:v})
      gradients = tape.gradient(training_loss, variables)  
      gradient_accumulator(gradients) 

      meta_train_lr = config.get("meta_train_lr",1.0)
      print("meta_train_lr: ", meta_train_lr)
      
      if config.get("stopping_gradient",True):
        print("apply stopping_gradient")
        for g, v in zip(gradients, variables):      
          args_dict.update({v.name: v-meta_train_lr*tf.stop_gradient(g)})
      else:
        print("passing gradient")
        for g, v in zip(gradients, variables):
          args_dict.update({v.name: update(v,g,lr=meta_train_lr)})
      
      #### Meta_loss:
        #### meta gradient for shared parameters
      outputs, _ = model.forward_fn(meta_test_source,
          args_dict,
          labels=meta_test_target,
          training=True,
          step=optimizer.iterations)
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      meta_training_loss = loss[0] / loss[1]
      gradients = tape.gradient(meta_training_loss, variables)
      gradient_accumulator(gradients)
      num_word_examples = tf.reduce_sum(meta_test_target["length"]) + tf.reduce_sum(meta_train_target["length"])
    
    return meta_training_loss, training_loss, num_word_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_meta_loss, per_replica_loss, per_replica_num_word_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      meta_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_meta_loss, None)
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_word_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_word_examples, None)    
    return meta_loss, loss, num_word_examples
    
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  
  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  meta_train_data_flow = iter(_meta_train_forward())
  _loss = []
  _meta_loss = []  
  _num_word_examples = []
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        meta_loss, loss, num_word_examples = next(meta_train_data_flow)  
        _loss.append(loss)
        _meta_loss.append(meta_loss)
        _num_word_examples.append(num_word_examples)
      _step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Meta_loss = %f; num_word_examples = %d; after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_meta_loss), np.sum(_num_word_examples), elapsed)
        _loss = []
        _meta_loss = []
        _num_word_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def meta_train_v11(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_multi_domain_meta_training_dataset_v1(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length)
  #####
  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target): 
     
    with tf.GradientTape(persistent=True) as tape:
      ##### Inner adaptation
      outputs, _ = model(
          meta_train_source,
          labels=meta_train_target,
          training=True,
          step=optimizer.iterations)    
      loss = model.compute_loss(outputs, meta_train_target, training=True)
      training_loss = loss[0] / loss[1]
      if config.get("ADAP_activity_regularizing",False):
        layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
        output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
        print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
        print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
        if isinstance(layer_activity_regularization_loss_scale, list):
          domain = meta_train_source["domain"][0]
          layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
          layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        if isinstance(output_activity_regularization_loss_scale, list):
          domain = meta_train_source["domain"][0]
          output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
          output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
        regularization_losses = model.losses
        print("model_name_scope", model.name_scope())
        print(regularization_losses)
        layer_activity_regularization_losses = []
        output_activity_regularization_losses = []
        for loss_ in regularization_losses:
          if "multi_adap__dense" in loss_.name:
            output_activity_regularization_losses.append(loss_)
          else:
            layer_activity_regularization_losses.append(loss_)
        print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
        print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
        if len(layer_activity_regularization_losses)>0:
          training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)
        if len(output_activity_regularization_losses)>0:
          training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)
      variables = model.trainable_variables       
      args_dict = dict()
      for v in variables:
        args_dict.update({v.name:v})
      gradients = tape.gradient(training_loss, variables)  
      gradient_accumulator(gradients) 
      
      meta_train_lr = learning_rate(optimizer.iterations)      
      if config.get("stopping_gradient",True):
        print("apply stopping_gradient")
        for g, v in zip(gradients, variables):      
          args_dict.update({v.name: v-meta_train_lr*tf.stop_gradient(g)})
      else:
        print("passing gradient")
        for g, v in zip(gradients, variables):
          args_dict.update({v.name: update(v,g,lr=meta_train_lr)})
      
      #### Meta_loss:
        #### meta gradient for shared parameters
      outputs, _ = model.forward_fn(meta_test_source,
          args_dict,
          labels=meta_test_target,
          training=True,
          step=optimizer.iterations)
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      meta_training_loss = loss[0] / loss[1]
      gradients = tape.gradient(meta_training_loss, variables)
      gradient_accumulator(gradients)
      num_word_examples = tf.reduce_sum(meta_test_target["length"]) + tf.reduce_sum(meta_train_target["length"])
    
    return meta_training_loss, training_loss, num_word_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_meta_loss, per_replica_loss, per_replica_num_word_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      meta_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_meta_loss, None)
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_word_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_word_examples, None)    
    return meta_loss, loss, num_word_examples
    
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  
  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  meta_train_data_flow = iter(_meta_train_forward())
  _loss = []
  _meta_loss = []  
  _num_word_examples = []
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        meta_loss, loss, num_word_examples = next(meta_train_data_flow)  
        _loss.append(loss)
        _meta_loss.append(meta_loss)
        _num_word_examples.append(num_word_examples)
      _step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Meta_loss = %f; num_word_examples = %d; after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_meta_loss), np.sum(_num_word_examples), elapsed)
        _loss = []
        _meta_loss = []
        _num_word_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def meta_train_v9(config,
          optimizer_1,
          optimizer_2,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  with strategy.scope():
    gradient_accumulator = optimizer_util.GradientAccumulator()  
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_multi_domain_meta_training_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length)
  #####
  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target): 
     
    with tf.GradientTape(persistent=True) as tape:
      ##### Inner adaptation
      outputs, _ = model(
          meta_train_source,
          labels=meta_train_target,
          training=True,
          step=optimizer_1.iterations)    
      loss = model.compute_loss(outputs, meta_train_target, training=True)
      training_loss = loss[0] / loss[1]
      if config.get("ADAP_activity_regularizing",False):
        layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
        output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
        print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
        print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
        if isinstance(layer_activity_regularization_loss_scale, list):
          domain = meta_train_source["domain"][0]
          layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
          layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        if isinstance(output_activity_regularization_loss_scale, list):
          domain = meta_train_source["domain"][0]
          output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
          output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
        regularization_losses = model.losses
        print("model_name_scope", model.name_scope())
        print(regularization_losses)
        layer_activity_regularization_losses = []
        output_activity_regularization_losses = []
        for loss_ in regularization_losses:
          if "multi_adap__dense" in loss_.name:
            output_activity_regularization_losses.append(loss_)
          else:
            layer_activity_regularization_losses.append(loss_)
        print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
        print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
        if len(layer_activity_regularization_losses)>0:
          training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)
        if len(output_activity_regularization_losses)>0:
          training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)
      variables = model.trainable_variables    
      args_dict = dict()
      for v in variables:
        args_dict.update({v.name:v})
      gradients = tape.gradient(training_loss, variables)  
      gradient_accumulator(gradients) 

      meta_train_lr = config.get("meta_train_lr",1.0)
      print("meta_train_lr: ", meta_train_lr)
      meta_train_lr = learning_rate(optimizer_1.iterations)
      if config.get("stopping_gradient",True):
        print("apply stopping_gradient")
        for g, v in zip(gradients, variables):      
          args_dict.update({v.name: v-meta_train_lr*tf.stop_gradient(g)})
      else:
        print("passing gradient")
        for g, v in zip(gradients, variables):
          args_dict.update({v.name: update(v,g,lr=meta_train_lr)})
      
      #### Meta_loss:
        #### meta gradient for shared parameters
      outputs, _ = model.forward_fn(meta_test_source,
          args_dict,
          labels=meta_test_target,
          training=True,
          step=optimizer_1.iterations)
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      meta_training_loss = loss[0] / loss[1]
      gradients = tape.gradient(meta_training_loss, variables)
      gradient_accumulator(gradients)
      num_word_examples = tf.reduce_sum(meta_test_target["length"]) + tf.reduce_sum(meta_train_target["length"])
    
    return meta_training_loss, training_loss, num_word_examples

  def _apply_gradients():
    variables = model.trainable_variables    
    grads_and_vars = []
    shared_grads_and_vars = []
    adap_grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      if "ADAP" in variable.name:
        adap_grads_and_vars.append((scaled_gradient, variable))
      else:
        shared_grads_and_vars.append((scaled_gradient, variable))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer_1.apply_gradients(shared_grads_and_vars)
    optimizer_2.apply_gradients(adap_grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_meta_loss, per_replica_loss, per_replica_num_word_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      meta_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_meta_loss, None)
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_word_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_word_examples, None)    
    return meta_loss, loss, num_word_examples
    
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  
  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  meta_train_data_flow = iter(_meta_train_forward())
  _loss = []
  _meta_loss = []  
  _num_word_examples = []
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        meta_loss, loss, num_word_examples = next(meta_train_data_flow)  
        _loss.append(loss)
        _meta_loss.append(meta_loss)
        _num_word_examples.append(num_word_examples)
      _step()
      step = optimizer_1.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Meta_loss = %f; num_word_examples = %d; after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_meta_loss), np.sum(_num_word_examples), elapsed)
        _loss = []
        _meta_loss = []
        _num_word_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def model_inspect(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          checkpoint_path=None,
          maximum_length=6,
          batch_size = 1,
          batch_type = "examples",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  
  #####
  
  if checkpoint_path is not None:
    checkpoint.restore(checkpoint_path).expect_partial()
    tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
  elif checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  else:
    exit()
  
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = batch_size
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))
  print("batch_size", batch_size)


  train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_train_size, batch_type, shuffle_buffer_size, maximum_length, multi_domain=False)
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)

  def _build_model(source, target):
    _, _ = model(
        source,
        labels=target,
        training=False,
        step=optimizer.iterations)
      
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      strategy.experimental_run_v2(
          _build_model, args=(per_replica_source, per_replica_target))
  
  # Runs the training loop.
  #train_data_flow = iter(_train_forward())
  #next(train_data_flow)
  """
  load_and_update_if_needed_from_ckpt(config["model_dir"],   
                        checkpoint_path,                        
                        trackables={"model":model},
                        vocab_update=False,
                        model_key="model")
  """
  #checkpoint_manager.save(checkpoint_number=0)
  
  """ with np.printoptions(threshold=np.inf):
    for v in model.trainable_variables:
      if "layer_norm" in v.name:
        print(v.name)
        #print(v.numpy())
        matrix = v.numpy()
        m = v.numpy().shape[0]
        dist = np.zeros((m,m))
        for i in range(m):
          for j in range(m):
            dist[i,j] = np.sum((matrix[i,:] - matrix[j,:]) * (matrix[i,:] - matrix[j,:]))
        print(dist) """

  # with np.printoptions(threshold=np.inf):
  #   for v in model.trainable_variables:
  #     if "latent_group_allocation_logit" in v.name:
  #       print(v.name)
  #       print(v.numpy())


  size = 0
  for v in model.trainable_variables:
    size += v.numpy().size
  print("total number of parameters: %d"%size)
  
  
  topK = config.get("domain_group_allocation_num")
  domain_dropout_masks = []
  vector_masks = []
  for domain in range(config.get("num_inspected_domains",8)):
    domain_dropout_mask = []
    for i in range(model.encoder.num_layers+model.decoder.num_layers+1):
      topK_ = tf.math.top_k(tf.nn.embedding_lookup(model.latent_group_allocation_logit_per_layer[i],domain),k=topK).indices.numpy()
      group_allocation = np.zeros(model.num_domain_unit_group)
      for j in topK_:
        group_allocation[j] = 1

      #tf.print("group_allocation:",group_allocation,"domain:",domain,"layer:",i,summarize=-1)

      #group_allocation = tf.repeat(tf.Variable(group_allocation,dtype=tf.float32),model.unit_group_size)

      #domain_dropout_mask.append(tf.concat([tf.ones(model.num_shared_units),group_allocation],-1))
      domain_dropout_mask.append(tf.Variable(group_allocation,dtype=tf.float32))
    domain_dropout_masks.append(domain_dropout_mask)
    vector_masks.append(tf.concat(domain_dropout_mask,0).numpy())
  acc_similarity_matrix = np.zeros((config.get("num_inspected_domains",8),config.get("num_inspected_domains",8)))

  m = np.array(vector_masks)
  np.savetxt("mask.csv", m, delimiter="\t")
  p = 0
  for layer in range(model.encoder.num_layers+model.decoder.num_layers+1):
    similarity_matrix = np.zeros((config.get("num_inspected_domains",8),config.get("num_inspected_domains",8)))
    for i in range(config.get("num_inspected_domains",8)):
      for j in range(config.get("num_inspected_domains",8)):
        m_i = domain_dropout_masks[i][layer]
        m_j = domain_dropout_masks[j][layer]
        #print("(%d,%d)"%(i,j),m_i * m_j)
        similarity_matrix[i,j]= tf.reduce_sum(m_i * m_j,0) / config.get("domain_group_allocation_num")
    m = tf.zeros_like(domain_dropout_masks[0][layer])
    for i in range(config.get("num_inspected_domains",8)):
      m_i = domain_dropout_masks[i][layer]
      m = m + m_i - m * m_i
    #print(m)
    p += tf.reduce_sum(m) / config.get("domain_group_allocation_num")
    acc_similarity_matrix += similarity_matrix
    print(similarity_matrix)
  print(acc_similarity_matrix/(model.encoder.num_layers+model.decoder.num_layers+1))
  print(p/(model.encoder.num_layers+model.decoder.num_layers+1))
  print(np.mean(acc_similarity_matrix/(model.encoder.num_layers+model.decoder.num_layers+1)))
  np.savetxt("similarity.csv", acc_similarity_matrix/(model.encoder.num_layers+model.decoder.num_layers+1), delimiter="\t")
  x_axis_labels = ["%d"%i for i in range(config.get("num_inspected_domains",8))]
  map_prod(x_axis_labels, x_axis_labels,acc_similarity_matrix/(model.encoder.num_layers+model.decoder.num_layers+1), "similarity.png")

  """
  checkpoint_path = checkpoint_manager.latest_checkpoint
  for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
    
    output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
    score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
    tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
  """
  # phi = [np.zeros(16)]
  # for i in range(1,8):
  #   number = i * 5000
  #   checkpoint.restore(os.path.join(config["model_dir"],"ckpt-%d"%number))
  #   phi.append(model.latent_group_allocation_logit_per_layer[4].numpy()[0,:])
  # print(np.asarray(phi))
  # np.savetxt("phi.csv",np.asarray(phi), delimiter="\t")

def src_wemb_pretrain(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_train_size, batch_type, shuffle_buffer_size, maximum_length)
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    training_loss = model.regularize_loss(training_loss, variables=variables)
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

def train_v3(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  train_dataset = create_training_dataset_v1(strategy, model, domain, source_file, target_file, 
                                                                        batch_train_size, batch_type, shuffle_buffer_size, maximum_length, multi_domain=(config["experiment"]!="baseline"))
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    
    if config.get("ADAP_activity_regularizing",False):
      layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
      output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
      print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
      print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
      if isinstance(layer_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
        layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
      if isinstance(output_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
        output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
      regularization_losses = model.losses
      print("model_name_scope", model.name_scope())
      print(regularization_losses)
      layer_activity_regularization_losses = []
      output_activity_regularization_losses = []
      for loss_ in regularization_losses:
        if "multi_adap__dense" in loss_.name:
          output_activity_regularization_losses.append(loss_)
        else:
          layer_activity_regularization_losses.append(loss_)
      print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
      print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
      if len(layer_activity_regularization_losses)>0:
        training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)
      if len(output_activity_regularization_losses)>0:
        training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples

  @dataset_util.function_on_next(train_dataset)
  def _train_iteration(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      return per_replica_source, per_replica_target
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  _, _ = next(train_data_flow)
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  _loss = []  
  _number_examples = []
  step = optimizer.iterations.numpy()
  if step <= 1:
    initializer = config.get("initializer","default")
    if initializer == "default":
      print("Initializing variables by tensorflow default")      
    elif initializer == "variance_scaling":
      print("Initializing variables by tf.variance_scaling")
      initial_value = []
      for v in model.trainable_variables:
        shape = tf.shape(v).numpy()
        initial_value.append(variance_scaling_initialier(shape, scale=1.0, mode="fan_avg", distribution="uniform"))
      weight_reset(initial_value)       

  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        loss, num_examples = next(train_data_flow)    
        _loss.append(loss)
        _number_examples.append(num_examples)
      _step()  
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      tf.summary.flush()
      if step > train_steps:
        break

def train_v8(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          picking_prob=None,
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_multi_domain_meta_training_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, 
                                                                        shuffle_buffer_size, maximum_length, picking_prob=picking_prob)
  #####
  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target):     
    ##### Inner adaptation
    outputs, _ = model(
        meta_train_source,
        labels=meta_train_target,
        training=True,
        step=optimizer.iterations)    
    loss = model.compute_loss(outputs, meta_train_target, training=True)
    training_loss = loss[0] / loss[1]
    if config.get("ADAP_activity_regularizing",False):
      layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
      output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
      print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
      print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
      if isinstance(layer_activity_regularization_loss_scale, list):
        domain = meta_train_source["domain"][0]
        layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
        layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
      if isinstance(output_activity_regularization_loss_scale, list):
        domain = meta_train_source["domain"][0]
        output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
        output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
      regularization_losses = model.losses
      print("model_name_scope", model.name_scope())
      print(regularization_losses)
      layer_activity_regularization_losses = []
      output_activity_regularization_losses = []
      for loss_ in regularization_losses:
        if "multi_adap__dense" in loss_.name:
          output_activity_regularization_losses.append(loss_)
        else:
          layer_activity_regularization_losses.append(loss_)
      print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
      print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
      if len(layer_activity_regularization_losses)>0:
        training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)
      if len(output_activity_regularization_losses)>0:
        training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)
    variables = model.trainable_variables       
    gradients = tf.gradients(training_loss, variables)  
    gradient_accumulator(gradients)
    
    outputs, _ = model(meta_test_source,
        labels=meta_test_target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, meta_test_target, training=True)
    meta_training_loss = loss[0] / loss[1]
    gradients = tf.gradients(meta_training_loss, variables)
    gradient_accumulator(gradients)
    num_word_examples = tf.reduce_sum(meta_test_target["length"]) + tf.reduce_sum(meta_train_target["length"])
  
    return meta_training_loss, training_loss, num_word_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_meta_loss, per_replica_loss, per_replica_num_word_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      meta_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_meta_loss, None)
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_word_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_word_examples, None)    
    return meta_loss, loss, num_word_examples
    
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  
  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  meta_train_data_flow = iter(_meta_train_forward())
  _loss = []
  _meta_loss = []  
  _num_word_examples = []
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        meta_loss, loss, num_word_examples = next(meta_train_data_flow)  
        _loss.append(loss)
        _meta_loss.append(meta_loss)
        _num_word_examples.append(num_word_examples)
      _step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Meta_loss = %f; num_word_examples = %d; after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_meta_loss), np.sum(_num_word_examples), elapsed)
        _loss = []
        _meta_loss = []
        _num_word_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def meta_train_v12(config,
          inner_optimizer,
          outer_optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          picking_prob=None,
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_datasets = create_multi_domain_meta_training_dataset_v2(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length)
  #####
  with strategy.scope():
    model.create_variables(optimizer=outer_optimizer)
    _outer_gradient_accumulator = optimizer_util.GradientAccumulator()  
    _inner_gradient_accumulator = optimizer_util.GradientAccumulator()

  def _accumulate_meta_train_gradients(source, target):
    #print("source: ", source)
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=outer_optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    gradients = outer_optimizer.get_gradients(training_loss, variables)
    _outer_gradient_accumulator(gradients)
    _inner_gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.print("domain:",source["domain"][0])
    return reported_loss, num_examples

  def _apply_inner_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(_inner_gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(_inner_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    inner_optimizer.apply_gradients(grads_and_vars)
    _inner_gradient_accumulator.reset()

  def _apply_outer_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(_outer_gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(_outer_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    outer_optimizer.apply_gradients(grads_and_vars)
    _outer_gradient_accumulator.reset()

  @tf.function
  def _inner_train_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_inner_gradients)
  @tf.function
  def _outer_train_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_outer_gradients)
  def _set_weight(v, w):
    v.assign(w)

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  meta_train_data_flows = []
  for meta_train_dataset in meta_train_datasets:
    @dataset_util.function_on_next(meta_train_dataset)
    def _meta_train_forward(next_fn):    
      with strategy.scope():
        per_replica_source, per_replica_target = next_fn()
        per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
            _accumulate_meta_train_gradients, args=(per_replica_source, per_replica_target))
        # TODO: these reductions could be delayed until _step is called.
        loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
        num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)    
      return loss, num_examples

    meta_train_data_flow = iter(_meta_train_forward())
    meta_train_data_flows.append(meta_train_data_flow)

  # Runs the training loop.
  import time
  start = time.time()  
  #print("meta_train_data_flows: ", meta_train_data_flows)
  datasets_size = [count_lines(src) for src in source_file]
  picking_prob = [data_size/sum(datasets_size) for data_size in datasets_size]
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  _loss = [[]] * len(meta_train_data_flows)
  _num_word_examples = []
  inner_loop_numb = [int(2)] * len(meta_train_data_flows)
  with _summary_writer.as_default():
    while True:  
      ##save current value of variables
      snapshots = [v.value() for v in model.trainable_variables]    
      domain = np.random.choice(len(meta_train_data_flows),1,p=picking_prob)[0]      
      ##inner loop
      for _ in range(inner_loop_numb[domain]):
        loss, num_word_examples = next(meta_train_data_flows[domain])  
        _loss[domain].append(loss)  
        _num_word_examples.append(num_word_examples)
        _inner_train_step()
      ##outer loop
      weight_reset(snapshots)
      _outer_train_step()
      ####      
      step = outer_optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; num_word_examples = %d; after %f seconds",
            step, learning_rate(step), np.mean([np.mean(losses) for losses in _loss]), np.sum(_num_word_examples), elapsed)
        _loss = [[]] * len(meta_train_data_flows)
        _num_word_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def meta_train_v13(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          picking_prob=None,
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  meta_train_dataset, meta_test_dataset = create_multi_domain_meta_training_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, 
                                                                        shuffle_buffer_size, maximum_length, picking_prob=picking_prob)
  #####
  def _accumulate_gradients(meta_train_source, meta_train_target, meta_test_source, meta_test_target): 
    #tf.print("meta_train_domain", meta_train_source["domain"][0], "meta_test_domain: ", meta_test_source["domain"][0], sep="|")
    meta_train_source["domain"] = tf.tile(tf.expand_dims(meta_test_source["domain"][0],0), meta_train_source["domain"].shape)
    meta_train_target["domain"] = tf.tile(tf.expand_dims(meta_test_target["domain"][0],0), meta_train_target["domain"].shape)
    with tf.GradientTape(persistent=True) as tape:
      ##### Inner adaptation
      outputs, _ = model(
          meta_train_source,
          labels=meta_train_target,
          training=True,
          step=optimizer.iterations)    
      loss = model.compute_loss(outputs, meta_train_target, training=True)
      training_loss = loss[0] / loss[1]
      if config.get("ADAP_activity_regularizing",False):
        layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
        output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
        print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
        print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
        if isinstance(layer_activity_regularization_loss_scale, list):
          domain = meta_train_source["domain"][0]
          layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
          layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        if isinstance(output_activity_regularization_loss_scale, list):
          domain = meta_train_source["domain"][0]
          output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
          output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
        regularization_losses = model.losses
        print("model_name_scope", model.name_scope())
        print(regularization_losses)
        layer_activity_regularization_losses = []
        output_activity_regularization_losses = []
        for loss_ in regularization_losses:
          if "multi_adap__dense" in loss_.name:
            output_activity_regularization_losses.append(loss_)
          else:
            layer_activity_regularization_losses.append(loss_)
        print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
        print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
        if len(layer_activity_regularization_losses)>0:
          training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)
        if len(output_activity_regularization_losses)>0:
          training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)
      variables = model.trainable_variables       
      args_dict = dict()
      for v in variables:
        args_dict.update({v.name:v})
      gradients = tape.gradient(training_loss, variables)  
      #gradient_accumulator(gradients) 

      meta_train_lr = config.get("meta_train_lr",1.0)
      print("meta_train_lr: ", meta_train_lr)
      
      if config.get("stopping_gradient",True):
        print("apply stopping_gradient")
        for g, v in zip(gradients, variables):      
          args_dict.update({v.name: v-meta_train_lr*tf.stop_gradient(g)})
      else:
        print("passing gradient")
        for g, v in zip(gradients, variables):
          args_dict.update({v.name: update(v,g,lr=meta_train_lr)})
      
      #### Meta_loss:
        #### meta gradient for shared parameters
      outputs, _ = model.forward_fn(meta_test_source,
          args_dict,
          labels=meta_test_target,
          training=True,
          step=optimizer.iterations)
      loss = model.compute_loss(outputs, meta_test_target, training=True)
      meta_training_loss = loss[0] / loss[1]
      gradients = tape.gradient(meta_training_loss, variables)
      gradient_accumulator(gradients)
      num_word_examples = tf.reduce_sum(meta_test_target["length"]) + tf.reduce_sum(meta_train_target["length"])
    
    return meta_training_loss, training_loss, num_word_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @utils.dataprocess.meta_learning_function_on_next(meta_train_dataset, meta_test_dataset)
  def _meta_train_forward(next_fn):    
    with strategy.scope():
      meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target = next_fn()
      per_replica_meta_loss, per_replica_loss, per_replica_num_word_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(meta_train_per_replica_source, meta_train_per_replica_target, meta_test_per_replica_source, meta_test_per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      meta_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_meta_loss, None)
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      num_word_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_word_examples, None)    
    return meta_loss, loss, num_word_examples
    
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  
  # Runs the training loop.
  import time
  start = time.time()  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  meta_train_data_flow = iter(_meta_train_forward())
  _loss = []
  _meta_loss = []  
  _num_word_examples = []
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        meta_loss, loss, num_word_examples = next(meta_train_data_flow)  
        _loss.append(loss)
        _meta_loss.append(meta_loss)
        _num_word_examples.append(num_word_examples)
      _step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Meta_loss = %f; num_word_examples = %d; after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_meta_loss), np.sum(_num_word_examples), elapsed)
        _loss = []
        _meta_loss = []
        _num_word_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if step > train_steps:
        break

def train_v12(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          picking_prob=None,
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=10000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_meta_train_size = config["batch_meta_train_size"]
  batch_meta_test_size = config["batch_meta_test_size"]
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  train_datasets = create_multi_domain_meta_training_dataset_v2(strategy, model, domain, source_file, target_file, 
                                                                        batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length)
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()

  def _accumulate_train_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.print("domain:",source["domain"][0])
    return reported_loss, num_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    #tf.print("accumulated_gradients: ",gradient_accumulator.step)
    gradient_accumulator.reset()

  @tf.function
  def train_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  train_data_flows = []
  for train_dataset in train_datasets:
    @dataset_util.function_on_next(train_dataset)
    def _train_forward(next_fn):    
      with strategy.scope():
        per_replica_source, per_replica_target = next_fn()
        per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
            _accumulate_train_gradients, args=(per_replica_source, per_replica_target))
        # TODO: these reductions could be delayed until _step is called.
        loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
        num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)    
      return loss, num_examples

    train_data_flow = iter(_train_forward())
    train_data_flows.append(train_data_flow)

  # Runs the training loop.
  import time
  start = time.time()  
  #datasets_size = [count_lines(src) for src in source_file]
  #picking_prob = [data_size/sum(datasets_size) for data_size in datasets_size]
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  _loss = [0.0] * len(train_data_flows)
  _num_word_examples = []
  step = 0
  importance_recalculate = config.get("importance_recalculate", 2000)
  save_stats = config.get("save_stats", 5000)
  domain_num = len(train_data_flows)
  # warmup_steps = config.get("warmup_steps",4000)
  # step_duration = config.get("step_duration",16)
  # prefinetuning_steps = config.get("prefinetuning_steps",200000)
  stats_path = os.path.join(config["model_dir"],"stats")
  if os.path.exists(stats_path):
    print("load %s"%stats_path)
    stats = np.load(stats_path)
  else:
    stats = {"consecutive_eval_drops": [0] * len(train_data_flows),
          "last_bleu_scores": [0] * len(train_data_flows),
          "last_training_loss": [20.0] * len(train_data_flows),
          "overfitting": [False] * len(train_data_flows),
          "consecutive_eval_drops:": [0] * len(train_data_flows),
          "importances": [1.0] * len(train_data_flows)}
  
  current_bleu_scores = [0] * domain_num
  current_training_loss = [0.0] * domain_num
  count = [1.0] * domain_num
  count_ = [1.0] * domain_num
  with _summary_writer.as_default():
    while True: 
      picking_prob = [importance/sum(stats["importances"]) for importance in stats["importances"]]
      domain = np.random.choice(domain_num,1,p=picking_prob)[0] 
      loss, num_word_examples = next(train_data_flows[domain])
      loss = loss.numpy()  
      _loss[domain] += loss
      count[domain] += 1
      current_training_loss[domain] += loss
      count_[domain] += 1
      _num_word_examples.append(num_word_examples)
      train_step()
      print("current_training_loss:",current_training_loss)
      ####      
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %s; num_word_examples = %d; after %f seconds; Importance = %s",
            step, learning_rate(step), " ".join([str(_loss[i]/count[i]) for i in range(len(_loss))]), np.sum(_num_word_examples), elapsed, " ".join([str(p) for p in picking_prob]))
        _loss = [0.0] * domain_num
        count = [1.0] * domain_num
        _num_word_examples = []
        start = time.time()

      if step % importance_recalculate:        
        current_training_loss = [current_training_loss[i]/count_[i] for i in range(domain_num)]
        print("last_training_loss:",stats["last_training_loss"])
        print("current_training_loss:",current_training_loss)
        for i in range(domain_num):
          if stats["last_training_loss"][i] < current_training_loss[i]:
            stats["importances"][i] = stats["importances"][i] * 2
          stats["last_training_loss"][i] = current_training_loss[i]
          current_training_loss[i] = 0.0
          count_[i] = 1.0

      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      
      if step % save_stats == 0:
        np.savez(os.path.join(config["model_dir"],"stats"), **stats)
      
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
          current_bleu_scores[i] = score

        for i in range(domain_num):
          if stats["last_bleu_scores"][i] > current_bleu_scores[i]:
            stats["consecutive_eval_drops"][i] +=1
          else:
            stats["consecutive_eval_drops"][i] = 0
          
          if stats["consecutive_eval_drops"][i] > 2:
            stats["overfitting"][i] = True

          if stats["overfitting"][i]:
            stats["importances"][i] = stats["importances"][i] / 2
          
      if step > train_steps:
        break

def domain_classification_on_top_encoder(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100):
  
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    checkpoint_path = checkpoint_manager.latest_checkpoint
    
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, 
                                                                        batch_train_size, batch_type, shuffle_buffer_size, maximum_length, multi_domain=(config["experiment"]!="baseline"),picking_prob=config.get("picking_prob",None))
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  

  def _accumulate_gradients(source, target):
    logits = model.classification_on_top_encoder(source, training=True)
    training_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(source["domain"], logits)    
    variables = [var for var in model.trainable_variables if "On_top_encoder_domain_classification" in var.name]
    print("var numb: ", len(variables))
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])    
    return tf.reduce_mean(training_loss), num_examples

  def _apply_gradients():
    variables = [var for var in model.trainable_variables if "On_top_encoder_domain_classification" in var.name]
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples

  @dataset_util.function_on_next(train_dataset)
  def _train_iteration(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      return per_replica_source, per_replica_target
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  _loss = []  
  _number_examples = []      

  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        loss, num_examples = next(train_data_flow)    
        _loss.append(loss)
        _number_examples.append(num_examples)
      _step()  
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          domain_predict(src, model, checkpoint_path, checkpoint, i, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
      tf.summary.flush()
      if step > train_steps:
        break

def domain_predict(source_file,
              model,
              checkpoint_path,
              checkpoint,
              domain,
              length_penalty,
              experiment="ldr",
              score_type="MultiBLEU",
              batch_size=10,
              beam_size=5):
  
  # Create the inference dataset.
  checkpoint.restore(checkpoint_path)
  tf.get_logger().info("Evaluating model %s", checkpoint_path)
  print("In domain %d"%domain)
  dataset = model.examples_inputter.make_inference_dataset(source_file, batch_size, domain)
  iterator = iter(dataset)

  # Create the mapping for target ids to tokens.

  @tf.function
  def predict_next():    
    source = next(iterator)  
    e, logits = model.classification_on_top_encoder(source, training=False)
    return tf.argmax(logits,-1)

  # Iterates on the dataset.
  
  predicted_domain = []
  
  while True:    
    try:
      predictions = predict_next()
      for d in predictions.numpy():          
        predicted_domain.append(d)
    except tf.errors.OutOfRangeError:
      break
  true_domain = [domain] * len(predicted_domain)
  from sklearn.metrics import classification_report
  from sklearn.metrics import accuracy_score
  print(classification_report(true_domain, predicted_domain))
  return accuracy_score(true_domain, predicted_domain)

def sentence_encode(source_file,
              model,
              checkpoint_manager,
              checkpoint,
              domain,
              output_file,
              experiment="ldr",
              batch_size=10):
  
  # Create the inference dataset.
  checkpoint.restore(checkpoint_manager.latest_checkpoint)
  tf.get_logger().info("Evaluating model %s", checkpoint_manager.latest_checkpoint)
  print("In domain %d"%domain)
  dataset = model.examples_inputter.make_inference_dataset(source_file, batch_size, domain)
  iterator = iter(dataset)

  # Create the mapping for target ids to tokens.

  @tf.function
  def encode_next():    
    source = next(iterator)  
    emb = model.sentence_encode(source, training=False)
    return emb

  # Iterates on the dataset.
  
  print("output file: ", output_file)
  src_sentence_embedding_list = []  
  maxcount = 1000000
  count = 0
  index = 0
  while True:    
    try:
      src_sentence_embedding_ = encode_next()
      src_sentence_embedding__ = src_sentence_embedding_.numpy()      
      src_sentence_embedding_list.append(src_sentence_embedding__)
      count += src_sentence_embedding__.shape[0]
      if count > maxcount:
        src_sentences = np.concatenate(src_sentence_embedding_list, axis=0)
        np.savez(output_file+str(index),sentence_embeddings=src_sentences)
        count = 0
        src_sentence_embedding_list = []
        index +=1
    except tf.errors.OutOfRangeError:
      break
  if len(src_sentence_embedding_list)>0:
    src_sentences = np.concatenate(src_sentence_embedding_list, axis=0)
    np.savez(output_file+str(index),sentence_embeddings=src_sentences)

def experimental_translate(source_file,
              reference,
              model,
              checkpoint_manager,
              checkpoint,              
              encoder_domain,
              decoder_domain,
              output_file,
              length_penalty,
              checkpoint_path=None,
              experiment="ldr",
              score_type="MultiBLEU",
              batch_size=10,
              beam_size=5):
  
  # Create the inference dataset.
  if checkpoint_path == None:
    checkpoint_path = checkpoint_manager.latest_checkpoint
  tf.get_logger().info("Evaluating model %s", checkpoint_path)
  print("encoder_domain: %d"%encoder_domain)
  print("decoder_domain: %s"%decoder_domain)
  checkpoint.restore(checkpoint_path)
  dataset = model.examples_inputter.make_inference_dataset(source_file, batch_size, encoder_domain)
  iterator = iter(dataset)

  # Create the mapping for target ids to tokens.
  ids_to_tokens = model.labels_inputter.ids_to_tokens

  @tf.function
  def predict_next():    
    source = next(iterator)
    source_length = source["length"]
    batch_size = tf.shape(source_length)[0]
    source_inputs = model.features_inputter(source)
    if experiment in ["residual","residualv15","residualv16","residualv17","residualv18","residualv2","residualv1","residualv3","residualv5","residualv13","residualv12","residualv6","residualv7","residualv11","residualv8","residualv9","baselinev1"]:
      encoder_outputs, _, _ = model.encoder([source_inputs, source["domain"]], source_length)
    else:
      encoder_outputs, _, _ = model.encoder(source_inputs, source_length)

    # Prepare the decoding strategy.
    if beam_size > 1:
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      source_length = tfa.seq2seq.tile_batch(source_length, beam_size)
      decoding_strategy = onmt.utils.BeamSearch(beam_size, length_penalty=length_penalty)
    else:
      decoding_strategy = onmt.utils.GreedySearch()

    # Run dynamic decoding.
    decoder_state = model.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=source_length)
    if experiment in ["residual","residualv15","residualv16","residualv17","residualv18","residualv2","residualv1","residualv3","residualv5","residualv6","residualv7","residualv13","residualv12","residualv11","residualv8","residualv9","baselinev1"]:
      map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}), tf.dtypes.cast(tf.fill(tf.expand_dims(tf.shape(ids)[0],0), decoder_domain), tf.int64)]
    elif experiment in ["DC"]:
      map_input_fn = lambda ids: model.labels_inputter({"ids": ids}, domain=decoder_domain)
    else:
      map_input_fn = lambda ids: model.labels_inputter({"ids": ids})
    decoded = model.decoder.dynamic_decode(
        map_input_fn,
        tf.fill([batch_size], START_OF_SENTENCE_ID),
        end_id=END_OF_SENTENCE_ID,
        initial_state=decoder_state,
        decoding_strategy=decoding_strategy,
        maximum_iterations=250)
    target_lengths = decoded.lengths
    target_tokens = ids_to_tokens.lookup(tf.cast(decoded.ids, tf.int64))
    return target_tokens, target_lengths

  # Iterates on the dataset.
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  print("output file: ", output_file)
  with open(output_file, "w") as output_:
    while True:    
      try:
        batch_tokens, batch_length = predict_next()
        for tokens, length in zip(batch_tokens.numpy(), batch_length.numpy()):
          sentence = b" ".join(tokens[0][:length[0]])
          print_bytes(sentence, output_)
          #print_bytes(sentence)
      except tf.errors.OutOfRangeError:
        break
  if reference!=None:
    print("score of model %s on test set %s: "%(checkpoint_manager.latest_checkpoint, source_file), scorer(reference, output_file))
    score = scorer(reference, output_file)
    if score is None:
      return 0.0
    else:
      return score

def visualize(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=None,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    checkpoint_path = checkpoint_manager.latest_checkpoint
  
  batch_train_size = config["batch_train_size"]  
  batch_type = config.get("batch_type","tokens")
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), single_pass=True,
                                            multi_domain=True,picking_prob=config.get("picking_prob",None))
  
  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=False,
        step=optimizer.iterations,
        internal_node_printing=True)
 
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
  
  # Runs the training loop.
  train_data_flow = iter(_train_forward())  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("Visualizing gating value")
  while True:  
    try:    
      next(train_data_flow)    
      tf.summary.flush()
    except StopIteration:
      break

def train_wdc(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    checkpoint_path = checkpoint_manager.latest_checkpoint
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=True,picking_prob=config.get("picking_prob",None))
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    non_adv_gradient_accumulator = optimizer_util.GradientAccumulator()  
    adv_gradient_accumulator = optimizer_util.GradientAccumulator()
  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    print(outputs)
    classification_logits_r = outputs["classification_logits_r"]
    classification_logits_s = outputs["classification_logits_s"]
    encoder_classification_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(source["domain"], classification_logits_r))
    probs = tf.nn.softmax(classification_logits_s, axis=1)
    prediction_probs = tf.map_fn(lambda x: x[0][x[1]], (probs, source["domain"]), dtype=tf.float32)
    adv_loss_1 = - tf.reduce_mean(tf.math.log(prediction_probs)) #tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(source["domain"], classification_logits_s))
    adv_loss_2 = - tf.reduce_mean(probs * tf.math.log(probs)) #- tf.reduce_mean(prediction_probs * tf.math.log(prediction_probs)) #- tf.reduce_mean(probs * tf.math.log(probs))
    #decoder_classification_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(source["domain"], outputs["state"])
    loss = model.compute_loss(outputs, target, training=True)  
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    total_loss = training_loss - adv_loss_2 * 0.2 + encoder_classification_loss 
    non_adv_vars = [v for v in model.trainable_variables if "On_top_decoder_domain_classification" not in v.name and "ADV_on_top_encoder_domain_classification" not in v.name] + \
                    [v for v in model.trainable_variables if "On_top_decoder_domain_classification" not in v.name and "ADV_on_top_encoder_domain_classification" in v.name and ("v_a" in v.name or "W_a" in v.name)]
    adv_vars = [v for v in model.trainable_variables if "ADV_on_top_encoder_domain_classification" in v.name and not ("v_a" in v.name or "W_a" in v.name)] 
    #####
    reported_loss = training_loss
    print("var numb: ", len(non_adv_vars))
    for v in non_adv_vars:
      print(v.name)
    gradients = optimizer.get_gradients(total_loss, non_adv_vars)
    non_adv_gradient_accumulator(gradients)
    #####
    print("adv_var_numb: ", len(adv_vars))
    for v in adv_vars:
      print(v.name)
    gradients = optimizer.get_gradients(adv_loss_1, adv_vars)
    adv_gradient_accumulator(gradients)
    #####
    num_examples = tf.reduce_sum(target["length"])
    return reported_loss, adv_loss_1, adv_loss_2, encoder_classification_loss, num_examples

  def _apply_gradients():
    grads_and_vars = []
    ####
    non_adv_vars = [v for v in model.trainable_variables if "On_top_decoder_domain_classification" not in v.name and "ADV_on_top_encoder_domain_classification" not in v.name] + \
                    [v for v in model.trainable_variables if "On_top_decoder_domain_classification" not in v.name and "ADV_on_top_encoder_domain_classification" in v.name and ("v_a" in v.name or "W_a" in v.name)]
    for gradient, variable in zip(non_adv_gradient_accumulator.gradients, non_adv_vars):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(non_adv_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    #####
    adv_vars = [v for v in model.trainable_variables if "ADV_on_top_encoder_domain_classification" in v.name and not ("v_a" in v.name or "W_a" in v.name)] 
    for gradient, variable in zip(adv_gradient_accumulator.gradients, adv_vars):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(adv_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    non_adv_gradient_accumulator.reset()
    adv_gradient_accumulator.reset()

  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_adv_loss_1, per_replica_adv_loss_2, per_replica_encoder_classification_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)    
      adv_loss_1 = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_adv_loss_1, None) 
      adv_loss_2 = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_adv_loss_2, None) 
      encoder_classification_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_encoder_classification_loss, None)   
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, adv_loss_1, adv_loss_2, encoder_classification_loss, num_examples
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  _loss = []  
  _adv_loss_1 = [] 
  _adv_loss_2 = []
  _encoder_classification_loss = []
  _number_examples = []

  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        loss, adv_loss_1, adv_loss_2, encoder_classification_loss, num_examples = next(train_data_flow)    
        _loss.append(loss)
        _adv_loss_1.append(adv_loss_1)
        _adv_loss_2.append(adv_loss_2)
        _encoder_classification_loss.append(encoder_classification_loss)
        _number_examples.append(num_examples)
      _step()  
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Adv_loss_1 = %f, Adv_loss_2 = %f, Encoder_classification_loss = %f, number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_adv_loss_1), np.mean(_adv_loss_2), np.mean(_encoder_classification_loss), np.sum(_number_examples), elapsed)
        _loss = []  
        _adv_loss_1 = [] 
        _adv_loss_2 = []
        _encoder_classification_loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      tf.summary.flush()
      if step > train_steps:
        break

def train_ldr(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    checkpoint_path = checkpoint_manager.latest_checkpoint
    
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=True,picking_prob=None)
  generic_dataset = create_training_dataset(strategy, model, config["generic_domain"], config["generic_source_file"], config["generic_target_file"], batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=True,picking_prob=config.get("picking_prob",None))
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    
    loss = model.compute_loss(outputs, target, training=True)
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss

    variables = model.trainable_variables
    print("var numb: ", len(variables))
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    return reported_loss, num_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples

  @dataset_util.function_on_next(generic_dataset)
  def _generic_train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  generic_data_flow = iter(_generic_train_forward())
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  _loss = []  
  _number_examples = []
  step = optimizer.iterations.numpy()    

  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        loss, num_examples = next(train_data_flow)    
        _, _ = next(generic_data_flow)
        _loss.append(loss)
        _number_examples.append(num_examples)
      _step()  
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      tf.summary.flush()
      if step > train_steps:
        break 

def train_denny_britz(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    checkpoint_path = checkpoint_manager.latest_checkpoint
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=True,picking_prob=config.get("picking_prob",None))
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    non_adv_gradient_accumulator = optimizer_util.GradientAccumulator()  
    adv_gradient_accumulator = optimizer_util.GradientAccumulator()

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        return_domain_classification_logits=True,
        training=True,
        step=optimizer.iterations)
    domain_classification_logits = outputs["domain_classification_logits"]
    #print("domain_classification_logits",domain_classification_logits)
    encoder_classification_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(source["domain"], domain_classification_logits))
    loss = model.compute_loss(outputs, target, training=True)  
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    if config["adv_training"]:
      print("adv_training")
      total_loss = training_loss - encoder_classification_loss
    else:
      total_loss = training_loss + encoder_classification_loss
    non_adv_vars = [v for v in model.trainable_variables if "On_top_encoder_domain_classification" not in v.name]
    adv_vars = [v for v in model.trainable_variables if "On_top_encoder_domain_classification" in v.name] 
    #####
    reported_loss = training_loss
    print("var numb: ", len(non_adv_vars))
    #for v in non_adv_vars:
    #  print(v.name)
    gradients = optimizer.get_gradients(total_loss, non_adv_vars)
    non_adv_gradient_accumulator(gradients)
    #####
    print("adv_var_numb: ", len(adv_vars))
    for v in adv_vars:
      print(v.name)
    gradients = optimizer.get_gradients(encoder_classification_loss, adv_vars)
    adv_gradient_accumulator(gradients)
    #####
    num_examples = tf.reduce_sum(target["length"])
    return reported_loss, encoder_classification_loss, num_examples

  def _apply_gradients():
    non_adv_vars = [v for v in model.trainable_variables if "On_top_encoder_domain_classification" not in v.name]
    grads_and_vars = []
    for gradient, variable in zip(non_adv_gradient_accumulator.gradients, non_adv_vars):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(non_adv_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    non_adv_gradient_accumulator.reset()

  def _apply_adv_gradients():
    adv_vars = [v for v in model.trainable_variables if "On_top_encoder_domain_classification" in v.name]  
    grads_and_vars = []
    for gradient, variable in zip(adv_gradient_accumulator.gradients, adv_vars):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(adv_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    adv_gradient_accumulator.reset()

  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_encoder_classification_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)     
      encoder_classification_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_encoder_classification_loss, None)   
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, encoder_classification_loss, num_examples
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  
  def _adv_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_adv_gradients)

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())

  ### Running one step to compile graph
  _, _, _ = next(train_data_flow)

  ### Initialize weights or update if needed for Continual Learning
  if config.get("continual_learning", False):
    assert config.get("checkpoint_path") != None
    checkpoint_path = config.get("checkpoint_path")
    load_and_update_if_needed_from_ckpt(config["model_dir"],   
                        checkpoint_path,
                        trackables={"model":model},
                        model_key="model")
                        
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  _loss = []  
  _encoder_classification_loss = []
  _number_examples = []

  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        loss, encoder_classification_loss, num_examples = next(train_data_flow)    
        _loss.append(loss)
        _encoder_classification_loss.append(encoder_classification_loss)
        _number_examples.append(num_examples)
      _step()  
      _adv_step()
      step = optimizer.iterations.numpy() // 2
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Encoder_classification_loss = %f, number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_encoder_classification_loss), np.sum(_number_examples), elapsed)
        _loss = []  
        _encoder_classification_loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      tf.summary.flush()
      if step > train_steps:
        break

def proxy_distance(config,
          optimizer,          
          learning_rate,
          model,  
          source_file,
          target_file,
          training_domain,
          eval_file,
          eval_domain,
          test_file,
          test_domain,
          strategy,  
          checkpoint_manager,
          checkpoint,
          save_dir,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100):
  
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    checkpoint_path = checkpoint_manager.latest_checkpoint  
    tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
    checkpoint.restore(checkpoint_path)
  output_dir = os.path.join(config["model_dir"],save_dir)
  new_checkpoint_manager = tf.train.CheckpointManager(checkpoint, output_dir, max_to_keep=None)
  
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  
  print("There are %d in-domain corpora"%len(source_file))
  print("batch type: ", batch_type)

  train_dataset = create_training_dataset(strategy, model, training_domain, source_file, target_file, 
                                                                        batch_train_size, batch_type, shuffle_buffer_size, maximum_length, multi_domain=(config["experiment"]!="baseline"),picking_prob=config.get("picking_prob",None))
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  

  def _accumulate_gradients(source, target):
    e, logits = model.classification_on_top_encoder(source, training=True)
    tf.print("logits: ", logits)
    training_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(source["domain"], logits)    
    #variables = [var for var in model.trainable_variables if "On_top_encoder_domain_classification" in var.name or "encoder" in var.name or "My_inputter_0" in var.name]
    variables = [var for var in model.trainable_variables if "On_top_encoder_domain_classification" in var.name]
    print("var numb: ", len(variables))
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.shape(source["length"])[0] 
    return tf.reduce_mean(training_loss), num_examples

  def _apply_gradients():
    #variables = model.trainable_variables
    variables = [var for var in model.trainable_variables if "On_top_encoder_domain_classification" in var.name]
    #variables = [var for var in model.trainable_variables if "On_top_encoder_domain_classification" in var.name or "encoder" in var.name or "My_inputter_0" in var.name]
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  _loss = []  
  _number_examples = []      

  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        loss, num_examples = next(train_data_flow)    
        _loss.append(loss)
        _number_examples.append(num_examples)
      _step()  
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      if step % eval_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        new_checkpoint_manager.save(checkpoint_number=step)
        checkpoint_path = new_checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,i in zip(eval_file, eval_domain):
          domain_predict(src, model, checkpoint_path, checkpoint, i, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
      tf.summary.flush()
      if step > train_steps:
        break
  errors = []
  for src,i in zip(test_file, test_domain):
    accuracy = domain_predict(src, model, checkpoint_manager, checkpoint, i, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
    errors.append(1-accuracy)
  return 2 * (1 - 2 * np.mean(errors))

def add_vocab(config,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=True,picking_prob=config.get("picking_prob",None))
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    if isinstance(loss, tuple):
      reported_loss = loss[0] / loss[2]
    else:
      _, reported_loss = loss, loss
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    num_examples = tf.reduce_sum(target["length"])
    return reported_loss, num_examples

  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples

  train_data_flow = iter(_train_forward())
  _,_ = next(train_data_flow)    
  step = optimizer.iterations.numpy()
  tf.get_logger().info("Saving checkpoint for step %d", step)
  for v in model.trainable_variables:
    if "_embedding" in v.name:
      v.assign(tf.cast(tf.concat([v, tf.Variable(np.zeros(1,512),dtype=v.dtype)],0),v.dtype), validate_shape=False)

  checkpoint_manager.save(checkpoint_number=step)
  return checkpoint_manager.latest_checkpoint

def averaged_checkpoint_translate(config, source_file,
              reference,
              model,
              checkpoint_manager,
              checkpoint,
              domain,
              output_file,
              length_penalty,
              is_noisy=1,
              experiment="ldr",
              score_type="MultiBLEU",
              batch_size=10,
              beam_size=10,
              max_count=3):
  
  # Create the inference dataset.
  from os import path
  """ if not path.exists(path.join("%s/averaged_checkpoint"%config["model_dir"],"ckpt-200000.data-00000-of-00002")):
    new_checkpoint_manager = average_checkpoints(config["model_dir"], output_dir="%s/averaged_checkpoint"%config["model_dir"], trackables={"model":model},
                        max_count=max_count,
                        model_key="model")
    checkpoint.restore(new_checkpoint_manager.latest_checkpoint)
    tf.get_logger().info("Evaluating model %s", new_checkpoint_manager.latest_checkpoint)
  else:
    checkpoint_path = path.join("%s/averaged_checkpoint"%config["model_dir"],"ckpt-200000")
    checkpoint.restore(checkpoint_path)
    tf.get_logger().info("Evaluating model %s", checkpoint_path) """
  if tf.__version__ == '2.3.0':
    new_checkpoint_manager = average_checkpoints_tf2_3(config["model_dir"], output_dir="%s/averaged_checkpoint"%config["model_dir"], trackables={"model":model},
                        max_count=max_count,
                        model_key="model")
  else:
    new_checkpoint_manager = average_checkpoints(config["model_dir"], output_dir="%s/averaged_checkpoint"%config["model_dir"], trackables={"model":model},
                        max_count=max_count,
                        model_key="model")
  checkpoint.restore(new_checkpoint_manager.latest_checkpoint)
  tf.get_logger().info("Evaluating model %s", new_checkpoint_manager.latest_checkpoint)
  print("In domain %d"%domain)
  if isinstance(model, onmt.models.Transformer):
    dataset = model.examples_inputter.make_inference_dataset(source_file, batch_size)
  else:
    dataset = model.examples_inputter.make_inference_dataset(source_file, batch_size, domain, is_noisy=is_noisy)
  
  iterator = iter(dataset)

  # Create the mapping for target ids to tokens.
  ids_to_tokens = model.labels_inputter.ids_to_tokens

  @tf.function
  def predict_next():    
    source = next(iterator)
    source_length = source["length"]
    batch_size = tf.shape(source_length)[0]
    source_inputs = model.features_inputter(source)
    if experiment in ["residual","residualv15","small_transformer","residualv25","residualv27","residualv28","residualv29","residual_big_transformer","residualv26","gated_residual_v5","residualv16","residualv19","residualv20","residualv21","residualv22","residualv23","residualv17","residualv18","residualv2","residualv1","residualv3","residualv5","residualv13","residualv12","residualv6","residualv11","residualv7","residualv8","residualv9","baselinev1"]:
      encoder_outputs, _, _ = model.encoder([source_inputs, source["domain"], source["is_noisy"]], source_length, training=False, internal_node_printing=True)
    else:
      encoder_outputs, _, _ = model.encoder(source_inputs, source_length, training=False)

    # Prepare the decoding strategy.
    if beam_size > 1:
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      source_length = tfa.seq2seq.tile_batch(source_length, beam_size)
      decoding_strategy = onmt.utils.BeamSearch(beam_size, length_penalty=length_penalty)
    else:
      decoding_strategy = onmt.utils.GreedySearch()

    # Run dynamic decoding.
    decoder_state = model.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=source_length)
    if experiment in ["residual","residualv15","small_transformer","residualv25","residualv27","residual_big_transformer","residualv26","gated_residual_v5","residualv16","residualv19","residualv20","residualv21","residualv22","residualv23","residualv17","residualv18","residualv2","residualv1","residualv3","residualv5","residualv6","residualv13","residualv12","residualv11","residualv7","residualv8","residualv9","baselinev1"]:
      map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}), tf.dtypes.cast(tf.fill(tf.expand_dims(tf.shape(ids)[0],0), domain), tf.int64)]
    elif experiment in ["DC","ldr"]:
      map_input_fn = lambda ids: model.labels_inputter({"ids": ids}, domain=domain)
    elif experiment in ["WDC"]:
      e_r, _ = model.classification_layer(encoder_outputs, source_length, training=False)
      e_s, _ = model.adv_classification_layer(encoder_outputs, source_length, training=False)
      g_s = model.share_gate(tf.concat([tf.tile(tf.expand_dims(e_s,1),[1,tf.shape(encoder_outputs)[1],1]),encoder_outputs],-1))
      g_r = model.specific_gate(tf.concat([tf.tile(tf.expand_dims(e_r,1),[1,tf.shape(encoder_outputs)[1],1]),encoder_outputs],-1))
      h_r = g_r * encoder_outputs
      h_s = g_s * encoder_outputs
      encoder_mask = model.encoder.build_mask(source_inputs, sequence_length=source_length)
      map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}, training=False), h_r, h_s, encoder_mask]
    elif experiment in ["residualv28","residualv29"]:
      map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}, training=False), source["domain"]]
    else:
      map_input_fn = lambda ids: model.labels_inputter({"ids": ids})
    decoded = model.decoder.dynamic_decode(
        map_input_fn,
        tf.fill([batch_size], START_OF_SENTENCE_ID),
        end_id=END_OF_SENTENCE_ID,
        initial_state=decoder_state,
        decoding_strategy=decoding_strategy,
        maximum_iterations=250)
    target_lengths = decoded.lengths
    target_tokens = ids_to_tokens.lookup(tf.cast(decoded.ids, tf.int64))
    return target_tokens, target_lengths

  # Iterates on the dataset.
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  print("output file: ", output_file)
  with open(output_file, "w") as output_:
    while True:    
      try:
        batch_tokens, batch_length = predict_next()
        for tokens, length in zip(batch_tokens.numpy(), batch_length.numpy()):
          sentence = b" ".join(tokens[0][:length[0]])
          print_bytes(sentence, output_)
          #print_bytes(sentence)
      except tf.errors.OutOfRangeError:
        break
  if reference!=None:
    print("score of model %s on test set %s: "%(checkpoint_manager.latest_checkpoint, source_file), scorer(reference, output_file))
    score = scorer(reference, output_file)
    if score is None:
      return 0.0
    else:
      return score

def debug_slurm_train(config,
          optimizer,          
          learning_rate,
          model,  
          hvd,  
          is_master,
          checkpoint_manager,
          checkpoint,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  #####
  num_replicas = hvd.size()
  is_master = hvd.rank() == 0
  #####
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if is_master:
    if checkpoint_manager.latest_checkpoint is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
      checkpoint.restore(checkpoint_manager.latest_checkpoint)
    else:
      if checkpoint_path is not None:
        tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
        checkpoint.restore(checkpoint_path)
    #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  dataset_fn = lambda input_context: create_training_dataset_hvd(model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                                                input_context.num_input_pipelines, input_context.input_pipeline_id, input_context.num_replicas_in_sync, 
                                                                maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                                                multi_domain=config.get("multi_domain", True),
                                                                picking_prob=config.get("picking_prob",None))
  #####
  gradient_accumulator = optimizer_util.GradientAccumulator()  
  
  dataset = dataset_fn(tf.distribute.InputContext(
          num_input_pipelines=hvd.size(),
          input_pipeline_id=hvd.rank(),
          num_replicas_in_sync=hvd.size()))

  counter = tf.Variable(
          tf.constant(0, dtype=tf.int64),
          trainable=False,
          synchronization=tf.VariableSynchronization.ON_READ,
          aggregation=tf.VariableAggregation.SUM)
    
  # Wrap forward and step with tf.function.
  def _all_reduce_sum(value):
    return hvd.allreduce(value, op=hvd.Sum)

  @tf.function(input_signature=dataset.element_spec)
  def _forward(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    # for var in variables:
    #   print(var.name)
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_words = tf.reduce_sum(target["length"])
    counter.assign_add(tf.cast(num_words, tf.int64))
    return reported_loss

  @tf.function
  def _step(is_first_batch):
    gradient_scale = gradient_accumulator.step * num_replicas
    gradients = [
        _all_reduce_sum(gradient / tf.cast(gradient_scale, gradient.dtype))
        for gradient in gradient_accumulator.gradients]
    variables = model.trainable_variables
    optimizer.apply_gradients(list(zip(gradients, variables)))
    if is_first_batch:
      hvd.broadcast_variables(model.variables, root_rank=0)
      hvd.broadcast_variables(optimizer.variables(), root_rank=0)
    gradient_accumulator.reset()

  @tf.function
  def _get_words_counters():
    tgt_word_counter = _all_reduce_sum(counter.read_value())
    counter.assign(tf.constant(0, dtype=tf.int64))
    return tgt_word_counter

  import time
  start = time.time()  
  _loss = []

  accum_steps = 1
  
  with _summary_writer.as_default():
    for step, (source, target) in enumerate(dataset):
      loss = _forward(source, target)
      _assert_loss_is_finite(loss)
      _loss.append(loss)
      _step(step==0)          
      
      if is_master and step % report_every == 0 and step>0:
        elapsed = time.time() - start
        _number_examples = _get_words_counters()
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        start = time.time()
      if is_master and step % save_every == 0 and step>0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if is_master and step % eval_every == 0 and step>0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        if config.get("unsupervised_clustering",False):
          tag_files = config.get("tag_files")
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          if config.get("unsupervised_clustering",False):
            score = translate_with_tag_file(src, tag_files[i], ref, model, checkpoint_manager, checkpoint, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          else:
            score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      if is_master:
        tf.summary.flush()
      if step > train_steps:
        break
      
def meta_train_v16(config,
          outer_optimizer,          
          inner_optimizer,
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  else:
    if checkpoint_path is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
      checkpoint.restore(checkpoint_path)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True),picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))
  
  #####
  with strategy.scope():
    model.create_variables(optimizer=outer_optimizer)
    inner_gradient_accumulator = optimizer_util.GradientAccumulator()  
    outer_gradient_accumulator = optimizer_util.GradientAccumulator()

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=outer_optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    domain = source["domain"][0]
    
    if config.get("ADAP_activity_regularizing",False):
      layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
      output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
      print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
      print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
      if isinstance(layer_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
        layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        #tf.print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale, "domain: ", domain)
      if isinstance(output_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
        output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
      regularization_losses = model.losses
      print("model_name_scope", model.name_scope())
      print(regularization_losses)
      layer_activity_regularization_losses = []
      output_activity_regularization_losses = []
      for loss_ in regularization_losses:
        if "multi_adap__dense" in loss_.name:
          output_activity_regularization_losses.append(loss_)
        else:
          layer_activity_regularization_losses.append(loss_)
      print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
      print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
      if len(layer_activity_regularization_losses)>0:
        training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)
      if len(output_activity_regularization_losses)>0:
        training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    for var in variables:
      print(var.name)
    gradients = inner_optimizer.get_gradients(training_loss, variables)
    inner_gradient_accumulator(gradients)
    outer_gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

  def _apply_inner_loop_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(inner_gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(inner_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    inner_optimizer.apply_gradients(grads_and_vars)
    inner_gradient_accumulator.reset()

  def _apply_outer_loop_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(outer_gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(outer_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    outer_optimizer.apply_gradients(grads_and_vars)
    outer_gradient_accumulator.reset()
 
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples

  @dataset_util.function_on_next(train_dataset)
  def _train_iteration(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      return per_replica_source, per_replica_target
  
  @tf.function
  def _inner_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_inner_loop_gradients)
  @tf.function
  def _outer_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_outer_loop_gradients)

  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _number_examples = []
  step = outer_optimizer.iterations.numpy()

  with _summary_writer.as_default():
    while True:
      #####
      snapshots = [v.value() for v in model.trainable_variables]
      #snapshots_example = [v.value() for v in model.trainable_variables if "multi_domain__sequence_to_sequence/multi_domain__self_attention_encoder_v12/self_attention_encoder_layer/transformer_layer_wrapper/multi_head_attention/dense/kernel" in v.name]
      #print(snapshots_example[0])
      for _ in range(int(config.get("inner_step",2))):
        loss, num_examples = next(train_data_flow)    
        _loss.append(loss)
        _number_examples.append(num_examples)
        _inner_step()
      #snapshot_1 = [v.value() for v in model.trainable_variables if "multi_domain__sequence_to_sequence/multi_domain__self_attention_encoder_v12/self_attention_encoder_layer/transformer_layer_wrapper/multi_head_attention/dense/kernel" in v.name]
      weight_reset(snapshots)
      #snapshot_2 = [v.value() for v in model.trainable_variables if "multi_domain__sequence_to_sequence/multi_domain__self_attention_encoder_v12/self_attention_encoder_layer/transformer_layer_wrapper/multi_head_attention/dense/kernel" in v.name]
      #print(snapshot_1[0])
      #print(snapshot_2[0])
      _outer_step()
      #####
      step = outer_optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        if config.get("unsupervised_clustering",False):
          tag_files = config.get("tag_files")
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          if config.get("unsupervised_clustering",False):
            score = translate_with_tag_file(src, tag_files[i], ref, model, checkpoint_manager, checkpoint, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          else:
            score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      tf.summary.flush()
      if step > train_steps:
        break

def train_wada(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  else:
    if checkpoint_path is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
      checkpoint.restore(checkpoint_path)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  
  print("There are %d in-domain corpora"%len(source_file))
  classification_loss_rate = tf.Variable(0.0,trainable=False)
  
  train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True),picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))
  from utils.dataprocess import count_lines
  datasets_size = [count_lines(src) for src in source_file]
  importance_weights = [data_size/sum(datasets_size) for data_size in datasets_size]
  temperature=config.get("temperature",1.0)
  importance_weights = [w ** temperature for w in importance_weights]
  importance_weights = [w/sum(importance_weights) for w in importance_weights]
  importance_weights = tf.constant(importance_weights)
  tf.print("importance_weights: ", importance_weights)
  #####
  with strategy.scope():
    classifier_optimizer = tfa.optimizers.LazyAdam(0.001)
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
    model_gradient_accumulator = optimizer_util.GradientAccumulator()
    classifier_gradient_accumulator = optimizer_util.GradientAccumulator()

  def _accumulate_model_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    domain = source["domain"][0]
    if config.get("apply_importance_weight", False):
      print("apply_importance_weight")
      training_loss = training_loss * importance_weights[domain]
    if config.get("ADAP_activity_regularizing",False):
        d_classification_gate_loss_scale = config.get("d_classification_gate_loss_scale",0.01)
        print("d_classification_gate_loss_scale: ", d_classification_gate_loss_scale)
                
        regularization_losses = model.losses
        print("model_name_scope", model.name_scope())
        print(regularization_losses)
        layer_activity_regularization_losses = []
        d_classification_gate_losses = []
        d_classifier_activity_regularization_losses = []
        d_classifier_weight_regularization_losses = []
        for loss_ in regularization_losses:
          if "multi_adap__dense" in loss_.name:
            continue
          elif "ADAP_gate" in loss_.name: #and "ActivityRegularizer" not in loss_.name and "Regularizer" not in loss_.name
            if "ActivityRegularizer" in loss_.name:
              d_classifier_activity_regularization_losses.append(loss_)
            elif "Regularizer" in loss_.name:
              d_classifier_weight_regularization_losses.append(loss_)
            else:
              d_classification_gate_losses.append(loss_)
          elif "ADAP_" in loss_.name:
            layer_activity_regularization_losses.append(loss_)

        print("There are %d adaptation regularization loss on domain classification gate_____"%len(d_classification_gate_losses))      

        if len(d_classification_gate_losses)>0 and d_classification_gate_loss_scale>0:
          classification_loss = d_classification_gate_loss_scale * tf.add_n(d_classification_gate_losses) / importance_weights[domain]
          training_loss -= classification_loss * classification_loss_rate
        

    variables = model.trainable_variables
    print("var numb: ", len(variables))
    model_vars = []
    classifier_vars = []
    for var in variables:
      if "ADAP_gate" in var.name:
        classifier_vars.append(var)
      else:
        model_vars.append(var)
    model_gradients = optimizer.get_gradients(training_loss, model_vars)
    model_gradient_accumulator(model_gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

  def _accumulate_classifier_gradients(source, target):
    _, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    domain = source["domain"][0]    
    regularization_losses = model.losses
    d_classification_gate_losses = []
    for loss_ in regularization_losses:
      if "multi_adap__dense" in loss_.name:
        continue
      elif "ADAP_gate" in loss_.name: #and "ActivityRegularizer" not in loss_.name and "Regularizer" not in loss_.name
        if "ActivityRegularizer" in loss_.name:
          continue
        elif "Regularizer" in loss_.name:
          continue
        else:
          d_classification_gate_losses.append(loss_)
    training_loss = tf.add_n(d_classification_gate_losses) / importance_weights[domain]
    reported_loss = tf.add_n(d_classification_gate_losses) #training_loss
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    model_vars = []
    classifier_vars = []
    for var in variables:
      if "ADAP_gate" in var.name:
        classifier_vars.append(var)
      else:
        model_vars.append(var)
    classifier_gradients = classifier_optimizer.get_gradients(training_loss, classifier_vars)
    classifier_gradient_accumulator(classifier_gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples
     
  def _apply_model_gradients():
    variables = model.trainable_variables
    model_vars = []
    classifier_vars = []
    for var in variables:
      if "ADAP_gate" in var.name:
        classifier_vars.append(var)
      else:
        model_vars.append(var)
    grads_and_vars = []
    for gradient, variable in zip(model_gradient_accumulator.gradients, model_vars):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(model_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    model_gradient_accumulator.reset()

  def _apply_classifier_gradients():
    variables = model.trainable_variables
    model_vars = []
    classifier_vars = []
    for var in variables:
      if "ADAP_gate" in var.name:
        classifier_vars.append(var)
      else:
        model_vars.append(var)
    grads_and_vars = []
    for gradient, variable in zip(classifier_gradient_accumulator.gradients, classifier_vars):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(classifier_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    classifier_optimizer.apply_gradients(grads_and_vars)
    classifier_gradient_accumulator.reset()

  @dataset_util.function_on_next(train_dataset)
  def _train_model_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_model_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples

  @dataset_util.function_on_next(train_dataset)
  def _train_classifier_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_classifier_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples

  @tf.function
  def _model_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_model_gradients)

  @tf.function
  def _classifier_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_classifier_gradients)

  # Runs the training loop.
  import time
  start = time.time()  
  #train_data_flow = iter(_train_forward())
  train_model_data_flow = iter(_train_model_forward())
  train_classifier_data_flow = iter(_train_classifier_forward())

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _d_classfication_loss = []
  _number_examples = []
  step = optimizer.iterations.numpy()
  if config.get("reset_step",None):
    print("start from %d-th step"%config.get("reset_step",150000))
    optimizer.iterations.assign(config.get("reset_step",150000))
  
  with _summary_writer.as_default():
    while True:
      #####Training batch
      if step == config.get("warm_start", 15000):
        classification_loss_rate.assign(1.0)
      if step >= config.get("warm_start", 15000):
        d_classfication_loss, _ = next(train_classifier_data_flow)
        _d_classfication_loss.append(d_classfication_loss)
        _classifier_step()
      loss, num_examples = next(train_model_data_flow)    
      _loss.append(loss)
      _number_examples.append(num_examples)
      _model_step()  
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
          "Step = %d ; Learning rate = %f ; Loss = %f; domain_classification_loss = %f, number_examples = %d, after %f seconds",
          step, learning_rate(step), np.mean(_loss), np.mean(_d_classfication_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _d_classfication_loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      tf.summary.flush()
      if step > train_steps:
        break

def finetune_wada(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  else:
    if checkpoint_path is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
      checkpoint.restore(checkpoint_path)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  
  print("There are %d in-domain corpora"%len(source_file))
  classification_loss_sign = tf.Variable(0.0,trainable=False)
  
  train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True),picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))
  from utils.dataprocess import count_lines
  datasets_size = [count_lines(src) for src in source_file]
  importance_weights = [data_size/sum(datasets_size) for data_size in datasets_size]
  temperature=config.get("temperature",1.0)
  importance_weights = [w ** temperature for w in importance_weights]
  importance_weights = [w/sum(importance_weights) for w in importance_weights]
  importance_weights = tf.constant(importance_weights)
  tf.print("importance_weights: ", importance_weights)
  #####
  with strategy.scope():
    classifier_optimizer = tfa.optimizers.LazyAdam(0.001)
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
    model_gradient_accumulator = optimizer_util.GradientAccumulator()
    classifier_gradient_accumulator = optimizer_util.GradientAccumulator()

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    domain = source["domain"][0]
    classification_loss = 0
    if config.get("apply_importance_weight", False):
      print("apply_importance_weight")
      training_loss = training_loss * importance_weights[domain]
    if config.get("ADAP_activity_regularizing",False):
        layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
        output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
        d_classification_gate_loss_scale = config.get("d_classification_gate_loss_scale",0.01)
        d_classifier_weight_regularization_losses_scale = config.get("d_classifier_weight_regularization_losses_scale",1.0)
        print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
        print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
        print("d_classification_gate_loss_scale: ", d_classification_gate_loss_scale)
        print("d_classifier_weight_regularization_losses_scale: ", d_classifier_weight_regularization_losses_scale)
        if isinstance(layer_activity_regularization_loss_scale, list):
          domain = source["domain"][0]
          layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
          layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
          #tf.print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale, "domain: ", domain)
        if isinstance(output_activity_regularization_loss_scale, list):
          domain = source["domain"][0]
          output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
          output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
        regularization_losses = model.losses
        print("model_name_scope", model.name_scope())
        print(regularization_losses)
        layer_activity_regularization_losses = []
        output_activity_regularization_losses = []
        d_classification_gate_losses = []
        d_classifier_activity_regularization_losses = []
        d_classifier_weight_regularization_losses = []
        for loss_ in regularization_losses:
          if "multi_adap__dense" in loss_.name:
            output_activity_regularization_losses.append(loss_)
          elif "ADAP_gate" in loss_.name: 
            if "Regularizer" in loss_.name:
              d_classifier_weight_regularization_losses.append(loss_)
            else:
              d_classification_gate_losses.append(loss_)
          elif "ADAP_" in loss_.name:
            layer_activity_regularization_losses.append(loss_)
        if (len(layer_activity_regularization_losses)>0) and layer_activity_regularization_loss_scale>0:
          print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
          training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)
        
        if len(d_classification_gate_losses)>0 and d_classification_gate_loss_scale>0:
          print("There are %d adaptation regularization loss on domain classification gate_____"%len(d_classification_gate_losses))
          classification_loss += d_classification_gate_loss_scale * tf.add_n(d_classification_gate_losses) / importance_weights[domain]

        if len(d_classifier_weight_regularization_losses)>0 and d_classifier_weight_regularization_losses_scale>0:
          print("There are %d d_classifier_weight_regularization_losses"%len(d_classifier_weight_regularization_losses))
          classification_loss += d_classifier_weight_regularization_losses_scale * tf.add_n(d_classifier_weight_regularization_losses)
        

    variables = model.trainable_variables
    model_vars = []
    classifier_vars = []
    for var in variables:
      if "ADAP_gate" in var.name:
        classifier_vars.append(var)
      elif "ADAP" in var.name:
        model_vars.append(var)
    variables = model_vars + classifier_vars
    print("var numb: ", len(variables))
    for var in model_vars:
      print(var.name)
    #model_gradients = optimizer.get_gradients(training_loss, model_vars)
    #gradients = optimizer.get_gradients(training_loss + classification_loss, variables)
    gradients = optimizer.get_gradients(training_loss, model_vars)
    #gradients = model_gradients + classifier_gradients
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, classification_loss, num_examples

  def _accumulate_model_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    domain = source["domain"][0]
    if config.get("apply_importance_weight", False):
      print("apply_importance_weight")
      training_loss = training_loss * importance_weights[domain]
    if config.get("ADAP_activity_regularizing",False):
        layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
        output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
        d_classification_gate_loss_scale = config.get("d_classification_gate_loss_scale",0.01)
        d_classifier_activity_regularization_loss_scale = config.get("d_classifier_activity_regularization_loss_scale",1.0)
        d_classifier_weight_regularization_losses_scale = config.get("d_classifier_weight_regularization_losses_scale",1.0)
        print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
        print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
        print("d_classification_gate_loss_scale: ", d_classification_gate_loss_scale)
        print("d_classifier_weight_regularization_losses_scale: ", d_classifier_weight_regularization_losses_scale)
        if isinstance(layer_activity_regularization_loss_scale, list):
          domain = source["domain"][0]
          layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
          layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        if isinstance(output_activity_regularization_loss_scale, list):
          domain = source["domain"][0]
          output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
          output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
        regularization_losses = model.losses
        print("model_name_scope", model.name_scope())
        print(regularization_losses)
        layer_activity_regularization_losses = []
        output_activity_regularization_losses = []
        d_classification_gate_losses = []
        d_classifier_activity_regularization_losses = []
        d_classifier_weight_regularization_losses = []
        for loss_ in regularization_losses:
          if "multi_adap__dense" in loss_.name:
            output_activity_regularization_losses.append(loss_)
          elif "ADAP_gate" in loss_.name: #and "ActivityRegularizer" not in loss_.name and "Regularizer" not in loss_.name
            if "ActivityRegularizer" in loss_.name:
              d_classifier_activity_regularization_losses.append(loss_)
            elif "Regularizer" in loss_.name:
              d_classifier_weight_regularization_losses.append(loss_)
            else:
              d_classification_gate_losses.append(loss_)
          elif "ADAP_" in loss_.name:
            layer_activity_regularization_losses.append(loss_)

        print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
        print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
        print("There are %d adaptation regularization loss on domain classification gate_____"%len(d_classification_gate_losses))
        print("There are %d d_classifier_activity_regularization_losses"%len(d_classifier_activity_regularization_losses))
        print("There are %d d_classifier_weight_regularization_losses"%len(d_classifier_weight_regularization_losses))
        if (len(layer_activity_regularization_losses)>0) and layer_activity_regularization_loss_scale>0:
          training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)

        if len(output_activity_regularization_losses)>0 and output_activity_regularization_loss_scale>0:
          training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)

        if len(d_classification_gate_losses)>0 and d_classification_gate_loss_scale>0:
          classification_loss = d_classification_gate_loss_scale * tf.add_n(d_classification_gate_losses) / importance_weights[domain]
          training_loss += classification_loss * classification_loss_sign

        if len(d_classifier_activity_regularization_losses)>0 and d_classifier_activity_regularization_loss_scale>0:
          training_loss += d_classifier_activity_regularization_loss_scale * tf.add_n(d_classifier_activity_regularization_losses)

        if len(d_classifier_weight_regularization_losses)>0 and d_classifier_weight_regularization_losses_scale>0:
          training_loss += d_classifier_weight_regularization_losses_scale * tf.add_n(d_classifier_weight_regularization_losses)    
        

    variables = model.trainable_variables
    print("var numb: ", len(variables))
    model_vars = []
    classifier_vars = []
    for var in variables:
      if "ADAP_gate/dense" in var.name:
        classifier_vars.append(var)
      else:
        model_vars.append(var)
    variables = model_vars + classifier_vars
    model_gradients = optimizer.get_gradients(training_loss, model_vars)
    model_gradient_accumulator(model_gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

  def _accumulate_classifier_gradients(source, target):
    _, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    domain = source["domain"][0]    
    regularization_losses = model.losses
    d_classification_gate_losses = []
    for loss_ in regularization_losses:
      if "multi_adap__dense" in loss_.name:
        continue
      elif "ADAP_gate" in loss_.name: #and "ActivityRegularizer" not in loss_.name and "Regularizer" not in loss_.name
        if "ActivityRegularizer" in loss_.name:
          continue
        elif "Regularizer" in loss_.name:
          continue
        else:
          d_classification_gate_losses.append(loss_)
    training_loss = tf.add_n(d_classification_gate_losses) / importance_weights[domain]
    reported_loss = training_loss
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    model_vars = []
    classifier_vars = []
    for var in variables:
      if "ADAP_gate/dense" in var.name:
        classifier_vars.append(var)
      else:
        model_vars.append(var)
    classifier_gradients = classifier_optimizer.get_gradients(training_loss, classifier_vars)
    classifier_gradient_accumulator(classifier_gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples
     
  def _apply_gradients():
    variables = model.trainable_variables
    model_vars = []
    classifier_vars = []
    for var in variables:
      if "ADAP_gate" in var.name:
        classifier_vars.append(var)
      elif "ADAP" in var.name:
        model_vars.append(var)
    variables = model_vars + classifier_vars
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, model_vars):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  def _apply_model_gradients():
    variables = model.trainable_variables
    model_vars = []
    classifier_vars = []
    for var in variables:
      if "ADAP_gate/dense" in var.name:
        classifier_vars.append(var)
      else:
        model_vars.append(var)
    variables = model_vars + classifier_vars
    grads_and_vars = []
    for gradient, variable in zip(model_gradient_accumulator.gradients, model_vars):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    model_gradient_accumulator.reset()

  def _apply_classifier_gradients():
    variables = model.trainable_variables
    model_vars = []
    classifier_vars = []
    for var in variables:
      if "ADAP_gate/dense" in var.name:
        classifier_vars.append(var)
      else:
        model_vars.append(var)
    variables = model_vars + classifier_vars
    grads_and_vars = []
    for gradient, variable in zip(classifier_gradient_accumulator.gradients, classifier_vars):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    classifier_optimizer.apply_gradients(grads_and_vars)
    classifier_gradient_accumulator.reset()

  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_classification_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      classification_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_classification_loss, None)
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, classification_loss, num_examples

  @dataset_util.function_on_next(train_dataset)
  def _train_model_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_model_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples

  @dataset_util.function_on_next(train_dataset)
  def _train_classifier_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_classifier_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples

  @dataset_util.function_on_next(train_dataset)
  def _train_iteration(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      return per_replica_source, per_replica_target
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  @tf.function
  def _model_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_model_gradients)

  @tf.function
  def _classifier_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_classifier_gradients)

  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  #train_model_data_flow = iter(_train_model_forward())
  #train_classifier_data_flow = iter(_train_classifier_forward())
  _, _, _ = next(train_data_flow)

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _d_classfication_loss = []
  _number_examples = []

  step = optimizer.iterations.numpy()
  if config.get("reset_step",None):
    print("start from %d-th step"%config.get("reset_step",150000))
    optimizer.iterations.assign(config.get("reset_step",150000))
  if step <= 1:
    initializer = config.get("initializer","default")
    if initializer == "default":
      print("Initializing variables by tensorflow default")      
    elif initializer == "variance_scaling":
      print("Initializing variables by tf.variance_scaling")
      initial_value = []
      for v in model.trainable_variables:
        shape = tf.shape(v).numpy()
        initial_value.append(variance_scaling_initialier(shape, scale=1.0, mode="fan_avg", distribution="uniform"))
      weight_reset(initial_value)       

  if config.get("continual_learning", False):
    print("Continual Learning needs to load from old model")
    assert config.get("checkpoint_path") != None
    checkpoint_path = config.get("checkpoint_path")
    load_and_update_if_needed_from_ckpt(config["model_dir"],   
                        checkpoint_path,                        
                        trackables={"model":model},
                        vocab_update=True,
                        model_key="model")

  with _summary_writer.as_default():
    while True:
      #####Training batch
      #for _ in range(int(config.get("accumulation_step",1))):
      loss, classification_loss, num_examples = next(train_data_flow)    
      _loss.append(loss)
      _d_classfication_loss.append(classification_loss)
      _number_examples.append(num_examples)
      _step()  
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
          "Step = %d ; Learning rate = %f ; Loss = %f; d_classfication_loss = %f, number_examples = %d, after %f seconds",
          step, learning_rate(step), np.mean(_loss), np.mean(_d_classfication_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        _d_classfication_loss = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      tf.summary.flush()
      if step > train_steps:
        break

def train_DRO(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  else:
    if checkpoint_path is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
      checkpoint.restore(checkpoint_path)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  update_z_every = config.get("update_z_every",50)
  print("There are %d in-domain corpora"%len(source_file))
  if experiment=="residualv28":
    prob_file = config["prob"]
    train_dataset = create_training_dataset_with_dprob(strategy, model, source_file, target_file, prob_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True),picking_prob=config.get("picking_prob",None))
  else:
    train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True),picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))
  

  #####
  datasets_size = [count_lines(src) for src in source_file]
  empirical_training_distribution = [data_size/sum(datasets_size) for data_size in datasets_size]
  empirical_training_distribution = tf.constant(empirical_training_distribution)
  z = tf.constant([1.0/len(datasets_size)] * len(datasets_size))

  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    domain = source["domain"][0]
    if config.get("apply_importance_weight", False):
      print("apply_importance_weight")
      training_loss = training_loss * z[domain] / empirical_training_distribution[domain]
    
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    for var in variables:
      print(var.name)
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  def update_z():
    return 0
  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  _, _ = next(train_data_flow)

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _number_examples = []
  step = optimizer.iterations.numpy()  
  _domain_loss = [None]* len(domain)
  if config.get("continual_learning", False):
    print("Continual Learning needs to load from old model")
    assert config.get("checkpoint_path") != None
    checkpoint_path = config.get("checkpoint_path")
    load_and_update_if_needed_from_ckpt(config["model_dir"],   
                        checkpoint_path,                        
                        trackables={"model":model},
                        vocab_update=True,
                        model_key="model")

  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        loss, num_examples = next(train_data_flow)    
        _loss.append(loss)
        _number_examples.append(num_examples)
      _step()  
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % update_z_every ==0:
        update_z()
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      tf.summary.flush()
      if step > train_steps:
        break

def finetune_wada_v1(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  else:
    if checkpoint_path is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
      checkpoint.restore(checkpoint_path)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  
  print("There are %d in-domain corpora"%len(source_file))
  
  train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True),picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))
  from utils.dataprocess import count_lines
  datasets_size = [count_lines(src) for src in source_file]
  if config.get("importance_weights",None):
    importance_weights = config.get("importance_weights",None)
  else:
    importance_weights = [data_size/sum(datasets_size) for data_size in datasets_size]
  temperature=config.get("temperature",1.0)
  importance_weights = [w ** temperature for w in importance_weights]
  importance_weights = [w/sum(importance_weights) for w in importance_weights]
  importance_weights = tf.constant(importance_weights)
  tf.print("importance_weights: ", importance_weights)
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    model_gradient_accumulator = optimizer_util.GradientAccumulator()
    classifier_gradient_accumulator = optimizer_util.GradientAccumulator()

  def _accumulate_model_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    domain = source["domain"][0]
    
    if config.get("ADAP_activity_regularizing",False):
        layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
        d_classification_gate_loss_scale = config.get("d_classification_gate_loss_scale",0.01)
        d_classifier_weight_regularization_losses_scale = config.get("d_classifier_weight_regularization_losses_scale",1.0)
        print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
        print("d_classification_gate_loss_scale: ", d_classification_gate_loss_scale)
        print("d_classifier_weight_regularization_losses_scale: ", d_classifier_weight_regularization_losses_scale)
        if isinstance(layer_activity_regularization_loss_scale, list):
          domain = source["domain"][0]
          layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
          layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        
        regularization_losses = model.losses
        print("model_name_scope", model.name_scope())
        print(regularization_losses)
        layer_activity_regularization_losses = []
        output_activity_regularization_losses = []
        d_classification_gate_losses = []
        d_classifier_activity_regularization_losses = []
        d_classifier_weight_regularization_losses = []
        for loss_ in regularization_losses:
          if "multi_adap__dense" in loss_.name:
            output_activity_regularization_losses.append(loss_)
          elif "ADAP_gate" in loss_.name: #and "ActivityRegularizer" not in loss_.name and "Regularizer" not in loss_.name
            if "ActivityRegularizer" in loss_.name:
              d_classifier_activity_regularization_losses.append(loss_)
            elif "Regularizer" in loss_.name:
              d_classifier_weight_regularization_losses.append(loss_)
            else:
              d_classification_gate_losses.append(loss_)
          elif "ADAP_" in loss_.name and not("noisy" in loss_.name) :
            layer_activity_regularization_losses.append(loss_)

        print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
        print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
        print("There are %d adaptation regularization loss on domain classification gate_____"%len(d_classification_gate_losses))
        print("There are %d d_classifier_activity_regularization_losses"%len(d_classifier_activity_regularization_losses))
        print("There are %d d_classifier_weight_regularization_losses"%len(d_classifier_weight_regularization_losses))
        if (len(layer_activity_regularization_losses)>0) and layer_activity_regularization_loss_scale>0:
          training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)
     
    regularization_losses = model.losses
    for loss_ in regularization_losses:
      if "multi_adap__dense" in loss_.name:
        continue
      elif "ADAP_gate" in loss_.name: 
        if "ActivityRegularizer" in loss_.name:
          continue
        elif "Regularizer" in loss_.name:
          d_classifier_weight_regularization_losses.append(loss_)
        else:
          d_classification_gate_losses.append(loss_)
    d_classifier_weight_regularization_losses_scale = config.get("d_classifier_weight_regularization_losses_scale",1.0)
    training_loss += 0.3 * tf.add_n(d_classification_gate_losses) / importance_weights[domain]
    if d_classifier_weight_regularization_losses_scale>0 and len(d_classifier_weight_regularization_losses)>0:
      print("There are %d d_classifier_weight_regularization_losses"%len(d_classifier_weight_regularization_losses))
      reported_loss_1 = tf.add_n(d_classifier_weight_regularization_losses) * d_classifier_weight_regularization_losses_scale
      training_loss += reported_loss_1

    variables = model.trainable_variables

    """ for v in variables:
      print(v.name) """
    model_vars = []
    classifier_vars = []
    for var in variables:
      if "ADAP_gate" in var.name:
        classifier_vars.append(var)
      elif "ADAP" in var.name and not("noisy" in var.name):
        model_vars.append(var)
      elif "enc_layernorm_2" in var.name:
        model_vars.append(var)
    variables = model_vars + classifier_vars
    print("model_vars numb: ", len(variables))
    
    for v in variables:
      print(v.name)
   
    model_gradients = optimizer.get_gradients(training_loss, variables)
    model_gradient_accumulator(model_gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, reported_loss_1, num_examples

  def _accumulate_classifier_gradients(source, target):
    _, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    domain = source["domain"][0]    
    regularization_losses = model.losses
    print(regularization_losses)
    d_classification_gate_losses = []
    d_classifier_weight_regularization_losses = []
    for loss_ in regularization_losses:
      if "multi_adap__dense" in loss_.name:
        continue
      elif "ADAP_gate" in loss_.name: 
        if "ActivityRegularizer" in loss_.name:
          continue
        elif "Regularizer" in loss_.name:
          d_classifier_weight_regularization_losses.append(loss_)
        else:
          d_classification_gate_losses.append(loss_)
    d_classifier_weight_regularization_losses_scale = config.get("d_classifier_weight_regularization_losses_scale",1.0)
    training_loss = tf.add_n(d_classification_gate_losses) / importance_weights[domain]
    if d_classifier_weight_regularization_losses_scale>0 and len(d_classifier_weight_regularization_losses)>0:
      print("There are %d d_classifier_weight_regularization_losses"%len(d_classifier_weight_regularization_losses))
      training_loss += tf.add_n(d_classifier_weight_regularization_losses) * d_classifier_weight_regularization_losses_scale
    reported_loss = training_loss
    variables = model.trainable_variables
    model_vars = []
    classifier_vars = []
    for var in variables:
      if "ADAP_gate" in var.name:
        classifier_vars.append(var)
      elif "ADAP" in var.name and not("noisy" in var.name):
        model_vars.append(var)
      elif "enc_layernorm_2" in var.name:
        model_vars.append(var)
    print("classifier_vars numb: ", len(classifier_vars))
    """ for v in classifier_vars:
      print(v.name) """
    classifier_gradients = optimizer.get_gradients(training_loss, classifier_vars)
    classifier_gradient_accumulator(classifier_gradients)
    num_examples = tf.reduce_sum(target["length"])
    return reported_loss, num_examples
 
  def _apply_model_gradients():
    variables = model.trainable_variables
    model_vars = []
    classifier_vars = []
    for var in variables:
      if "ADAP_gate" in var.name:
        classifier_vars.append(var)
      elif "ADAP" in var.name and not("noisy" in var.name):
        model_vars.append(var)
      elif "enc_layernorm_2" in var.name:
        model_vars.append(var)
    variables = model_vars + classifier_vars
    grads_and_vars = []
    for gradient, variable in zip(model_gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(model_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    model_gradient_accumulator.reset()

  def _apply_classifier_gradients():
    variables = model.trainable_variables
    model_vars = []
    classifier_vars = []
    for var in variables:
      if "ADAP_gate" in var.name:
        classifier_vars.append(var)
      elif "ADAP" in var.name and not("noisy" in var.name):
        model_vars.append(var)
      elif "enc_layernorm_2" in var.name:
        model_vars.append(var)
    variables = model_vars + classifier_vars
    grads_and_vars = []
    for gradient, variable in zip(classifier_gradient_accumulator.gradients, classifier_vars):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(classifier_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    classifier_gradient_accumulator.reset()

  @dataset_util.function_on_next(train_dataset)
  def _train_classifier_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_classifier_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples

  @dataset_util.function_on_next(train_dataset)
  def _train_model_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_loss_1, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_model_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)  
      classification_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss_1, None)  
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, classification_loss, num_examples
  
  @tf.function
  def _model_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_model_gradients)

  @tf.function
  def _classifier_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_classifier_gradients)

  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  # Runs the training loop.
  import time
  start = time.time()  
  train_model_data_flow = iter(_train_model_forward())
  train_classifier_data_flow = iter(_train_classifier_forward())

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _d_classfication_loss = []
  _number_examples = []

  step = optimizer.iterations.numpy()
  if config.get("reset_step",None):
    print("start from %d-th step"%config.get("reset_step",150000))
    optimizer.iterations.assign(config.get("reset_step",150000))
    step = optimizer.iterations.numpy()
  
  with _summary_writer.as_default():
    while True:
      if step < config.get("classifier_training_step",250000):
        classification_loss, num_examples = next(train_classifier_data_flow)    
        _d_classfication_loss.append(classification_loss)
        _number_examples.append(num_examples)
        _classifier_step()
      else:
        loss, classification_loss, num_examples = next(train_model_data_flow)  
        _loss.append(loss)  
        _d_classfication_loss.append(classification_loss)
        _number_examples.append(num_examples)
        _model_step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        if step < config.get("classifier_training_step",250000):
          elapsed = time.time() - start
          tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; d_classfication_loss = %f, number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_d_classfication_loss), np.sum(_number_examples), elapsed)
          _number_examples = []
          _d_classfication_loss = []
          start = time.time()
        else:
          elapsed = time.time() - start
          tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; classification_loss = %f, number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_d_classfication_loss), np.sum(_number_examples), elapsed)
          _loss = []
          _d_classfication_loss = []
          _number_examples = []
          start = time.time()

      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)

      if step % eval_every == 0 and step > config.get("classifier_training_step",250000): 
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      tf.summary.flush()
      if step > train_steps:
        break

def finetune_noisy_v1(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  else:
    if checkpoint_path is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
      checkpoint.restore(checkpoint_path)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  is_noisy = config.get("is_noisy",None)
  
  print("There are %d in-domain corpora"%len(source_file))
  
  train_dataset = create_training_dataset_robustness(strategy, model, domain, is_noisy, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True),picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))

  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    model_gradient_accumulator = optimizer_util.GradientAccumulator()
    classifier_gradient_accumulator = optimizer_util.GradientAccumulator()

  def _accumulate_model_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    noisy_layer_activity_regularization_loss_scale = config.get("noisy_layer_activity_regularization_loss_scale",0.001)
    print("noisy_layer_activity_regularization_loss_scale: ", noisy_layer_activity_regularization_loss_scale)
    noisy_layer_activity_regularization_loss_scale = tf.constant(noisy_layer_activity_regularization_loss_scale)
        
    regularization_losses = model.losses
    print("model_name_scope", model.name_scope())
    print(regularization_losses)
    noisy_layer_activity_regularization_losses = []
    
    for loss_ in regularization_losses:
      if "noisy_ADAP" in loss_.name:
        noisy_layer_activity_regularization_losses.append(loss_)

    print("There are %d noisy_layer_activity_regularization_losses"%len(noisy_layer_activity_regularization_losses))
    if (len(noisy_layer_activity_regularization_losses)>0) and noisy_layer_activity_regularization_loss_scale>0:
      training_loss += noisy_layer_activity_regularization_loss_scale * tf.add_n(noisy_layer_activity_regularization_losses)
     
    variables = model.trainable_variables
    
    model_vars = []
    classifier_vars = []
    for var in variables:
      if "noisy_gate" in var.name:
        classifier_vars.append(var)
      elif "noisy_ADAP" in var.name:
        model_vars.append(var)
    variables = model_vars + classifier_vars
    print("model_vars numb: ", len(model_vars))
    model_gradients = optimizer.get_gradients(training_loss, model_vars)
    model_gradient_accumulator(model_gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

  def _accumulate_classifier_gradients(source, target):
    _, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    regularization_losses = model.losses
    d_classification_gate_losses = []
    d_classifier_weight_regularization_losses = []
    for loss_ in regularization_losses:
      if "noisy_gate" in loss_.name: 
        if "ActivityRegularizer" in loss_.name:
          continue
        elif "Regularizer" in loss_.name:
          d_classifier_weight_regularization_losses.append(loss_)
        else:
          d_classification_gate_losses.append(loss_)
    d_classifier_weight_regularization_losses_scale = config.get("d_classifier_weight_regularization_losses_scale",1.0)
    training_loss = tf.add_n(d_classification_gate_losses)
    if d_classifier_weight_regularization_losses_scale>0 and len(d_classifier_weight_regularization_losses)>0:
      print("There are %d d_classifier_weight_regularization_losses"%len(d_classifier_weight_regularization_losses))
      training_loss += tf.add_n(d_classifier_weight_regularization_losses) * d_classifier_weight_regularization_losses_scale
    reported_loss = training_loss
    variables = model.trainable_variables
    model_vars = []
    classifier_vars = []
    for var in variables:
      if "noisy_gate" in var.name:
        classifier_vars.append(var)
      elif "noisy_ADAP" in var.name:
        model_vars.append(var)
    print("classifier_vars numb: ", len(classifier_vars))
    classifier_gradients = optimizer.get_gradients(training_loss, classifier_vars)
    classifier_gradient_accumulator(classifier_gradients)
    num_examples = tf.reduce_sum(target["length"])
    return reported_loss, num_examples
 
  def _apply_model_gradients():
    variables = model.trainable_variables
    model_vars = []
    classifier_vars = []
    for var in variables:
      if "noisy_gate" in var.name:
        classifier_vars.append(var)
      elif "noisy_ADAP" in var.name:
        model_vars.append(var)
    grads_and_vars = []
    for gradient, variable in zip(model_gradient_accumulator.gradients, model_vars):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(model_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    model_gradient_accumulator.reset()

  def _apply_classifier_gradients():
    variables = model.trainable_variables
    model_vars = []
    classifier_vars = []
    for var in variables:
      if "noisy_gate" in var.name:
        classifier_vars.append(var)
      elif "noisy_ADAP" in var.name:
        model_vars.append(var)
    grads_and_vars = []
    for gradient, variable in zip(classifier_gradient_accumulator.gradients, classifier_vars):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(classifier_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    classifier_gradient_accumulator.reset()

  @dataset_util.function_on_next(train_dataset)
  def _train_classifier_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_classifier_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples

  @dataset_util.function_on_next(train_dataset)
  def _train_model_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_model_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples

  @tf.function
  def _model_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_model_gradients)

  @tf.function
  def _classifier_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_classifier_gradients)

  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  # Runs the training loop.
  import time
  start = time.time()  
  train_model_data_flow = iter(_train_model_forward())
  train_classifier_data_flow = iter(_train_classifier_forward())

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _d_classfication_loss = []
  _number_examples = []

  step = optimizer.iterations.numpy()
  if config.get("reset_step",None):
    print("start from %d-th step"%config.get("reset_step",150000))
    optimizer.iterations.assign(config.get("reset_step",150000))
    step = optimizer.iterations.numpy()
  
  with _summary_writer.as_default():
    while True:
      if step < config.get("classifier_training_step",250000):
        classification_loss, num_examples = next(train_classifier_data_flow)    
        _d_classfication_loss.append(classification_loss)
        _number_examples.append(num_examples)
        _classifier_step()
      else:
        loss, num_examples = next(train_model_data_flow)  
        _loss.append(loss)  
        _number_examples.append(num_examples)
        _model_step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        if step < config.get("classifier_training_step",250000):
          elapsed = time.time() - start
          tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; d_classfication_loss = %f, number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_d_classfication_loss), np.sum(_number_examples), elapsed)
          _number_examples = []
          _d_classfication_loss = []
          start = time.time()
        else:
          elapsed = time.time() - start
          tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
          _loss = []
          _number_examples = []
          start = time.time()

      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)

      if step % eval_every == 0 and step > config.get("classifier_training_step",250000): 
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      tf.summary.flush()
      if step > train_steps:
        break
   
def translate_farajan(source_file,
              context_src_file,
              context_tgt_file,
              context_score,
              reference,
              model,
              config,
              optimizer,
              checkpoint_manager,
              checkpoint,              
              domain,
              output_file,
              length_penalty,
              is_noisy=1,
              checkpoint_path=None,
              probs_file=None,
              experiment="ldr",
              score_type="MultiBLEU",
              batch_size=5,
              beam_size=5):
  
  # Create the inference dataset.
  if checkpoint_path == None:
    checkpoint_path = checkpoint_manager.latest_checkpoint
  tf.get_logger().info("Evaluating model %s", checkpoint_path)
  print("In domain %d"%domain)
  checkpoint.restore(checkpoint_path)
  dataset = model.examples_inputter.make_inference_dataset(source_file, 1, domain)
  iterator = iter(dataset)
  if "baseline" in experiment:
    context_dataset = model.examples_inputter.make_training_dataset(context_src_file, context_tgt_file, batch_size=1, batch_type="example", single_pass=True)
  else:
    context_dataset = model.examples_inputter.make_training_dataset(context_src_file, context_tgt_file, 1, domain, batch_type="example", single_pass=True)
  context_iteration = iter(context_dataset)
  ids_to_tokens = model.labels_inputter.ids_to_tokens
  learning_rate = tf.Variable(config.get("farajan_lr",0.001),trainable=False)
  optimizer = tfa.optimizers.LazyAdam(learning_rate)
  model.create_variables(optimizer=optimizer)
  @tf.function(experimental_relax_shapes=True)
  def minifinetune(source, target, step_num):
    tf.print("context_src: ", source["tokens"], "context_target: ", target["tokens"])
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
    else:
      training_loss, _ = loss, loss        
    variables = model.trainable_variables
    gradients = optimizer.get_gradients(training_loss, variables)
    grads_and_vars = []
    step_num = config.get("farajan_steps",9)
    for gradient, variable in zip(gradients, variables):
      grads_and_vars.append((gradient, variable))
    
    for i in range(step_num):
      optimizer.apply_gradients(grads_and_vars)

  @tf.function
  def predict_next():    
    source = next(iterator)
    tf.print("source: ", source["tokens"])
    #tf.print("source: ", source, "src_context: ", context_src, "tgt_context: ", context_tgt)
    source_length = source["length"]
    batch_size = tf.shape(source_length)[0]
    source_inputs = model.features_inputter(source)
    if experiment in ["residual","residualv15","DRO","residualv25","residualv27","residualv28","residualv29","residual_big_transformer","residualv26","gated_residual_v5","residualv16","residualv19","residualv20","residualv21","residualv22","residualv23","residualv17","residualv18","residualv2","residualv1","residualv3","residualv5","residualv13","residualv12","residualv6","residualv7","residualv11","residualv8","residualv9","baselinev1"]:
      encoder_outputs, _, _ = model.encoder([source_inputs, source["domain"], source["is_noisy"]], source_length, training=False, internal_node_printing=True)
    else:
      encoder_outputs, _, _ = model.encoder(source_inputs, source_length, training=False)

    # Prepare the decoding strategy.
    if beam_size > 1:
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      source_length = tfa.seq2seq.tile_batch(source_length, beam_size)
      decoding_strategy = onmt.utils.BeamSearch(beam_size, length_penalty=length_penalty)
    else:
      decoding_strategy = onmt.utils.GreedySearch()

    # Run dynamic decoding.
    decoder_state = model.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=source_length)
    if experiment in ["residual","residualv2","DRO","residualv15","residualv25","residualv27","residual_big_transformer","residualv26","gated_residual_v5","residualv16","residualv19","residualv20","residualv21","residualv22","residualv23","residualv17","residualv18","residualv1","residualv3","residualv5","residualv6","residualv7","residualv13","residualv12","residualv11","residualv8","residualv9","baselinev1"]:
      map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}, training=False), tf.dtypes.cast(tf.fill(tf.expand_dims(tf.shape(ids)[0],0), domain), tf.int64)]
    elif experiment in ["DC"]:
      map_input_fn = lambda ids: model.labels_inputter({"ids": ids}, domain=domain, training=False)
    elif experiment in ["WDC"]:
      e_r, _ = model.classification_layer(encoder_outputs, source_length, training=False)
      e_s, _ = model.adv_classification_layer(encoder_outputs, source_length, training=False)
      g_s = model.share_gate(tf.concat([tf.tile(tf.expand_dims(e_s,1),[1,tf.shape(encoder_outputs)[1],1]),encoder_outputs],-1))
      g_r = model.specific_gate(tf.concat([tf.tile(tf.expand_dims(e_r,1),[1,tf.shape(encoder_outputs)[1],1]),encoder_outputs],-1))
      h_r = g_r * encoder_outputs
      h_s = g_s * encoder_outputs
      encoder_mask = model.encoder.build_mask(source_inputs, sequence_length=source_length)
      map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}, training=False), h_r, h_s, encoder_mask]
    elif experiment in ["residualv28","residualv29"]:
      map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}, training=False), source["domain"]]
    else:
      map_input_fn = lambda ids: model.labels_inputter({"ids": ids}, training=False)
    decoded = model.decoder.dynamic_decode(
        map_input_fn,
        tf.fill([batch_size], START_OF_SENTENCE_ID),
        end_id=END_OF_SENTENCE_ID,
        initial_state=decoder_state,
        decoding_strategy=decoding_strategy,
        maximum_iterations=250)
    target_lengths = decoded.lengths
    target_tokens = ids_to_tokens.lookup(tf.cast(decoded.ids, tf.int64)) 
    return target_tokens, target_lengths
  
  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    for snap, var in zip(snapshots, model.trainable_variables):
      _set_weight(var, snap)
  # Iterates on the dataset.

  print("output file: ", output_file)
  step = optimizer.iterations.numpy()
  f_score = open(context_score,"r")
  with open(output_file, "w") as output_:
    while True:    
      try:
        # save values
        snapshots = [v.value() for v in model.trainable_variables]
        #finetuning phase
        src, tgt = next(context_iteration)
        score = f_score.readline()
        if src["length"].numpy()>1:
          score = float(score)
          if 0.5 <= score and score <= 0.6:
            learning_rate.assign(0.001)
            step_num = 6
          elif 0.6 <= score and score <= 0.7:
            learning_rate.assign(0.001)
            step_num = 7
          elif 0.7 <= score and score <= 0.8:
            learning_rate.assign(0.001)
            step_num = 8
          elif 0.8 <= score and score <= 0.9:
            learning_rate.assign(0.001)
            step_num = 9
          elif 0.9 <= score and score <= 1.0:
            learning_rate.assign(0.001)
            step_num = 9
          minifinetune(src,tgt,tf.constant(step_num))
        #translating phase
        batch_tokens, batch_length = predict_next()
        #reset parameters
        weight_reset(snapshots)
        #reset step
        optimizer.iterations.assign(step)
        for tokens, length in zip(batch_tokens.numpy(), batch_length.numpy()):
          sentence = b" ".join(tokens[0][:length[0]])
          print_bytes(sentence, output_)
          #print_bytes(sentence)
      except tf.errors.OutOfRangeError:
        break
      except StopIteration: 
        break
  
  return 0

def score(source_file,
              translation_file,
              model,
              config,
              strategy,
              optimizer,
              checkpoint_manager,
              checkpoint,              
              domain,
              output_file,
              length_penalty,
              is_noisy=1,
              checkpoint_path=None,
              probs_file=None,
              experiment="ldr",
              score_type="MultiBLEU",
              batch_size=5,
              beam_size=5):
  
  # Create the inference dataset.
  if checkpoint_path == None:
    checkpoint_path = checkpoint_manager.latest_checkpoint
  tf.get_logger().info("Evaluating model %s", checkpoint_path)
  print("In domain %d"%domain)
  checkpoint.restore(checkpoint_path)

  dataset = model.examples_inputter.make_training_dataset(source_file, translation_file, batch_size=64, batch_type="example", single_pass=True)
  iteration = iter(dataset)
  ids_to_tokens = model.labels_inputter.ids_to_tokens
  model.create_variables()
  def translation_scoring():
    source,target=next(iteration)
    #tf.print("src: ", source["tokens"], "trans: ", target["tokens"])
    scores = model.score(source,target)
    return tf.nest.map_structure(lambda t: t.numpy(), scores)
  
  while True:    
    params = {"with_token_level": True, "with_alignments":None}
    try:
      results = translation_scoring()
      for batch in misc.extract_batches(results):
        model.print_score(batch, params=params)
    except tf.errors.OutOfRangeError:
      break
    except StopIteration:
      break
  
  return 0

def EWC_stat(source_file,
              reference,
              model,
              config,
              strategy,
              optimizer,
              checkpoint_manager,
              checkpoint,
              maximum_length=80,
              checkpoint_path=None):
  
  # Create the inference dataset.
  if checkpoint_path == None:
    checkpoint_path = checkpoint_manager.latest_checkpoint
  tf.get_logger().info("Evaluating model %s", checkpoint_path)
  checkpoint.restore(checkpoint_path)
  """ dataset = model.examples_inputter.make_training_dataset(source_file, reference, 1, 0, batch_type="example", single_pass=True, maximum_features_length=maximum_length,
                                maximum_labels_length=maximum_length) """
  batch_train_size = 1  
  batch_type = "examples"
  source_file = config["eval_src"]
  target_file = config["eval_tgt"]
  domain = config.get("domain",None)
  shuffle_buffer_size = 5000000
  dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), single_pass=True,
                                            multi_domain=config.get("multi_domain", True), picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))
  iterator = iter(dataset)
  model.create_variables(optimizer=optimizer)
  EWC_weights = []
  def star_vars_init():
    variables = model.trainable_variables
    for var in variables:
      EWC_weights.append(tf.Variable(tf.zeros_like(var), trainable=False))

  def EWC_accumulate(source, target):
    with tf.GradientTape() as tape:
      variables = model.trainable_variables
      tape.watch(variables)
      outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
      loss = model.compute_loss(outputs, target, training=True)

      if isinstance(loss, tuple):
        training_loss = loss[0] / loss[1]
      else:
        training_loss, _ = loss, loss   
      gradients = tape.gradient(training_loss, variables)
    for gradient, EWC_weight in zip(gradients, EWC_weights):
      EWC_accum(EWC_weight, gradient)

  star_vars_init()
  count = 0
  import time
  begin = time.time()
  while True:    
    try:
      source, target = next(iterator)
      EWC_accumulate(source, target)
      count +=1
      if count%1000==0:
        end = time.time()
        print(end-begin)
        begin = end
      #if count>12000:
      #  break
    except tf.errors.OutOfRangeError:
      break
    except StopIteration:
      break
  
  for w in EWC_weights:
    print(w/count)
  
  EWC_dict = dict()
  for v, EWC_weight in zip(model.trainable_variables, EWC_weights):
    EWC_dict[v.name] = EWC_weight/count
  print(EWC_dict)
  dir_name = os.path.dirname(checkpoint_path)
  np.savez(os.path.join(dir_name,"EWC_%s"%checkpoint_path.split("/")[-1]),**EWC_dict)
  
  return 0

def EWC_res_stat(source_file,
              reference,
              model,
              config,
              strategy,
              optimizer,
              checkpoint_manager,
              checkpoint,
              maximum_length=80,
              checkpoint_path=None):
  
  # Create the inference dataset.
  if checkpoint_path == None:
    checkpoint_path = checkpoint_manager.latest_checkpoint
  tf.get_logger().info("Evaluating model %s", checkpoint_path)
  checkpoint.restore(checkpoint_path)
  batch_train_size = 1  
  batch_type = "examples"
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  shuffle_buffer_size = 5000000
  dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))
  iterator = iter(dataset)
  model.create_variables(optimizer=optimizer)
  EWC_weights = []
  def star_vars_init():
    variables = []
    for var in model.trainable_variables:
      if "ADAP_" in var.name and "layer_norm" in var.name:
        variables.append(var)
    for var in variables:
        EWC_weights.append(tf.Variable(tf.zeros_like(var), trainable=False))
        var.assign(tf.zeros_like(var))

  def EWC_accumulate(source, target):
    with tf.GradientTape() as tape:
      variables = []
      for var in model.trainable_variables:
        if "ADAP_" in var.name and "layer_norm" in var.name:
          variables.append(var)
      tape.watch(variables)
      outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
      loss = model.compute_loss(outputs, target, training=True)

      if isinstance(loss, tuple):
        training_loss = loss[0] / loss[1]
      else:
        training_loss, _ = loss, loss   
      gradients = tape.gradient(training_loss, variables)
    for gradient, EWC_weight in zip(gradients, EWC_weights):
      EWC_accum(EWC_weight, gradient)

  star_vars_init()
  count = 0
  import time
  begin = time.time()
  while True:    
    try:
      source, target = next(iterator)
      EWC_accumulate(source, target)
      count +=1
      if count%1000==0:
        end = time.time()
        print(end-begin)
        begin = end
      if count>6000:
        break
    except tf.errors.OutOfRangeError:
      break
    except StopIteration:
      break
  
  for w in EWC_weights:
    print(w/count)
  
  EWC_dict = dict()
  for v, EWC_weight in zip(model.trainable_variables, EWC_weights):
    EWC_dict[v.name] = EWC_weight/count
  print(EWC_dict)
  dir_name = os.path.dirname(checkpoint_path)
  np.savez(os.path.join(dir_name,"EWC_%s"%checkpoint_path.split("/")[-1]),**EWC_dict)
  
  return 0

def translate_farajan_residual(source_file,
              context_src_file,
              context_tgt_file,
              reference,
              model,
              config,
              strategy,
              optimizer,
              checkpoint_manager,
              checkpoint,              
              domain,
              output_file,
              length_penalty,
              is_noisy=1,
              checkpoint_path=None,
              probs_file=None,
              experiment="ldr",
              score_type="MultiBLEU",
              batch_size=5,
              beam_size=5):
  
  # Create the inference dataset.
  if checkpoint_path == None:
    checkpoint_path = checkpoint_manager.latest_checkpoint
  tf.get_logger().info("Evaluating model %s", checkpoint_path)
  print("In domain %d"%domain)
  checkpoint.restore(checkpoint_path)
  dataset = model.examples_inputter.make_inference_dataset(source_file, 1, domain)
  iterator = iter(dataset)
  if "baseline" in experiment:
    context_dataset = model.examples_inputter.make_training_dataset(context_src_file, context_tgt_file, batch_size=1, batch_type="example")
  else:
    context_dataset = model.examples_inputter.make_training_dataset(context_src_file, context_tgt_file, 20, domain, batch_type="example", single_pass=True)
  context_iteration = iter(context_dataset)
  ids_to_tokens = model.labels_inputter.ids_to_tokens
  optimizer = tfa.optimizers.LazyAdam(config.get("farajan_lr",0.001))
  model.create_variables(optimizer=optimizer)
  @tf.function(experimental_relax_shapes=True)
  def minifinetune(source, target):
    tf.print("context_src: ", source["tokens"], "context_target: ", target["tokens"])
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
    else:
      training_loss, _ = loss, loss        
    #variables = model.trainable_variables
    variables = []
    for v in model.trainable_variables:
      if "ADAP_" in v.name or "ldr_embedding" in v.name or "ldr_inputter" in v.name:
        print(v.name)
        variables.append(v)
    #print("var numb: ", len(variables))
    gradients = optimizer.get_gradients(training_loss, variables)
    grads_and_vars = []
    for gradient, variable in zip(gradients, variables):
      grads_and_vars.append((gradient, variable))
    for i in range(config.get("farajan_steps",9)):
      optimizer.apply_gradients(grads_and_vars)

  @tf.function
  def predict_next():    
    source = next(iterator)
    tf.print("source: ", source["tokens"])
    #context_src, context_tgt = next(context_iteration)
    #tf.print("source: ", source, "src_context: ", context_src, "tgt_context: ", context_tgt)
    source_length = source["length"]
    batch_size = tf.shape(source_length)[0]
    source_inputs = model.features_inputter(source)
    if experiment in ["residual","residualv15","DRO","residualv25","residualv27","residualv28","residualv29","residual_big_transformer","residualv26","gated_residual_v5","residualv16","residualv19","residualv20","residualv21","residualv22","residualv23","residualv17","residualv18","residualv2","residualv1","residualv3","residualv5","residualv13","residualv12","residualv6","residualv7","residualv11","residualv8","residualv9","baselinev1"]:
      encoder_outputs, _, _ = model.encoder([source_inputs, source["domain"], source["is_noisy"]], source_length, training=False, internal_node_printing=True)
    else:
      encoder_outputs, _, _ = model.encoder(source_inputs, source_length, training=False)

    # Prepare the decoding strategy.
    if beam_size > 1:
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      source_length = tfa.seq2seq.tile_batch(source_length, beam_size)
      decoding_strategy = onmt.utils.BeamSearch(beam_size, length_penalty=length_penalty)
    else:
      decoding_strategy = onmt.utils.GreedySearch()

    # Run dynamic decoding.
    decoder_state = model.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=source_length)
    if experiment in ["residual","residualv2","DRO","residualv15","residualv25","residualv27","residual_big_transformer","residualv26","gated_residual_v5","residualv16","residualv19","residualv20","residualv21","residualv22","residualv23","residualv17","residualv18","residualv1","residualv3","residualv5","residualv6","residualv7","residualv13","residualv12","residualv11","residualv8","residualv9","baselinev1"]:
      map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}, training=False), tf.dtypes.cast(tf.fill(tf.expand_dims(tf.shape(ids)[0],0), domain), tf.int64)]
    elif experiment in ["DC"]:
      map_input_fn = lambda ids: model.labels_inputter({"ids": ids}, domain=domain, training=False)
    elif experiment in ["WDC"]:
      e_r, _ = model.classification_layer(encoder_outputs, source_length, training=False)
      e_s, _ = model.adv_classification_layer(encoder_outputs, source_length, training=False)
      g_s = model.share_gate(tf.concat([tf.tile(tf.expand_dims(e_s,1),[1,tf.shape(encoder_outputs)[1],1]),encoder_outputs],-1))
      g_r = model.specific_gate(tf.concat([tf.tile(tf.expand_dims(e_r,1),[1,tf.shape(encoder_outputs)[1],1]),encoder_outputs],-1))
      h_r = g_r * encoder_outputs
      h_s = g_s * encoder_outputs
      encoder_mask = model.encoder.build_mask(source_inputs, sequence_length=source_length)
      map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}, training=False), h_r, h_s, encoder_mask]
    elif experiment in ["residualv28","residualv29"]:
      map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}, training=False), source["domain"]]
    else:
      map_input_fn = lambda ids: model.labels_inputter({"ids": ids}, training=False)
    decoded = model.decoder.dynamic_decode(
        map_input_fn,
        tf.fill([batch_size], START_OF_SENTENCE_ID),
        end_id=END_OF_SENTENCE_ID,
        initial_state=decoder_state,
        decoding_strategy=decoding_strategy,
        maximum_iterations=250)
    target_lengths = decoded.lengths
    target_tokens = ids_to_tokens.lookup(tf.cast(decoded.ids, tf.int64)) 
    return target_tokens, target_lengths
  
  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    for snap, var in zip(snapshots, model.trainable_variables):
      _set_weight(var, snap)
  # Iterates on the dataset.

  print("output file: ", output_file)
  step = optimizer.iterations.numpy()
  with open(output_file, "w") as output_:
    while True:    
      try:
        # save values
        snapshots = [v.value() for v in model.trainable_variables]
        #finetuning phase
        src, tgt = next(context_iteration)
        if src["length"].numpy()>1:
          minifinetune(src,tgt)
        #translating phase
        batch_tokens, batch_length = predict_next()
        #reset parameters
        weight_reset(snapshots)
        #reset step
        optimizer.iterations.assign(step)
        for tokens, length in zip(batch_tokens.numpy(), batch_length.numpy()):
          sentence = b" ".join(tokens[0][:length[0]])
          print_bytes(sentence, output_)
          #print_bytes(sentence)
      except tf.errors.OutOfRangeError:
        break
  
  return 0

def train_NGD(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  if config.get("report_every",None)!=None:
    report_every = config.get("report_every")
  hessian_update_every = config.get("hessian_update_every",100)
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  else:
    if checkpoint_path is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
      checkpoint.restore(checkpoint_path)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_hessian_size = config.get("batch_hessian_size",10)
  print("batch_hessian_size: ", batch_hessian_size, "examples")
  print("batch_train_size: ", batch_train_size, batch_type)
  hessian_accum_step = config.get("hessian_accum_step",1)
  step = optimizer.iterations.numpy()
  print("current learning rate: ", learning_rate(step))
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  
  print("There are %d in-domain corpora"%len(source_file))
  
  train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=config.get("picking_prob",None), 
                                            temperature=config.get("temperature",1.0))
  hessian_datasets = create_training_dataset(strategy, model, domain, config.get("hessian_src", source_file), 
                                            config.get("hessian_ref", target_file), batch_hessian_size, "examples", shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=None, 
                                            temperature=config.get("temperature",1.0), pick_in_order=True)

  from utils.dataprocess import count_lines
  datasets_size = [count_lines(src) for src in source_file]
  importance_weights = [data_size/sum(datasets_size) for data_size in datasets_size]
  temperature=config.get("hessian_temperature",1.0)
  importance_weights = [w ** (temperature) for w in importance_weights]
  importance_weights = [w/sum(importance_weights) for w in importance_weights]
  
  ### update factore of diag hessians
  alpha = config.get("hessian_moving_rate",0.1)
  epsilon = config.get("posterior_weight",1e-8)
  print("epsilon: ", epsilon)
  print("hessian_moving_rate: ", alpha)
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator() 
    hessian_accumulators = optimizer_util.DiagHessianAccumulator()
    # hessian_accumulators = [tf.Variable(
    #         tf.zeros_like(var),
    #         trainable=False, synchronization=tf.VariableSynchronization.ON_READ) for var in model.trainable_variables]
    rescale_sum = tf.Variable(0.0, trainable=False, synchronization=tf.VariableSynchronization.ON_READ)
    hessian_moving_stats = [tf.Variable(
            tf.zeros_like(var),
            trainable=False, aggregation=tf.compat.v1.VariableAggregation.MEAN, synchronization=tf.VariableSynchronization.AUTO) for var in model.trainable_variables]
    normalized_hessian_moving_stats = [tf.Variable(
            tf.zeros_like(var),
            trainable=False, aggregation=tf.compat.v1.VariableAggregation.MEAN, synchronization=tf.VariableSynchronization.AUTO) for var in model.trainable_variables]
    importance_weights = tf.constant(importance_weights)
    tf.print("importance_weights: ", importance_weights)
  
  #########  
  def _accumulate_diag_hessians(source,target): 
    with tf.GradientTape(persistent=True) as tape:  
      variables = model.trainable_variables
      tape.watch(variables)
      outputs, _ = model(
          source,
          labels=target,
          training=False,
          step=optimizer.iterations,
          inference=False)
      _dom = source["domain"][0]
      loss = model.compute_individual_loss(outputs, target, training=True)
      def hessian_accum_along_loss(diag_hessian_acc, x):
        gradients = tape.gradient(x,variables)
        _hessians = []
        for grad in gradients:
          _hessians.append(tf.square(grad))
        hessian_accumulators(_hessians)
        return diag_hessian_acc
      tf.scan(hessian_accum_along_loss, loss, parallel_iterations=batch_hessian_size)
  def _accumulate_NGD_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    if config.get("ADAP_activity_regularizing",False):
      layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
      output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
      d_classification_gate_loss_scale = config.get("d_classification_gate_loss_scale",0.01)
      d_classifier_activity_regularization_loss_scale = config.get("d_classifier_activity_regularization_loss_scale",1.0)
      d_classifier_weight_regularization_losses_scale = config.get("d_classifier_weight_regularization_losses_scale",1.0)
      # print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
      # print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
      # print("d_classification_gate_loss_scale: ", d_classification_gate_loss_scale)
      # print("d_classifier_weight_regularization_losses_scale: ", d_classifier_weight_regularization_losses_scale)
      if isinstance(layer_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
        layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        #tf.print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale, "domain: ", domain)
      if isinstance(output_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
        output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
      regularization_losses = model.losses
      # print("model_name_scope", model.name_scope())
      # print(regularization_losses)
      layer_activity_regularization_losses = []
      output_activity_regularization_losses = []
      d_classification_gate_losses = []
      d_classifier_activity_regularization_losses = []
      d_classifier_weight_regularization_losses = []
      for loss_ in regularization_losses:
        if "multi_adap__dense" in loss_.name:
          output_activity_regularization_losses.append(loss_)
        elif "ADAP_gate" in loss_.name: #and "ActivityRegularizer" not in loss_.name and "Regularizer" not in loss_.name
          if "ActivityRegularizer" in loss_.name:
            d_classifier_activity_regularization_losses.append(loss_)
          elif "Regularizer" in loss_.name:
            d_classifier_weight_regularization_losses.append(loss_)
          else:
            d_classification_gate_losses.append(loss_)
        elif "ADAP_" in loss_.name:
          layer_activity_regularization_losses.append(loss_)

      # print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
      # print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
      # print("There are %d adaptation regularization loss on domain classification gate_____"%len(d_classification_gate_losses))
      # print("There are %d d_classifier_activity_regularization_losses"%len(d_classifier_activity_regularization_losses))
      # print("There are %d d_classifier_weight_regularization_losses"%len(d_classifier_weight_regularization_losses))
      if (len(layer_activity_regularization_losses)>0) and layer_activity_regularization_loss_scale>0:
        training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)

      if len(output_activity_regularization_losses)>0 and output_activity_regularization_loss_scale>0:
        training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)

      if len(d_classification_gate_losses)>0 and d_classification_gate_loss_scale>0:
        training_loss += d_classification_gate_loss_scale * tf.add_n(d_classification_gate_losses) / importance_weights[domain]

      if len(d_classifier_activity_regularization_losses)>0 and d_classifier_activity_regularization_loss_scale>0:
        training_loss += d_classifier_activity_regularization_loss_scale * tf.add_n(d_classifier_activity_regularization_losses)

      if len(d_classifier_weight_regularization_losses)>0 and d_classifier_weight_regularization_losses_scale>0:
        training_loss += d_classifier_weight_regularization_losses_scale * tf.add_n(d_classifier_weight_regularization_losses)
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    #for var in variables:
    #  print(var.name)
    gradients = optimizer.get_gradients(training_loss, variables)
    new_gradients = []
    rescale_sum.assign(0.0)
    for gradient, hessian_moving_stat, var in zip(gradients, normalized_hessian_moving_stats, variables):
      if isinstance(gradient,tf.IndexedSlices):
        if "embedding" in var.name:
          rescale_sum.assign_add(tf.reduce_sum(tf.square(gradient.values)/ (tf.nn.embedding_lookup(hessian_moving_stat, gradient.indices) + epsilon)))
        else:
          rescale_sum.assign_add(tf.reduce_sum(tf.square(gradient.values)/ epsilon))
        #tf.print("hessian %s: "%var.name, tf.nn.embedding_lookup(hessian_moving_stat.value(), gradient.indices), "indices: ", gradient.indices, sep="|")
        #tf.print("hessian_stat: ", hessian_moving_stat.value())
        #continue
      else:
        rescale_sum.assign_add(tf.reduce_sum(tf.square(gradient) / (hessian_moving_stat + epsilon)))
        #tf.print("hessian %s: "%var.name, hessian_moving_stat.value())
    #tf.print("rescale_sum: ", rescale_sum)
    for gradient, hessian_moving_stat, var in zip(gradients, normalized_hessian_moving_stats, variables):
      if isinstance(gradient,tf.IndexedSlices):
        # new_gradients.append(gradient)
        # new_gradients.append(tf.IndexedSlices(gradient.values / (tf.nn.embedding_lookup(hessian_moving_stat.value(), gradient.indices) + epsilon) 
        # * 1 / tf.sqrt(tf.reduce_sum(tf.square(gradient.values)/ (tf.nn.embedding_lookup(hessian_moving_stat.value(), gradient.indices) + epsilon))), 
        # gradient.indices, dense_shape=gradient.dense_shape))
        if "embedding" in var.name:
          new_gradients.append(tf.IndexedSlices(gradient.values / (tf.nn.embedding_lookup(hessian_moving_stat, gradient.indices) + epsilon) 
         * 1 / tf.sqrt(rescale_sum.value()), 
         gradient.indices, dense_shape=gradient.dense_shape))
        else:
          new_gradients.append(tf.IndexedSlices(gradient.values / epsilon * 1 / tf.sqrt(rescale_sum.value()), 
         gradient.indices, dense_shape=gradient.dense_shape))
        # tf.print("hessian_%s: "%var.name, tf.nn.embedding_lookup(hessian_moving_stat.value(), gradient.indices))
      else:
        # new_gradients.append(gradient / (hessian_moving_stat.value() +epsilon) * 1 / tf.sqrt(tf.reduce_sum(tf.square(gradient) / (hessian_moving_stat.value()+epsilon))))
        new_gradients.append(gradient / (hessian_moving_stat + epsilon) * 1 / tf.sqrt(rescale_sum.value()))
    gradient_accumulator(new_gradients)
    num_examples = tf.reduce_sum(target["length"])
    return reported_loss, num_examples
  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    if config.get("ADAP_activity_regularizing",False):
      layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
      output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
      d_classification_gate_loss_scale = config.get("d_classification_gate_loss_scale",0.01)
      d_classifier_activity_regularization_loss_scale = config.get("d_classifier_activity_regularization_loss_scale",1.0)
      d_classifier_weight_regularization_losses_scale = config.get("d_classifier_weight_regularization_losses_scale",1.0)
      # print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
      # print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
      # print("d_classification_gate_loss_scale: ", d_classification_gate_loss_scale)
      # print("d_classifier_weight_regularization_losses_scale: ", d_classifier_weight_regularization_losses_scale)
      if isinstance(layer_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
        layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        #tf.print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale, "domain: ", domain)
      if isinstance(output_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
        output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
      regularization_losses = model.losses
      # print("model_name_scope", model.name_scope())
      # print(regularization_losses)
      layer_activity_regularization_losses = []
      output_activity_regularization_losses = []
      d_classification_gate_losses = []
      d_classifier_activity_regularization_losses = []
      d_classifier_weight_regularization_losses = []
      for loss_ in regularization_losses:
        if "multi_adap__dense" in loss_.name:
          output_activity_regularization_losses.append(loss_)
        elif "ADAP_gate" in loss_.name: #and "ActivityRegularizer" not in loss_.name and "Regularizer" not in loss_.name
          if "ActivityRegularizer" in loss_.name:
            d_classifier_activity_regularization_losses.append(loss_)
          elif "Regularizer" in loss_.name:
            d_classifier_weight_regularization_losses.append(loss_)
          else:
            d_classification_gate_losses.append(loss_)
        elif "ADAP_" in loss_.name:
          layer_activity_regularization_losses.append(loss_)

      # print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
      # print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
      # print("There are %d adaptation regularization loss on domain classification gate_____"%len(d_classification_gate_losses))
      # print("There are %d d_classifier_activity_regularization_losses"%len(d_classifier_activity_regularization_losses))
      # print("There are %d d_classifier_weight_regularization_losses"%len(d_classifier_weight_regularization_losses))
      if (len(layer_activity_regularization_losses)>0) and layer_activity_regularization_loss_scale>0:
        training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)

      if len(output_activity_regularization_losses)>0 and output_activity_regularization_loss_scale>0:
        training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)

      if len(d_classification_gate_losses)>0 and d_classification_gate_loss_scale>0:
        training_loss += d_classification_gate_loss_scale * tf.add_n(d_classification_gate_losses) / importance_weights[domain]

      if len(d_classifier_activity_regularization_losses)>0 and d_classifier_activity_regularization_loss_scale>0:
        training_loss += d_classifier_activity_regularization_loss_scale * tf.add_n(d_classifier_activity_regularization_losses)

      if len(d_classifier_weight_regularization_losses)>0 and d_classifier_weight_regularization_losses_scale>0:
        training_loss += d_classifier_weight_regularization_losses_scale * tf.add_n(d_classifier_weight_regularization_losses)
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    #for var in variables:
    #  print(var.name)
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    return reported_loss, num_examples
  
  #########
  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
  def update_hessian_moving_stats():
    for accum, stat in zip(hessian_accumulators.hessians, hessian_moving_stats):
      stat.assign(accum / tf.cast(hessian_accum_step * batch_hessian_size, tf.float32))
    for hessian, normalized_hessian in zip(hessian_moving_stats, normalized_hessian_moving_stats):
      normalized_hessian.assign(hessian/tf.reduce_sum(hessian))
    
  #########
  @dataset_util.function_on_next(train_dataset)
  def _NGD_train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_NGD_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples
  @dataset_util.function_on_next(hessian_datasets)
  def _hessian_acc_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      strategy.experimental_run_v2(
          _accumulate_diag_hessians, args=(per_replica_source, per_replica_target))
  ##########
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  @tf.function
  def _hessian_stats_update_step():
    strategy.experimental_run_v2(update_hessian_moving_stats)
  ##########
  # Runs the training loop.
  import time
  start = time.time()  
  NGD_train_data_flow = iter(_NGD_train_forward())
  _hessian_accumulator_flow = iter(_hessian_acc_forward())
  train_data_flow = iter(_train_forward())
  _, _ = next(train_data_flow)
  last_eval = [0.0] * len(domain)
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _number_examples = []
  
  score_type = config.get("score_type","MultiBLEU")
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  ref_eval_concat = file_concatenate(config["eval_ref"],"ref_eval_concat",dir_name=os.path.join(config["model_dir"],"eval"))
  
  new_picking_prob = [1.0/len(domain)] * len(domain)
  overfitting = [False] * (len(domain))
  if step>0:
    checkpoint_path = checkpoint_manager.latest_checkpoint
    tf.summary.experimental.set_step(step)
    output_files = []
    eval_scores = []
    for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
        output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
        score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
        tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
        output_files.append(output_file)
        eval_scores.append(score)
    ##### BLEU on concat dev set.
    output_file_concat = file_concatenate(output_files,"output_file_concat.%s"%os.path.basename(checkpoint_path))
    score = scorer(ref_eval_concat, output_file_concat)
    print("score of model %s on concat dev set: "%checkpoint_manager.latest_checkpoint, score)
    tf.summary.scalar("concat_eval_score", score, description="BLEU on concat dev set")
    if config.get("dynamic_domain_batch",False):
      ##### check overfitting
      for i in range(len(domain)):
        if new_picking_prob[i] > 2*datasets_size[i]/float(sum(datasets_size)) and last_eval[i] > eval_scores[i]:
          overfitting[i] = True
          print("Domain %d overfitted"%i)
        else:
          overfitting[i] = False
        last_eval[i] = eval_scores[i]
      #############################
      tf.summary.flush()
      target_scores = config.get("eval_target_scores",None)
      achivement_percentage = [1-e/float(t) for e,t in zip(eval_scores, target_scores)]
      new_picking_prob = [p/sum(achivement_percentage) for p in achivement_percentage]
      new_picking_prob = [p if not overfitted else p/3.0 for p, overfitted, data_size in zip(new_picking_prob, overfitting, datasets_size)]
      new_picking_prob = [p/sum(new_picking_prob) for p in new_picking_prob]
      print("new_picking_prob: ", new_picking_prob)
      train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                          maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                          multi_domain=config.get("multi_domain", True), picking_prob= new_picking_prob, 
                                          temperature=0.5)
      @dataset_util.function_on_next(train_dataset)
      def _NGD_train_forward(next_fn):    
        with strategy.scope():
          per_replica_source, per_replica_target = next_fn()
          per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
              _accumulate_NGD_gradients, args=(per_replica_source, per_replica_target))
          # TODO: these reductions could be delayed until _step is called.
          loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
          num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
        return loss, num_examples
      NGD_train_data_flow = iter(_NGD_train_forward())

  with _summary_writer.as_default():
    while True:
      #####Training batch
      if step % hessian_update_every == 0 and step >= config.get("NGD_warm_start",0):
        for i in range(hessian_accum_step):
          next(_hessian_accumulator_flow)
        _hessian_stats_update_step()
      if step >= config.get("NGD_warm_start",0):
        loss, num_examples = next(NGD_train_data_flow)    
        _loss.append(loss)
        _number_examples.append(num_examples)
      else:
        loss, num_examples = next(train_data_flow)    
        _loss.append(loss)
        _number_examples.append(num_examples)
      _step()  
      step = optimizer.iterations.numpy()
      
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
          "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
          step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        output_files = []
        eval_scores = []
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
            output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
            score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
            tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
            output_files.append(output_file)
            eval_scores.append(score)
        ##### BLEU on concat dev set.
        output_file_concat = file_concatenate(output_files,"output_file_concat.%s"%os.path.basename(checkpoint_path))
        score = scorer(ref_eval_concat, output_file_concat)
        print("score of model %s on concat dev set: "%checkpoint_manager.latest_checkpoint, score)
        tf.summary.scalar("concat_eval_score", score, description="BLEU on concat dev set")
        #############################
        tf.summary.flush()
        if config.get("dynamic_domain_batch",False):
          ##### check overfitting
          for i in range(len(domain)):
            if new_picking_prob[i] > 2*datasets_size[i]/float(sum(datasets_size)) and last_eval[i] > eval_scores[i]:
              overfitting[i] = True
              print("Domain %d overfitted"%i)
            else:
              overfitting[i] = False
            last_eval[i] = eval_scores[i]
          target_scores = config.get("eval_target_scores",None)
          achivement_percentage = [1-e/float(t) for e,t in zip(eval_scores, target_scores)]
          new_picking_prob = [p/sum(achivement_percentage) for p in achivement_percentage]
          new_picking_prob = [p if not overfitted else p/3.0 for p, overfitted, data_size in zip(new_picking_prob, overfitting, datasets_size)]
          new_picking_prob = [p/sum(new_picking_prob) for p in new_picking_prob]
          print("new_picking_prob: ", new_picking_prob)
          train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                              maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                              multi_domain=config.get("multi_domain", True), picking_prob= new_picking_prob, 
                                              temperature=0.5)
          @dataset_util.function_on_next(train_dataset)
          def _NGD_train_forward(next_fn):    
            with strategy.scope():
              per_replica_source, per_replica_target = next_fn()
              per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
                  _accumulate_NGD_gradients, args=(per_replica_source, per_replica_target))
              # TODO: these reductions could be delayed until _step is called.
              loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
              num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
            return loss, num_examples
          NGD_train_data_flow = iter(_NGD_train_forward())
      if step > train_steps:
        break

def continue_NGD(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  if config.get("report_every",None)!=None:
    report_every = config.get("report_every")
  hessian_update_every = config.get("hessian_update_every",100)
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  else:
    if checkpoint_path is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
      checkpoint.restore(checkpoint_path)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_hessian_size = config.get("batch_hessian_size",10)
  print("batch_hessian_size: ", batch_hessian_size, "examples")
  print("batch_train_size: ", batch_train_size, batch_type)
  hessian_accum_step = config.get("hessian_accum_step",1)
  step = optimizer.iterations.numpy()
  print("current learning rate: ", learning_rate(step))
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  
  print("There are %d in-domain corpora"%len(source_file))
  
  train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=config.get("picking_prob",None), 
                                            temperature=config.get("temperature",1.0))
  hessian_datasets = create_training_dataset(strategy, model, domain, config["previous_src"], config["previous_tgt"] , batch_hessian_size, "examples", shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=None, 
                                            temperature=config.get("temperature",1.0), pick_in_order=True)

  from utils.dataprocess import count_lines
  datasets_size = [count_lines(src) for src in source_file]
  importance_weights = [data_size/sum(datasets_size) for data_size in datasets_size]
  temperature=config.get("hessian_temperature",1.0)
  importance_weights = [w ** (temperature) for w in importance_weights]
  importance_weights = [w/sum(importance_weights) for w in importance_weights]
  
  ### update factore of diag hessians
  alpha = config.get("hessian_moving_rate",0.1)
  epsilon = config.get("posterior_weight",1e-8)
  print("epsilon: ", epsilon)
  print("hessian_moving_rate: ", alpha)
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator() 
    hessian_accumulators = optimizer_util.DiagHessianAccumulator()
    # hessian_accumulators = [tf.Variable(
    #         tf.zeros_like(var),
    #         trainable=False, synchronization=tf.VariableSynchronization.ON_READ) for var in model.trainable_variables]
    rescale_sum = tf.Variable(0.0, trainable=False, synchronization=tf.VariableSynchronization.ON_READ)
    hessian_moving_stats = [tf.Variable(
            tf.zeros_like(var),
            trainable=False, aggregation=tf.compat.v1.VariableAggregation.MEAN, synchronization=tf.VariableSynchronization.AUTO) for var in model.trainable_variables]
    normalized_hessian_moving_stats = [tf.Variable(
            tf.zeros_like(var),
            trainable=False, aggregation=tf.compat.v1.VariableAggregation.MEAN, synchronization=tf.VariableSynchronization.AUTO) for var in model.trainable_variables]
    importance_weights = tf.constant(importance_weights)
    tf.print("importance_weights: ", importance_weights)
  #########  
  def _accumulate_diag_hessians(source,target): 
    with tf.GradientTape(persistent=True) as tape:  
      variables = model.trainable_variables
      tape.watch(variables)
      outputs, _ = model(
          source,
          labels=target,
          training=True,
          step=optimizer.iterations)
      _dom = source["domain"][0]
      loss = model.compute_individual_loss(outputs, target, training=True)
      def hessian_accum_along_loss(diag_hessian_acc, x):
        gradients = tape.gradient(x,variables)
        _hessians = []
        for grad in gradients:
          _hessians.append(tf.square(grad))
        hessian_accumulators(_hessians)
        return diag_hessian_acc
      tf.scan(hessian_accum_along_loss, loss, parallel_iterations=batch_hessian_size)
  def _accumulate_NGD_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    if config.get("ADAP_activity_regularizing",False):
      layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
      output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
      d_classification_gate_loss_scale = config.get("d_classification_gate_loss_scale",0.01)
      d_classifier_activity_regularization_loss_scale = config.get("d_classifier_activity_regularization_loss_scale",1.0)
      d_classifier_weight_regularization_losses_scale = config.get("d_classifier_weight_regularization_losses_scale",1.0)
      # print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
      # print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
      # print("d_classification_gate_loss_scale: ", d_classification_gate_loss_scale)
      # print("d_classifier_weight_regularization_losses_scale: ", d_classifier_weight_regularization_losses_scale)
      if isinstance(layer_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
        layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        #tf.print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale, "domain: ", domain)
      if isinstance(output_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
        output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
      regularization_losses = model.losses
      # print("model_name_scope", model.name_scope())
      # print(regularization_losses)
      layer_activity_regularization_losses = []
      output_activity_regularization_losses = []
      d_classification_gate_losses = []
      d_classifier_activity_regularization_losses = []
      d_classifier_weight_regularization_losses = []
      for loss_ in regularization_losses:
        if "multi_adap__dense" in loss_.name:
          output_activity_regularization_losses.append(loss_)
        elif "ADAP_gate" in loss_.name: #and "ActivityRegularizer" not in loss_.name and "Regularizer" not in loss_.name
          if "ActivityRegularizer" in loss_.name:
            d_classifier_activity_regularization_losses.append(loss_)
          elif "Regularizer" in loss_.name:
            d_classifier_weight_regularization_losses.append(loss_)
          else:
            d_classification_gate_losses.append(loss_)
        elif "ADAP_" in loss_.name:
          layer_activity_regularization_losses.append(loss_)

      # print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
      # print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
      # print("There are %d adaptation regularization loss on domain classification gate_____"%len(d_classification_gate_losses))
      # print("There are %d d_classifier_activity_regularization_losses"%len(d_classifier_activity_regularization_losses))
      # print("There are %d d_classifier_weight_regularization_losses"%len(d_classifier_weight_regularization_losses))
      if (len(layer_activity_regularization_losses)>0) and layer_activity_regularization_loss_scale>0:
        training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)

      if len(output_activity_regularization_losses)>0 and output_activity_regularization_loss_scale>0:
        training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)

      if len(d_classification_gate_losses)>0 and d_classification_gate_loss_scale>0:
        training_loss += d_classification_gate_loss_scale * tf.add_n(d_classification_gate_losses) / importance_weights[domain]

      if len(d_classifier_activity_regularization_losses)>0 and d_classifier_activity_regularization_loss_scale>0:
        training_loss += d_classifier_activity_regularization_loss_scale * tf.add_n(d_classifier_activity_regularization_losses)

      if len(d_classifier_weight_regularization_losses)>0 and d_classifier_weight_regularization_losses_scale>0:
        training_loss += d_classifier_weight_regularization_losses_scale * tf.add_n(d_classifier_weight_regularization_losses)
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    #for var in variables:
    #  print(var.name)
    gradients = optimizer.get_gradients(training_loss, variables)
    new_gradients = []
    rescale_sum.assign(0.0)
    for gradient, hessian_moving_stat, var in zip(gradients, normalized_hessian_moving_stats, variables):
      if isinstance(gradient,tf.IndexedSlices):
        if "embedding" in var.name:
          rescale_sum.assign_add(tf.reduce_sum(tf.square(gradient.values)/ (tf.nn.embedding_lookup(hessian_moving_stat, gradient.indices) + epsilon)))
        else:
          rescale_sum.assign_add(tf.reduce_sum(tf.square(gradient.values)/ epsilon))
        #tf.print("hessian %s: "%var.name, tf.nn.embedding_lookup(hessian_moving_stat.value(), gradient.indices), "indices: ", gradient.indices, sep="|")
        #tf.print("hessian_stat: ", hessian_moving_stat.value())
        #continue
      else:
        rescale_sum.assign_add(tf.reduce_sum(tf.square(gradient) / (hessian_moving_stat + epsilon)))
        #tf.print("hessian %s: "%var.name, hessian_moving_stat.value())
    #tf.print("rescale_sum: ", rescale_sum)
    for gradient, hessian_moving_stat, var in zip(gradients, normalized_hessian_moving_stats, variables):
      if isinstance(gradient,tf.IndexedSlices):
        # new_gradients.append(gradient)
        # new_gradients.append(tf.IndexedSlices(gradient.values / (tf.nn.embedding_lookup(hessian_moving_stat.value(), gradient.indices) + epsilon) 
        # * 1 / tf.sqrt(tf.reduce_sum(tf.square(gradient.values)/ (tf.nn.embedding_lookup(hessian_moving_stat.value(), gradient.indices) + epsilon))), 
        # gradient.indices, dense_shape=gradient.dense_shape))
        if "embedding" in var.name:
          new_gradients.append(tf.IndexedSlices(gradient.values / (tf.nn.embedding_lookup(hessian_moving_stat, gradient.indices) + epsilon) 
         * 1 / tf.sqrt(rescale_sum.value()), 
         gradient.indices, dense_shape=gradient.dense_shape))
        else:
          new_gradients.append(tf.IndexedSlices(gradient.values / epsilon * 1 / tf.sqrt(rescale_sum.value()), 
         gradient.indices, dense_shape=gradient.dense_shape))
        # tf.print("hessian_%s: "%var.name, tf.nn.embedding_lookup(hessian_moving_stat.value(), gradient.indices))
      else:
        # new_gradients.append(gradient / (hessian_moving_stat.value() +epsilon) * 1 / tf.sqrt(tf.reduce_sum(tf.square(gradient) / (hessian_moving_stat.value()+epsilon))))
        new_gradients.append(gradient / (hessian_moving_stat + epsilon) * 1 / tf.sqrt(rescale_sum.value()))
    gradient_accumulator(new_gradients)
    num_examples = tf.reduce_sum(target["length"])
    return reported_loss, num_examples
  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    if config.get("ADAP_activity_regularizing",False):
      layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
      output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
      d_classification_gate_loss_scale = config.get("d_classification_gate_loss_scale",0.01)
      d_classifier_activity_regularization_loss_scale = config.get("d_classifier_activity_regularization_loss_scale",1.0)
      d_classifier_weight_regularization_losses_scale = config.get("d_classifier_weight_regularization_losses_scale",1.0)
      # print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
      # print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
      # print("d_classification_gate_loss_scale: ", d_classification_gate_loss_scale)
      # print("d_classifier_weight_regularization_losses_scale: ", d_classifier_weight_regularization_losses_scale)
      if isinstance(layer_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
        layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        #tf.print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale, "domain: ", domain)
      if isinstance(output_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
        output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
      regularization_losses = model.losses
      # print("model_name_scope", model.name_scope())
      # print(regularization_losses)
      layer_activity_regularization_losses = []
      output_activity_regularization_losses = []
      d_classification_gate_losses = []
      d_classifier_activity_regularization_losses = []
      d_classifier_weight_regularization_losses = []
      for loss_ in regularization_losses:
        if "multi_adap__dense" in loss_.name:
          output_activity_regularization_losses.append(loss_)
        elif "ADAP_gate" in loss_.name: #and "ActivityRegularizer" not in loss_.name and "Regularizer" not in loss_.name
          if "ActivityRegularizer" in loss_.name:
            d_classifier_activity_regularization_losses.append(loss_)
          elif "Regularizer" in loss_.name:
            d_classifier_weight_regularization_losses.append(loss_)
          else:
            d_classification_gate_losses.append(loss_)
        elif "ADAP_" in loss_.name:
          layer_activity_regularization_losses.append(loss_)

      # print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
      # print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
      # print("There are %d adaptation regularization loss on domain classification gate_____"%len(d_classification_gate_losses))
      # print("There are %d d_classifier_activity_regularization_losses"%len(d_classifier_activity_regularization_losses))
      # print("There are %d d_classifier_weight_regularization_losses"%len(d_classifier_weight_regularization_losses))
      if (len(layer_activity_regularization_losses)>0) and layer_activity_regularization_loss_scale>0:
        training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)

      if len(output_activity_regularization_losses)>0 and output_activity_regularization_loss_scale>0:
        training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)

      if len(d_classification_gate_losses)>0 and d_classification_gate_loss_scale>0:
        training_loss += d_classification_gate_loss_scale * tf.add_n(d_classification_gate_losses) / importance_weights[domain]

      if len(d_classifier_activity_regularization_losses)>0 and d_classifier_activity_regularization_loss_scale>0:
        training_loss += d_classifier_activity_regularization_loss_scale * tf.add_n(d_classifier_activity_regularization_losses)

      if len(d_classifier_weight_regularization_losses)>0 and d_classifier_weight_regularization_losses_scale>0:
        training_loss += d_classifier_weight_regularization_losses_scale * tf.add_n(d_classifier_weight_regularization_losses)
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    #for var in variables:
    #  print(var.name)
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    return reported_loss, num_examples
  #########
  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
  def update_hessian_moving_stats():
    for accum, stat in zip(hessian_accumulators.hessians, hessian_moving_stats):
      stat.assign(accum / tf.cast(hessian_accum_step * batch_hessian_size, tf.float32))
    for hessian, normalized_hessian in zip(hessian_moving_stats, normalized_hessian_moving_stats):
      normalized_hessian.assign(hessian/tf.reduce_sum(hessian))
    
  #########
  @dataset_util.function_on_next(train_dataset)
  def _NGD_train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_NGD_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples
  @dataset_util.function_on_next(hessian_datasets)
  def _hessian_acc_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      strategy.experimental_run_v2(
          _accumulate_diag_hessians, args=(per_replica_source, per_replica_target))
  ##########
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  @tf.function
  def _hessian_stats_update_step():
    strategy.experimental_run_v2(update_hessian_moving_stats)
  ##########
  # Runs the training loop.
  import time
  start = time.time()  
  NGD_train_data_flow = iter(_NGD_train_forward())
  _hessian_accumulator_flow = iter(_hessian_acc_forward())
  train_data_flow = iter(_train_forward())
  _, _ = next(train_data_flow)
  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _number_examples = []
  
  score_type = config.get("score_type","MultiBLEU")
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  ref_eval_concat = file_concatenate(config["eval_ref"],"ref_eval_concat",dir_name=os.path.join(config["model_dir"],"eval"))

  with _summary_writer.as_default():
    while True:
      #####Training batch
      if step % hessian_update_every == 0 and step >= config.get("NGD_warm_start",0):
        for i in range(hessian_accum_step):
          next(_hessian_accumulator_flow)
        _hessian_stats_update_step()
      
      if step >= config.get("NGD_warm_start",0):
        loss, num_examples = next(NGD_train_data_flow)    
        _loss.append(loss)
        _number_examples.append(num_examples)
      else:
        loss, num_examples = next(train_data_flow)    
        _loss.append(loss)
        _number_examples.append(num_examples)
      _step()  
      step = optimizer.iterations.numpy()

      # if step % report_every == 0:
      #   for h, n_h, var in zip(hessian_moving_stats, normalized_hessian_moving_stats, model.trainable_variables):
      #       #print("hessian %s: "%var.name, h)
      #       print("normalized hessian %s: "%var.name, tf.reduce_sum(tf.square(n_h)))
      #   #break
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
          "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
          step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        output_files = []
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
            output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
            score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
            tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
            output_files.append(output_file)
        ##### BLEU on concat dev set.
        output_file_concat = file_concatenate(output_files,"output_file_concat.%s"%os.path.basename(checkpoint_path))
        score = scorer(ref_eval_concat, output_file_concat)
        print("score of model %s on concat dev set: "%checkpoint_manager.latest_checkpoint, score)
        tf.summary.scalar("concat_eval_score", score, description="BLEU on concat dev set")
        #############################
        tf.summary.flush()
      if step > train_steps:
        break

def debug_NGD(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  if config.get("report_every",None)!=None:
    report_every = config.get("report_every")
  hessian_update_every = config.get("hessian_update_every",100)
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  else:
    if checkpoint_path is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
      checkpoint.restore(checkpoint_path)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_hessian_size = config.get("batch_hessian_size",10)
  print("batch_hessian_size: ", batch_hessian_size, "examples")
  print("batch_train_size: ", batch_train_size, batch_type)
  hessian_accum_step = config.get("hessian_accum_step",1)
  step = optimizer.iterations.numpy()
  print("current learning rate: ", learning_rate(step))
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  
  print("There are %d in-domain corpora"%len(source_file))
  
  train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=config.get("picking_prob",None), 
                                            temperature=config.get("temperature",1.0))
  hessian_datasets = create_training_dataset(strategy, model, domain, source_file, target_file, batch_hessian_size, "examples", shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=None, 
                                            temperature=config.get("temperature",1.0), pick_in_order=True)

  from utils.dataprocess import count_lines
  datasets_size = [count_lines(src) for src in source_file]
  importance_weights = [data_size/sum(datasets_size) for data_size in datasets_size]
  temperature=config.get("temperature",1.0)
  importance_weights = [w ** temperature for w in importance_weights]
  importance_weights = [w/sum(importance_weights) for w in importance_weights]
  importance_weights = tf.constant(importance_weights)
  tf.print("importance_weights: ", importance_weights)
  ### update factore of diag hessians
  alpha = config.get("hessian_moving_rate",0.1)
  epsilon = config.get("posterior_weight",1e-8)
  print("epsilon: ", epsilon)
  print("hessian_moving_rate: ", alpha)
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator() 
    hessian_accumulators = optimizer_util.DiagHessianAccumulator()
    # hessian_accumulators = [tf.Variable(
    #         tf.zeros_like(var),
    #         trainable=False, synchronization=tf.VariableSynchronization.ON_READ) for var in model.trainable_variables]
    rescale_sum = tf.Variable(0.0, trainable=False, synchronization=tf.VariableSynchronization.ON_READ)
    hessian_moving_stats = [tf.Variable(
            tf.zeros_like(var),
            trainable=False, synchronization=tf.VariableSynchronization.ON_READ) for var in model.trainable_variables]
    normalized_hessian_moving_stats = [tf.Variable(
            tf.zeros_like(var),
            trainable=False, aggregation=tf.compat.v1.VariableAggregation.MEAN, synchronization=tf.VariableSynchronization.AUTO) for var in model.trainable_variables]

  #########  
  def _accumulate_diag_hessians(source,target): 
    with tf.GradientTape(persistent=True) as tape:  
      variables = model.trainable_variables
      tape.watch(variables)
      outputs, _ = model(
          source,
          labels=target,
          training=True,
          step=optimizer.iterations)
      loss = model.compute_individual_loss(outputs, target, training=True)
      def hessian_accum_along_loss(diag_hessian_acc, x):
        gradients = tape.gradient(x,variables)
        _hessians = []
        for grad in gradients:
          _hessians.append(tf.square(grad))
        hessian_accumulators(_hessians)
        return diag_hessian_acc
      tf.scan(hessian_accum_along_loss, loss, parallel_iterations=batch_hessian_size)
  def _accumulate_NGD_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    if config.get("ADAP_activity_regularizing",False):
      layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
      output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
      d_classification_gate_loss_scale = config.get("d_classification_gate_loss_scale",0.01)
      d_classifier_activity_regularization_loss_scale = config.get("d_classifier_activity_regularization_loss_scale",1.0)
      d_classifier_weight_regularization_losses_scale = config.get("d_classifier_weight_regularization_losses_scale",1.0)
      # print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
      # print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
      # print("d_classification_gate_loss_scale: ", d_classification_gate_loss_scale)
      # print("d_classifier_weight_regularization_losses_scale: ", d_classifier_weight_regularization_losses_scale)
      if isinstance(layer_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
        layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        #tf.print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale, "domain: ", domain)
      if isinstance(output_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
        output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
      regularization_losses = model.losses
      # print("model_name_scope", model.name_scope())
      # print(regularization_losses)
      layer_activity_regularization_losses = []
      output_activity_regularization_losses = []
      d_classification_gate_losses = []
      d_classifier_activity_regularization_losses = []
      d_classifier_weight_regularization_losses = []
      for loss_ in regularization_losses:
        if "multi_adap__dense" in loss_.name:
          output_activity_regularization_losses.append(loss_)
        elif "ADAP_gate" in loss_.name: #and "ActivityRegularizer" not in loss_.name and "Regularizer" not in loss_.name
          if "ActivityRegularizer" in loss_.name:
            d_classifier_activity_regularization_losses.append(loss_)
          elif "Regularizer" in loss_.name:
            d_classifier_weight_regularization_losses.append(loss_)
          else:
            d_classification_gate_losses.append(loss_)
        elif "ADAP_" in loss_.name:
          layer_activity_regularization_losses.append(loss_)

      # print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
      # print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
      # print("There are %d adaptation regularization loss on domain classification gate_____"%len(d_classification_gate_losses))
      # print("There are %d d_classifier_activity_regularization_losses"%len(d_classifier_activity_regularization_losses))
      # print("There are %d d_classifier_weight_regularization_losses"%len(d_classifier_weight_regularization_losses))
      if (len(layer_activity_regularization_losses)>0) and layer_activity_regularization_loss_scale>0:
        training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)

      if len(output_activity_regularization_losses)>0 and output_activity_regularization_loss_scale>0:
        training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)

      if len(d_classification_gate_losses)>0 and d_classification_gate_loss_scale>0:
        training_loss += d_classification_gate_loss_scale * tf.add_n(d_classification_gate_losses) / importance_weights[domain]

      if len(d_classifier_activity_regularization_losses)>0 and d_classifier_activity_regularization_loss_scale>0:
        training_loss += d_classifier_activity_regularization_loss_scale * tf.add_n(d_classifier_activity_regularization_losses)

      if len(d_classifier_weight_regularization_losses)>0 and d_classifier_weight_regularization_losses_scale>0:
        training_loss += d_classifier_weight_regularization_losses_scale * tf.add_n(d_classifier_weight_regularization_losses)
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    #for var in variables:
    #  print(var.name)
    gradients = optimizer.get_gradients(training_loss, variables)
    new_gradients = []
    rescale_sum.assign(0.0)
    for gradient, hessian_moving_stat, var in zip(gradients, normalized_hessian_moving_stats, variables):
      if isinstance(gradient,tf.IndexedSlices):
        rescale_sum.assign_add(tf.reduce_sum(tf.square(gradient.values)/ (tf.nn.embedding_lookup(hessian_moving_stat, gradient.indices) + epsilon)))
        #tf.print("hessian %s: "%var.name, tf.nn.embedding_lookup(hessian_moving_stat.value(), gradient.indices), "indices: ", gradient.indices, sep="|")
        #tf.print("hessian_stat: ", hessian_moving_stat.value())
      else:
        rescale_sum.assign_add(tf.reduce_sum(tf.square(gradient) / (hessian_moving_stat + epsilon)))
        #tf.print("hessian %s: "%var.name, hessian_moving_stat.value())
    #tf.print("rescale_sum: ", rescale_sum)
    for gradient, hessian_moving_stat, var in zip(gradients, normalized_hessian_moving_stats, variables):
      if isinstance(gradient,tf.IndexedSlices):
        # new_gradients.append(gradient)
        # new_gradients.append(tf.IndexedSlices(gradient.values / (tf.nn.embedding_lookup(hessian_moving_stat.value(), gradient.indices) + epsilon) 
        # * 1 / tf.sqrt(tf.reduce_sum(tf.square(gradient.values)/ (tf.nn.embedding_lookup(hessian_moving_stat.value(), gradient.indices) + epsilon))), 
        # gradient.indices, dense_shape=gradient.dense_shape))
        new_gradients.append(tf.IndexedSlices(gradient.values / (tf.nn.embedding_lookup(hessian_moving_stat, gradient.indices) + epsilon) 
         * 1 / tf.sqrt(rescale_sum.value()), 
         gradient.indices, dense_shape=gradient.dense_shape))
        # tf.print("hessian_%s: "%var.name, tf.nn.embedding_lookup(hessian_moving_stat.value(), gradient.indices))
      else:
        # new_gradients.append(gradient / (hessian_moving_stat.value() +epsilon) * 1 / tf.sqrt(tf.reduce_sum(tf.square(gradient) / (hessian_moving_stat.value()+epsilon))))
        new_gradients.append(gradient / (hessian_moving_stat + epsilon) * 1 / tf.sqrt(rescale_sum.value()))
    gradient_accumulator(new_gradients)
    num_examples = tf.reduce_sum(target["length"])
    return reported_loss, num_examples
  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    if config.get("ADAP_activity_regularizing",False):
      layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
      output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
      d_classification_gate_loss_scale = config.get("d_classification_gate_loss_scale",0.01)
      d_classifier_activity_regularization_loss_scale = config.get("d_classifier_activity_regularization_loss_scale",1.0)
      d_classifier_weight_regularization_losses_scale = config.get("d_classifier_weight_regularization_losses_scale",1.0)
      # print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
      # print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
      # print("d_classification_gate_loss_scale: ", d_classification_gate_loss_scale)
      # print("d_classifier_weight_regularization_losses_scale: ", d_classifier_weight_regularization_losses_scale)
      if isinstance(layer_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
        layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        #tf.print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale, "domain: ", domain)
      if isinstance(output_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
        output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
      regularization_losses = model.losses
      # print("model_name_scope", model.name_scope())
      # print(regularization_losses)
      layer_activity_regularization_losses = []
      output_activity_regularization_losses = []
      d_classification_gate_losses = []
      d_classifier_activity_regularization_losses = []
      d_classifier_weight_regularization_losses = []
      for loss_ in regularization_losses:
        if "multi_adap__dense" in loss_.name:
          output_activity_regularization_losses.append(loss_)
        elif "ADAP_gate" in loss_.name: #and "ActivityRegularizer" not in loss_.name and "Regularizer" not in loss_.name
          if "ActivityRegularizer" in loss_.name:
            d_classifier_activity_regularization_losses.append(loss_)
          elif "Regularizer" in loss_.name:
            d_classifier_weight_regularization_losses.append(loss_)
          else:
            d_classification_gate_losses.append(loss_)
        elif "ADAP_" in loss_.name:
          layer_activity_regularization_losses.append(loss_)

      # print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
      # print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
      # print("There are %d adaptation regularization loss on domain classification gate_____"%len(d_classification_gate_losses))
      # print("There are %d d_classifier_activity_regularization_losses"%len(d_classifier_activity_regularization_losses))
      # print("There are %d d_classifier_weight_regularization_losses"%len(d_classifier_weight_regularization_losses))
      if (len(layer_activity_regularization_losses)>0) and layer_activity_regularization_loss_scale>0:
        training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)

      if len(output_activity_regularization_losses)>0 and output_activity_regularization_loss_scale>0:
        training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)

      if len(d_classification_gate_losses)>0 and d_classification_gate_loss_scale>0:
        training_loss += d_classification_gate_loss_scale * tf.add_n(d_classification_gate_losses) / importance_weights[domain]

      if len(d_classifier_activity_regularization_losses)>0 and d_classifier_activity_regularization_loss_scale>0:
        training_loss += d_classifier_activity_regularization_loss_scale * tf.add_n(d_classifier_activity_regularization_losses)

      if len(d_classifier_weight_regularization_losses)>0 and d_classifier_weight_regularization_losses_scale>0:
        training_loss += d_classifier_weight_regularization_losses_scale * tf.add_n(d_classifier_weight_regularization_losses)
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    #for var in variables:
    #  print(var.name)
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    return reported_loss, num_examples
  #########
  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
  def update_hessian_moving_stats():
    for accum, stat in zip(hessian_accumulators.hessians, hessian_moving_stats):
      stat.assign(accum / tf.cast(hessian_accum_step * batch_hessian_size, tf.float32))
    for accum, normalized_accum in zip(hessian_moving_stats, normalized_hessian_moving_stats):
      normalized_accum.assign(accum.value()/tf.reduce_sum(accum.value()))
  #########
  @dataset_util.function_on_next(train_dataset)
  def _NGD_train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_NGD_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples
  @dataset_util.function_on_next(hessian_datasets)
  def _hessian_acc_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      strategy.experimental_run_v2(
          _accumulate_diag_hessians, args=(per_replica_source, per_replica_target))
  ##########
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  @tf.function
  def _hessian_stats_update_step():
    strategy.experimental_run_v2(update_hessian_moving_stats)
  ##########
  # Runs the training loop.
  import time
  start = time.time()  
  NGD_train_data_flow = iter(_NGD_train_forward())
  _hessian_accumulator_flow = iter(_hessian_acc_forward())
  train_data_flow = iter(_train_forward())
  _, _ = next(train_data_flow)
  
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _number_examples = []
      
  for _ in range(3):    
    for i in range(hessian_accum_step):
      next(_hessian_accumulator_flow)
    _hessian_stats_update_step()
    loss, num_examples = next(NGD_train_data_flow)    
    _loss.append(loss)
    _number_examples.append(num_examples)
    _step()  
    step = optimizer.iterations.numpy()

    if step % report_every == 0:
      for h, n_h, var in zip(hessian_moving_stats, normalized_hessian_moving_stats, model.trainable_variables):
        #print("hessian %s: "%var.name, h)
        print("normalized hessian %s: "%var.name, tf.reduce_sum(tf.square(n_h)))
    if step % report_every == 0:
      elapsed = time.time() - start
      tf.get_logger().info(
        "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
        step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
      _loss = []
      _number_examples = []
      start = time.time()
          
  # with _summary_writer.as_default():
  #   while True:
  #     #####Training batch
  #     if step % hessian_update_every == 0 and step >= config.get("NGD_warm_start",0):
  #       for i in range(hessian_accum_step):
  #         next(_hessian_accumulator_flow)
  #       _hessian_stats_update_step()
      
  #     if step >= config.get("NGD_warm_start",0):
  #       loss, num_examples = next(NGD_train_data_flow)    
  #       _loss.append(loss)
  #       _number_examples.append(num_examples)
  #     else:
  #       loss, num_examples = next(train_data_flow)    
  #       _loss.append(loss)
  #       _number_examples.append(num_examples)
  #     _step()  
  #     step = optimizer.iterations.numpy()

  #     if step % report_every == 0:
  #       for h, n_h, var in zip(hessian_moving_stats, normalized_hessian_moving_stats, model.trainable_variables):
  #           #print("hessian %s: "%var.name, h)
  #           print("normalized hessian %s: "%var.name, tf.reduce_sum(tf.square(n_h)))
  #       #break
  #     if step % report_every == 0:
  #       elapsed = time.time() - start
  #       tf.get_logger().info(
  #         "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
  #         step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
  #       _loss = []
  #       _number_examples = []
  #       start = time.time()
  #     if step % save_every == 0:
  #       tf.get_logger().info("Saving checkpoint for step %d", step)
  #       checkpoint_manager.save(checkpoint_number=step)
  #     if step % eval_every == 0:
  #       checkpoint_path = checkpoint_manager.latest_checkpoint
  #       tf.summary.experimental.set_step(step)
  #       output_files = []
  #       for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
  #           output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
  #           score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
  #           tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
  #           output_files.append(output_file)
  #       ##### BLEU on concat dev set.
  #       output_file_concat = file_concatenate(output_files,"output_file_concat.%s"%os.path.basename(checkpoint_path))
  #       score = scorer(ref_eval_concat, output_file_concat)
  #       print("score of model %s on concat dev set: "%checkpoint_manager.latest_checkpoint, score)
  #       tf.summary.scalar("concat_eval_score", score, description="BLEU on concat dev set")
  #       #############################
  #       tf.summary.flush()
  #     if step > train_steps:
  #       break
      
def train_L2W(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          adapter_optimizer=None,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  redistribute_every = config.get("redistribute_every",2000)
  #####
  if checkpoint_path is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
    checkpoint.restore(checkpoint_path)
  else:
    if checkpoint_manager.latest_checkpoint is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
      checkpoint.restore(checkpoint_manager.latest_checkpoint)    
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  eval_domain = config.get("eval_domain")
  ###### early stopping criterion
  current_max_eval_bleu = 0.0
  descending_streak = 0
  ######
  if not config.get("domain_importances",None):
    domain_importances = [1.0/len(eval_domain)]*len(eval_domain)
  else:
    domain_importances = config.get("domain_importances")
  print("There are %d in-domain corpora"%len(source_file))
  ###############
  # train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
  #                                           maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
  #                                           multi_domain=config.get("multi_domain", True), picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))

  print("maximum_length", maximum_length)
  train_datasets_p = [] 
  datasets_size = [count_lines(src) for src in source_file]
  picking_prob = [data_size/sum(datasets_size) for data_size in datasets_size]
  if config.get("picking_prob",None):
    picking_prob = config.get("picking_prob",None)
  print("initial domain picking probability: ", picking_prob)
  for i,src,tgt in zip(domain, source_file, target_file):
    train_datasets_p.append(model.examples_inputter.make_training_dataset(src, tgt,
            batch_size=batch_train_size,
            batch_type=batch_type,
            domain=i,
            single_pass=False,
            shuffle_buffer_size=shuffle_buffer_size,
            length_bucket_width=1,  # Bucketize sequences by the same length for efficiency.
            maximum_features_length=maximum_length,
            maximum_labels_length=maximum_length))
  train_dataset = tf.data.experimental.sample_from_datasets(train_datasets_p, weights=picking_prob)
  with strategy.scope():
    base_dataset = train_dataset
    train_dataset = strategy.experimental_distribute_datasets_from_function(
          lambda _: base_dataset)  
  #############
  train_datasets = [create_training_dataset(strategy, model, [domain], [source_file], [target_file], batch_train_size//2, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=None, temperature=config.get("temperature",1.0))
                                            for domain, source_file, target_file in zip(config.get("domain"), config.get("src"), config.get("tgt"))]

  dev_datasets = [create_training_dataset(strategy, model, [domain], [source_file], [target_file], batch_train_size//2, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=None, temperature=config.get("temperature",1.0))
                                            for domain, source_file, target_file in zip(config.get("eval_domain"), config.get("eval_src"), config.get("eval_ref"))]
  #############
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
    sub_gradient_accumulator = optimizer_util.GradientAccumulator()
    #dev_gradient_accumulators = [optimizer_util.GradientAccumulator() for _ in domain]
    #train_gradient_accumulators = [optimizer_util.GradientAccumulator() for _ in domain]
    dev_gradient_accumulator = optimizer_util.GradientAccumulator()
    train_gradient_accumulator = optimizer_util.GradientAccumulator()
    domain_rewards = tf.Variable([0.0]*len(domain), trainable=False, aggregation=tf.compat.v1.VariableAggregation.MEAN, synchronization=tf.VariableSynchronization.AUTO)
    #domain_logits = tf.Variable([0.0]*len(domain), trainable=True)
    #d_logits_grad_accumulator = optimizer_util.GradientAccumulator()
    #domain_importances = tf.Variable(domain_importances, trainable=False, aggregation=tf.compat.v1.VariableAggregation.MEAN, synchronization=tf.VariableSynchronization.AUTO)
    #sampler_optimizer = tf.keras.optimizers.Adam(learning_rate=config.get("sampler_optim_lr",0.01))
    #sampler_vars = [domain_logits]
    #sampler_optimizer._create_slots(sampler_vars)
  domain_logits = tf.Variable([0.0]*len(domain), trainable=True)
  grad_domain_logits_accum = tf.Variable(tf.zeros_like(domain_logits), trainable=False)
  sampler_optimizer = tf.keras.optimizers.Adam(learning_rate=config.get("sampler_optim_lr",0.01))
  #sampler_vars = [domain_logits]
  print("domain_rewards: ", domain_rewards)
  print("domain_importances: ", domain_importances)
  
  @tf.function
  def _grad_sampler_accum():
    loss = - tf.reduce_sum(tf.stop_gradient(tf.nn.softmax(domain_logits)) * tf.nn.log_softmax(domain_logits) * domain_rewards)
    if config.get("sampler_entropy_constraint",False):
      print("sampler_entropy_constraint_weight",config.get("sampler_entropy_constraint_weight",1e-3))
      loss +=  tf.reduce_sum(config.get("sampler_entropy_constraint_weight",1e-3) * tf.nn.log_softmax(domain_logits) * tf.nn.softmax(domain_logits))
    grad = sampler_optimizer.get_gradients(loss,[domain_logits])
    grad_domain_logits_accum.assign_add(grad[0])
    return tf.reduce_sum(tf.stop_gradient(tf.nn.softmax(domain_logits)) * domain_rewards)

  @tf.function
  def _sampler_step_1():
    sampler_optimizer.apply_gradients([(grad_domain_logits_accum, domain_logits)])
    grad_domain_logits_accum.assign(tf.zeros_like(domain_logits))

  def update_sampling_distribution(logits):
    logits = logits.numpy()
    for i, l in enumerate(logits):
        if logits[i] < 0:
            logits[i] = 0
    if sum(logits) == 0:
        logits = [0.1 for _ in range(len(logits))]
    p = np.array(logits) / sum(logits)
    print("new domain probs")
    print(p)
    return p

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    if config.get("ADAP_activity_regularizing",False):
      if experiment=="residualv28":
        layer_activity_regularization_losses = []
        output_activity_regularization_losses = []
        regularization_losses = model.losses
        for loss_ in regularization_losses:
          if "multi_adap__dense" in loss_.name:
            output_activity_regularization_losses.append(loss_)
          else:
            layer_activity_regularization_losses.append(loss_)
        layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
        if len(layer_activity_regularization_losses)>0:
          #print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
          training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)
      else:
        layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
        output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
        d_classification_gate_loss_scale = config.get("d_classification_gate_loss_scale",0.01)
        d_classifier_activity_regularization_loss_scale = config.get("d_classifier_activity_regularization_loss_scale",1.0)
        d_classifier_weight_regularization_losses_scale = config.get("d_classifier_weight_regularization_losses_scale",1.0)
        # print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
        # print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
        # print("d_classification_gate_loss_scale: ", d_classification_gate_loss_scale)
        # print("d_classifier_weight_regularization_losses_scale: ", d_classifier_weight_regularization_losses_scale)
        if isinstance(layer_activity_regularization_loss_scale, list):
          domain = source["domain"][0]
          layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
          layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
          #tf.print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale, "domain: ", domain)
        if isinstance(output_activity_regularization_loss_scale, list):
          domain = source["domain"][0]
          output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
          output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
        regularization_losses = model.losses
        # print("model_name_scope", model.name_scope())
        # print(regularization_losses)
        layer_activity_regularization_losses = []
        output_activity_regularization_losses = []
        d_classification_gate_losses = []
        d_classifier_activity_regularization_losses = []
        d_classifier_weight_regularization_losses = []
        for loss_ in regularization_losses:
          if "multi_adap__dense" in loss_.name:
            output_activity_regularization_losses.append(loss_)
          elif "ADAP_gate" in loss_.name: #and "ActivityRegularizer" not in loss_.name and "Regularizer" not in loss_.name
            if "ActivityRegularizer" in loss_.name:
              d_classifier_activity_regularization_losses.append(loss_)
            elif "Regularizer" in loss_.name:
              d_classifier_weight_regularization_losses.append(loss_)
            else:
              d_classification_gate_losses.append(loss_)
          elif "ADAP_" in loss_.name:
            layer_activity_regularization_losses.append(loss_)

        # print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
        # print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
        # print("There are %d adaptation regularization loss on domain classification gate_____"%len(d_classification_gate_losses))
        # print("There are %d d_classifier_activity_regularization_losses"%len(d_classifier_activity_regularization_losses))
        # print("There are %d d_classifier_weight_regularization_losses"%len(d_classifier_weight_regularization_losses))
        if (len(layer_activity_regularization_losses)>0) and layer_activity_regularization_loss_scale>0:
          training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)

        if len(output_activity_regularization_losses)>0 and output_activity_regularization_loss_scale>0:
          training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)

        # if len(d_classification_gate_losses)>0 and d_classification_gate_loss_scale>0:
        #   training_loss += d_classification_gate_loss_scale * tf.add_n(d_classification_gate_losses) / importance_weights[domain]

        if len(d_classifier_activity_regularization_losses)>0 and d_classifier_activity_regularization_loss_scale>0:
          training_loss += d_classifier_activity_regularization_loss_scale * tf.add_n(d_classifier_activity_regularization_losses)

        if len(d_classifier_weight_regularization_losses)>0 and d_classifier_weight_regularization_losses_scale>0:
          training_loss += d_classifier_weight_regularization_losses_scale * tf.add_n(d_classifier_weight_regularization_losses)
    variables = model.trainable_variables
    #print("var numb: ", len(variables))
    """ for var in variables:
      print(var.name) """
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

  def _accumulate_dev_train_gradients(source, target):
    with tf.GradientTape() as tape:
      variables = model.trainable_variables    
      tape.watch(variables)
      outputs, _ = model(
          source,
          labels=target,
          training=True,
          step=optimizer.iterations)
      loss = model.compute_loss(outputs, target, training=True)

      if isinstance(loss, tuple):
        training_loss = loss[0] / loss[1]
        reported_loss = loss[0] / loss[2]
      else:
        training_loss, reported_loss = loss, loss

      gradients = tape.gradient(training_loss, variables)
      sub_gradient_accumulator(gradients)
      return training_loss
  
  def _reset_dev_train_gradients():
    dev_gradient_accumulator.reset() # for dev_gradient_accumulator in dev_gradient_accumulators]
    train_gradient_accumulator.reset() #for train_gradient_accumulator in train_gradient_accumulators]

  def _reset_sub_gradients():
    sub_gradient_accumulator.reset()

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
  
  def _apply_dev_train_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(sub_gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(sub_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    sub_gradient_accumulator.reset()
 
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples

  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  @tf.function
  def _dev_train_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_dev_train_gradients)

  @tf.function
  def _reset_dev_train_grad_accum_step():
    with strategy.scope():
      _reset_dev_train_gradients()

  @tf.function
  def _reset_sub_grad_accum_step():
    with strategy.scope():
      _reset_sub_gradients()
  
  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  dev_iterators = [iter(dev_dataset) for dev_dataset in dev_datasets]
  train_iterators = [iter(train_dataset) for train_dataset in train_datasets]
 
  _, _ = next(train_data_flow)

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _number_examples = []
  step = optimizer.iterations.numpy()
  if config.get("reset_step",None):
    print("start from %d-th step"%config.get("reset_step",150000))
    optimizer.iterations.assign(config.get("reset_step",150000))
  
  if step <= 1:
    initializer = config.get("initializer","default")
    if initializer == "default":
      print("Initializing variables by tensorflow default")      
    elif initializer == "variance_scaling":
      print("Initializing variables by tf.variance_scaling")
      initial_value = []
      for v in model.trainable_variables:
        shape = tf.shape(v).numpy()
        initial_value.append(variance_scaling_initialier(shape, scale=1.0, mode="fan_avg", distribution="uniform"))
      weight_reset(initial_value)   
  else:
    print("current domain_logits", config.get("domain_logits",[0.0]*len(domain)))
    domain_logits.assign(config.get("domain_logits",[0.0]*len(domain)))
    # compute domain rewards
    rewards = [0.0] * len(domain)
    snapshots = [v.value() for v in model.trainable_variables]
    saved_step = optimizer.iterations.numpy()
    #######
    current_probs = tf.nn.softmax(domain_logits).numpy()
    print("current_probs: ", current_probs)
    #######
    ##### compute theta_t+1
    for k in np.random.choice(domain,config.get("update_theta_train_batch_per_run_num",len(domain)),p=current_probs): 
      src, tgt = next(train_iterators[k])
      strategy.experimental_run_v2(_accumulate_dev_train_gradients, args=(src, tgt))
    strategy.experimental_run_v2(_apply_dev_train_gradients)
    snapshots_1 = [v.value() for v in model.trainable_variables]
    for i, train_iter in enumerate(train_iterators):
      _reward = 0.0
      ##### accumulate gradient over training set of src domain i at theta_t
      weight_reset(snapshots)
      with strategy.scope():
        for _ in range(config.get("train_batch_per_run_num",10)):
          src, tgt = next(train_iter)
          strategy.experimental_run_v2(_accumulate_dev_train_gradients, args=(src, tgt))
        train_gradient_accumulator(sub_gradient_accumulator.gradients)
        strategy.experimental_run_v2(sub_gradient_accumulator.reset)
      ##### accumulate gradient over dev set of k tgt domains at theta_t+1
      weight_reset(snapshots_1)
      with strategy.scope():
        for j, dev_iter in enumerate(dev_iterators):
          _sum = 0.0
          _dev_norm = 0.0
          _tr_norm = 0.0
          for _ in range(config.get("dev_batch_per_run_num",10)):
            src, tgt = next(dev_iter)
            strategy.experimental_run_v2(_accumulate_dev_train_gradients, args=(src, tgt))
          dev_gradient_accumulator(sub_gradient_accumulator.gradients)
          strategy.experimental_run_v2(sub_gradient_accumulator.reset)         
          for dev_grad, tr_grad in zip(dev_gradient_accumulator.gradients, train_gradient_accumulator.gradients):
            _sum += tf.reduce_sum(dev_grad * tr_grad)
            _dev_norm += tf.reduce_sum(dev_grad * dev_grad)
            _tr_norm += tf.reduce_sum(tr_grad * tr_grad)
          _reward += _sum / (tf.sqrt(_dev_norm * _tr_norm) + 1e-10) * domain_importances[j]
          # reset dev gradient accumulations to zero
          strategy.experimental_run_v2(dev_gradient_accumulator.reset)
          #print(dev_gradient_accumulator.gradients[0])
        # reset train dev gradient accumulations to zero
        strategy.experimental_run_v2(train_gradient_accumulator.reset)
        #print(sub_gradient_accumulator.gradients[0])
        #print(train_gradient_accumulator.gradients[0])
      #_reward /= len(domain)
      rewards[i] = _reward.numpy()
      # reset model parameters
      weight_reset(snapshots)
      optimizer.iterations.assign(saved_step)
    domain_rewards.assign(tf.constant(rewards))
    # compute new domain distribution
    print("domain rewards", domain_rewards)
    for _ in range(config.get("domain_sampler_optim_step", 30)):
      #loss = _sampler_flow()
      #_sampler_step()
      _ = _grad_sampler_accum()
      _sampler_step_1()
      
    print("domain_logits: ", domain_logits.numpy())
    probs = tf.nn.softmax(domain_logits)
    new_picking_prob = update_sampling_distribution(probs)
    tf.summary.experimental.set_step(saved_step)
    for i in range(len(domain)):
      tf.summary.scalar("reward_%d"%i, rewards[i], description="reward of using training set %d"%(i))
      tf.summary.scalar("domain_prob_%d"%i, new_picking_prob[i], description="probability of using training set %d"%(i))
    tf.summary.flush()
    # create new training course with updated domain distribution
    train_dataset = tf.data.experimental.sample_from_datasets(train_datasets_p, weights=new_picking_prob)
    with strategy.scope():
      base_dataset = train_dataset
      train_dataset = strategy.experimental_distribute_datasets_from_function(
            lambda _: base_dataset)
    @dataset_util.function_on_next(train_dataset)
    def _train_forward(next_fn):    
      with strategy.scope():
        per_replica_source, per_replica_target = next_fn()
        per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
            _accumulate_gradients, args=(per_replica_source, per_replica_target))
        # TODO: these reductions could be delayed until _step is called.
        loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
        num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
      return loss, num_examples
    train_data_flow = iter(_train_forward())
    #######
    weight_reset(snapshots)
    optimizer.iterations.assign(saved_step)
    #######

  if config.get("continual_learning", False):
    print("Continual Learning needs to load from old model")
    assert config.get("checkpoint_path") != None
    checkpoint_path = config.get("checkpoint_path")
    load_and_update_if_needed_from_ckpt(config["model_dir"],   
                        checkpoint_path,                        
                        trackables={"model":model},
                        vocab_update=True,
                        model_key="model")

  score_type = config.get("score_type","MultiBLEU")
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  ref_eval_concat = file_concatenate(config["eval_ref"],"ref_eval_concat",dir_name=os.path.join(config["model_dir"],"eval"))

  with _summary_writer.as_default():
    while True:
      ####Training batch
      loss, num_examples = next(train_data_flow)    
      _loss.append(loss.numpy())
      _number_examples.append(num_examples.numpy())
      _step()  
      step = optimizer.iterations.numpy()
      
      if step % redistribute_every == 0 and step > config.get("warm_start",5000):
        # compute domain rewards
        rewards = [0.0] * len(domain)
        snapshots = [v.value() for v in model.trainable_variables]
        saved_step = optimizer.iterations.numpy()
        #######
        current_probs = tf.nn.softmax(domain_logits).numpy()
        print("current_probs: ", current_probs)
        #######
        ##### compute theta_t+1
        for k in np.random.choice(domain,config.get("update_theta_train_batch_per_run_num",len(domain)),p=current_probs): 
          src, tgt = next(train_iterators[k])
          strategy.experimental_run_v2(_accumulate_dev_train_gradients, args=(src, tgt))
        strategy.experimental_run_v2(_apply_dev_train_gradients)
        snapshots_1 = [v.value() for v in model.trainable_variables]
        for i, train_iter in enumerate(train_iterators):
          _reward = 0.0
          ##### accumulate gradient over training set of src domain i at theta_t
          weight_reset(snapshots)
          with strategy.scope():
            for _ in range(config.get("train_batch_per_run_num",10)):
              src, tgt = next(train_iter)
              #print("domain of training set %d is %d"%(i,src["domain"].values[0].numpy()[0]))
              strategy.experimental_run_v2(_accumulate_dev_train_gradients, args=(src, tgt))
            train_gradient_accumulator(sub_gradient_accumulator.gradients)
            strategy.experimental_run_v2(sub_gradient_accumulator.reset)
          ##### accumulate gradient over dev set of k tgt domains at theta_t+1
          weight_reset(snapshots_1)
          with strategy.scope():
            for j, dev_iter in enumerate(dev_iterators):
              _sum = 0.0
              _dev_norm = 0.0
              _tr_norm = 0.0
              for _ in range(config.get("dev_batch_per_run_num",10)):
                src, tgt = next(dev_iter)
                #print("domain of dev set %d is %d"%(j,src["domain"].values[0].numpy()[0]))
                strategy.experimental_run_v2(_accumulate_dev_train_gradients, args=(src, tgt))
              dev_gradient_accumulator(sub_gradient_accumulator.gradients)
              strategy.experimental_run_v2(sub_gradient_accumulator.reset)         
              for dev_grad, tr_grad, var in zip(dev_gradient_accumulator.gradients, train_gradient_accumulator.gradients, model.trainable_variables):
                if True:#"ADAP_" not in var.name:
                  _sum += tf.reduce_sum(dev_grad * tr_grad)
                  _dev_norm += tf.reduce_sum(dev_grad * dev_grad)
                  _tr_norm += tf.reduce_sum(tr_grad * tr_grad)
              if config.get("cosine_reward",True):
                _reward += _sum / (tf.sqrt(_dev_norm * _tr_norm) + 1e-10) * domain_importances[j]
              else:
                _reward += _sum * learning_rate(saved_step) * domain_importances[j]
              #print("reward of using training set %d to improve dev set %d: %f"%(i,j, _sum / (tf.sqrt(_dev_norm * _tr_norm) + 1e-10) * domain_importances[j]))
              # reset dev gradient accumulations to zero
              strategy.experimental_run_v2(dev_gradient_accumulator.reset)
              #print(dev_gradient_accumulator.gradients[0])
            # reset train dev gradient accumulations to zero
            strategy.experimental_run_v2(train_gradient_accumulator.reset)
            #print(sub_gradient_accumulator.gradients[0])
            #print(train_gradient_accumulator.gradients[0])
          #_reward /= len(domain)
          rewards[i] = _reward.numpy()
          # reset model parameters
          weight_reset(snapshots)
          optimizer.iterations.assign(saved_step)
        domain_rewards.assign(tf.constant(rewards))
        # compute new domain distribution
        print("domain rewards", domain_rewards)
        for _ in range(config.get("domain_sampler_optim_step", 30)):
          _ = _grad_sampler_accum()
          _sampler_step_1()
          
        print("domain_logits: ", domain_logits.numpy())
        probs = tf.nn.softmax(domain_logits)
        new_picking_prob = update_sampling_distribution(probs)
        tf.summary.experimental.set_step(saved_step)
        for i in range(len(domain)):
          tf.summary.scalar("reward_%d"%i, rewards[i], description="reward of using training set %d"%(i))
          tf.summary.scalar("domain_prob_%d"%i, new_picking_prob[i], description="probability of using training set %d"%(i))
        tf.summary.flush()
        # create new training course with updated domain distribution
        train_dataset = tf.data.experimental.sample_from_datasets(train_datasets_p, weights=new_picking_prob)
        with strategy.scope():
          base_dataset = train_dataset
          train_dataset = strategy.experimental_distribute_datasets_from_function(
                lambda _: base_dataset)
        @dataset_util.function_on_next(train_dataset)
        def _train_forward(next_fn):    
          with strategy.scope():
            per_replica_source, per_replica_target = next_fn()
            per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
                _accumulate_gradients, args=(per_replica_source, per_replica_target))
            # TODO: these reductions could be delayed until _step is called.
            loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
            num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
          return loss, num_examples
        train_data_flow = iter(_train_forward())
        #######
        weight_reset(snapshots)
        optimizer.iterations.assign(saved_step)
        #######

      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
          "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
          step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        output_files = []
        new_bleu = 0.0
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
          output_files.append(output_file)
          new_bleu += score * domain_importances[i]
        ##### BLEU on concat dev set.
        output_file_concat = file_concatenate(output_files,"output_file_concat.%s"%os.path.basename(checkpoint_path))
        score = scorer(ref_eval_concat, output_file_concat)
        print("score of model %s on concat dev set: "%checkpoint_manager.latest_checkpoint, score)
        tf.summary.scalar("concat_eval_score", score, description="BLEU on concat dev set")
        #############################
        if new_bleu >= current_max_eval_bleu:
          current_max_eval_bleu = new_bleu
          descending_streak = 0
        else:
          descending_streak += 1
      if descending_streak >= 5:
        break
      tf.summary.flush()
      if step > train_steps:
        break

def train_NGD_L2W(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  if config.get("report_every",None)!=None:
    report_every = config.get("report_every")
  hessian_update_every = config.get("hessian_update_every",100)
  redistribute_every = config.get("redistribute_every",2000)
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  else:
    if checkpoint_path is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
      checkpoint.restore(checkpoint_path)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_hessian_size = config.get("batch_hessian_size",10)
  print("batch_hessian_size: ", batch_hessian_size, "examples")
  print("batch_train_size: ", batch_train_size, batch_type)
  hessian_accum_step = config.get("hessian_accum_step",1)
  step = optimizer.iterations.numpy()
  print("current learning rate: ", learning_rate(step))
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  eval_domain = config.get("eval_domain")
  ###### early stopping criterion
  current_max_eval_bleu = 0.0
  descending_streak = 0
  ######
  if not config.get("domain_importances",None):
    domain_importances = [1.0/len(eval_domain)]*len(eval_domain)
  else:
    domain_importances = config.get("domain_importances")
  print("There are %d in-domain corpora"%len(source_file))
  ###############
  print("maximum_length", maximum_length)
  train_datasets_p = [] 
  datasets_size = [count_lines(src) for src in source_file]
  picking_prob = [data_size/sum(datasets_size) for data_size in datasets_size]
  if config.get("picking_prob",None):
    picking_prob = config.get("picking_prob",None)
  print("initial domain picking probability: ", picking_prob)
  for i,src,tgt in zip(domain, source_file, target_file):
    train_datasets_p.append(model.examples_inputter.make_training_dataset(src, tgt,
            batch_size=batch_train_size,
            batch_type=batch_type,
            domain=i,
            single_pass=False,
            shuffle_buffer_size=shuffle_buffer_size,
            length_bucket_width=1,  # Bucketize sequences by the same length for efficiency.
            maximum_features_length=maximum_length,
            maximum_labels_length=maximum_length))
  train_dataset = tf.data.experimental.sample_from_datasets(train_datasets_p, weights=picking_prob)
  with strategy.scope():
    base_dataset = train_dataset
    train_dataset = strategy.experimental_distribute_datasets_from_function(
          lambda _: base_dataset)  
  #############
  train_datasets = [create_training_dataset(strategy, model, [domain], [source_file], [target_file], batch_train_size//2, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=None, temperature=config.get("temperature",1.0))
                                            for domain, source_file, target_file in zip(config.get("domain"), config.get("src"), config.get("tgt"))]

  dev_datasets = [create_training_dataset(strategy, model, [domain], [source_file], [target_file], batch_train_size//2, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=None, temperature=config.get("temperature",1.0))
                                            for domain, source_file, target_file in zip(config.get("eval_domain"), config.get("eval_src"), config.get("eval_ref"))]
  #############
  hessian_datasets = create_training_dataset(strategy, model, domain, config.get("hessian_src", source_file), 
                                            config.get("hessian_ref", target_file), batch_hessian_size, "examples", shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=None, 
                                            temperature=config.get("temperature",1.0), pick_in_order=True)
  #############
  
  ### update factore of diag hessians
  alpha = config.get("hessian_moving_rate",0.1)
  epsilon = config.get("posterior_weight",1e-8)
  print("epsilon: ", epsilon)
  print("hessian_moving_rate: ", alpha)
  #####from utils.dataprocess import count_lines
  datasets_size = [count_lines(src) for src in source_file]
  importance_weights = [data_size/sum(datasets_size) for data_size in datasets_size]
  temperature=config.get("hessian_temperature",1.0)
  importance_weights = [w ** (temperature) for w in importance_weights]
  importance_weights = [w/sum(importance_weights) for w in importance_weights]
  
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator() 
    hessian_accumulators = optimizer_util.DiagHessianAccumulator()
    # hessian_accumulators = [tf.Variable(
    #         tf.zeros_like(var),
    #         trainable=False, synchronization=tf.VariableSynchronization.ON_READ) for var in model.trainable_variables]
    rescale_sum = tf.Variable(0.0, trainable=False, synchronization=tf.VariableSynchronization.ON_READ)
    hessian_moving_stats = [tf.Variable(
            tf.zeros_like(var),
            trainable=False, aggregation=tf.compat.v1.VariableAggregation.MEAN, synchronization=tf.VariableSynchronization.AUTO) for var in model.trainable_variables]
    normalized_hessian_moving_stats = [tf.Variable(
            tf.zeros_like(var),
            trainable=False, aggregation=tf.compat.v1.VariableAggregation.MEAN, synchronization=tf.VariableSynchronization.AUTO) for var in model.trainable_variables]
    importance_weights = tf.constant(importance_weights)
    tf.print("importance_weights: ", importance_weights)
    sub_gradient_accumulator = optimizer_util.GradientAccumulator()
    dev_gradient_accumulator = optimizer_util.GradientAccumulator()
    train_gradient_accumulator = optimizer_util.GradientAccumulator()
    domain_rewards = tf.Variable([0.0]*len(domain), trainable=False, aggregation=tf.compat.v1.VariableAggregation.MEAN, synchronization=tf.VariableSynchronization.AUTO)
    d_logits_grad_accumulator = optimizer_util.GradientAccumulator()
  domain_logits = tf.Variable([0.0]*len(domain), trainable=True)
  grad_domain_logits_accum = tf.Variable(tf.zeros_like(domain_logits), trainable=False)
  sampler_optimizer = tf.keras.optimizers.Adam(learning_rate=config.get("sampler_optim_lr",0.01))
  print("domain_rewards: ", domain_rewards)
  print("domain_importances: ", domain_importances)
  
  @tf.function
  def _grad_sampler_accum():
    loss = - tf.reduce_sum(tf.stop_gradient(tf.nn.softmax(domain_logits)) * tf.nn.log_softmax(domain_logits) * domain_rewards)
    if config.get("sampler_entropy_constraint",False):
      print("sampler_entropy_constraint_weight",config.get("sampler_entropy_constraint_weight",1e-3))
      loss +=  tf.reduce_sum(config.get("sampler_entropy_constraint_weight",1e-3) * tf.nn.log_softmax(domain_logits) * tf.nn.softmax(domain_logits))
    grad = sampler_optimizer.get_gradients(loss,[domain_logits])
    grad_domain_logits_accum.assign_add(grad[0])
    return tf.reduce_sum(tf.stop_gradient(tf.nn.softmax(domain_logits)) * domain_rewards)
  @tf.function
  def _sampler_step_1():
    sampler_optimizer.apply_gradients([(grad_domain_logits_accum, domain_logits)])
    grad_domain_logits_accum.assign(tf.zeros_like(domain_logits))
  def update_sampling_distribution(logits):
    logits = logits.numpy()
    for i, l in enumerate(logits):
        if logits[i] < 0:
            logits[i] = 0
    if sum(logits) == 0:
        logits = [0.1 for _ in range(len(logits))]
    p = np.array(logits) / sum(logits)
    print("new domain probs")
    print(p)
    return p
  #########  
  def _accumulate_diag_hessians(source,target): 
    with tf.GradientTape(persistent=True) as tape:  
      variables = model.trainable_variables
      tape.watch(variables)
      outputs, _ = model(
          source,
          labels=target,
          training=True,
          step=optimizer.iterations)
      _dom = source["domain"][0]
      loss = model.compute_individual_loss(outputs, target, training=True)
      def hessian_accum_along_loss(diag_hessian_acc, x):
        gradients = tape.gradient(x,variables)
        _hessians = []
        for grad in gradients:
          _hessians.append(tf.square(grad))
        hessian_accumulators(_hessians)
        return diag_hessian_acc
      tf.scan(hessian_accum_along_loss, loss, parallel_iterations=batch_hessian_size)
  def _accumulate_NGD_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    if config.get("ADAP_activity_regularizing",False):
      layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
      output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
      d_classification_gate_loss_scale = config.get("d_classification_gate_loss_scale",0.01)
      d_classifier_activity_regularization_loss_scale = config.get("d_classifier_activity_regularization_loss_scale",1.0)
      d_classifier_weight_regularization_losses_scale = config.get("d_classifier_weight_regularization_losses_scale",1.0)
      
      if isinstance(layer_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
        layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        #tf.print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale, "domain: ", domain)
      if isinstance(output_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
        output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
      regularization_losses = model.losses
      # print("model_name_scope", model.name_scope())
      # print(regularization_losses)
      layer_activity_regularization_losses = []
      output_activity_regularization_losses = []
      d_classification_gate_losses = []
      d_classifier_activity_regularization_losses = []
      d_classifier_weight_regularization_losses = []
      for loss_ in regularization_losses:
        if "multi_adap__dense" in loss_.name:
          output_activity_regularization_losses.append(loss_)
        elif "ADAP_gate" in loss_.name: #and "ActivityRegularizer" not in loss_.name and "Regularizer" not in loss_.name
          if "ActivityRegularizer" in loss_.name:
            d_classifier_activity_regularization_losses.append(loss_)
          elif "Regularizer" in loss_.name:
            d_classifier_weight_regularization_losses.append(loss_)
          else:
            d_classification_gate_losses.append(loss_)
        elif "ADAP_" in loss_.name:
          layer_activity_regularization_losses.append(loss_)

      if (len(layer_activity_regularization_losses)>0) and layer_activity_regularization_loss_scale>0:
        training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)

      if len(output_activity_regularization_losses)>0 and output_activity_regularization_loss_scale>0:
        training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)

      if len(d_classification_gate_losses)>0 and d_classification_gate_loss_scale>0:
        training_loss += d_classification_gate_loss_scale * tf.add_n(d_classification_gate_losses) 

      if len(d_classifier_activity_regularization_losses)>0 and d_classifier_activity_regularization_loss_scale>0:
        training_loss += d_classifier_activity_regularization_loss_scale * tf.add_n(d_classifier_activity_regularization_losses)

      if len(d_classifier_weight_regularization_losses)>0 and d_classifier_weight_regularization_losses_scale>0:
        training_loss += d_classifier_weight_regularization_losses_scale * tf.add_n(d_classifier_weight_regularization_losses)
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    #for var in variables:
    #  print(var.name)
    gradients = optimizer.get_gradients(training_loss, variables)
    new_gradients = []
    rescale_sum.assign(0.0)
    for gradient, hessian_moving_stat, var in zip(gradients, normalized_hessian_moving_stats, variables):
      if isinstance(gradient,tf.IndexedSlices):
        if "embedding" in var.name:
          rescale_sum.assign_add(tf.reduce_sum(tf.square(gradient.values)/ (tf.nn.embedding_lookup(hessian_moving_stat, gradient.indices) + epsilon)))
        else:
          rescale_sum.assign_add(tf.reduce_sum(tf.square(gradient.values)/ epsilon))
        #tf.print("hessian %s: "%var.name, tf.nn.embedding_lookup(hessian_moving_stat.value(), gradient.indices), "indices: ", gradient.indices, sep="|")
        #tf.print("hessian_stat: ", hessian_moving_stat.value())
        #continue
      else:
        rescale_sum.assign_add(tf.reduce_sum(tf.square(gradient) / (hessian_moving_stat + epsilon)))
        #tf.print("hessian %s: "%var.name, hessian_moving_stat.value())
    #tf.print("rescale_sum: ", rescale_sum)
    for gradient, hessian_moving_stat, var in zip(gradients, normalized_hessian_moving_stats, variables):
      if isinstance(gradient,tf.IndexedSlices):
        # new_gradients.append(gradient)
        # new_gradients.append(tf.IndexedSlices(gradient.values / (tf.nn.embedding_lookup(hessian_moving_stat.value(), gradient.indices) + epsilon) 
        # * 1 / tf.sqrt(tf.reduce_sum(tf.square(gradient.values)/ (tf.nn.embedding_lookup(hessian_moving_stat.value(), gradient.indices) + epsilon))), 
        # gradient.indices, dense_shape=gradient.dense_shape))
        if "embedding" in var.name:
          new_gradients.append(tf.IndexedSlices(gradient.values / (tf.nn.embedding_lookup(hessian_moving_stat, gradient.indices) + epsilon) 
         * 1 / tf.sqrt(rescale_sum.value()), 
         gradient.indices, dense_shape=gradient.dense_shape))
        else:
          new_gradients.append(tf.IndexedSlices(gradient.values / epsilon * 1 / tf.sqrt(rescale_sum.value()), 
         gradient.indices, dense_shape=gradient.dense_shape))
        # tf.print("hessian_%s: "%var.name, tf.nn.embedding_lookup(hessian_moving_stat.value(), gradient.indices))
      else:
        # new_gradients.append(gradient / (hessian_moving_stat.value() +epsilon) * 1 / tf.sqrt(tf.reduce_sum(tf.square(gradient) / (hessian_moving_stat.value()+epsilon))))
        new_gradients.append(gradient / (hessian_moving_stat + epsilon) * 1 / tf.sqrt(rescale_sum.value()))
    gradient_accumulator(new_gradients)
    num_examples = tf.reduce_sum(target["length"])
    return reported_loss, num_examples
  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    if config.get("ADAP_activity_regularizing",False):
      layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
      output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
      d_classification_gate_loss_scale = config.get("d_classification_gate_loss_scale",0.01)
      d_classifier_activity_regularization_loss_scale = config.get("d_classifier_activity_regularization_loss_scale",1.0)
      d_classifier_weight_regularization_losses_scale = config.get("d_classifier_weight_regularization_losses_scale",1.0)
      
      if isinstance(layer_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
        layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        #tf.print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale, "domain: ", domain)
      if isinstance(output_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
        output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
      regularization_losses = model.losses
      # print("model_name_scope", model.name_scope())
      # print(regularization_losses)
      layer_activity_regularization_losses = []
      output_activity_regularization_losses = []
      d_classification_gate_losses = []
      d_classifier_activity_regularization_losses = []
      d_classifier_weight_regularization_losses = []
      for loss_ in regularization_losses:
        if "multi_adap__dense" in loss_.name:
          output_activity_regularization_losses.append(loss_)
        elif "ADAP_gate" in loss_.name: #and "ActivityRegularizer" not in loss_.name and "Regularizer" not in loss_.name
          if "ActivityRegularizer" in loss_.name:
            d_classifier_activity_regularization_losses.append(loss_)
          elif "Regularizer" in loss_.name:
            d_classifier_weight_regularization_losses.append(loss_)
          else:
            d_classification_gate_losses.append(loss_)
        elif "ADAP_" in loss_.name:
          layer_activity_regularization_losses.append(loss_)

      
      if (len(layer_activity_regularization_losses)>0) and layer_activity_regularization_loss_scale>0:
        training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)

      if len(output_activity_regularization_losses)>0 and output_activity_regularization_loss_scale>0:
        training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)

      if len(d_classification_gate_losses)>0 and d_classification_gate_loss_scale>0:
        training_loss += d_classification_gate_loss_scale * tf.add_n(d_classification_gate_losses)

      if len(d_classifier_activity_regularization_losses)>0 and d_classifier_activity_regularization_loss_scale>0:
        training_loss += d_classifier_activity_regularization_loss_scale * tf.add_n(d_classifier_activity_regularization_losses)

      if len(d_classifier_weight_regularization_losses)>0 and d_classifier_weight_regularization_losses_scale>0:
        training_loss += d_classifier_weight_regularization_losses_scale * tf.add_n(d_classifier_weight_regularization_losses)
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    #for var in variables:
    #  print(var.name)
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    return reported_loss, num_examples
  def _accumulate_dev_train_gradients(source, target):
    with tf.GradientTape() as tape:
      variables = model.trainable_variables    
      tape.watch(variables)
      outputs, _ = model(
          source,
          labels=target,
          training=True,
          step=optimizer.iterations)
      loss = model.compute_loss(outputs, target, training=True)

      if isinstance(loss, tuple):
        training_loss = loss[0] / loss[1]
        reported_loss = loss[0] / loss[2]
      else:
        training_loss, reported_loss = loss, loss

      gradients = tape.gradient(training_loss, variables)
      sub_gradient_accumulator(gradients)
      return loss
  def _reset_dev_train_gradients():
    dev_gradient_accumulator.reset() # for dev_gradient_accumulator in dev_gradient_accumulators]
    train_gradient_accumulator.reset() #for train_gradient_accumulator in train_gradient_accumulators]
  def _reset_sub_gradients():
    sub_gradient_accumulator.reset()
  def _apply_dev_train_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(sub_gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(sub_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    sub_gradient_accumulator.reset()
  def _apply_sampler_gradients():
    grads_and_vars = []
    scaled_gradient = d_logits_grad_accumulator.gradients[0] / (strategy.num_replicas_in_sync * tf.cast(d_logits_grad_accumulator.step, tf.float32))
    grads_and_vars.append((scaled_gradient, domain_logits))
    sampler_optimizer.apply_gradients(grads_and_vars)
    d_logits_grad_accumulator.reset()
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  @tf.function
  def _dev_train_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_dev_train_gradients)

  @tf.function
  def _sampler_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_sampler_gradients)

  @tf.function
  def _reset_dev_train_grad_accum_step():
    with strategy.scope():
      _reset_dev_train_gradients()

  @tf.function
  def _reset_sub_grad_accum_step():
    with strategy.scope():
      _reset_sub_gradients()
  
  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  #########
  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
  def update_hessian_moving_stats():
    for accum, stat in zip(hessian_accumulators.hessians, hessian_moving_stats):
      stat.assign(accum / tf.cast(hessian_accum_step * batch_hessian_size, tf.float32))
    for hessian, normalized_hessian in zip(hessian_moving_stats, normalized_hessian_moving_stats):
      normalized_hessian.assign(hessian/tf.reduce_sum(hessian))
    
  #########
  @dataset_util.function_on_next(train_dataset)
  def _NGD_train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_NGD_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples
  @dataset_util.function_on_next(hessian_datasets)
  def _hessian_acc_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      strategy.experimental_run_v2(
          _accumulate_diag_hessians, args=(per_replica_source, per_replica_target))
  ##########
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  @tf.function
  def _hessian_stats_update_step():
    strategy.experimental_run_v2(update_hessian_moving_stats)
  ##########

  # Runs the training loop.
  import time
  start = time.time()  
  NGD_train_data_flow = iter(_NGD_train_forward())
  _hessian_accumulator_flow = iter(_hessian_acc_forward())
  train_data_flow = iter(_train_forward())
  dev_iterators = [iter(dev_dataset) for dev_dataset in dev_datasets]
  train_iterators = [iter(train_dataset) for train_dataset in train_datasets]
  
  _, _ = next(train_data_flow)
  last_eval = [0.0] * len(domain)
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _number_examples = []
  
  score_type = config.get("score_type","MultiBLEU")
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  ref_eval_concat = file_concatenate(config["eval_ref"],"ref_eval_concat",dir_name=os.path.join(config["model_dir"],"eval"))
  if step >= config.get("NGD_warm_start",0):
    for i in range(hessian_accum_step):
      next(_hessian_accumulator_flow)
    _hessian_stats_update_step()
    print("normalized_hessian_moving_stats: [3]", normalized_hessian_moving_stats[3].numpy())
  if step <= 1:
    initializer = config.get("initializer","default")
    if initializer == "default":
      print("Initializing variables by tensorflow default")      
    elif initializer == "variance_scaling":
      print("Initializing variables by tf.variance_scaling")
      initial_value = []
      for v in model.trainable_variables:
        shape = tf.shape(v).numpy()
        initial_value.append(variance_scaling_initialier(shape, scale=1.0, mode="fan_avg", distribution="uniform"))
      weight_reset(initial_value)   
  else:
    print("current domain_logits", config.get("domain_logits",[0.0]*len(domain)))
    domain_logits.assign(config.get("domain_logits",[0.0]*len(domain)))
    # compute domain rewards
    rewards = [0.0] * len(domain)
    snapshots = [v.value() for v in model.trainable_variables]
    saved_step = optimizer.iterations.numpy()
    #######
    current_probs = tf.nn.softmax(domain_logits).numpy()
    print("current_probs: ", current_probs)
    #######
    ##### compute theta_t+1
    for k in np.random.choice(domain,config.get("update_theta_train_batch_per_run_num",len(domain)),p=current_probs): 
      src, tgt = next(train_iterators[k])
      strategy.experimental_run_v2(_accumulate_dev_train_gradients, args=(src, tgt))
    strategy.experimental_run_v2(_apply_dev_train_gradients)
    snapshots_1 = [v.value() for v in model.trainable_variables]
    for i, train_iter in enumerate(train_iterators):
      _reward = 0.0
      ##### accumulate gradient over training set of src domain i at theta_t
      weight_reset(snapshots)
      with strategy.scope():
        for _ in range(config.get("train_batch_per_run_num",10)):
          src, tgt = next(train_iter)
          strategy.experimental_run_v2(_accumulate_dev_train_gradients, args=(src, tgt))
        train_gradient_accumulator(sub_gradient_accumulator.gradients)
        strategy.experimental_run_v2(sub_gradient_accumulator.reset)
      ##### accumulate gradient over dev set of k tgt domains at theta_t+1
      weight_reset(snapshots_1)
      with strategy.scope():
        for j, dev_iter in enumerate(dev_iterators):
          _sum = 0.0
          _dev_norm = 0.0
          _tr_norm = 0.0
          for _ in range(config.get("dev_batch_per_run_num",10)):
            src, tgt = next(dev_iter)
            strategy.experimental_run_v2(_accumulate_dev_train_gradients, args=(src, tgt))
          dev_gradient_accumulator(sub_gradient_accumulator.gradients)
          strategy.experimental_run_v2(sub_gradient_accumulator.reset)         
          for dev_grad, tr_grad in zip(dev_gradient_accumulator.gradients, train_gradient_accumulator.gradients):
            _sum += tf.reduce_sum(dev_grad * tr_grad)
            _dev_norm += tf.reduce_sum(dev_grad * dev_grad)
            _tr_norm += tf.reduce_sum(tr_grad * tr_grad)
          _reward += _sum / (tf.sqrt(_dev_norm * _tr_norm) + 1e-10) * domain_importances[j]
          # reset dev gradient accumulations to zero
          strategy.experimental_run_v2(dev_gradient_accumulator.reset)
          #print(dev_gradient_accumulator.gradients[0])
        # reset train dev gradient accumulations to zero
        strategy.experimental_run_v2(train_gradient_accumulator.reset)
        #print(sub_gradient_accumulator.gradients[0])
        #print(train_gradient_accumulator.gradients[0])
      #_reward /= len(domain)
      rewards[i] = _reward.numpy()
      # reset model parameters
      weight_reset(snapshots)
      optimizer.iterations.assign(saved_step)
    domain_rewards.assign(tf.constant(rewards))
    # compute new domain distribution
    print("domain rewards", domain_rewards)
    for _ in range(config.get("domain_sampler_optim_step", 30)):
      #loss = _sampler_flow()
      #_sampler_step()
      _ = _grad_sampler_accum()
      _sampler_step_1()
      
    print("domain_logits: ", domain_logits.numpy())
    probs = tf.nn.softmax(domain_logits)
    new_picking_prob = update_sampling_distribution(probs)
    tf.summary.experimental.set_step(saved_step)
    for i in range(len(domain)):
      tf.summary.scalar("reward_%d"%i, rewards[i], description="reward of using training set %d"%(i))
      tf.summary.scalar("domain_prob_%d"%i, new_picking_prob[i], description="probability of using training set %d"%(i))
    tf.summary.flush()
    # create new training course with updated domain distribution
    train_dataset = tf.data.experimental.sample_from_datasets(train_datasets_p, weights=new_picking_prob)
    with strategy.scope():
      base_dataset = train_dataset
      train_dataset = strategy.experimental_distribute_datasets_from_function(
            lambda _: base_dataset)
    @dataset_util.function_on_next(train_dataset)
    def _train_forward(next_fn):    
      with strategy.scope():
        per_replica_source, per_replica_target = next_fn()
        per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
            _accumulate_gradients, args=(per_replica_source, per_replica_target))
        # TODO: these reductions could be delayed until _step is called.
        loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
        num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
      return loss, num_examples
    train_data_flow = iter(_train_forward())
    #######
    weight_reset(snapshots)
    optimizer.iterations.assign(saved_step)
    #######
  with _summary_writer.as_default():
    while True:
      #####Training batch
      if step % hessian_update_every == 0 and step >= config.get("NGD_warm_start",0):
        for i in range(hessian_accum_step):
          next(_hessian_accumulator_flow)
        _hessian_stats_update_step()
      if step >= config.get("NGD_warm_start",0):
        loss, num_examples = next(NGD_train_data_flow)    
        _loss.append(loss)
        _number_examples.append(num_examples)
      else:
        loss, num_examples = next(train_data_flow)    
        _loss.append(loss)
        _number_examples.append(num_examples)
      _step()  
      step = optimizer.iterations.numpy()
      
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
          "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
          step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      if step % redistribute_every == 0 and step > config.get("warm_start",5000):
        # compute domain rewards
        rewards = [0.0] * len(domain)
        snapshots = [v.value() for v in model.trainable_variables]
        saved_step = optimizer.iterations.numpy()
        #######
        current_probs = tf.nn.softmax(domain_logits).numpy()
        print("current_probs: ", current_probs)
        #######
        ##### compute theta_t+1
        for k in np.random.choice(domain,config.get("update_theta_train_batch_per_run_num",len(domain)),p=current_probs): 
          src, tgt = next(train_iterators[k])
          strategy.experimental_run_v2(_accumulate_dev_train_gradients, args=(src, tgt))
        strategy.experimental_run_v2(_apply_dev_train_gradients)
        snapshots_1 = [v.value() for v in model.trainable_variables]
        for i, train_iter in enumerate(train_iterators):
          _reward = 0.0
          ##### accumulate gradient over training set of src domain i at theta_t
          weight_reset(snapshots)
          with strategy.scope():
            for _ in range(config.get("train_batch_per_run_num",10)):
              src, tgt = next(train_iter)
              strategy.experimental_run_v2(_accumulate_dev_train_gradients, args=(src, tgt))
            train_gradient_accumulator(sub_gradient_accumulator.gradients)
            strategy.experimental_run_v2(sub_gradient_accumulator.reset)
          ##### accumulate gradient over dev set of k tgt domains at theta_t+1
          weight_reset(snapshots_1)
          with strategy.scope():
            for j, dev_iter in enumerate(dev_iterators):
              _sum = 0.0
              _dev_norm = 0.0
              _tr_norm = 0.0
              for _ in range(config.get("dev_batch_per_run_num",10)):
                src, tgt = next(dev_iter)
                strategy.experimental_run_v2(_accumulate_dev_train_gradients, args=(src, tgt))
              dev_gradient_accumulator(sub_gradient_accumulator.gradients)
              strategy.experimental_run_v2(sub_gradient_accumulator.reset)         
              for dev_grad, tr_grad, var in zip(dev_gradient_accumulator.gradients, train_gradient_accumulator.gradients, model.trainable_variables):
                if True:#"ADAP_" not in var.name:
                  _sum += tf.reduce_sum(dev_grad * tr_grad)
                  _dev_norm += tf.reduce_sum(dev_grad * dev_grad)
                  _tr_norm += tf.reduce_sum(tr_grad * tr_grad)
              if config.get("cosine_reward",True):
                _reward += _sum / (tf.sqrt(_dev_norm * _tr_norm) + 1e-10) * domain_importances[j]
              else:
                _reward += _sum * learning_rate(saved_step) * domain_importances[j]
              # reset dev gradient accumulations to zero
              strategy.experimental_run_v2(dev_gradient_accumulator.reset)
              #print(dev_gradient_accumulator.gradients[0])
            # reset train dev gradient accumulations to zero
            strategy.experimental_run_v2(train_gradient_accumulator.reset)
            #print(sub_gradient_accumulator.gradients[0])
            #print(train_gradient_accumulator.gradients[0])
          #_reward /= len(domain)
          rewards[i] = _reward.numpy()
          # reset model parameters
          weight_reset(snapshots)
          optimizer.iterations.assign(saved_step)
        domain_rewards.assign(tf.constant(rewards))
        # compute new domain distribution
        print("domain rewards", domain_rewards)
        for _ in range(config.get("domain_sampler_optim_step", 30)):
          _ = _grad_sampler_accum()
          _sampler_step_1()
          
        print("domain_logits: ", domain_logits.numpy())
        probs = tf.nn.softmax(domain_logits)
        new_picking_prob = update_sampling_distribution(probs)
        tf.summary.experimental.set_step(saved_step)
        for i in range(len(domain)):
          tf.summary.scalar("reward_%d"%i, rewards[i], description="reward of using training set %d"%(i))
          tf.summary.scalar("domain_prob_%d"%i, new_picking_prob[i], description="probability of using training set %d"%(i))
        tf.summary.flush()
        # create new training course with updated domain distribution
        train_dataset = tf.data.experimental.sample_from_datasets(train_datasets_p, weights=new_picking_prob)
        with strategy.scope():
          base_dataset = train_dataset
          train_dataset = strategy.experimental_distribute_datasets_from_function(
                lambda _: base_dataset)
        if step < config.get("NGD_warm_start",0):
          @dataset_util.function_on_next(train_dataset)
          def _train_forward(next_fn):    
            with strategy.scope():
              per_replica_source, per_replica_target = next_fn()
              per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
                  _accumulate_gradients, args=(per_replica_source, per_replica_target))
              # TODO: these reductions could be delayed until _step is called.
              loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
              num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
            return loss, num_examples
          train_data_flow = iter(_train_forward())
        else:
          @dataset_util.function_on_next(train_dataset)
          def _NGD_train_forward(next_fn):    
            with strategy.scope():
              per_replica_source, per_replica_target = next_fn()
              per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
                  _accumulate_NGD_gradients, args=(per_replica_source, per_replica_target))
              # TODO: these reductions could be delayed until _step is called.
              loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
              num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
            return loss, num_examples
          NGD_train_data_flow = iter(_NGD_train_forward())
        #######
        weight_reset(snapshots)
        optimizer.iterations.assign(saved_step)
        #######
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        output_files = []
        new_bleu = 0.0
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
            output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
            score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
            tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
            output_files.append(output_file)
            new_bleu += score * domain_importances[i]
        ##### BLEU on concat dev set.
        output_file_concat = file_concatenate(output_files,"output_file_concat.%s"%os.path.basename(checkpoint_path))
        score = scorer(ref_eval_concat, output_file_concat)
        print("score of model %s on concat dev set: "%checkpoint_manager.latest_checkpoint, score)
        tf.summary.scalar("concat_eval_score", score, description="BLEU on concat dev set")
        #############################
        if new_bleu >= current_max_eval_bleu:
          current_max_eval_bleu = new_bleu
          descending_streak = 0
        else:
          descending_streak += 1
      if descending_streak >= 5:
        break
      if step > train_steps:
        break

def train_L2W_v1(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          adapter_optimizer=None,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  redistribute_every = config.get("redistribute_every",2000)
  if config.get("use_meta_optimizer",False):
    inner_optimizer = tf.keras.optimizers.SGD(config.get("meta_train_lr",0.001))
  else:
    inner_optimizer = optimizer
  #####
  if checkpoint_path is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
    checkpoint.restore(checkpoint_path)
  else:
    if checkpoint_manager.latest_checkpoint is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
      checkpoint.restore(checkpoint_manager.latest_checkpoint)    
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  eval_domain = config.get("eval_domain")
  ###### early stopping criterion
  current_max_eval_bleu = 0.0
  descending_streak = 0
  ######
  if not config.get("domain_importances",None):
    domain_importances = [1.0/len(eval_domain)]*len(eval_domain)
  else:
    domain_importances = config.get("domain_importances")
  print("There are %d in-domain corpora"%len(source_file))
  ###############
  print("cosine_reward: ",config.get("cosine_reward",True))
  ###############
  # train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
  #                                           maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
  #                                           multi_domain=config.get("multi_domain", True), picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))

  print("maximum_length", maximum_length)
  train_datasets_p = [] 
  datasets_size = [count_lines(src) for src in source_file]
  picking_prob = [data_size/sum(datasets_size) for data_size in datasets_size]
  if config.get("picking_prob",None):
    picking_prob = config.get("picking_prob",None)
  print("initial domain picking probability: ", picking_prob)
  for i,src,tgt in zip(domain, source_file, target_file):
    train_datasets_p.append(model.examples_inputter.make_training_dataset(src, tgt,
            batch_size=batch_train_size,
            batch_type=batch_type,
            domain=i,
            single_pass=False,
            shuffle_buffer_size=shuffle_buffer_size,
            length_bucket_width=1,  # Bucketize sequences by the same length for efficiency.
            maximum_features_length=maximum_length,
            maximum_labels_length=maximum_length))
  train_dataset = tf.data.experimental.sample_from_datasets(train_datasets_p, weights=picking_prob)
  with strategy.scope():
    base_dataset = train_dataset
    train_dataset = strategy.experimental_distribute_datasets_from_function(
          lambda _: base_dataset)  
  #############
  train_datasets = [create_training_dataset(strategy, model, [domain], [source_file], [target_file], batch_train_size//2, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=None, temperature=config.get("temperature",1.0))
                                            for domain, source_file, target_file in zip(config.get("domain"), config.get("src"), config.get("tgt"))]

  dev_datasets = [create_training_dataset(strategy, model, [domain], [source_file], [target_file], batch_train_size//2, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=None, temperature=config.get("temperature",1.0))
                                            for domain, source_file, target_file in zip(config.get("eval_domain"), config.get("eval_src"), config.get("eval_ref"))]
  #############
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
    sub_gradient_accumulator = optimizer_util.GradientAccumulator()
    dev_gradient_accumulator = optimizer_util.GradientAccumulator()
    train_gradient_accumulator = optimizer_util.GradientAccumulator()
    domain_rewards = tf.Variable([0.0]*len(domain), trainable=False, aggregation=tf.compat.v1.VariableAggregation.MEAN, synchronization=tf.VariableSynchronization.AUTO)
    d_logits_grad_accumulator = optimizer_util.GradientAccumulator()

  domain_logits = tf.Variable([0.0]*len(domain), trainable=True)
  grad_domain_logits_accum = tf.Variable(tf.zeros_like(domain_logits), trainable=False)
  sampler_optimizer = tf.keras.optimizers.Adam(learning_rate=config.get("sampler_optim_lr",0.01))
  sampler_vars = [domain_logits]
  print("domain_rewards: ", domain_rewards)
  print("domain_importances: ", domain_importances)
  
  @tf.function
  def _grad_sampler_accum():
    loss = - tf.reduce_sum(tf.stop_gradient(tf.nn.softmax(domain_logits)) * tf.nn.log_softmax(domain_logits) * domain_rewards)
    if config.get("sampler_entropy_constraint",False):
      print("sampler_entropy_constraint_weight",config.get("sampler_entropy_constraint_weight",1e-3))
      loss +=  tf.reduce_sum(config.get("sampler_entropy_constraint_weight",1e-3) * tf.nn.log_softmax(domain_logits) * tf.nn.softmax(domain_logits))
    grad = sampler_optimizer.get_gradients(loss,[domain_logits])
    grad_domain_logits_accum.assign_add(grad[0])
    return tf.reduce_sum(tf.stop_gradient(tf.nn.softmax(domain_logits)) * domain_rewards)

  @tf.function
  def _sampler_step_1():
    sampler_optimizer.apply_gradients([(grad_domain_logits_accum, domain_logits)])
    grad_domain_logits_accum.assign(tf.zeros_like(domain_logits))

  def update_sampling_distribution(logits):
    logits = logits.numpy()
    for i, l in enumerate(logits):
        if logits[i] < 0:
            logits[i] = 0
    if sum(logits) == 0:
        logits = [0.1 for _ in range(len(logits))]
    p = np.array(logits) / sum(logits)
    print("new domain probs")
    print(p)
    return p

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    domain = source["domain"][0]
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    
    variables = model.trainable_variables
    #print("var numb: ", len(variables))
    """ for var in variables:
      print(var.name) """
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

  def _accumulate_dev_train_gradients(source, target):
    with tf.GradientTape() as tape:
      variables = model.trainable_variables    
      tape.watch(variables)
      outputs, _ = model(
          source,
          labels=target,
          training=True,
          step=optimizer.iterations)
      loss = model.compute_loss(outputs, target, training=True)

      if isinstance(loss, tuple):
        training_loss = loss[0] / loss[1]
        reported_loss = loss[0] / loss[2]
      else:
        training_loss, reported_loss = loss, loss

      gradients = tape.gradient(training_loss, variables)
      sub_gradient_accumulator(gradients)
      return training_loss
  
  def _reset_dev_train_gradients():
    dev_gradient_accumulator.reset() # for dev_gradient_accumulator in dev_gradient_accumulators]
    [train_gradient_accumulator.reset() for train_gradient_accumulator in train_gradient_accumulators]

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
  
  def _apply_dev_train_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(sub_gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(sub_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    inner_optimizer.apply_gradients(grads_and_vars)
    sub_gradient_accumulator.reset()
 
  def _apply_sampler_gradients():
    grads_and_vars = []
    scaled_gradient = d_logits_grad_accumulator.gradients[0] / (strategy.num_replicas_in_sync * tf.cast(d_logits_grad_accumulator.step, tf.float32))
    grads_and_vars.append((scaled_gradient, domain_logits))
    sampler_optimizer.apply_gradients(grads_and_vars)
    d_logits_grad_accumulator.reset()

  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples
  
  @tf.function
  def _sampler_flow():
    with strategy.scope():
      per_replica_loss = strategy.experimental_run_v2(_sampler_loss)
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
    return loss

  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  @tf.function
  def _dev_train_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_dev_train_gradients)

  @tf.function
  def _sampler_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_sampler_gradients)

  @tf.function
  def _reset_dev_train_grad_accum_step():
    with strategy.scope():
      _reset_dev_train_gradients()
  
  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  dev_iterators = [iter(dev_dataset) for dev_dataset in dev_datasets]
  train_iterators = [iter(train_dataset) for train_dataset in train_datasets]
 
  _, _ = next(train_data_flow)

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _number_examples = []
  step = optimizer.iterations.numpy()
  if config.get("reset_step",None):
    print("start from %d-th step"%config.get("reset_step",150000))
    optimizer.iterations.assign(config.get("reset_step",150000))
  
  if step <= 1:
    initializer = config.get("initializer","default")
    if initializer == "default":
      print("Initializing variables by tensorflow default")      
    elif initializer == "variance_scaling":
      print("Initializing variables by tf.variance_scaling")
      initial_value = []
      for v in model.trainable_variables:
        shape = tf.shape(v).numpy()
        initial_value.append(variance_scaling_initialier(shape, scale=1.0, mode="fan_avg", distribution="uniform"))
      weight_reset(initial_value)   
  else:
    print("current domain_logits", config.get("domain_logits",[0.0]*len(domain)))
    domain_logits.assign(config.get("domain_logits",[0.0]*len(domain)))
    # compute domain rewards
    rewards = [0.0] * len(domain)
    snapshots = [v.value() for v in model.trainable_variables]
    saved_step = optimizer.iterations.numpy()
    #######
    current_probs = tf.nn.softmax(domain_logits).numpy()
    print("current_probs: ", current_probs)
    #######
    
    for i, train_iter in enumerate(train_iterators):
      _reward = 0.0
      weight_reset(snapshots)
      with strategy.scope():
        ##### compute theta_t+1
        for _ in range(config.get("train_batch_per_run_num",10)): 
          src, tgt = next(train_iterators[i])
          strategy.experimental_run_v2(_accumulate_dev_train_gradients, args=(src, tgt))
        ##### accumulate gradient over training set of src domain i at theta_t
        # for _ in range(config.get("train_batch_per_run_num",10)):
        #   src, tgt = next(train_iter)
        #   strategy.experimental_run_v2(_accumulate_dev_train_gradients, args=(src, tgt))
        train_gradient_accumulator(sub_gradient_accumulator.gradients)
        strategy.experimental_run_v2(_apply_dev_train_gradients)
        #strategy.experimental_run_v2(sub_gradient_accumulator.reset)
      ##### accumulate gradient over dev set of k tgt domains at theta_t+1
      with strategy.scope():
        for j, dev_iter in enumerate(dev_iterators):
          _sum = 0.0
          _dev_norm = 0.0
          _tr_norm = 0.0
          for _ in range(config.get("dev_batch_per_run_num",10)):
            src, tgt = next(dev_iter)
            strategy.experimental_run_v2(_accumulate_dev_train_gradients, args=(src, tgt))
          dev_gradient_accumulator(sub_gradient_accumulator.gradients)
          strategy.experimental_run_v2(sub_gradient_accumulator.reset)         
          for dev_grad, tr_grad in zip(dev_gradient_accumulator.gradients, train_gradient_accumulator.gradients):
            _sum += tf.reduce_sum(dev_grad * tr_grad)
            _dev_norm += tf.reduce_sum(dev_grad * dev_grad)
            _tr_norm += tf.reduce_sum(tr_grad * tr_grad)
          _reward += _sum / (tf.sqrt(_dev_norm * _tr_norm) + 1e-10) * domain_importances[j]
          # reset dev gradient accumulations to zero
          strategy.experimental_run_v2(dev_gradient_accumulator.reset)
          #print(dev_gradient_accumulator.gradients[0])
        # reset train dev gradient accumulations to zero
        strategy.experimental_run_v2(train_gradient_accumulator.reset)
        #print(sub_gradient_accumulator.gradients[0])
        #print(train_gradient_accumulator.gradients[0])
      #_reward /= len(domain)
      rewards[i] = _reward.numpy()
      # reset model parameters
      weight_reset(snapshots)
      optimizer.iterations.assign(saved_step)
    domain_rewards.assign(tf.constant(rewards))
    # compute new domain distribution
    print("domain rewards", domain_rewards)
    for _ in range(config.get("domain_sampler_optim_step", 30)):
      #loss = _sampler_flow()
      #_sampler_step()
      _ = _grad_sampler_accum()
      _sampler_step_1()
      
    print("domain_logits: ", domain_logits.numpy())
    probs = tf.nn.softmax(domain_logits)
    new_picking_prob = update_sampling_distribution(probs)
    tf.summary.experimental.set_step(saved_step)
    for i in range(len(domain)):
      tf.summary.scalar("reward_%d"%i, rewards[i], description="reward of using training set %d"%(i))
      tf.summary.scalar("domain_prob_%d"%i, new_picking_prob[i], description="probability of using training set %d"%(i))
    tf.summary.flush()
    # create new training course with updated domain distribution
    train_dataset = tf.data.experimental.sample_from_datasets(train_datasets_p, weights=new_picking_prob)
    with strategy.scope():
      base_dataset = train_dataset
      train_dataset = strategy.experimental_distribute_datasets_from_function(
            lambda _: base_dataset)
    @dataset_util.function_on_next(train_dataset)
    def _train_forward(next_fn):    
      with strategy.scope():
        per_replica_source, per_replica_target = next_fn()
        per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
            _accumulate_gradients, args=(per_replica_source, per_replica_target))
        # TODO: these reductions could be delayed until _step is called.
        loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
        num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
      return loss, num_examples
    train_data_flow = iter(_train_forward())
    #######
    weight_reset(snapshots)
    optimizer.iterations.assign(saved_step)
    #######

  if config.get("continual_learning", False):
    print("Continual Learning needs to load from old model")
    assert config.get("checkpoint_path") != None
    checkpoint_path = config.get("checkpoint_path")
    load_and_update_if_needed_from_ckpt(config["model_dir"],   
                        checkpoint_path,                        
                        trackables={"model":model},
                        vocab_update=True,
                        model_key="model")

  score_type = config.get("score_type","MultiBLEU")
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  ref_eval_concat = file_concatenate(config["eval_ref"],"ref_eval_concat",dir_name=os.path.join(config["model_dir"],"eval"))
  ########
  excluded_params = []
  for var in model.trainable_variables:
    if tf.shape(var)[-1].numpy()==31266 or tf.shape(var)[0].numpy()==31266:
      print(var.name)
      excluded_params.append(var.name)
  ########
  with _summary_writer.as_default():
    while True:
      ####Training batch
      loss, num_examples = next(train_data_flow)    
      _loss.append(loss.numpy())
      _number_examples.append(num_examples.numpy())
      _step()  
      step = optimizer.iterations.numpy()
      
      if step % redistribute_every == 0 and step > config.get("warm_start",5000):
        # compute domain rewards
        rewards = [0.0] * len(domain)
        snapshots = [v.value() for v in model.trainable_variables]
        saved_step = optimizer.iterations.numpy()
        #######
        current_probs = tf.nn.softmax(domain_logits).numpy()
        print("current_probs: ", current_probs)
        #######        
        for i, train_iter in enumerate(train_iterators):
          _reward = 0.0
          weight_reset(snapshots)
          with strategy.scope():
            ##### compute theta_t+1
            for _ in range(config.get("train_batch_per_run_num",10)): 
              src, tgt = next(train_iterators[i])
              strategy.experimental_run_v2(_accumulate_dev_train_gradients, args=(src, tgt))
            train_gradient_accumulator(sub_gradient_accumulator.gradients)
            strategy.experimental_run_v2(_apply_dev_train_gradients)
            strategy.experimental_run_v2(sub_gradient_accumulator.reset)
          ##### accumulate gradient over dev set of k tgt domains at theta_t+1
          with strategy.scope():
            for j, dev_iter in enumerate(dev_iterators):
              _sum = 0.0
              _dev_norm = 0.0
              _tr_norm = 0.0
              #count = 0
              for _ in range(config.get("dev_batch_per_run_num",10)):
                src, tgt = next(dev_iter)
                strategy.experimental_run_v2(_accumulate_dev_train_gradients, args=(src, tgt))
              dev_gradient_accumulator(sub_gradient_accumulator.gradients)
              strategy.experimental_run_v2(sub_gradient_accumulator.reset)         
              for dev_grad, tr_grad, var in zip(dev_gradient_accumulator._gradients, train_gradient_accumulator._gradients, model.trainable_variables):
                _sum += tf.reduce_sum(dev_grad * tr_grad)
                _dev_norm += tf.reduce_sum(dev_grad * dev_grad)
                _tr_norm += tf.reduce_sum(tr_grad * tr_grad)
              #print("number_of_parameters_in_reward: %d"%(count))
              if config.get("cosine_reward",True):
                _reward += _sum / (tf.sqrt(_dev_norm * _tr_norm) + 1e-10) * domain_importances[j]
              else:
                _reward += _sum * domain_importances[j]
              # reset dev gradient accumulations to zero
              strategy.experimental_run_v2(dev_gradient_accumulator.reset)
              #print(dev_gradient_accumulator.gradients[0])
            # reset train dev gradient accumulations to zero
            strategy.experimental_run_v2(train_gradient_accumulator.reset)
            #print(sub_gradient_accumulator.gradients[0])
            #print(train_gradient_accumulator.gradients[0])
          #_reward /= len(domain)
          rewards[i] = _reward.numpy()
          # reset model parameters
          weight_reset(snapshots)
          optimizer.iterations.assign(saved_step)
        domain_rewards.assign(tf.constant(rewards))
        # compute new domain distribution
        print("domain rewards", domain_rewards)
        for _ in range(config.get("domain_sampler_optim_step", 30)):
          _ = _grad_sampler_accum()
          _sampler_step_1()
          
        print("domain_logits: ", domain_logits.numpy())
        probs = tf.nn.softmax(domain_logits)
        new_picking_prob = update_sampling_distribution(probs)
        tf.summary.experimental.set_step(saved_step)
        for i in range(len(domain)):
          tf.summary.scalar("reward_%d"%i, rewards[i], description="reward of using training set %d"%(i))
          tf.summary.scalar("domain_prob_%d"%i, new_picking_prob[i], description="probability of using training set %d"%(i))
        tf.summary.flush()
        # create new training course with updated domain distribution
        del train_dataset
        del train_data_flow
        train_dataset = tf.data.experimental.sample_from_datasets(train_datasets_p, weights=new_picking_prob)
        with strategy.scope():
          base_dataset = train_dataset
          train_dataset = strategy.experimental_distribute_datasets_from_function(
                lambda _: base_dataset)
        @dataset_util.function_on_next(train_dataset)
        def _train_forward(next_fn):    
          with strategy.scope():
            per_replica_source, per_replica_target = next_fn()
            per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
                _accumulate_gradients, args=(per_replica_source, per_replica_target))
            # TODO: these reductions could be delayed until _step is called.
            loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
            num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
          return loss, num_examples
        train_data_flow = iter(_train_forward())
        #######
        weight_reset(snapshots)
        optimizer.iterations.assign(saved_step)
        #######

      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
          "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
          step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        output_files = []
        new_bleu = 0.0
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
          output_files.append(output_file)
          new_bleu += score * domain_importances[i]
        ##### BLEU on concat dev set.
        output_file_concat = file_concatenate(output_files,"output_file_concat.%s"%os.path.basename(checkpoint_path))
        score = scorer(ref_eval_concat, output_file_concat)
        print("score of model %s on concat dev set: "%checkpoint_manager.latest_checkpoint, score)
        tf.summary.scalar("concat_eval_score", score, description="BLEU on concat dev set")
        #############################
        if new_bleu >= current_max_eval_bleu:
          current_max_eval_bleu = new_bleu
          descending_streak = 0
        else:
          descending_streak += 1
      tf.summary.flush()
      if config.get("early_stopping",True) and descending_streak >= 5:
        break
      if step > train_steps:
        break

def train_L2W_v2(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          adapter_optimizer=None,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=3000000,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  redistribute_every = config.get("redistribute_every",2000)
  if config.get("use_meta_optimizer",False):
    inner_optimizer = tf.keras.optimizers.SGD(config.get("meta_train_lr",0.001))
  else:
    inner_optimizer = optimizer
  #####
  if checkpoint_path is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
    checkpoint.restore(checkpoint_path)
  else:
    if checkpoint_manager.latest_checkpoint is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
      checkpoint.restore(checkpoint_manager.latest_checkpoint)    
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  eval_domain = config.get("eval_domain")
  ###### early stopping criterion
  current_max_eval_bleu = 0.0
  descending_streak = 0
  ######
  if not config.get("domain_importances",None):
    domain_importances = [1.0/len(eval_domain)]*len(eval_domain)
  else:
    domain_importances = config.get("domain_importances")
  print("There are %d in-domain corpora"%len(source_file))
  domain_num = len(source_file)
  ###############
  print("cosine_reward: ",config.get("cosine_reward",True))
  ###############
  # train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
  #                                           maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
  #                                           multi_domain=config.get("multi_domain", True), picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))

  print("maximum_length", maximum_length)
  train_datasets_p = [] 
  datasets_size = [count_lines(src) for src in source_file]
  picking_prob = [data_size/sum(datasets_size) for data_size in datasets_size]
  temperature=config.get("temperature",1.0)
  picking_prob = [w ** temperature for w in picking_prob]
  picking_prob = [w/sum(picking_prob) for w in picking_prob]
  if config.get("picking_prob",None):
    picking_prob = config.get("picking_prob",None)
  picking_prob = [p/sum(picking_prob) for p in picking_prob]
  print("initial domain picking probability: ", picking_prob)
  for i,src,tgt in zip(domain, source_file, target_file):
    train_datasets_p.append(model.examples_inputter.make_training_dataset(src, tgt,
            batch_size=batch_train_size,
            batch_type=batch_type,
            domain=i,
            single_pass=False,
            shuffle_buffer_size=shuffle_buffer_size,
            length_bucket_width=1,  # Bucketize sequences by the same length for efficiency.
            maximum_features_length=maximum_length,
            maximum_labels_length=maximum_length))
  train_dataset = tf.data.experimental.sample_from_datasets(train_datasets_p, weights=picking_prob)
  with strategy.scope():
    base_dataset = train_dataset
    train_dataset = strategy.experimental_distribute_datasets_from_function(
          lambda _: base_dataset)  
  #############
  train_datasets = [create_training_dataset(strategy, model, [domain], [source_file], [target_file], batch_train_size//2, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=None, temperature=config.get("temperature",1.0))
                                            for domain, source_file, target_file in zip(config.get("domain"), config.get("src"), config.get("tgt"))]

  dev_datasets = [create_training_dataset(strategy, model, [domain], [source_file], [target_file], batch_train_size//2, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=None, temperature=config.get("temperature",1.0))
                                            for domain, source_file, target_file in zip(config.get("eval_domain"), config.get("eval_src"), config.get("eval_ref"))]
  #############
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
    sub_gradient_accumulator = optimizer_util.GradientAccumulator()
    dev_gradient_accumulator = optimizer_util.GradientAccumulator()
    train_gradient_accumulator = optimizer_util.GradientAccumulator()
    domain_rewards = tf.Variable([0.0]*len(domain), trainable=False, aggregation=tf.compat.v1.VariableAggregation.MEAN, synchronization=tf.VariableSynchronization.AUTO)
    d_logits_grad_accumulator = optimizer_util.GradientAccumulator()
    
  print("actor_parameterization: ",config.get("actor_parameterization","softmax"))
  if config.get("actor_parameterization","softmax") =="softmax":
    if config.get("picking_prob",None):
      domain_logits = tf.Variable(np.log(np.array(picking_prob)), dtype=tf.float32, trainable=True)
    else:
      domain_logits = tf.Variable([1.0/domain_num]*domain_num, trainable=True)
  elif config.get("actor_parameterization","softmax") =="linear":
    domain_logits = tf.Variable(picking_prob, trainable=True)
  elif config.get("actor_parameterization","softmax") =="sparsemax":
    domain_logits = tf.Variable([1.0/domain_num]*domain_num, trainable=True)
  elif config.get("actor_parameterization","softmax") =="taylor":
    if config.get("picking_prob",None):
      domain_logits = tf.Variable(np.sqrt(np.array(picking_prob))/sum(np.sqrt(np.array(picking_prob))), dtype=tf.float32, trainable=True)
    else:
      domain_logits = tf.Variable([1.0/domain_num]*domain_num, trainable=True)
    #domain_logits = tf.Variable(np.sqrt(np.array(picking_prob)), dtype=tf.float32, trainable=True)

  
  grad_domain_logits_accum = tf.Variable(tf.zeros_like(domain_logits), trainable=False)
  print("sampler_opt: ", config.get("sampler_opt", "SGD"))
  if config.get("sampler_opt", "SGD") == "SGD":
    sampler_optimizer = tf.keras.optimizers.SGD(learning_rate=config.get("sampler_optim_lr",0.01))
  elif config.get("sampler_opt", "SGD") == "Adam":
    sampler_optimizer = tf.keras.optimizers.Adam(learning_rate=config.get("sampler_optim_lr",0.01)) 
  uniform_prob = tf.Variable([1.0/len(domain)]*len(domain), trainable=False)
  print("init domain_logits: ", domain_logits)
  print("domain_rewards: ", domain_rewards)
  print("domain_importances: ", domain_importances)
  temp = config.get("actor_temperature",1.0)
  print("actor_temperature: ", temp)
  @tf.function
  def _grad_sampler_accum():
    if config.get("actor_parameterization","softmax") =="softmax":
      loss = - tf.reduce_sum(tf.nn.softmax(domain_logits*temp) * domain_rewards)
    elif config.get("actor_parameterization","softmax") =="linear":
      loss = - tf.reduce_sum(domain_logits * domain_rewards)
    elif config.get("actor_parameterization","softmax") =="taylor":
      loss = - tf.reduce_sum(tf.math.square(domain_logits*temp)/tf.reduce_sum(tf.math.square(domain_logits*temp)) * domain_rewards)
    elif config.get("actor_parameterization","softmax") =="sparsemax":
      loss = - tf.reduce_sum(tfa.activations.sparsemax(domain_logits*temp) * domain_rewards)
    
    if config.get("sampler_entropy_constraint",False):
      print("sampler_entropy_constraint_weight",config.get("sampler_entropy_constraint_weight",1e-3))
      if config.get("actor_parameterization","softmax") =="softmax":
        loss +=  tf.reduce_sum(config.get("sampler_entropy_constraint_weight",1e-3) * tf.nn.log_softmax(domain_logits) * tf.nn.softmax(domain_logits))
      elif config.get("actor_parameterization","softmax") =="linear":
        loss +=  tf.reduce_sum(config.get("sampler_entropy_constraint_weight",1e-3) * tf.square(domain_logits - uniform_prob))
      elif config.get("actor_parameterization","softmax") =="taylor":
        loss +=  tf.reduce_sum(config.get("sampler_entropy_constraint_weight",1e-3) * tf.math.square(domain_logits)/tf.reduce_sum(tf.math.square(domain_logits)) * tf.math.log(tf.math.square(domain_logits)/tf.reduce_sum(tf.math.square(domain_logits))))
    grad = sampler_optimizer.get_gradients(loss,[domain_logits])
    grad_domain_logits_accum.assign_add(grad[0])
    return tf.reduce_sum(tf.stop_gradient(tf.nn.softmax(domain_logits)) * domain_rewards)

  @tf.function
  def _sampler_step_1():
    sampler_optimizer.apply_gradients([(grad_domain_logits_accum, domain_logits)])
    if config.get("actor_parameterization","softmax") =="linear":
      domain_logits.assign(tf.clip_by_value(domain_logits, clip_value_min=0.0, clip_value_max=10.0))
      domain_logits.assign(domain_logits/tf.reduce_sum(domain_logits))
    grad_domain_logits_accum.assign(tf.zeros_like(domain_logits))

  def update_sampling_distribution(logits):
    logits = logits.numpy()
    for i, l in enumerate(logits):
        if logits[i] < 0:
            logits[i] = 0
    if sum(logits) == 0:
        logits = [0.1 for _ in range(len(logits))]
    p = np.array(logits) / sum(logits)
    print("new domain probs")
    print(p)
    return p

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    _domain = source["domain"][0]
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    
    if config.get("ADAP_activity_regularizing",False):
      layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
      output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
      d_classification_gate_loss_scale = config.get("d_classification_gate_loss_scale",0.01)
      d_classifier_activity_regularization_loss_scale = config.get("d_classifier_activity_regularization_loss_scale",1.0)
      d_classifier_weight_regularization_losses_scale = config.get("d_classifier_weight_regularization_losses_scale",1.0)
      #print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
      #print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
      #print("d_classification_gate_loss_scale: ", d_classification_gate_loss_scale)
      #print("d_classifier_weight_regularization_losses_scale: ", d_classifier_weight_regularization_losses_scale)
      if isinstance(layer_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
        layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)

      if isinstance(output_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
        output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
      regularization_losses = model.losses
      #print("model_name_scope", model.name_scope())
      #print(regularization_losses)
      layer_activity_regularization_losses = []
      output_activity_regularization_losses = []
      d_classification_gate_losses = []
      d_classifier_activity_regularization_losses = []
      d_classifier_weight_regularization_losses = []
      for loss_ in regularization_losses:
        if "multi_adap__dense" in loss_.name:
          output_activity_regularization_losses.append(loss_)
        elif "ADAP_gate" in loss_.name: #and "ActivityRegularizer" not in loss_.name and "Regularizer" not in loss_.name
          if "ActivityRegularizer" in loss_.name:
            d_classifier_activity_regularization_losses.append(loss_)
          elif "Regularizer" in loss_.name:
            d_classifier_weight_regularization_losses.append(loss_)
          else:
            d_classification_gate_losses.append(loss_)
        elif "ADAP_" in loss_.name:
          layer_activity_regularization_losses.append(loss_)

      #print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
      #print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
      #print("There are %d adaptation regularization loss on domain classification gate_____"%len(d_classification_gate_losses))
      #print("There are %d d_classifier_activity_regularization_losses"%len(d_classifier_activity_regularization_losses))
      #print("There are %d d_classifier_weight_regularization_losses"%len(d_classifier_weight_regularization_losses))
      if (len(layer_activity_regularization_losses)>0) and layer_activity_regularization_loss_scale>0:
        training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)

      if len(output_activity_regularization_losses)>0 and output_activity_regularization_loss_scale>0:
        training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)

      if len(d_classification_gate_losses)>0 and d_classification_gate_loss_scale>0:
        training_loss += d_classification_gate_loss_scale * tf.add_n(d_classification_gate_losses) / importance_weights[domain]

      if len(d_classifier_activity_regularization_losses)>0 and d_classifier_activity_regularization_loss_scale>0:
        training_loss += d_classifier_activity_regularization_loss_scale * tf.add_n(d_classifier_activity_regularization_losses)

      if len(d_classifier_weight_regularization_losses)>0 and d_classifier_weight_regularization_losses_scale>0:
        training_loss += d_classifier_weight_regularization_losses_scale * tf.add_n(d_classifier_weight_regularization_losses)

    variables = model.trainable_variables
    
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples, _domain
  
  def _loss(src, tgt):
    outputs, _ = model(
        src,
        labels=tgt,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, tgt, training=True)
    
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    return reported_loss

  def _accumulate_train_gradients(source, target):
    with tf.GradientTape() as tape:
      variables = model.trainable_variables    
      tape.watch(variables)
      outputs, _ = model(
          source,
          labels=target,
          training=True,
          step=optimizer.iterations,
          inference=False)
      loss = model.compute_loss(outputs, target, training=True)

      if isinstance(loss, tuple):
        training_loss = loss[0] / loss[1]
        reported_loss = loss[0] / loss[2]
      else:
        training_loss, reported_loss = loss, loss

      gradients = tape.gradient(training_loss, variables)
      sub_gradient_accumulator(gradients)
      return training_loss
  
  def _accumulate_dev_gradients(source, target):
    with tf.GradientTape() as tape:
      variables = model.trainable_variables    
      tape.watch(variables)
      outputs, _ = model(
          source,
          labels=target,
          training=False,
          step=optimizer.iterations,
          inference=False)
      loss = model.compute_loss(outputs, target, training=False)

      if isinstance(loss, tuple):
        training_loss = loss[0] / loss[1]
        reported_loss = loss[0] / loss[2]
      else:
        training_loss, reported_loss = loss, loss
      gradients = tape.gradient(training_loss, variables)
      sub_gradient_accumulator(gradients)
      return training_loss
  
  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
  
  def _apply_dev_train_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(sub_gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(sub_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    inner_optimizer.apply_gradients(grads_and_vars)
    sub_gradient_accumulator.reset()

  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples, per_replica_domain = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
      _domain = per_replica_domain
    return loss, num_examples, _domain
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  
  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  def _compute_loss(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=False,
        step=optimizer.iterations,
        inference=False)
    loss = model.compute_loss(outputs, target, training=False)
    
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    return reported_loss
 
  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  dev_iterators = [iter(dev_dataset) for dev_dataset in dev_datasets]
  train_iterators = [iter(train_dataset) for train_dataset in train_datasets]
 
  _, _, _ = next(train_data_flow)

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _number_examples = []
  step = optimizer.iterations.numpy()
  ########
  excluded_params = []
  for var in model.trainable_variables:
    if tf.shape(var)[-1].numpy()==31266 or tf.shape(var)[0].numpy()==31266:
      print(var.name)
      excluded_params.append(var.name)
  ########
  if config.get("reset_step",None):
    print("start from %d-th step"%config.get("reset_step",150000))
    optimizer.iterations.assign(config.get("reset_step",150000))
  
  if step <= 1:
    initializer = config.get("initializer","default")
    if initializer == "default":
      print("Initializing variables by tensorflow default")      
    elif initializer == "variance_scaling":
      print("Initializing variables by tf.variance_scaling")
      initial_value = []
      for v in model.trainable_variables:
        shape = tf.shape(v).numpy()
        initial_value.append(variance_scaling_initialier(shape, scale=1.0, mode="fan_avg", distribution="uniform"))
      weight_reset(initial_value)   
  else:
    print("current domain_logits", config.get("domain_logits",[0.0]*len(domain)))
    domain_logits.assign(config.get("domain_logits",[0.0]*len(domain)))
    if config.get("actor_parameterization","softmax") =="softmax":
      probs = tf.nn.softmax(domain_logits)
    elif config.get("actor_parameterization","softmax") =="linear":
      probs = domain_logits
    elif config.get("actor_parameterization","softmax") =="taylor":
      probs = tf.math.square(domain_logits)/tf.reduce_sum(tf.math.square(domain_logits))
    elif config.get("actor_parameterization","softmax") =="sparsemax":
      probs = tfa.activations.sparsemax(domain_logits)

    new_picking_prob = update_sampling_distribution(probs)
    # create new training course with updated domain distribution
    train_dataset = tf.data.experimental.sample_from_datasets(train_datasets_p, weights=new_picking_prob)
    with strategy.scope():
      base_dataset = train_dataset
      train_dataset = strategy.experimental_distribute_datasets_from_function(
            lambda _: base_dataset)
    @dataset_util.function_on_next(train_dataset)
    def _train_forward(next_fn):    
      with strategy.scope():
        per_replica_source, per_replica_target = next_fn()
        per_replica_loss, per_replica_num_examples, per_replica_domain = strategy.experimental_run_v2(
            _accumulate_gradients, args=(per_replica_source, per_replica_target))
        # TODO: these reductions could be delayed until _step is called.
        loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
        num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
        _domain = per_replica_domain
      return loss, num_examples, _domain
    train_data_flow = iter(_train_forward())

  if config.get("continual_learning", False):
    print("Continual Learning needs to load from old model")
    assert config.get("checkpoint_path") != None
    checkpoint_path = config.get("checkpoint_path")
    load_and_update_if_needed_from_ckpt(config["model_dir"],   
                        checkpoint_path,                        
                        trackables={"model":model},
                        vocab_update=True,
                        model_key="model")

  score_type = config.get("score_type","MultiBLEU")
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  ref_eval_concat = file_concatenate(config["eval_ref"],"ref_eval_concat",dir_name=os.path.join(config["model_dir"],"eval"))
  domain_counts = [0.0] * len(domain)
  with _summary_writer.as_default():
    while True:
      ####Training batch
      loss, num_examples, _domain = next(train_data_flow)    
      _loss.append(loss.numpy())
      _number_examples.append(num_examples.numpy())
      _step()  
      step = optimizer.iterations.numpy()
      if strategy.num_replicas_in_sync >1:
        for v in _domain.values:
          domain_counts[int(v.numpy())] +=1
      else:
        domain_counts[int(_domain.numpy())] +=1

      if step % redistribute_every == 0 and step > config.get("warm_start",5000):
        # compute domain rewards
        rewards = [0.0] * len(domain)
        snapshots = [v.value() for v in model.trainable_variables]
        saved_step = optimizer.iterations.numpy()
        #######
        if config.get("actor_parameterization","softmax") =="softmax":
          current_probs = tf.nn.softmax(domain_logits*temp).numpy()
        elif config.get("actor_parameterization","softmax") =="linear":
          current_probs = domain_logits.numpy()
        elif config.get("actor_parameterization","softmax") =="taylor":
          current_probs = tf.math.square(domain_logits*temp)/tf.reduce_sum(tf.math.square(domain_logits*temp))
        elif config.get("actor_parameterization","softmax") =="sparsemax":
          current_probs = tfa.activations.sparsemax(domain_logits*temp)

        print("current_probs: ", current_probs)
        ####### Prepare dev batch
        dev_batches = []
        for j, dev_iter in enumerate(dev_iterators):
          dev_batches_domain_i = []
          for _ in range(config.get("dev_batch_per_run_num",10)):
            src, tgt = next(dev_iter)
            dev_batches_domain_i.append((src,tgt))
          dev_batches.append(dev_batches_domain_i)
        ####### loss of dev batch at theta_t
        loss_t = [0.0] * len(dev_iterators)
        with strategy.scope():
          for j, dev_iter in enumerate(dev_iterators):
            loss_ = 0
            for src, tgt in dev_batches[j]:
              loss_per_device = strategy.experimental_run_v2(_compute_loss, args=(src, tgt))
              loss_ += strategy.reduce(tf.distribute.ReduceOp.MEAN, loss_per_device, None)
            print("average loss at theta_t on %s: %f"%(config.get("eval_src")[j], loss_.numpy()/len(dev_batches[j])))
            loss_t[j] = loss_.numpy()/len(dev_batches[j])
        #######        
        for i, train_iter in enumerate(train_iterators):
          loss_t_1 = [0.0] * len(dev_iterators)
          _reward = 0.0
          weight_reset(snapshots)
          with strategy.scope():
            ##### compute theta_t+1
            for _ in range(config.get("train_batch_per_run_num",10)): 
              for _ in range(config.get("train_batch_step_accum",10)):
                src, tgt = next(train_iterators[i])
                strategy.experimental_run_v2(_accumulate_train_gradients, args=(src, tgt))
              strategy.experimental_run_v2(lambda: train_gradient_accumulator(sub_gradient_accumulator.gradients))
              strategy.experimental_run_v2(_apply_dev_train_gradients)
            strategy.experimental_run_v2(sub_gradient_accumulator.reset)
            ####### loss of dev batch at theta_t+1
            for j, dev_iter in enumerate(dev_iterators):
              loss_ = 0
              for src, tgt in dev_batches[j]:
                loss_per_device = strategy.experimental_run_v2(_compute_loss, args=(src, tgt))
                loss_ += strategy.reduce(tf.distribute.ReduceOp.MEAN, loss_per_device, None)
              print("average loss at theta_t+1 on %s: %f"%(config.get("eval_src")[j], loss_.numpy()/len(dev_batches[j])))
              loss_t_1[j] = loss_.numpy()/len(dev_batches[j])
            rewards[i] = sum([(max(0,l-l1))*importance for l,l1,importance in zip(loss_t, loss_t_1, domain_importances)])
          
          weight_reset(snapshots)
          optimizer.iterations.assign(saved_step)
        if config.get("reward_rescaling",False):
          rewards = reward_rescale(rewards)
        domain_rewards.assign(tf.cast(tf.constant(rewards), dtype=domain_rewards.dtype))
        
        print("domain rewards", domain_rewards)
        for _ in range(config.get("domain_sampler_optim_step", 30)):
          _ = _grad_sampler_accum()
          _sampler_step_1()
          
        print("domain_logits: ", domain_logits.numpy())
        if config.get("actor_parameterization","softmax") =="softmax":
          probs = tf.nn.softmax(domain_logits*temp)
        elif config.get("actor_parameterization","softmax") =="linear":
          probs = domain_logits
        elif config.get("actor_parameterization","softmax") =="taylor":
          probs = tf.math.square(domain_logits*temp)/tf.reduce_sum(tf.math.square(domain_logits*temp))
        elif config.get("actor_parameterization","softmax") =="sparsemax":
          probs = tfa.activations.sparsemax(domain_logits*temp)
          
        new_picking_prob = update_sampling_distribution(probs)
        tf.summary.experimental.set_step(saved_step)
        for i in range(len(domain)):
          tf.summary.scalar("reward_%d"%i, rewards[i], description="reward of using training set %d"%(i))
          tf.summary.scalar("domain_prob_%d"%i, new_picking_prob[i], description="probability of using training set %d"%(i))
        tf.summary.flush()
        # create new training course with updated domain distribution
        del train_dataset
        del train_data_flow
        train_dataset = tf.data.experimental.sample_from_datasets(train_datasets_p, weights=new_picking_prob)
        with strategy.scope():
          base_dataset = train_dataset
          train_dataset = strategy.experimental_distribute_datasets_from_function(
                lambda _: base_dataset)
        @dataset_util.function_on_next(train_dataset)
        def _train_forward(next_fn):    
          with strategy.scope():
            per_replica_source, per_replica_target = next_fn()
            per_replica_loss, per_replica_num_examples, per_replica_domain = strategy.experimental_run_v2(
                _accumulate_gradients, args=(per_replica_source, per_replica_target))
            # TODO: these reductions could be delayed until _step is called.
            loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
            num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
            _domain = per_replica_domain #strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_domain, None)
          return loss, num_examples, _domain

        train_data_flow = iter(_train_forward())
        #######
        weight_reset(snapshots)
        optimizer.iterations.assign(saved_step)
        #print("previous domain count: ", domain_counts)
        print("previous domain count in percentage: ",[d/sum(domain_counts) for d in domain_counts])
        domain_counts = [0.0] * len(domain)
        #######

      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
          "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
          step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        output_files = []
        new_bleu = 0.0
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
          output_files.append(output_file)
          new_bleu += score * domain_importances[i]
        ##### BLEU on concat dev set.
        output_file_concat = file_concatenate(output_files,"output_file_concat.%s"%os.path.basename(checkpoint_path))
        score = scorer(ref_eval_concat, output_file_concat)
        print("score of model %s on concat dev set: "%checkpoint_manager.latest_checkpoint, score)
        tf.summary.scalar("concat_eval_score", score, description="BLEU on concat dev set")
        #############################
        if new_bleu >= current_max_eval_bleu:
          current_max_eval_bleu = new_bleu
          descending_streak = 0
        else:
          descending_streak += 1
      tf.summary.flush()
      if config.get("early_stopping",True) and descending_streak >= 5:
        break
      if step > train_steps:
        break

def train_L2W_g(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          adapter_optimizer=None,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=3000000,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  redistribute_every = config.get("redistribute_every",2000)
  if config.get("use_meta_optimizer",False):
    inner_optimizer = tf.keras.optimizers.SGD(config.get("meta_train_lr",0.001))
  else:
    inner_optimizer = optimizer
  #####
  if checkpoint_path is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
    checkpoint.restore(checkpoint_path)
  else:
    if checkpoint_manager.latest_checkpoint is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
      checkpoint.restore(checkpoint_manager.latest_checkpoint)    
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  eval_domain = config.get("eval_domain")
  ###### early stopping criterion
  current_max_eval_bleu = 0.0
  descending_streak = 0
  ######
  if not config.get("domain_importances",None):
    domain_importances = [1.0/len(eval_domain)]*len(eval_domain)
  else:
    domain_importances = config.get("domain_importances")
  print("There are %d in-domain corpora"%len(source_file))
  ###############
  print("cosine_reward: ",config.get("cosine_reward",True))
  ###############
  # train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
  #                                           maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
  #                                           multi_domain=config.get("multi_domain", True), picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))

  print("maximum_length", maximum_length)
  train_datasets_p = [] 
  domain_num = len(domain)
  datasets_size = [count_lines(src) for src in source_file]
  picking_prob = [data_size/sum(datasets_size) for data_size in datasets_size]
  temperature=config.get("temperature",1.0)
  picking_prob = [w ** temperature for w in picking_prob]
  picking_prob = [w/sum(picking_prob) for w in picking_prob]
  if config.get("picking_prob",None):
    picking_prob = config.get("picking_prob",None)
  print("initial domain picking probability: ", picking_prob)
  for i,src,tgt in zip(domain, source_file, target_file):
    train_datasets_p.append(model.examples_inputter.make_training_dataset(src, tgt,
            batch_size=batch_train_size,
            batch_type=batch_type,
            domain=i,
            single_pass=False,
            shuffle_buffer_size=shuffle_buffer_size,
            length_bucket_width=1,  # Bucketize sequences by the same length for efficiency.
            maximum_features_length=maximum_length,
            maximum_labels_length=maximum_length))
  train_dataset = tf.data.experimental.sample_from_datasets(train_datasets_p, weights=picking_prob)
  with strategy.scope():
    base_dataset = train_dataset
    train_dataset = strategy.experimental_distribute_datasets_from_function(
          lambda _: base_dataset)  
  #############
  train_datasets = [create_training_dataset(strategy, model, [domain], [source_file], [target_file], batch_train_size//2, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=None, temperature=config.get("temperature",1.0))
                                            for domain, source_file, target_file in zip(config.get("domain"), config.get("src"), config.get("tgt"))]

  dev_datasets = [create_training_dataset(strategy, model, [domain], [source_file], [target_file], batch_train_size//2, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=None, temperature=config.get("temperature",1.0))
                                            for domain, source_file, target_file in zip(config.get("eval_domain"), config.get("eval_src"), config.get("eval_ref"))]
  #############
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
    sub_gradient_accumulator = optimizer_util.GradientAccumulator()
    
    dev_gradient_accumulator = optimizer_util.GradientAccumulator()
    train_gradient_accumulator = optimizer_util.GradientAccumulator()
    domain_rewards = tf.Variable([0.0]*len(domain), trainable=False, aggregation=tf.compat.v1.VariableAggregation.MEAN, synchronization=tf.VariableSynchronization.AUTO)
    d_logits_grad_accumulator = optimizer_util.GradientAccumulator()
    
  print("actor_parameterization: ",config.get("actor_parameterization","softmax"))
  temp = config.get("actor_temperature",1.0)
  if config.get("actor_parameterization","softmax") =="softmax":
    if config.get("picking_prob",None):
      domain_logits = tf.Variable(np.log(np.array(picking_prob)), dtype=tf.float32, trainable=True)
    else:
      domain_logits = tf.Variable([1.0/domain_num]*domain_num, trainable=True)
  elif config.get("actor_parameterization","softmax") =="linear":
    domain_logits = tf.Variable(picking_prob, trainable=True)
  elif config.get("actor_parameterization","softmax") =="sparsemax":
    domain_logits = tf.Variable([1.0/domain_num]*domain_num, trainable=True)
  elif config.get("actor_parameterization","softmax") =="taylor":
    """ if config.get("picking_prob",None):
      domain_logits = tf.Variable(np.sqrt(np.array(picking_prob)), dtype=tf.float32, trainable=True)
    else:
      domain_logits = tf.Variable([1.0/domain_num]*domain_num, trainable=True) """
    domain_logits = tf.Variable(np.sqrt(np.array(picking_prob)), dtype=tf.float32, trainable=True)
  
  grad_domain_logits_accum = tf.Variable(tf.zeros_like(domain_logits), trainable=False)
  print("sampler_opt: ", config.get("sampler_opt", "SGD"))
  if config.get("sampler_opt", "SGD") == "SGD":
    sampler_optimizer = tf.keras.optimizers.SGD(learning_rate=config.get("sampler_optim_lr",0.01))
  elif config.get("sampler_opt", "SGD") == "Adam":
    sampler_optimizer = tf.keras.optimizers.Adam(learning_rate=config.get("sampler_optim_lr",0.01)) 
  sampler_vars = [domain_logits]
  uniform_prob = tf.Variable([1.0/len(domain)]*len(domain), trainable=False)
  print("init domain_logits: ", domain_logits)
  print("domain_rewards: ", domain_rewards)
  print("domain_importances: ", domain_importances)
  
  @tf.function
  def _grad_sampler_accum():
    if config.get("actor_parameterization","softmax") =="softmax":
      loss = - tf.reduce_sum(tf.stop_gradient(tf.nn.softmax(domain_logits)) * tf.nn.log_softmax(domain_logits) * domain_rewards)
    elif config.get("actor_parameterization","softmax") =="linear":
      loss = - tf.reduce_sum(domain_logits * domain_rewards)
    if config.get("sampler_entropy_constraint",False):
      print("sampler_entropy_constraint_weight",config.get("sampler_entropy_constraint_weight",1e-3))
      if config.get("actor_parameterization","softmax") =="softmax":
        loss +=  tf.reduce_sum(config.get("sampler_entropy_constraint_weight",1e-3) * tf.nn.log_softmax(domain_logits) * tf.nn.softmax(domain_logits))
      elif config.get("actor_parameterization","softmax") =="linear":
        loss +=  tf.reduce_sum(config.get("sampler_entropy_constraint_weight",1e-3) * tf.square(domain_logits - uniform_prob))#tf.reduce_sum(config.get("sampler_entropy_constraint_weight",1e-3) * domain_logits * tf.math.log(domain_logits))
    grad = sampler_optimizer.get_gradients(loss,[domain_logits])
    grad_domain_logits_accum.assign_add(grad[0])
    return tf.reduce_sum(tf.stop_gradient(tf.nn.softmax(domain_logits)) * domain_rewards)

  @tf.function
  def _sampler_step_1():
    sampler_optimizer.apply_gradients([(grad_domain_logits_accum, domain_logits)])
    if config.get("actor_parameterization","softmax") =="linear":
      #domain_logits.assign(domain_logits - tf.reduce_min(domain_logits))
      domain_logits.assign(tf.clip_by_value(domain_logits, clip_value_min=0.0, clip_value_max=10.0))
      domain_logits.assign(domain_logits/tf.reduce_sum(domain_logits))
    grad_domain_logits_accum.assign(tf.zeros_like(domain_logits))

  def update_sampling_distribution(logits):
    logits = logits.numpy()
    for i, l in enumerate(logits):
        if logits[i] < 0:
            logits[i] = 0
    if sum(logits) == 0:
        logits = [0.1 for _ in range(len(logits))]
    p = np.array(logits) / sum(logits)
    print("new domain probs")
    print(p)
    return p

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    _domain = source["domain"][0]
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss

    variables = model.trainable_variables
    
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples, _domain
  
  def _loss(src, tgt):
    outputs, _ = model(
        src,
        labels=tgt,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, tgt, training=True)
    
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    return reported_loss

  def _accumulate_train_gradients(source, target):
    with tf.GradientTape() as tape:
      variables = model.trainable_variables    
      tape.watch(variables)
      outputs, _ = model(
          source,
          labels=target,
          training=True,
          step=optimizer.iterations,
          inference=False)
      loss = model.compute_loss(outputs, target, training=True)

      if isinstance(loss, tuple):
        training_loss = loss[0] / loss[1]
        reported_loss = loss[0] / loss[2]
      else:
        training_loss, reported_loss = loss, loss

      #tf.print(loss[1],loss[2],sep="|")
      gradients = tape.gradient(training_loss, variables)
      sub_gradient_accumulator(gradients)
      return training_loss
  
  def _accumulate_dev_gradients(source, target):
    with tf.GradientTape() as tape:
      variables = model.trainable_variables    
      tape.watch(variables)
      outputs, _ = model(
          source,
          labels=target,
          training=False,
          step=optimizer.iterations,
          inference=False)
      loss = model.compute_loss(outputs, target, training=False)

      if isinstance(loss, tuple):
        training_loss = loss[0] / loss[1]
        reported_loss = loss[0] / loss[2]
      else:
        training_loss, reported_loss = loss, loss
      #tf.print(loss[1],loss[2],sep="|")
      gradients = tape.gradient(training_loss, variables)
      sub_gradient_accumulator(gradients)
      return training_loss
  
  def _reset_dev_train_gradients():
    dev_gradient_accumulator.reset() # for dev_gradient_accumulator in dev_gradient_accumulators]
    [train_gradient_accumulator.reset() for train_gradient_accumulator in train_gradient_accumulators]

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
  
  def _apply_dev_train_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(sub_gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(sub_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    inner_optimizer.apply_gradients(grads_and_vars)
    sub_gradient_accumulator.reset()
 
  def _apply_sampler_gradients():
    grads_and_vars = []
    scaled_gradient = d_logits_grad_accumulator.gradients[0] / (strategy.num_replicas_in_sync * tf.cast(d_logits_grad_accumulator.step, tf.float32))
    grads_and_vars.append((scaled_gradient, domain_logits))
    sampler_optimizer.apply_gradients(grads_and_vars)
    d_logits_grad_accumulator.reset()

  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples, per_replica_domain = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
      _domain = per_replica_domain
    return loss, num_examples, _domain
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  @tf.function
  def _dev_train_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_dev_train_gradients)

  @tf.function
  def _sampler_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_sampler_gradients)

  @tf.function
  def _reset_dev_train_grad_accum_step():
    with strategy.scope():
      _reset_dev_train_gradients()
  
  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  def _compute_loss(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=False,
        step=optimizer.iterations,
        inference=False)
    loss = model.compute_loss(outputs, target, training=False)
    
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    return reported_loss
 
  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  dev_iterators = [iter(dev_dataset) for dev_dataset in dev_datasets]
  train_iterators = [iter(train_dataset) for train_dataset in train_datasets]
 
  _, _, _ = next(train_data_flow)

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _number_examples = []
  step = optimizer.iterations.numpy()
  ########
  excluded_params = []
  for var in model.trainable_variables:
    if tf.shape(var)[-1].numpy()==31266 or tf.shape(var)[0].numpy()==31266:
      print(var.name)
      excluded_params.append(var.name)
  ########
  if config.get("reset_step",None):
    print("start from %d-th step"%config.get("reset_step",150000))
    optimizer.iterations.assign(config.get("reset_step",150000))
  
  if step <= 1:
    initializer = config.get("initializer","default")
    if initializer == "default":
      print("Initializing variables by tensorflow default")      
    elif initializer == "variance_scaling":
      print("Initializing variables by tf.variance_scaling")
      initial_value = []
      for v in model.trainable_variables:
        shape = tf.shape(v).numpy()
        initial_value.append(variance_scaling_initialier(shape, scale=1.0, mode="fan_avg", distribution="uniform"))
      weight_reset(initial_value)   
  else:
    print("current domain_logits", config.get("domain_logits",[0.0]*len(domain)))
    domain_logits.assign(config.get("domain_logits",[0.0]*len(domain)))
    if config.get("actor_parameterization","softmax") =="softmax":
      probs = tf.nn.softmax(domain_logits)
    elif config.get("actor_parameterization","softmax") =="linear":
      probs = domain_logits
    elif config.get("actor_parameterization","softmax") =="taylor":
      probs = tf.math.square(domain_logits)/tf.reduce_sum(tf.math.square(domain_logits))
    elif config.get("actor_parameterization","softmax") =="sparsemax":
      probs = tfa.activations.sparsemax(domain_logits)
    
    new_picking_prob = update_sampling_distribution(probs)
    # create new training course with updated domain distribution
    train_dataset = tf.data.experimental.sample_from_datasets(train_datasets_p, weights=new_picking_prob)
    with strategy.scope():
      base_dataset = train_dataset
      train_dataset = strategy.experimental_distribute_datasets_from_function(
            lambda _: base_dataset)
    @dataset_util.function_on_next(train_dataset)
    def _train_forward(next_fn):    
      with strategy.scope():
        per_replica_source, per_replica_target = next_fn()
        per_replica_loss, per_replica_num_examples, per_replica_domain = strategy.experimental_run_v2(
            _accumulate_gradients, args=(per_replica_source, per_replica_target))
        # TODO: these reductions could be delayed until _step is called.
        loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
        num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
        _domain = per_replica_domain
      return loss, num_examples, _domain
    train_data_flow = iter(_train_forward())

  if config.get("continual_learning", False):
    print("Continual Learning needs to load from old model")
    assert config.get("checkpoint_path") != None
    checkpoint_path = config.get("checkpoint_path")
    load_and_update_if_needed_from_ckpt(config["model_dir"],   
                        checkpoint_path,                        
                        trackables={"model":model},
                        vocab_update=True,
                        model_key="model")

  score_type = config.get("score_type","MultiBLEU")
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  ref_eval_concat = file_concatenate(config["eval_ref"],"ref_eval_concat",dir_name=os.path.join(config["model_dir"],"eval"))
  domain_counts = [0.0] * len(domain)
  print("reward_formula: ", config.get("reward_formula","g-cosine"))
  with _summary_writer.as_default():
    while True:
      ####Training batch
      loss, num_examples, _domain = next(train_data_flow)    
      _loss.append(loss.numpy())
      _number_examples.append(num_examples.numpy())
      _step()  
      step = optimizer.iterations.numpy()
      if strategy.num_replicas_in_sync >1:
        for v in _domain.values:
          domain_counts[int(v.numpy())] +=1
      else:
        domain_counts[int(_domain.numpy())] +=1

      if step % redistribute_every == 0 and step > config.get("warm_start",5000):
        # compute domain rewards
        rewards = [0.0] * len(domain)
        snapshots = [v.value() for v in model.trainable_variables]
        saved_step = optimizer.iterations.numpy()
        #######
        if config.get("actor_parameterization","softmax") =="softmax":
          current_probs = tf.nn.softmax(domain_logits).numpy()
        elif config.get("actor_parameterization","softmax") =="linear":
          current_probs = domain_logits.numpy()
        print("current_probs: ", current_probs)
        ####### Prepare dev batch
        dev_batches = []
        for j, dev_iter in enumerate(dev_iterators):
          dev_batches_domain_i = []
          for _ in range(config.get("dev_batch_per_run_num",10)):
            src, tgt = next(dev_iter)
            dev_batches_domain_i.append((src,tgt))
          dev_batches.append(dev_batches_domain_i)
        #######        
        for i, train_iter in enumerate(train_iterators):
          _reward = 0.0
          weight_reset(snapshots)
          with strategy.scope():
            ##### compute theta_t+1
            for _ in range(config.get("train_batch_per_run_num",10)): 
              for _ in range(config.get("train_batch_step_accum",10)):
                src, tgt = next(train_iterators[i])
                strategy.experimental_run_v2(_accumulate_train_gradients, args=(src, tgt))
              strategy.experimental_run_v2(lambda: train_gradient_accumulator(sub_gradient_accumulator.gradients))
              strategy.experimental_run_v2(_apply_dev_train_gradients)
            strategy.experimental_run_v2(sub_gradient_accumulator.reset)
          with strategy.scope():
            for j, dev_iter in enumerate(dev_iterators):
              _sum = 0.0
              _dev_norm = 0.0
              _tr_norm = 0.0
              #count = 0
              for src, tgt in dev_batches[j]:
                #print("valid domain: %d: "%j,src["domain"])
                strategy.experimental_run_v2(_accumulate_dev_gradients, args=(src, tgt))
              strategy.experimental_run_v2(lambda: dev_gradient_accumulator(sub_gradient_accumulator.gradients))
              strategy.experimental_run_v2(sub_gradient_accumulator.reset)         
              for dev_grad, tr_grad, var, snapshot in zip(dev_gradient_accumulator._gradients, train_gradient_accumulator._gradients, model.trainable_variables, snapshots):
                if config.get("reward_formula","g-cosine")=="g-cosine":
                  _sum += tf.reduce_sum(dev_grad * tr_grad)
                  _dev_norm += tf.reduce_sum(dev_grad * dev_grad)
                  _tr_norm += tf.reduce_sum(tr_grad * tr_grad)
                elif config.get("reward_formula","g-cosine")=="u-cosine":
                  tr_grad = snapshot - var
                  _sum += tf.reduce_sum(dev_grad * tr_grad)
                  _dev_norm += tf.reduce_sum(dev_grad * dev_grad)
                  _tr_norm += tf.reduce_sum(tr_grad * tr_grad)
              if config.get("cosine_reward",True):
                _reward += _sum / (tf.sqrt(_dev_norm * _tr_norm) + 1e-10) * domain_importances[j]
              else:
                _reward += _sum * domain_importances[j] #_sum * learning_rate(saved_step) * domain_importances[j]
              # reset dev gradient accumulations to zero
              strategy.experimental_run_v2(dev_gradient_accumulator.reset)
            # reset train dev gradient accumulations to zero
            strategy.experimental_run_v2(train_gradient_accumulator.reset)
          rewards[i] = _reward.numpy()
          #reset model parameters
          weight_reset(snapshots)
          optimizer.iterations.assign(saved_step)
        domain_rewards.assign(tf.cast(tf.constant(rewards), dtype=domain_rewards.dtype))
        # if not config.get("cosine_reward",True):
        #   domain_rewards.assign(tf.clip_by_value(domain_rewards, clip_value_min=-1.0, clip_value_max=1.0))
        # compute new domain distribution
        print("domain rewards", domain_rewards)
        for _ in range(config.get("domain_sampler_optim_step", 30)):
          _ = _grad_sampler_accum()
          _sampler_step_1()
          
        print("domain_logits: ", domain_logits.numpy())
        if config.get("actor_parameterization","softmax") =="softmax":
          probs = tf.nn.softmax(domain_logits)
        elif config.get("actor_parameterization","softmax") =="linear":
          probs = domain_logits
        new_picking_prob = update_sampling_distribution(probs)
        tf.summary.experimental.set_step(saved_step)
        for i in range(len(domain)):
          tf.summary.scalar("reward_%d"%i, rewards[i], description="reward of using training set %d"%(i))
          tf.summary.scalar("domain_prob_%d"%i, new_picking_prob[i], description="probability of using training set %d"%(i))
        tf.summary.flush()
        # create new training course with updated domain distribution
        del train_dataset
        del train_data_flow
        train_dataset = tf.data.experimental.sample_from_datasets(train_datasets_p, weights=new_picking_prob)
        with strategy.scope():
          base_dataset = train_dataset
          train_dataset = strategy.experimental_distribute_datasets_from_function(
                lambda _: base_dataset)
        @dataset_util.function_on_next(train_dataset)
        def _train_forward(next_fn):    
          with strategy.scope():
            per_replica_source, per_replica_target = next_fn()
            per_replica_loss, per_replica_num_examples, per_replica_domain = strategy.experimental_run_v2(
                _accumulate_gradients, args=(per_replica_source, per_replica_target))
            # TODO: these reductions could be delayed until _step is called.
            loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
            num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
            _domain = per_replica_domain #strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_domain, None)
          return loss, num_examples, _domain
        train_data_flow = iter(_train_forward())
        #######
        weight_reset(snapshots)
        optimizer.iterations.assign(saved_step)
        #print("previous domain count: ", domain_counts)
        print("previous domain count in percentage: ",[d/sum(domain_counts) for d in domain_counts])
        domain_counts = [0.0] * len(domain)
        #######

      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
          "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
          step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        output_files = []
        new_bleu = 0.0
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
          output_files.append(output_file)
          new_bleu += score * domain_importances[i]
        ##### BLEU on concat dev set.
        output_file_concat = file_concatenate(output_files,"output_file_concat.%s"%os.path.basename(checkpoint_path))
        score = scorer(ref_eval_concat, output_file_concat)
        print("score of model %s on concat dev set: "%checkpoint_manager.latest_checkpoint, score)
        tf.summary.scalar("concat_eval_score", score, description="BLEU on concat dev set")
        #############################
        if new_bleu >= current_max_eval_bleu:
          current_max_eval_bleu = new_bleu
          descending_streak = 0
        else:
          descending_streak += 1
      tf.summary.flush()
      if config.get("early_stopping",True) and descending_streak >= 5:
        break
      if step > train_steps:
        break

def train_NGD_L2W_v1(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  if config.get("report_every",None)!=None:
    report_every = config.get("report_every")
  hessian_update_every = config.get("hessian_update_every",100)
  redistribute_every = config.get("redistribute_every",2000)
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  else:
    if checkpoint_path is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
      checkpoint.restore(checkpoint_path)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_hessian_size = config.get("batch_hessian_size",10)
  print("batch_hessian_size: ", batch_hessian_size, "examples")
  print("batch_train_size: ", batch_train_size, batch_type)
  hessian_accum_step = config.get("hessian_accum_step",1)
  step = optimizer.iterations.numpy()
  print("current learning rate: ", learning_rate(step))
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  eval_domain = config.get("eval_domain")
  ###### early stopping criterion
  current_max_eval_bleu = 0.0
  descending_streak = 0
  ######
  if not config.get("domain_importances",None):
    domain_importances = [1.0/len(eval_domain)]*len(eval_domain)
  else:
    domain_importances = config.get("domain_importances")
  print("There are %d in-domain corpora"%len(source_file))
  ###############
  print("maximum_length", maximum_length)
  train_datasets_p = [] 
  datasets_size = [count_lines(src) for src in source_file]
  picking_prob = [data_size/sum(datasets_size) for data_size in datasets_size]
  if config.get("picking_prob",None):
    picking_prob = config.get("picking_prob",None)
  print("initial domain picking probability: ", picking_prob)
  for i,src,tgt in zip(domain, source_file, target_file):
    train_datasets_p.append(model.examples_inputter.make_training_dataset(src, tgt,
            batch_size=batch_train_size,
            batch_type=batch_type,
            domain=i,
            single_pass=False,
            shuffle_buffer_size=shuffle_buffer_size,
            length_bucket_width=1,  # Bucketize sequences by the same length for efficiency.
            maximum_features_length=maximum_length,
            maximum_labels_length=maximum_length))
  train_dataset = tf.data.experimental.sample_from_datasets(train_datasets_p, weights=picking_prob)
  with strategy.scope():
    base_dataset = train_dataset
    train_dataset = strategy.experimental_distribute_datasets_from_function(
          lambda _: base_dataset)  
  #############
  train_datasets = [create_training_dataset(strategy, model, [domain], [source_file], [target_file], batch_train_size//2, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=None, temperature=config.get("temperature",1.0))
                                            for domain, source_file, target_file in zip(config.get("domain"), config.get("src"), config.get("tgt"))]

  dev_datasets = [create_training_dataset(strategy, model, [domain], [source_file], [target_file], batch_train_size//2, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=None, temperature=config.get("temperature",1.0))
                                            for domain, source_file, target_file in zip(config.get("eval_domain"), config.get("eval_src"), config.get("eval_ref"))]
  #############
  hessian_datasets = create_training_dataset(strategy, model, domain, config.get("hessian_src", source_file), 
                                            config.get("hessian_ref", target_file), batch_hessian_size, "examples", shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=None, 
                                            temperature=config.get("temperature",1.0), pick_in_order=True)
  #############
  
  ### update factore of diag hessians
  alpha = config.get("hessian_moving_rate",0.1)
  epsilon = config.get("posterior_weight",1e-8)
  print("epsilon: ", epsilon)
  print("hessian_moving_rate: ", alpha)
  #####from utils.dataprocess import count_lines
  datasets_size = [count_lines(src) for src in source_file]
  importance_weights = [data_size/sum(datasets_size) for data_size in datasets_size]
  temperature=config.get("hessian_temperature",1.0)
  importance_weights = [w ** (temperature) for w in importance_weights]
  importance_weights = [w/sum(importance_weights) for w in importance_weights]
  
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator() 
    hessian_accumulators = optimizer_util.DiagHessianAccumulator()
    # hessian_accumulators = [tf.Variable(
    #         tf.zeros_like(var),
    #         trainable=False, synchronization=tf.VariableSynchronization.ON_READ) for var in model.trainable_variables]
    rescale_sum = tf.Variable(0.0, trainable=False, synchronization=tf.VariableSynchronization.ON_READ)
    hessian_moving_stats = [tf.Variable(
            tf.zeros_like(var),
            trainable=False, aggregation=tf.compat.v1.VariableAggregation.MEAN, synchronization=tf.VariableSynchronization.AUTO) for var in model.trainable_variables]
    normalized_hessian_moving_stats = [tf.Variable(
            tf.zeros_like(var),
            trainable=False, aggregation=tf.compat.v1.VariableAggregation.MEAN, synchronization=tf.VariableSynchronization.AUTO) for var in model.trainable_variables]
    importance_weights = tf.constant(importance_weights)
    tf.print("importance_weights: ", importance_weights)
    sub_gradient_accumulator = optimizer_util.GradientAccumulator()
    dev_gradient_accumulator = optimizer_util.GradientAccumulator()
    train_gradient_accumulator = optimizer_util.GradientAccumulator()
    domain_rewards = tf.Variable([0.0]*len(domain), trainable=False, aggregation=tf.compat.v1.VariableAggregation.MEAN, synchronization=tf.VariableSynchronization.AUTO)
    d_logits_grad_accumulator = optimizer_util.GradientAccumulator()
  domain_logits = tf.Variable([0.0]*len(domain), trainable=True)
  grad_domain_logits_accum = tf.Variable(tf.zeros_like(domain_logits), trainable=False)
  sampler_optimizer = tf.keras.optimizers.Adam(learning_rate=config.get("sampler_optim_lr",0.01))
  print("domain_rewards: ", domain_rewards)
  print("domain_importances: ", domain_importances)
  
  @tf.function
  def _grad_sampler_accum():
    loss = - tf.reduce_sum(tf.stop_gradient(tf.nn.softmax(domain_logits)) * tf.nn.log_softmax(domain_logits) * domain_rewards)
    if config.get("sampler_entropy_constraint",False):
      print("sampler_entropy_constraint_weight",config.get("sampler_entropy_constraint_weight",1e-3))
      loss +=  tf.reduce_sum(config.get("sampler_entropy_constraint_weight",1e-3) * tf.nn.log_softmax(domain_logits) * tf.nn.softmax(domain_logits))
    grad = sampler_optimizer.get_gradients(loss,[domain_logits])
    grad_domain_logits_accum.assign_add(grad[0])
    return tf.reduce_sum(tf.stop_gradient(tf.nn.softmax(domain_logits)) * domain_rewards)
  @tf.function
  def _sampler_step_1():
    sampler_optimizer.apply_gradients([(grad_domain_logits_accum, domain_logits)])
    grad_domain_logits_accum.assign(tf.zeros_like(domain_logits))
  def update_sampling_distribution(logits):
    logits = logits.numpy()
    for i, l in enumerate(logits):
        if logits[i] < 0:
            logits[i] = 0
    if sum(logits) == 0:
        logits = [0.1 for _ in range(len(logits))]
    p = np.array(logits) / sum(logits)
    print("new domain probs")
    print(p)
    return p
  #########  
  def _accumulate_diag_hessians(source,target): 
    with tf.GradientTape(persistent=True) as tape:  
      variables = model.trainable_variables
      tape.watch(variables)
      outputs, _ = model(
          source,
          labels=target,
          training=True,
          step=optimizer.iterations)
      _dom = source["domain"][0]
      loss = model.compute_individual_loss(outputs, target, training=True)
      def hessian_accum_along_loss(diag_hessian_acc, x):
        gradients = tape.gradient(x,variables)
        _hessians = []
        for grad in gradients:
          _hessians.append(tf.square(grad))
        hessian_accumulators(_hessians)
        return diag_hessian_acc
      tf.scan(hessian_accum_along_loss, loss, parallel_iterations=batch_hessian_size)
  def _accumulate_NGD_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    if config.get("ADAP_activity_regularizing",False):
      layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
      output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
      d_classification_gate_loss_scale = config.get("d_classification_gate_loss_scale",0.01)
      d_classifier_activity_regularization_loss_scale = config.get("d_classifier_activity_regularization_loss_scale",1.0)
      d_classifier_weight_regularization_losses_scale = config.get("d_classifier_weight_regularization_losses_scale",1.0)
      
      if isinstance(layer_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
        layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        #tf.print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale, "domain: ", domain)
      if isinstance(output_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
        output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
      regularization_losses = model.losses
      # print("model_name_scope", model.name_scope())
      # print(regularization_losses)
      layer_activity_regularization_losses = []
      output_activity_regularization_losses = []
      d_classification_gate_losses = []
      d_classifier_activity_regularization_losses = []
      d_classifier_weight_regularization_losses = []
      for loss_ in regularization_losses:
        if "multi_adap__dense" in loss_.name:
          output_activity_regularization_losses.append(loss_)
        elif "ADAP_gate" in loss_.name: #and "ActivityRegularizer" not in loss_.name and "Regularizer" not in loss_.name
          if "ActivityRegularizer" in loss_.name:
            d_classifier_activity_regularization_losses.append(loss_)
          elif "Regularizer" in loss_.name:
            d_classifier_weight_regularization_losses.append(loss_)
          else:
            d_classification_gate_losses.append(loss_)
        elif "ADAP_" in loss_.name:
          layer_activity_regularization_losses.append(loss_)

      if (len(layer_activity_regularization_losses)>0) and layer_activity_regularization_loss_scale>0:
        training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)

      if len(output_activity_regularization_losses)>0 and output_activity_regularization_loss_scale>0:
        training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)

      if len(d_classification_gate_losses)>0 and d_classification_gate_loss_scale>0:
        training_loss += d_classification_gate_loss_scale * tf.add_n(d_classification_gate_losses) 

      if len(d_classifier_activity_regularization_losses)>0 and d_classifier_activity_regularization_loss_scale>0:
        training_loss += d_classifier_activity_regularization_loss_scale * tf.add_n(d_classifier_activity_regularization_losses)

      if len(d_classifier_weight_regularization_losses)>0 and d_classifier_weight_regularization_losses_scale>0:
        training_loss += d_classifier_weight_regularization_losses_scale * tf.add_n(d_classifier_weight_regularization_losses)
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    #for var in variables:
    #  print(var.name)
    gradients = optimizer.get_gradients(training_loss, variables)
    new_gradients = []
    rescale_sum.assign(0.0)
    for gradient, hessian_moving_stat, var in zip(gradients, normalized_hessian_moving_stats, variables):
      if isinstance(gradient,tf.IndexedSlices):
        if "embedding" in var.name:
          rescale_sum.assign_add(tf.reduce_sum(tf.square(gradient.values)/ (tf.nn.embedding_lookup(hessian_moving_stat, gradient.indices) + epsilon)))
        else:
          rescale_sum.assign_add(tf.reduce_sum(tf.square(gradient.values)/ epsilon))
        #tf.print("hessian %s: "%var.name, tf.nn.embedding_lookup(hessian_moving_stat.value(), gradient.indices), "indices: ", gradient.indices, sep="|")
        #tf.print("hessian_stat: ", hessian_moving_stat.value())
        #continue
      else:
        rescale_sum.assign_add(tf.reduce_sum(tf.square(gradient) / (hessian_moving_stat + epsilon)))
        #tf.print("hessian %s: "%var.name, hessian_moving_stat.value())
    #tf.print("rescale_sum: ", rescale_sum)
    for gradient, hessian_moving_stat, var in zip(gradients, normalized_hessian_moving_stats, variables):
      if isinstance(gradient,tf.IndexedSlices):
        # new_gradients.append(gradient)
        # new_gradients.append(tf.IndexedSlices(gradient.values / (tf.nn.embedding_lookup(hessian_moving_stat.value(), gradient.indices) + epsilon) 
        # * 1 / tf.sqrt(tf.reduce_sum(tf.square(gradient.values)/ (tf.nn.embedding_lookup(hessian_moving_stat.value(), gradient.indices) + epsilon))), 
        # gradient.indices, dense_shape=gradient.dense_shape))
        if "embedding" in var.name:
          new_gradients.append(tf.IndexedSlices(gradient.values / (tf.nn.embedding_lookup(hessian_moving_stat, gradient.indices) + epsilon) 
         * 1 / tf.sqrt(rescale_sum.value()), 
         gradient.indices, dense_shape=gradient.dense_shape))
        else:
          new_gradients.append(tf.IndexedSlices(gradient.values / epsilon * 1 / tf.sqrt(rescale_sum.value()), 
         gradient.indices, dense_shape=gradient.dense_shape))
        # tf.print("hessian_%s: "%var.name, tf.nn.embedding_lookup(hessian_moving_stat.value(), gradient.indices))
      else:
        # new_gradients.append(gradient / (hessian_moving_stat.value() +epsilon) * 1 / tf.sqrt(tf.reduce_sum(tf.square(gradient) / (hessian_moving_stat.value()+epsilon))))
        new_gradients.append(gradient / (hessian_moving_stat + epsilon) * 1 / tf.sqrt(rescale_sum.value()))
    gradient_accumulator(new_gradients)
    num_examples = tf.reduce_sum(target["length"])
    return reported_loss, num_examples
  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    if config.get("ADAP_activity_regularizing",False):
      layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
      output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
      d_classification_gate_loss_scale = config.get("d_classification_gate_loss_scale",0.01)
      d_classifier_activity_regularization_loss_scale = config.get("d_classifier_activity_regularization_loss_scale",1.0)
      d_classifier_weight_regularization_losses_scale = config.get("d_classifier_weight_regularization_losses_scale",1.0)
      
      if isinstance(layer_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
        layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)
        #tf.print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale, "domain: ", domain)
      if isinstance(output_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
        output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
      regularization_losses = model.losses
      # print("model_name_scope", model.name_scope())
      # print(regularization_losses)
      layer_activity_regularization_losses = []
      output_activity_regularization_losses = []
      d_classification_gate_losses = []
      d_classifier_activity_regularization_losses = []
      d_classifier_weight_regularization_losses = []
      for loss_ in regularization_losses:
        if "multi_adap__dense" in loss_.name:
          output_activity_regularization_losses.append(loss_)
        elif "ADAP_gate" in loss_.name: #and "ActivityRegularizer" not in loss_.name and "Regularizer" not in loss_.name
          if "ActivityRegularizer" in loss_.name:
            d_classifier_activity_regularization_losses.append(loss_)
          elif "Regularizer" in loss_.name:
            d_classifier_weight_regularization_losses.append(loss_)
          else:
            d_classification_gate_losses.append(loss_)
        elif "ADAP_" in loss_.name:
          layer_activity_regularization_losses.append(loss_)

      
      if (len(layer_activity_regularization_losses)>0) and layer_activity_regularization_loss_scale>0:
        training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)

      if len(output_activity_regularization_losses)>0 and output_activity_regularization_loss_scale>0:
        training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)

      if len(d_classification_gate_losses)>0 and d_classification_gate_loss_scale>0:
        training_loss += d_classification_gate_loss_scale * tf.add_n(d_classification_gate_losses)

      if len(d_classifier_activity_regularization_losses)>0 and d_classifier_activity_regularization_loss_scale>0:
        training_loss += d_classifier_activity_regularization_loss_scale * tf.add_n(d_classifier_activity_regularization_losses)

      if len(d_classifier_weight_regularization_losses)>0 and d_classifier_weight_regularization_losses_scale>0:
        training_loss += d_classifier_weight_regularization_losses_scale * tf.add_n(d_classifier_weight_regularization_losses)
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    #for var in variables:
    #  print(var.name)
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    return reported_loss, num_examples
  def _accumulate_dev_train_gradients(source, target):
    with tf.GradientTape() as tape:
      variables = model.trainable_variables    
      tape.watch(variables)
      outputs, _ = model(
          source,
          labels=target,
          training=True,
          step=optimizer.iterations)
      loss = model.compute_loss(outputs, target, training=True)

      if isinstance(loss, tuple):
        training_loss = loss[0] / loss[1]
        reported_loss = loss[0] / loss[2]
      else:
        training_loss, reported_loss = loss, loss

      gradients = tape.gradient(training_loss, variables)
      sub_gradient_accumulator(gradients)
      return loss
  def _reset_dev_train_gradients():
    dev_gradient_accumulator.reset() # for dev_gradient_accumulator in dev_gradient_accumulators]
    train_gradient_accumulator.reset() #for train_gradient_accumulator in train_gradient_accumulators]
  def _reset_sub_gradients():
    sub_gradient_accumulator.reset()
  def _apply_dev_train_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(sub_gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(sub_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    sub_gradient_accumulator.reset()
  def _apply_sampler_gradients():
    grads_and_vars = []
    scaled_gradient = d_logits_grad_accumulator.gradients[0] / (strategy.num_replicas_in_sync * tf.cast(d_logits_grad_accumulator.step, tf.float32))
    grads_and_vars.append((scaled_gradient, domain_logits))
    sampler_optimizer.apply_gradients(grads_and_vars)
    d_logits_grad_accumulator.reset()
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  @tf.function
  def _dev_train_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_dev_train_gradients)

  @tf.function
  def _sampler_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_sampler_gradients)

  @tf.function
  def _reset_dev_train_grad_accum_step():
    with strategy.scope():
      _reset_dev_train_gradients()

  @tf.function
  def _reset_sub_grad_accum_step():
    with strategy.scope():
      _reset_sub_gradients()
  
  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  #########
  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
  def update_hessian_moving_stats():
    for accum, stat in zip(hessian_accumulators.hessians, hessian_moving_stats):
      stat.assign(accum / tf.cast(hessian_accum_step * batch_hessian_size, tf.float32))
    for hessian, normalized_hessian in zip(hessian_moving_stats, normalized_hessian_moving_stats):
      normalized_hessian.assign(hessian/tf.reduce_sum(hessian))
    
  #########
  @dataset_util.function_on_next(train_dataset)
  def _NGD_train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_NGD_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples
  @dataset_util.function_on_next(hessian_datasets)
  def _hessian_acc_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      strategy.experimental_run_v2(
          _accumulate_diag_hessians, args=(per_replica_source, per_replica_target))
  ##########
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  @tf.function
  def _hessian_stats_update_step():
    strategy.experimental_run_v2(update_hessian_moving_stats)
  ##########

  # Runs the training loop.
  import time
  start = time.time()  
  NGD_train_data_flow = iter(_NGD_train_forward())
  _hessian_accumulator_flow = iter(_hessian_acc_forward())
  train_data_flow = iter(_train_forward())
  dev_iterators = [iter(dev_dataset) for dev_dataset in dev_datasets]
  train_iterators = [iter(train_dataset) for train_dataset in train_datasets]
  
  _, _ = next(train_data_flow)
  last_eval = [0.0] * len(domain)
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _number_examples = []
  
  score_type = config.get("score_type","MultiBLEU")
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  ref_eval_concat = file_concatenate(config["eval_ref"],"ref_eval_concat",dir_name=os.path.join(config["model_dir"],"eval"))
  if step >= config.get("NGD_warm_start",0):
    for i in range(hessian_accum_step):
      next(_hessian_accumulator_flow)
    _hessian_stats_update_step()
    print("normalized_hessian_moving_stats: [3]", normalized_hessian_moving_stats[3].numpy())
  if step <= 1:
    initializer = config.get("initializer","default")
    if initializer == "default":
      print("Initializing variables by tensorflow default")      
    elif initializer == "variance_scaling":
      print("Initializing variables by tf.variance_scaling")
      initial_value = []
      for v in model.trainable_variables:
        shape = tf.shape(v).numpy()
        initial_value.append(variance_scaling_initialier(shape, scale=1.0, mode="fan_avg", distribution="uniform"))
      weight_reset(initial_value)   
  else:
    print("current domain_logits", config.get("domain_logits",[0.0]*len(domain)))
    domain_logits.assign(config.get("domain_logits",[0.0]*len(domain)))
    # compute domain rewards
    rewards = [0.0] * len(domain)
    snapshots = [v.value() for v in model.trainable_variables]
    saved_step = optimizer.iterations.numpy()
    #######
    current_probs = tf.nn.softmax(domain_logits).numpy()
    print("current_probs: ", current_probs)
    #######
    ##### compute theta_t+1
    for k in np.random.choice(domain,config.get("update_theta_train_batch_per_run_num",len(domain)),p=current_probs): 
      src, tgt = next(train_iterators[k])
      strategy.experimental_run_v2(_accumulate_dev_train_gradients, args=(src, tgt))
    strategy.experimental_run_v2(_apply_dev_train_gradients)
    snapshots_1 = [v.value() for v in model.trainable_variables]
    for i, train_iter in enumerate(train_iterators):
      _reward = 0.0
      ##### accumulate gradient over training set of src domain i at theta_t
      weight_reset(snapshots)
      with strategy.scope():
        for _ in range(config.get("train_batch_per_run_num",10)):
          src, tgt = next(train_iter)
          strategy.experimental_run_v2(_accumulate_dev_train_gradients, args=(src, tgt))
        train_gradient_accumulator(sub_gradient_accumulator.gradients)
        strategy.experimental_run_v2(sub_gradient_accumulator.reset)
      ##### accumulate gradient over dev set of k tgt domains at theta_t+1
      weight_reset(snapshots_1)
      with strategy.scope():
        for j, dev_iter in enumerate(dev_iterators):
          _sum = 0.0
          _dev_norm = 0.0
          _tr_norm = 0.0
          for _ in range(config.get("dev_batch_per_run_num",10)):
            src, tgt = next(dev_iter)
            strategy.experimental_run_v2(_accumulate_dev_train_gradients, args=(src, tgt))
          dev_gradient_accumulator(sub_gradient_accumulator.gradients)
          strategy.experimental_run_v2(sub_gradient_accumulator.reset)         
          for dev_grad, tr_grad in zip(dev_gradient_accumulator.gradients, train_gradient_accumulator.gradients):
            _sum += tf.reduce_sum(dev_grad * tr_grad)
            _dev_norm += tf.reduce_sum(dev_grad * dev_grad)
            _tr_norm += tf.reduce_sum(tr_grad * tr_grad)
          _reward += _sum / (tf.sqrt(_dev_norm * _tr_norm) + 1e-10) * domain_importances[j]
          # reset dev gradient accumulations to zero
          strategy.experimental_run_v2(dev_gradient_accumulator.reset)
          #print(dev_gradient_accumulator.gradients[0])
        # reset train dev gradient accumulations to zero
        strategy.experimental_run_v2(train_gradient_accumulator.reset)
        #print(sub_gradient_accumulator.gradients[0])
        #print(train_gradient_accumulator.gradients[0])
      #_reward /= len(domain)
      rewards[i] = _reward.numpy()
      # reset model parameters
      weight_reset(snapshots)
      optimizer.iterations.assign(saved_step)
    domain_rewards.assign(tf.constant(rewards))
    # compute new domain distribution
    print("domain rewards", domain_rewards)
    for _ in range(config.get("domain_sampler_optim_step", 30)):
      #loss = _sampler_flow()
      #_sampler_step()
      _ = _grad_sampler_accum()
      _sampler_step_1()
      
    print("domain_logits: ", domain_logits.numpy())
    probs = tf.nn.softmax(domain_logits)
    new_picking_prob = update_sampling_distribution(probs)
    tf.summary.experimental.set_step(saved_step)
    for i in range(len(domain)):
      tf.summary.scalar("reward_%d"%i, rewards[i], description="reward of using training set %d"%(i))
      tf.summary.scalar("domain_prob_%d"%i, new_picking_prob[i], description="probability of using training set %d"%(i))
    tf.summary.flush()
    # create new training course with updated domain distribution
    train_dataset = tf.data.experimental.sample_from_datasets(train_datasets_p, weights=new_picking_prob)
    with strategy.scope():
      base_dataset = train_dataset
      train_dataset = strategy.experimental_distribute_datasets_from_function(
            lambda _: base_dataset)
    @dataset_util.function_on_next(train_dataset)
    def _train_forward(next_fn):    
      with strategy.scope():
        per_replica_source, per_replica_target = next_fn()
        per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
            _accumulate_gradients, args=(per_replica_source, per_replica_target))
        # TODO: these reductions could be delayed until _step is called.
        loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
        num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
      return loss, num_examples
    train_data_flow = iter(_train_forward())
    #######
    weight_reset(snapshots)
    optimizer.iterations.assign(saved_step)
    #######
  with _summary_writer.as_default():
    while True:
      #####Training batch
      if step % hessian_update_every == 0 and step >= config.get("NGD_warm_start",0):
        for i in range(hessian_accum_step):
          next(_hessian_accumulator_flow)
        _hessian_stats_update_step()
      if step >= config.get("NGD_warm_start",0):
        loss, num_examples = next(NGD_train_data_flow)    
        _loss.append(loss)
        _number_examples.append(num_examples)
      else:
        loss, num_examples = next(train_data_flow)    
        _loss.append(loss)
        _number_examples.append(num_examples)
      _step()  
      step = optimizer.iterations.numpy()
      
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
          "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
          step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      if step % redistribute_every == 0 and step > config.get("warm_start",5000):
        # compute domain rewards
        rewards = [0.0] * len(domain)
        snapshots = [v.value() for v in model.trainable_variables]
        saved_step = optimizer.iterations.numpy()
        #######
        current_probs = tf.nn.softmax(domain_logits).numpy()
        print("current_probs: ", current_probs)
        #######
        for i, train_iter in enumerate(train_iterators):
          _reward = 0.0
          ##### accumulate gradient over training set of src domain i at theta_t
          weight_reset(snapshots)
          with strategy.scope():
            for _ in range(config.get("train_batch_per_run_num",10)):
              src, tgt = next(train_iter)
              strategy.experimental_run_v2(_accumulate_dev_train_gradients, args=(src, tgt))
            train_gradient_accumulator(sub_gradient_accumulator.gradients)

            strategy.experimental_run_v2(sub_gradient_accumulator.reset)
          ##### accumulate gradient over dev set of k tgt domains at theta_t+1
          with strategy.scope():
            for j, dev_iter in enumerate(dev_iterators):
              _sum = 0.0
              _dev_norm = 0.0
              _tr_norm = 0.0
              for _ in range(config.get("dev_batch_per_run_num",10)):
                src, tgt = next(dev_iter)
                strategy.experimental_run_v2(_accumulate_dev_train_gradients, args=(src, tgt))
              dev_gradient_accumulator(sub_gradient_accumulator.gradients)
              strategy.experimental_run_v2(sub_gradient_accumulator.reset)         
              for dev_grad, tr_grad, var in zip(dev_gradient_accumulator.gradients, train_gradient_accumulator.gradients, model.trainable_variables):
                if True:#"ADAP_" not in var.name:
                  _sum += tf.reduce_sum(dev_grad * tr_grad)
                  _dev_norm += tf.reduce_sum(dev_grad * dev_grad)
                  _tr_norm += tf.reduce_sum(tr_grad * tr_grad)
              if config.get("cosine_reward",True):
                _reward += _sum / (tf.sqrt(_dev_norm * _tr_norm) + 1e-10) * domain_importances[j]
              else:
                _reward += _sum * learning_rate(saved_step) * domain_importances[j]
              # reset dev gradient accumulations to zero
              strategy.experimental_run_v2(dev_gradient_accumulator.reset)
              #print(dev_gradient_accumulator.gradients[0])
            # reset train dev gradient accumulations to zero
            strategy.experimental_run_v2(train_gradient_accumulator.reset)
            #print(sub_gradient_accumulator.gradients[0])
            #print(train_gradient_accumulator.gradients[0])
          #_reward /= len(domain)
          rewards[i] = _reward.numpy()
          # reset model parameters
          weight_reset(snapshots)
          optimizer.iterations.assign(saved_step)
        domain_rewards.assign(tf.constant(rewards))
        # compute new domain distribution
        print("domain rewards", domain_rewards)
        for _ in range(config.get("domain_sampler_optim_step", 30)):
          _ = _grad_sampler_accum()
          _sampler_step_1()
          
        print("domain_logits: ", domain_logits.numpy())
        probs = tf.nn.softmax(domain_logits)
        new_picking_prob = update_sampling_distribution(probs)
        tf.summary.experimental.set_step(saved_step)
        for i in range(len(domain)):
          tf.summary.scalar("reward_%d"%i, rewards[i], description="reward of using training set %d"%(i))
          tf.summary.scalar("domain_prob_%d"%i, new_picking_prob[i], description="probability of using training set %d"%(i))
        tf.summary.flush()
        # create new training course with updated domain distribution
        train_dataset = tf.data.experimental.sample_from_datasets(train_datasets_p, weights=new_picking_prob)
        with strategy.scope():
          base_dataset = train_dataset
          train_dataset = strategy.experimental_distribute_datasets_from_function(
                lambda _: base_dataset)
        if step < config.get("NGD_warm_start",0):
          @dataset_util.function_on_next(train_dataset)
          def _train_forward(next_fn):    
            with strategy.scope():
              per_replica_source, per_replica_target = next_fn()
              per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
                  _accumulate_gradients, args=(per_replica_source, per_replica_target))
              # TODO: these reductions could be delayed until _step is called.
              loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
              num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
            return loss, num_examples
          train_data_flow = iter(_train_forward())
        else:
          @dataset_util.function_on_next(train_dataset)
          def _NGD_train_forward(next_fn):    
            with strategy.scope():
              per_replica_source, per_replica_target = next_fn()
              per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
                  _accumulate_NGD_gradients, args=(per_replica_source, per_replica_target))
              # TODO: these reductions could be delayed until _step is called.
              loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)      
              num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
            return loss, num_examples
          NGD_train_data_flow = iter(_NGD_train_forward())
        #######
        weight_reset(snapshots)
        optimizer.iterations.assign(saved_step)
        #######
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        output_files = []
        new_bleu = 0.0
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
            output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
            score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
            tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
            output_files.append(output_file)
            new_bleu += score * domain_importances[i]
        ##### BLEU on concat dev set.
        output_file_concat = file_concatenate(output_files,"output_file_concat.%s"%os.path.basename(checkpoint_path))
        score = scorer(ref_eval_concat, output_file_concat)
        print("score of model %s on concat dev set: "%checkpoint_manager.latest_checkpoint, score)
        tf.summary.scalar("concat_eval_score", score, description="BLEU on concat dev set")
        #############################
        if new_bleu >= current_max_eval_bleu:
          current_max_eval_bleu = new_bleu
          descending_streak = 0
        else:
          descending_streak += 1
      if descending_streak >= 5:
        break
      if step > train_steps:
        break

def train_L2W_v3(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          adapter_optimizer=None,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  redistribute_every = config.get("redistribute_every",2000)
  if config.get("use_meta_optimizer",False):
    inner_optimizer = tf.keras.optimizers.SGD(config.get("meta_train_lr",0.001))
  else:
    inner_optimizer = optimizer
  #####
  if checkpoint_path is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
    checkpoint.restore(checkpoint_path)
  else:
    if checkpoint_manager.latest_checkpoint is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
      checkpoint.restore(checkpoint_manager.latest_checkpoint)    
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  eval_domain = config.get("eval_domain")
  ###### early stopping criterion
  current_max_eval_bleu = 0.0
  descending_streak = 0
  ######
  if not config.get("domain_importances",None):
    domain_importances = [1.0/len(eval_domain)]*len(eval_domain)
  else:
    domain_importances = config.get("domain_importances")
  print("There are %d in-domain corpora"%len(source_file))
  ###############
  print("cosine_reward: ",config.get("cosine_reward",True))
  ###############
  print("maximum_length", maximum_length)
  train_datasets_p = [] 
  datasets_size = [count_lines(src) for src in source_file]
  picking_prob = [data_size/sum(datasets_size) for data_size in datasets_size]
  temperature=config.get("temperature",1.0)
  picking_prob = [w ** temperature for w in picking_prob]
  picking_prob = [w/sum(picking_prob) for w in picking_prob]
  if config.get("picking_prob",None):
    picking_prob = config.get("picking_prob",None)
  print("initial domain picking probability: ", picking_prob)
  for i,src,tgt in zip(domain, source_file, target_file):
    train_datasets_p.append(model.examples_inputter.make_training_dataset(src, tgt,
            batch_size=batch_train_size,
            batch_type=batch_type,
            domain=i,
            single_pass=False,
            shuffle_buffer_size=shuffle_buffer_size,
            length_bucket_width=1,  # Bucketize sequences by the same length for efficiency.
            maximum_features_length=maximum_length,
            maximum_labels_length=maximum_length))
  train_dataset = tf.data.experimental.sample_from_datasets(train_datasets_p, weights=picking_prob)
  with strategy.scope():
    base_dataset = train_dataset
    train_dataset = strategy.experimental_distribute_datasets_from_function(
          lambda _: base_dataset)  
  #############
  train_datasets = [create_training_dataset(strategy, model, [domain], [source_file], [target_file], batch_train_size//2, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=None, temperature=config.get("temperature",1.0))
                                            for domain, source_file, target_file in zip(config.get("domain"), config.get("src"), config.get("tgt"))]

  dev_datasets = []
  batch_hessian_size = config.get("batch_hessian_size",32)
  for d, _source_file, _target_file in zip(config.get("eval_domain"), config.get("eval_src"), config.get("eval_ref")):
    dev_dataset = model.examples_inputter.make_training_dataset(_source_file, _target_file,
              batch_size=batch_hessian_size,
              batch_type="examples",
              domain=d,
              single_pass=False,
              shuffle_buffer_size=shuffle_buffer_size,
              length_bucket_width=config.get("length_bucket_width",1),  # Bucketize sequences by the same length for efficiency.
              maximum_features_length=maximum_length,
              maximum_labels_length=maximum_length)
    with strategy.scope():
      base_dataset_ = dev_dataset
      dev_dataset = strategy.experimental_distribute_datasets_from_function(
          lambda _: base_dataset_)
    dev_datasets.append(dev_dataset)
  #############
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
    hessian_accumulators = optimizer_util.DiagHessianAccumulator()
    sub_gradient_accumulator = optimizer_util.GradientAccumulator()
    dev_gradient_accumulator = optimizer_util.GradientAccumulator()
    train_gradient_accumulator = optimizer_util.GradientAccumulator()
    domain_rewards = tf.Variable([0.0]*len(domain), trainable=False, aggregation=tf.compat.v1.VariableAggregation.MEAN, synchronization=tf.VariableSynchronization.AUTO)
    d_logits_grad_accumulator = optimizer_util.GradientAccumulator()
  print("actor_parameterization: ",config.get("actor_parameterization","softmax"))
  if config.get("actor_parameterization","softmax") =="softmax":
    domain_logits = tf.Variable([0.0]*len(domain), trainable=True)
  elif config.get("actor_parameterization","softmax") =="linear":
    domain_logits = tf.Variable(picking_prob, trainable=True)
  grad_domain_logits_accum = tf.Variable(tf.zeros_like(domain_logits), trainable=False)
  sampler_optimizer = tf.keras.optimizers.SGD(learning_rate=config.get("sampler_optim_lr",0.01)) #tf.keras.optimizers.Adam(learning_rate=config.get("sampler_optim_lr",0.01))
  sampler_vars = [domain_logits]
  print("init domain_logits: ", domain_logits)
  print("domain_rewards: ", domain_rewards)
  print("domain_importances: ", domain_importances)
  epsilon = config.get("posterior_weight",1e-8)
  print("epsilon: ", epsilon)
  @tf.function
  def _grad_sampler_accum():
    if config.get("actor_parameterization","softmax") =="softmax":
      loss = - tf.reduce_sum(tf.stop_gradient(tf.nn.softmax(domain_logits)) * tf.nn.log_softmax(domain_logits) * domain_rewards)
    elif config.get("actor_parameterization","softmax") =="linear":
      loss = - tf.reduce_sum(domain_logits * domain_rewards)
    if config.get("sampler_entropy_constraint",False):
      print("sampler_entropy_constraint_weight",config.get("sampler_entropy_constraint_weight",1e-3))
      if config.get("actor_parameterization","softmax") =="softmax":
        loss +=  tf.reduce_sum(config.get("sampler_entropy_constraint_weight",1e-3) * tf.nn.log_softmax(domain_logits) * tf.nn.softmax(domain_logits))
      elif config.get("actor_parameterization","softmax") =="linear":
        loss +=  tf.reduce_sum(config.get("sampler_entropy_constraint_weight",1e-3) * domain_logits * tf.math.log(domain_logits))
    grad = sampler_optimizer.get_gradients(loss,[domain_logits])
    grad_domain_logits_accum.assign_add(grad[0])
    return tf.reduce_sum(tf.stop_gradient(tf.nn.softmax(domain_logits)) * domain_rewards)

  @tf.function
  def _sampler_step_1():
    sampler_optimizer.apply_gradients([(grad_domain_logits_accum, domain_logits)])
    if config.get("actor_parameterization","softmax") =="linear":
      #domain_logits.assign(domain_logits - tf.reduce_min(domain_logits))
      domain_logits.assign(tf.clip_by_value(domain_logits, clip_value_min=0.0, clip_value_max=10.0))
      domain_logits.assign(domain_logits/tf.reduce_sum(domain_logits))
    grad_domain_logits_accum.assign(tf.zeros_like(domain_logits))

  def update_sampling_distribution(logits):
    logits = logits.numpy()
    for i, l in enumerate(logits):
        if logits[i] < 0:
            logits[i] = 0
    if sum(logits) == 0:
        logits = [0.1 for _ in range(len(logits))]
    p = np.array(logits) / sum(logits)
    print("new domain probs")
    print(p)
    return p

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    _domain = source["domain"][0]
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    
    variables = model.trainable_variables
    
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples, _domain

  def _accumulate_dev_train_gradients(source, target):
    with tf.GradientTape() as tape:
      variables = model.trainable_variables    
      tape.watch(variables)
      outputs, _ = model(
          source,
          labels=target,
          training=True,
          step=optimizer.iterations)
      loss = model.compute_loss(outputs, target, training=True)

      if isinstance(loss, tuple):
        training_loss = loss[0] / loss[1]
        reported_loss = loss[0] / loss[2]
      else:
        training_loss, reported_loss = loss, loss

      gradients = tape.gradient(training_loss, variables)
      sub_gradient_accumulator(gradients)
      return training_loss
  
  def _accumulate_diag_hessians(source,target): 
    with tf.GradientTape(persistent=True) as tape:  
      variables = model.trainable_variables
      tape.watch(variables)
      outputs, _ = model(
          source,
          labels=target,
          training=True,
          step=optimizer.iterations)
      _dom = source["domain"][0]
      loss = model.compute_individual_loss(outputs, target, training=True)
      def hessian_accum_along_loss(diag_hessian_acc, x):
        gradients = tape.gradient(x,variables)
        _hessians = []
        for grad in gradients:
          _hessians.append(tf.square(grad))
        hessian_accumulators(_hessians)
        return diag_hessian_acc
      tf.scan(hessian_accum_along_loss, loss, parallel_iterations=batch_hessian_size)

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
  
  def _apply_dev_train_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(sub_gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(sub_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    inner_optimizer.apply_gradients(grads_and_vars)
    sub_gradient_accumulator.reset()
 
  def _apply_sampler_gradients():
    grads_and_vars = []
    scaled_gradient = d_logits_grad_accumulator.gradients[0] / (strategy.num_replicas_in_sync * tf.cast(d_logits_grad_accumulator.step, tf.float32))
    grads_and_vars.append((scaled_gradient, domain_logits))
    sampler_optimizer.apply_gradients(grads_and_vars)
    d_logits_grad_accumulator.reset()

  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples, per_replica_domain = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
      _domain = per_replica_domain
    return loss, num_examples, _domain
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  @tf.function
  def _dev_train_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_dev_train_gradients)

  @tf.function
  def _sampler_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_sampler_gradients)
  
  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  dev_iterators = [iter(dev_dataset) for dev_dataset in dev_datasets]
  train_iterators = [iter(train_dataset) for train_dataset in train_datasets]
 
  _, _, _ = next(train_data_flow)

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _number_examples = []
  step = optimizer.iterations.numpy()
  ########
  excluded_params = []
  for var in model.trainable_variables:
    if tf.shape(var)[-1].numpy()==31266 or tf.shape(var)[0].numpy()==31266:
      print(var.name)
      excluded_params.append(var.name)
  ########
  if config.get("reset_step",None):
    print("start from %d-th step"%config.get("reset_step",150000))
    optimizer.iterations.assign(config.get("reset_step",150000))
  
  if step <= 1:
    initializer = config.get("initializer","default")
    if initializer == "default":
      print("Initializing variables by tensorflow default")      
    elif initializer == "variance_scaling":
      print("Initializing variables by tf.variance_scaling")
      initial_value = []
      for v in model.trainable_variables:
        shape = tf.shape(v).numpy()
        initial_value.append(variance_scaling_initialier(shape, scale=1.0, mode="fan_avg", distribution="uniform"))
      weight_reset(initial_value)   
  else:
    print("current domain_logits", config.get("domain_logits",[0.0]*len(domain)))
    domain_logits.assign(config.get("domain_logits",[0.0]*len(domain)))
    if config.get("actor_parameterization","softmax") =="softmax":
      probs = tf.nn.softmax(domain_logits)
    elif config.get("actor_parameterization","softmax") =="linear":
      probs = domain_logits
    new_picking_prob = update_sampling_distribution(probs)
    # create new training course with updated domain distribution
    train_dataset = tf.data.experimental.sample_from_datasets(train_datasets_p, weights=new_picking_prob)
    with strategy.scope():
      base_dataset = train_dataset
      train_dataset = strategy.experimental_distribute_datasets_from_function(
            lambda _: base_dataset)
    @dataset_util.function_on_next(train_dataset)
    def _train_forward(next_fn):    
      with strategy.scope():
        per_replica_source, per_replica_target = next_fn()
        per_replica_loss, per_replica_num_examples, per_replica_domain = strategy.experimental_run_v2(
            _accumulate_gradients, args=(per_replica_source, per_replica_target))
        # TODO: these reductions could be delayed until _step is called.
        loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
        num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
        _domain = per_replica_domain
      return loss, num_examples, _domain
    train_data_flow = iter(_train_forward())

  if config.get("continual_learning", False):
    print("Continual Learning needs to load from old model")
    assert config.get("checkpoint_path") != None
    checkpoint_path = config.get("checkpoint_path")
    load_and_update_if_needed_from_ckpt(config["model_dir"],   
                        checkpoint_path,                        
                        trackables={"model":model},
                        vocab_update=True,
                        model_key="model")

  score_type = config.get("score_type","MultiBLEU")
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  ref_eval_concat = file_concatenate(config["eval_ref"],"ref_eval_concat",dir_name=os.path.join(config["model_dir"],"eval"))
  domain_counts = [0.0] * len(domain)
  with _summary_writer.as_default():
    while True:
      ####Training batch
      loss, num_examples, _domain = next(train_data_flow)    
      _loss.append(loss.numpy())
      _number_examples.append(num_examples.numpy())
      _step()  
      step = optimizer.iterations.numpy()
      if strategy.num_replicas_in_sync > 1:
        for v in _domain.values:
          domain_counts[int(v.numpy())] +=1
      else:
        v = _domain
        domain_counts[int(v.numpy())] +=1

      if step % redistribute_every == 0 and step > config.get("warm_start",5000):
        # compute domain rewards
        rewards = [0.0] * len(domain)
        snapshots = [v.value() for v in model.trainable_variables]
        saved_step = optimizer.iterations.numpy()
        #######
        if config.get("actor_parameterization","softmax") =="softmax":
          current_probs = tf.nn.softmax(domain_logits).numpy()
        elif config.get("actor_parameterization","softmax") =="linear":
          current_probs = domain_logits.numpy()
        print("current_probs: ", current_probs)
        ####### Prepare dev batch
        dev_batches = []
        for j, dev_iter in enumerate(dev_iterators):
          dev_batches_domain_i = []
          for _ in range(config.get("dev_batch_per_run_num",10)):
            src, tgt = next(dev_iter)
            dev_batches_domain_i.append((src,tgt))
          dev_batches.append(dev_batches_domain_i)
        #######        
        for i, train_iter in enumerate(train_iterators):
          _reward = 0.0
          weight_reset(snapshots)
          with strategy.scope():
            ##### compute theta_t+1
            for _ in range(config.get("train_batch_per_run_num",10)): 
              src, tgt = next(train_iterators[i])
              strategy.experimental_run_v2(_accumulate_dev_train_gradients, args=(src, tgt))
              train_gradient_accumulator(sub_gradient_accumulator.gradients)
              strategy.experimental_run_v2(_apply_dev_train_gradients)
            strategy.experimental_run_v2(sub_gradient_accumulator.reset)
          ##### accumulate gradient over dev set of k tgt domains at theta_t+1
          with strategy.scope():
            for j, dev_iter in enumerate(dev_iterators):
              _sum = 0.0
              _dev_norm = 0.0
              _tr_norm = 0.0
              #count = 0
              for src, tgt in dev_batches[j]:
                strategy.experimental_run_v2(_accumulate_dev_train_gradients, args=(src, tgt))
                strategy.experimental_run_v2(_accumulate_diag_hessians, args=(src, tgt))
              dev_gradient_accumulator(sub_gradient_accumulator.gradients)
              strategy.experimental_run_v2(sub_gradient_accumulator.reset)     
              for dev_grad, tr_grad, var, hessian_moving_stat in zip(dev_gradient_accumulator.gradients, train_gradient_accumulator.gradients, model.trainable_variables, hessian_accumulators.hessians):
                if isinstance(dev_grad,tf.IndexedSlices):
                  dev_grad = tf.IndexedSlices(dev_grad.values / (tf.nn.embedding_lookup(hessian_moving_stat, dev_grad.indices) + epsilon), 
                                              dev_grad.indices, dense_shape=dev_grad.dense_shape)
                else:
                  dev_grad = dev_grad / (hessian_moving_stat + epsilon)
                _sum += tf.reduce_sum(dev_grad * tr_grad)
                _dev_norm += tf.reduce_sum(dev_grad * dev_grad)
                _tr_norm += tf.reduce_sum(tr_grad * tr_grad)    
              if config.get("cosine_reward",True):
                _reward += _sum / (tf.sqrt(_dev_norm * _tr_norm) + 1e-10) * domain_importances[j]
              else:
                _reward += _sum * domain_importances[j] #_sum * learning_rate(saved_step) * domain_importances[j]
              # reset dev gradient accumulations to zero
              strategy.experimental_run_v2(dev_gradient_accumulator.reset)
              strategy.experimental_run_v2(hessian_accumulators.reset)
              #print(dev_gradient_accumulator.gradients[0])
            # reset train dev gradient accumulations to zero
            strategy.experimental_run_v2(train_gradient_accumulator.reset)
            #print(sub_gradient_accumulator.gradients[0])
            #print(train_gradient_accumulator.gradients[0])
          #_reward /= len(domain)
          rewards[i] = _reward.numpy()
          # reset model parameters
          weight_reset(snapshots)
          optimizer.iterations.assign(saved_step)
        domain_rewards.assign(tf.constant(rewards))
        if not config.get("cosine_reward",True):
          domain_rewards.assign(tf.clip_by_value(domain_rewards, clip_value_min=-1.0, clip_value_max=1.0))
        # compute new domain distribution
        print("domain rewards", domain_rewards)
        for _ in range(config.get("domain_sampler_optim_step", 30)):
          _ = _grad_sampler_accum()
          _sampler_step_1()
          
        print("domain_logits: ", domain_logits.numpy())
        if config.get("actor_parameterization","softmax") =="softmax":
          probs = tf.nn.softmax(domain_logits)
        elif config.get("actor_parameterization","softmax") =="linear":
          probs = domain_logits
        new_picking_prob = update_sampling_distribution(probs)
        tf.summary.experimental.set_step(saved_step)
        for i in range(len(domain)):
          tf.summary.scalar("reward_%d"%i, rewards[i], description="reward of using training set %d"%(i))
          tf.summary.scalar("domain_prob_%d"%i, new_picking_prob[i], description="probability of using training set %d"%(i))
        tf.summary.flush()
        # create new training course with updated domain distribution
        train_dataset = tf.data.experimental.sample_from_datasets(train_datasets_p, weights=new_picking_prob)
        with strategy.scope():
          base_dataset = train_dataset
          train_dataset = strategy.experimental_distribute_datasets_from_function(
                lambda _: base_dataset)
        @dataset_util.function_on_next(train_dataset)
        def _train_forward(next_fn):    
          with strategy.scope():
            per_replica_source, per_replica_target = next_fn()
            per_replica_loss, per_replica_num_examples, per_replica_domain = strategy.experimental_run_v2(
                _accumulate_gradients, args=(per_replica_source, per_replica_target))
            # TODO: these reductions could be delayed until _step is called.
            loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
            num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
            _domain = per_replica_domain #strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_domain, None)
          return loss, num_examples, _domain
        train_data_flow = iter(_train_forward())
        #######
        weight_reset(snapshots)
        optimizer.iterations.assign(saved_step)
        #print("previous domain count: ", domain_counts)
        print("previous domain count in percentage: ",[d/sum(domain_counts) for d in domain_counts])
        domain_counts = [0.0] * len(domain)
        #######

      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
          "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
          step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        output_files = []
        new_bleu = 0.0
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
          output_files.append(output_file)
          new_bleu += score * domain_importances[i]
        ##### BLEU on concat dev set.
        output_file_concat = file_concatenate(output_files,"output_file_concat.%s"%os.path.basename(checkpoint_path))
        score = scorer(ref_eval_concat, output_file_concat)
        print("score of model %s on concat dev set: "%checkpoint_manager.latest_checkpoint, score)
        tf.summary.scalar("concat_eval_score", score, description="BLEU on concat dev set")
        #############################
        if new_bleu >= current_max_eval_bleu:
          current_max_eval_bleu = new_bleu
          descending_streak = 0
        else:
          descending_streak += 1
      tf.summary.flush()
      if config.get("early_stopping",True) and descending_streak >= 5:
        break
      if step > train_steps:
        break

def debug_L2W_v1(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          adapter_optimizer=None,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  beam_size = 5
  length_penalty = 0.6
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  redistribute_every = config.get("redistribute_every",2000)
  if config.get("use_meta_optimizer",False):
    inner_optimizer = tf.keras.optimizers.SGD(config.get("meta_train_lr",0.001))
  else:
    inner_optimizer = optimizer
  #####
  if checkpoint_path is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
    checkpoint.restore(checkpoint_path)
  else:
    if checkpoint_manager.latest_checkpoint is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
      checkpoint.restore(checkpoint_manager.latest_checkpoint)    
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  eval_domain = config.get("eval_domain")
  ###### early stopping criterion
  current_max_eval_bleu = 0.0
  descending_streak = 0
  ######
  if not config.get("domain_importances",None):
    domain_importances = [1.0/len(eval_domain)]*len(eval_domain)
  else:
    domain_importances = config.get("domain_importances")
  print("There are %d in-domain corpora"%len(source_file))
  ###############
  print("cosine_reward: ",config.get("cosine_reward",True))
  ###############
  # train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
  #                                           maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
  #                                           multi_domain=config.get("multi_domain", True), picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))
   
  #############
  train_datasets = [create_training_dataset(strategy, model, [domain], [source_file], [target_file], batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=None, single_pass=config.get("single_pass",False), temperature=config.get("temperature",1.0))
                                            for domain, source_file, target_file in zip(config.get("domain"), config.get("src"), config.get("tgt"))]

  dev_datasets = [create_training_dataset(strategy, model, [domain], [source_file], [target_file], batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=None, temperature=config.get("temperature",1.0))
                                            for domain, source_file, target_file in zip(config.get("eval_domain"), config.get("eval_src"), config.get("eval_ref"))]
  #############
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
    sub_gradient_accumulator = optimizer_util.GradientAccumulator()
    #dev_gradient_accumulators = [optimizer_util.GradientAccumulator() for _ in domain]
    #train_gradient_accumulators = [optimizer_util.GradientAccumulator() for _ in domain]
    dev_gradient_accumulator = optimizer_util.GradientAccumulator()
    train_gradient_accumulator = optimizer_util.GradientAccumulator()
    domain_rewards = tf.Variable([0.0]*len(domain), trainable=False, aggregation=tf.compat.v1.VariableAggregation.MEAN, synchronization=tf.VariableSynchronization.AUTO)
    #domain_logits = tf.Variable([0.0]*len(domain), trainable=True)
    d_logits_grad_accumulator = optimizer_util.GradientAccumulator()
    #domain_importances = tf.Variable(domain_importances, trainable=False, aggregation=tf.compat.v1.VariableAggregation.MEAN, synchronization=tf.VariableSynchronization.AUTO)
    #sampler_optimizer = tf.keras.optimizers.Adam(learning_rate=config.get("sampler_optim_lr",0.01))
    #sampler_vars = [domain_logits]
    #sampler_optimizer._create_slots(sampler_vars)
  domain_logits = tf.Variable([0.0]*len(domain), trainable=True)
  grad_domain_logits_accum = tf.Variable(tf.zeros_like(domain_logits), trainable=False)
  sampler_optimizer = tf.keras.optimizers.Adam(learning_rate=config.get("sampler_optim_lr",0.01))
  sampler_vars = [domain_logits]
  print("domain_rewards: ", domain_rewards)
  print("domain_importances: ", domain_importances)
  
  @tf.function
  def _grad_sampler_accum():
    loss = - tf.reduce_sum(tf.stop_gradient(tf.nn.softmax(domain_logits)) * tf.nn.log_softmax(domain_logits) * domain_rewards)
    if config.get("sampler_entropy_constraint",False):
      print("sampler_entropy_constraint_weight",config.get("sampler_entropy_constraint_weight",1e-3))
      loss +=  tf.reduce_sum(config.get("sampler_entropy_constraint_weight",1e-3) * tf.nn.log_softmax(domain_logits) * tf.nn.softmax(domain_logits))
    grad = sampler_optimizer.get_gradients(loss,[domain_logits])
    grad_domain_logits_accum.assign_add(grad[0])
    return tf.reduce_sum(tf.stop_gradient(tf.nn.softmax(domain_logits)) * domain_rewards)

  @tf.function
  def _sampler_step_1():
    sampler_optimizer.apply_gradients([(grad_domain_logits_accum, domain_logits)])
    grad_domain_logits_accum.assign(tf.zeros_like(domain_logits))

  def update_sampling_distribution(logits):
    logits = logits.numpy()
    for i, l in enumerate(logits):
        if logits[i] < 0:
            logits[i] = 0
    if sum(logits) == 0:
        logits = [0.1 for _ in range(len(logits))]
    p = np.array(logits) / sum(logits)
    print("new domain probs")
    print(p)
    return p

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    domain = source["domain"][0]
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    
    variables = model.trainable_variables
    #print("var numb: ", len(variables))
    """ for var in variables:
      print(var.name) """
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

  def _accumulate_train_gradients(source, target):
    with tf.GradientTape() as tape:
      variables = model.trainable_variables    
      tape.watch(variables)
      outputs, _ = model(
          source,
          labels=target,
          training=True,
          step=optimizer.iterations,
          inference=False)
      loss = model.compute_loss(outputs, target, training=True)

      if isinstance(loss, tuple):
        training_loss = loss[0] / loss[1]
        reported_loss = loss[0] / loss[2]
      else:
        training_loss, reported_loss = loss, loss
      #tf.print(loss[1],loss[2],sep="|")
      gradients = tape.gradient(training_loss, variables)
      sub_gradient_accumulator(gradients)
      return training_loss
  
  def _accumulate_dev_gradients(source, target):
    with tf.GradientTape() as tape:
      variables = model.trainable_variables    
      tape.watch(variables)
      outputs, _ = model(
          source,
          labels=target,
          training=False,
          step=optimizer.iterations,
          inference=False)
      loss = model.compute_loss(outputs, target, training=False)

      if isinstance(loss, tuple):
        training_loss = loss[0] / loss[1]
        reported_loss = loss[0] / loss[2]
      else:
        training_loss, reported_loss = loss, loss
      #tf.print(loss[1],loss[2],sep="|")
      gradients = tape.gradient(training_loss, variables)
      sub_gradient_accumulator(gradients)
      return training_loss

  def _reset_dev_train_gradients():
    dev_gradient_accumulator.reset() # for dev_gradient_accumulator in dev_gradient_accumulators]
    [train_gradient_accumulator.reset() for train_gradient_accumulator in train_gradient_accumulators]

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
  
  def _apply_dev_train_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(sub_gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(sub_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    inner_optimizer.apply_gradients(grads_and_vars)
    sub_gradient_accumulator.reset()
  
  @tf.function
  def _dev_train_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_dev_train_gradients)

  @tf.function
  def _sampler_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_sampler_gradients)

  @tf.function
  def _reset_dev_train_grad_accum_step():
    with strategy.scope():
      _reset_dev_train_gradients()
  
  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  def _compute_loss(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=False,
        step=optimizer.iterations,
        inference=False)
    loss = model.compute_loss(outputs, target, training=False)
    
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    return reported_loss

  @tf.function
  def predict_next(source):    
    source_length = source["length"]
    batch_size = tf.shape(source_length)[0]
    source_inputs = model.features_inputter(source)
    if experiment in ["residual","residualv15","DRO","residualv25","residualv27","residualv28","residualv29","residual_big_transformer","residualv26","gated_residual_v5","residualv16","residualv19","residualv20","residualv21","residualv22","residualv23","residualv17","residualv18","residualv2","residualv1","residualv3","residualv5","residualv13","residualv12","residualv6","residualv7","residualv11","residualv8","residualv9","baselinev1"]:
      encoder_outputs, _, _ = model.encoder([source_inputs, source["domain"], source["is_noisy"]], source_length, training=False, internal_node_printing=True)
    else:
      encoder_outputs, _, _ = model.encoder(source_inputs, source_length, training=False)

    # Prepare the decoding strategy.
    if beam_size > 1:
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      source_length = tfa.seq2seq.tile_batch(source_length, beam_size)
      decoding_strategy = onmt.utils.BeamSearch(beam_size, length_penalty=length_penalty)
    else:
      decoding_strategy = onmt.utils.GreedySearch()

    # Run dynamic decoding.
    decoder_state = model.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=source_length)
    if experiment in ["residual","residualv2","DRO","residualv15","residualv25","residualv27","residual_big_transformer","residualv26","gated_residual_v5","residualv16","residualv19","residualv20","residualv21","residualv22","residualv23","residualv17","residualv18","residualv1","residualv3","residualv5","residualv6","residualv7","residualv13","residualv12","residualv11","residualv8","residualv9","baselinev1"]:
      map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}, training=False), tf.dtypes.cast(tf.fill(tf.expand_dims(tf.shape(ids)[0],0), domain), tf.int64)]
    elif experiment in ["DC"]:
      map_input_fn = lambda ids: model.labels_inputter({"ids": ids}, domain=domain, training=False)
    elif experiment in ["WDC"]:
      e_r, _ = model.classification_layer(encoder_outputs, source_length, training=False)
      e_s, _ = model.adv_classification_layer(encoder_outputs, source_length, training=False)
      g_s = model.share_gate(tf.concat([tf.tile(tf.expand_dims(e_s,1),[1,tf.shape(encoder_outputs)[1],1]),encoder_outputs],-1))
      g_r = model.specific_gate(tf.concat([tf.tile(tf.expand_dims(e_r,1),[1,tf.shape(encoder_outputs)[1],1]),encoder_outputs],-1))
      h_r = g_r * encoder_outputs
      h_s = g_s * encoder_outputs
      encoder_mask = model.encoder.build_mask(source_inputs, sequence_length=source_length)
      map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}, training=False), h_r, h_s, encoder_mask]
    elif experiment in ["residualv28","residualv29"]:
      map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}, training=False), source["domain"]]
    else:
      map_input_fn = lambda ids: model.labels_inputter({"ids": ids}, training=False)
    decoded = model.decoder.dynamic_decode(
        map_input_fn,
        tf.fill([batch_size], START_OF_SENTENCE_ID),
        end_id=END_OF_SENTENCE_ID,
        initial_state=decoder_state,
        decoding_strategy=decoding_strategy,
        maximum_iterations=250)
    target_lengths = decoded.lengths
    target_tokens = ids_to_tokens.lookup(tf.cast(decoded.ids, tf.int64)) 
    return target_tokens, target_lengths
  # Runs the training loop.
  import time
  start = time.time()  
  dev_iterators = [iter(dev_dataset) for dev_dataset in dev_datasets]
  train_iterators = [iter(train_dataset) for train_dataset in train_datasets]

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _number_examples = []
  step = optimizer.iterations.numpy()
  
  excluded_params = []
  for var in model.trainable_variables:
    if tf.shape(var)[-1].numpy()==31266 or tf.shape(var)[0].numpy()==31266:
      print(var.name)
      excluded_params.append(var.name)
  if step <= 1:
    initializer = config.get("initializer","default")
    if initializer == "default":
      print("Initializing variables by tensorflow default")      
    elif initializer == "variance_scaling":
      print("Initializing variables by tf.variance_scaling")
      initial_value = []
      for v in model.trainable_variables:
        shape = tf.shape(v).numpy()
        initial_value.append(variance_scaling_initialier(shape, scale=1.0, mode="fan_avg", distribution="uniform"))
      weight_reset(initial_value)   
  else:
    print("current domain_logits", config.get("domain_logits",[0.0]*len(domain)))
    domain_logits.assign(config.get("domain_logits",[0.0]*len(domain)))
    # compute domain rewards
    rewards = [0.0] * len(domain)
    snapshots = [v.value() for v in model.trainable_variables]
    saved_step = optimizer.iterations.numpy()
    #######
    current_probs = tf.nn.softmax(domain_logits).numpy()
    print("current_probs: ", current_probs)
    #######
    
  domain_num = len(domain)
  excluded_reward_acc = np.zeros((domain_num, domain_num))
  excluded_norm_acc = np.zeros((domain_num, domain_num))
  included_reward_acc = np.zeros((domain_num, domain_num))
  included_norm_acc = np.zeros((domain_num, domain_num))
  reward_acc = np.zeros((domain_num, domain_num, 10))
  output_file_1 = "/gpfsdswork/projects/rech/sfz/utt84zy/DAFE/output_1"
  output_file_2 = "/gpfsdswork/projects/rech/sfz/utt84zy/DAFE/output_2"
  ref_file = "/gpfsdswork/projects/rech/sfz/utt84zy/DAFE/ref"
  loss_diff = []
  rewards_acc = []
  with _summary_writer.as_default(): 
      for it in range(15):#while True:
        try:
          # compute domain rewards
          rewards = [0.0] * len(domain)
          snapshots = [v.value() for v in model.trainable_variables]
          saved_step = optimizer.iterations.numpy()
          #######
          current_probs = tf.nn.softmax(domain_logits).numpy()
          print("current_probs: ", current_probs)
          ####### Prepare dev batch
          dev_batches = []
          for j, dev_iter in enumerate(dev_iterators):
            dev_batches_domain_i = []
            batch_count = 0
            for _ in range(config.get("dev_batch_per_run_num",10)):
              src, tgt = next(dev_iter)
              batch_count += sum([tf.shape(val)[0].numpy() for val in src["domain"].values])
              dev_batches_domain_i.append((src,tgt))
            dev_batches.append(dev_batches_domain_i)
            print("number of dev batches of domain %s: %d"%(config.get("eval_src")[j], batch_count))
          #######        
          total_reward = 0
          count = 0
          loss_t = [0] * len(dev_iterators)
          #reward_ = np.zeros((len(train_iterators), len(dev_iterators)))
          reward_ = dict()
          for i in range(len(train_iterators)):
            for j in range(len(dev_iterators)):
              reward_[(i,j)] = []
          for i, train_iter in enumerate(train_iterators):
            weight_reset(snapshots)
            with strategy.scope():
              ####### loss of dev batch at theta_t
              for j, dev_iter in enumerate(dev_iterators):
                loss_ = 0
                for src, tgt in dev_batches[j]:
                  loss_per_device = strategy.experimental_run_v2(_compute_loss, args=(src, tgt))
                  loss_ += strategy.reduce(tf.distribute.ReduceOp.MEAN, loss_per_device, None)
                print("average loss at theta_t on %s: %f"%(config.get("eval_src")[j], loss_/len(dev_batches[j])))
                loss_t[j] = loss_/len(dev_batches[j])
              ##### compute theta_t+1
              for _ in range(config.get("train_batch_per_run_num",10)): 
                old_snapshots = [v.value() for v in model.trainable_variables]
                for _ in range(config.get("train_batch_step_accum",10)):
                  src, tgt = next(train_iterators[i])
                  #print("training domain: ", [d[0].numpy() for d in src["domain"].values])
                  strategy.experimental_run_v2(_accumulate_train_gradients, args=(src, tgt))
                strategy.experimental_run_v2(lambda: train_gradient_accumulator(sub_gradient_accumulator.gradients))
                strategy.experimental_run_v2(_apply_dev_train_gradients)
                new_snapshots = [v.value() for v in model.trainable_variables]
                strategy.experimental_run_v2(sub_gradient_accumulator.reset)
                for j, dev_iter in enumerate(dev_iterators):
                  _sum = 0.0
                  _dev_norm = 0.0
                  _tr_norm = 0.0
                  for src, tgt in dev_batches[j]:
                    strategy.experimental_run_v2(_accumulate_dev_gradients, args=(src, tgt))
                  strategy.experimental_run_v2(lambda: dev_gradient_accumulator(sub_gradient_accumulator.gradients))
                  strategy.experimental_run_v2(sub_gradient_accumulator.reset)         
                  for dev_grad, tr_grad, var, snapshot, old_snapshot, new_snapshot in zip(dev_gradient_accumulator._gradients, train_gradient_accumulator._gradients, model.trainable_variables, snapshots, old_snapshots, new_snapshots):
                    tr_grad = -new_snapshot + old_snapshot
                    _sum += tf.reduce_sum(dev_grad * tr_grad)
                    _dev_norm += tf.reduce_sum(dev_grad * dev_grad)
                    _tr_norm += tf.reduce_sum(tr_grad * tr_grad)
                  if config.get("cosine_reward",True):
                    _reward_ij = _sum / (tf.sqrt(_dev_norm * _tr_norm) + 1e-10) * domain_importances[j]
                  else:
                    _reward_ij = _sum * learning_rate(saved_step) * domain_importances[j]
                  reward_[(i,j)].append(_reward_ij.numpy())
                  loss_ = 0
                  for src, tgt in dev_batches[j]:
                    loss_per_device = strategy.experimental_run_v2(_compute_loss, args=(src, tgt))
                    loss_ += strategy.reduce(tf.distribute.ReduceOp.MEAN, loss_per_device, None)
                  print("average loss at theta_t+1 simulated from training domain %d on %s: %f"%(i, config.get("eval_src")[j], loss_/len(dev_batches[j])))
                  #print("reward of training set %d to dev set %d: "%(i,j),_reward_ij)
                  # reset dev gradient accumulations to zero
                  strategy.experimental_run_v2(dev_gradient_accumulator.reset)
                # reset train dev gradient accumulations to zero
                strategy.experimental_run_v2(train_gradient_accumulator.reset)
              #######
              for j, dev_iter in enumerate(dev_iterators):
                #print("reward of training set %d to dev set %d: "%(i,j),reward_[(i,j)])
                rewards_acc.append(sum(reward_[(i,j)]))
              ####### loss of dev batch at theta_t+1
              for j, dev_iter in enumerate(dev_iterators):
                loss_ = 0
                for src, tgt in dev_batches[j]:
                  loss_per_device = strategy.experimental_run_v2(_compute_loss, args=(src, tgt))
                  loss_ += strategy.reduce(tf.distribute.ReduceOp.MEAN, loss_per_device, None)
                print("average loss at theta_t+1 from training domain %d on %s: %f"%(i, config.get("eval_src")[j], loss_/len(dev_batches[j])))
                loss_ = loss_/len(dev_batches[j])
                loss_diff.append(- loss_ + loss_t[j])
            # reset model parameters
            weight_reset(snapshots)
            optimizer.iterations.assign(saved_step)
          #######
          weight_reset(snapshots)
          optimizer.iterations.assign(saved_step)
          #######
        except tf.errors.OutOfRangeError:
          print("average reward: ", total_reward/count)

  print(loss_diff)
  print(rewards_acc)
  print(np.cov(loss_diff, rewards_acc))

def debug_L2W_v2(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          adapter_optimizer=None,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  beam_size = 5
  length_penalty = 0.6
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  redistribute_every = config.get("redistribute_every",2000)
  if config.get("use_meta_optimizer",False):
    inner_optimizer = tf.keras.optimizers.SGD(config.get("meta_train_lr",0.001))
  else:
    inner_optimizer = optimizer
  #####
  if checkpoint_path is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
    checkpoint.restore(checkpoint_path)
  else:
    if checkpoint_manager.latest_checkpoint is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
      checkpoint.restore(checkpoint_manager.latest_checkpoint)    
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  eval_domain = config.get("eval_domain")
  ###### early stopping criterion
  current_max_eval_bleu = 0.0
  descending_streak = 0
  ######
  if not config.get("domain_importances",None):
    domain_importances = [1.0/len(eval_domain)]*len(eval_domain)
  else:
    domain_importances = config.get("domain_importances")
  print("There are %d in-domain corpora"%len(source_file))
  ###############
  print("cosine_reward: ",config.get("cosine_reward",True))
  ###############
  # train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
  #                                           maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
  #                                           multi_domain=config.get("multi_domain", True), picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))
   
  #############
  train_datasets = [create_training_dataset(strategy, model, [domain], [source_file], [target_file], batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=None, single_pass=config.get("single_pass",False), temperature=config.get("temperature",1.0))
                                            for domain, source_file, target_file in zip(config.get("domain"), config.get("src"), config.get("tgt"))]

  dev_datasets = [create_training_dataset(strategy, model, [domain], [source_file], [target_file], batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=None, temperature=config.get("temperature",1.0))
                                            for domain, source_file, target_file in zip(config.get("eval_domain"), config.get("eval_src"), config.get("eval_ref"))]
  #############
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
    sub_gradient_accumulator = optimizer_util.GradientAccumulator()
    #dev_gradient_accumulators = [optimizer_util.GradientAccumulator() for _ in domain]
    #train_gradient_accumulators = [optimizer_util.GradientAccumulator() for _ in domain]
    dev_gradient_accumulator = optimizer_util.GradientAccumulator()
    train_gradient_accumulator = optimizer_util.GradientAccumulator()
    domain_rewards = tf.Variable([0.0]*len(domain), trainable=False, aggregation=tf.compat.v1.VariableAggregation.MEAN, synchronization=tf.VariableSynchronization.AUTO)
    #domain_logits = tf.Variable([0.0]*len(domain), trainable=True)
    d_logits_grad_accumulator = optimizer_util.GradientAccumulator()
    #domain_importances = tf.Variable(domain_importances, trainable=False, aggregation=tf.compat.v1.VariableAggregation.MEAN, synchronization=tf.VariableSynchronization.AUTO)
    #sampler_optimizer = tf.keras.optimizers.Adam(learning_rate=config.get("sampler_optim_lr",0.01))
    #sampler_vars = [domain_logits]
    #sampler_optimizer._create_slots(sampler_vars)
  domain_logits = tf.Variable([0.0]*len(domain), trainable=True)
  grad_domain_logits_accum = tf.Variable(tf.zeros_like(domain_logits), trainable=False)
  sampler_optimizer = tf.keras.optimizers.Adam(learning_rate=config.get("sampler_optim_lr",0.01))
  sampler_vars = [domain_logits]
  print("domain_rewards: ", domain_rewards)
  print("domain_importances: ", domain_importances)
  
  @tf.function
  def _grad_sampler_accum():
    loss = - tf.reduce_sum(tf.stop_gradient(tf.nn.softmax(domain_logits)) * tf.nn.log_softmax(domain_logits) * domain_rewards)
    if config.get("sampler_entropy_constraint",False):
      print("sampler_entropy_constraint_weight",config.get("sampler_entropy_constraint_weight",1e-3))
      loss +=  tf.reduce_sum(config.get("sampler_entropy_constraint_weight",1e-3) * tf.nn.log_softmax(domain_logits) * tf.nn.softmax(domain_logits))
    grad = sampler_optimizer.get_gradients(loss,[domain_logits])
    grad_domain_logits_accum.assign_add(grad[0])
    return tf.reduce_sum(tf.stop_gradient(tf.nn.softmax(domain_logits)) * domain_rewards)

  @tf.function
  def _sampler_step_1():
    sampler_optimizer.apply_gradients([(grad_domain_logits_accum, domain_logits)])
    grad_domain_logits_accum.assign(tf.zeros_like(domain_logits))

  def update_sampling_distribution(logits):
    logits = logits.numpy()
    for i, l in enumerate(logits):
        if logits[i] < 0:
            logits[i] = 0
    if sum(logits) == 0:
        logits = [0.1 for _ in range(len(logits))]
    p = np.array(logits) / sum(logits)
    print("new domain probs")
    print(p)
    return p

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    domain = source["domain"][0]
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    
    variables = model.trainable_variables
    #print("var numb: ", len(variables))
    """ for var in variables:
      print(var.name) """
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples

  def _accumulate_train_gradients(source, target):
    with tf.GradientTape() as tape:
      variables = model.trainable_variables    
      tape.watch(variables)
      outputs, _ = model(
          source,
          labels=target,
          training=True,
          step=optimizer.iterations,
          inference=False)
      loss = model.compute_loss(outputs, target, training=True)

      if isinstance(loss, tuple):
        training_loss = loss[0] / loss[1]
        reported_loss = loss[0] / loss[2]
      else:
        training_loss, reported_loss = loss, loss
      #tf.print(loss[1],loss[2],sep="|")
      gradients = tape.gradient(training_loss, variables)
      sub_gradient_accumulator(gradients)
      return training_loss
  
  def _accumulate_dev_gradients(source, target):
    with tf.GradientTape() as tape:
      variables = model.trainable_variables    
      tape.watch(variables)
      outputs, _ = model(
          source,
          labels=target,
          training=False,
          step=optimizer.iterations,
          inference=False)
      loss = model.compute_loss(outputs, target, training=False)

      if isinstance(loss, tuple):
        training_loss = loss[0] / loss[1]
        reported_loss = loss[0] / loss[2]
      else:
        training_loss, reported_loss = loss, loss
      #tf.print(loss[1],loss[2],sep="|")
      gradients = tape.gradient(training_loss, variables)
      sub_gradient_accumulator(gradients)
      return training_loss

  def _reset_dev_train_gradients():
    dev_gradient_accumulator.reset() # for dev_gradient_accumulator in dev_gradient_accumulators]
    [train_gradient_accumulator.reset() for train_gradient_accumulator in train_gradient_accumulators]

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
  
  def _apply_dev_train_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(sub_gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(sub_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    inner_optimizer.apply_gradients(grads_and_vars)
    sub_gradient_accumulator.reset()
  
  @tf.function
  def _dev_train_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_dev_train_gradients)

  @tf.function
  def _sampler_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_sampler_gradients)

  @tf.function
  def _reset_dev_train_grad_accum_step():
    with strategy.scope():
      _reset_dev_train_gradients()
  
  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  def _compute_loss(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=False,
        step=optimizer.iterations,
        inference=False)
    loss = model.compute_loss(outputs, target, training=False)
    
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    return reported_loss

  @tf.function
  def predict_next(source):    
    source_length = source["length"]
    batch_size = tf.shape(source_length)[0]
    source_inputs = model.features_inputter(source)
    if experiment in ["residual","residualv15","DRO","residualv25","residualv27","residualv28","residualv29","residual_big_transformer","residualv26","gated_residual_v5","residualv16","residualv19","residualv20","residualv21","residualv22","residualv23","residualv17","residualv18","residualv2","residualv1","residualv3","residualv5","residualv13","residualv12","residualv6","residualv7","residualv11","residualv8","residualv9","baselinev1"]:
      encoder_outputs, _, _ = model.encoder([source_inputs, source["domain"], source["is_noisy"]], source_length, training=False, internal_node_printing=True)
    else:
      encoder_outputs, _, _ = model.encoder(source_inputs, source_length, training=False)

    # Prepare the decoding strategy.
    if beam_size > 1:
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      source_length = tfa.seq2seq.tile_batch(source_length, beam_size)
      decoding_strategy = onmt.utils.BeamSearch(beam_size, length_penalty=length_penalty)
    else:
      decoding_strategy = onmt.utils.GreedySearch()

    # Run dynamic decoding.
    decoder_state = model.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=source_length)
    if experiment in ["residual","residualv2","DRO","residualv15","residualv25","residualv27","residual_big_transformer","residualv26","gated_residual_v5","residualv16","residualv19","residualv20","residualv21","residualv22","residualv23","residualv17","residualv18","residualv1","residualv3","residualv5","residualv6","residualv7","residualv13","residualv12","residualv11","residualv8","residualv9","baselinev1"]:
      map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}, training=False), tf.dtypes.cast(tf.fill(tf.expand_dims(tf.shape(ids)[0],0), domain), tf.int64)]
    elif experiment in ["DC"]:
      map_input_fn = lambda ids: model.labels_inputter({"ids": ids}, domain=domain, training=False)
    elif experiment in ["WDC"]:
      e_r, _ = model.classification_layer(encoder_outputs, source_length, training=False)
      e_s, _ = model.adv_classification_layer(encoder_outputs, source_length, training=False)
      g_s = model.share_gate(tf.concat([tf.tile(tf.expand_dims(e_s,1),[1,tf.shape(encoder_outputs)[1],1]),encoder_outputs],-1))
      g_r = model.specific_gate(tf.concat([tf.tile(tf.expand_dims(e_r,1),[1,tf.shape(encoder_outputs)[1],1]),encoder_outputs],-1))
      h_r = g_r * encoder_outputs
      h_s = g_s * encoder_outputs
      encoder_mask = model.encoder.build_mask(source_inputs, sequence_length=source_length)
      map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}, training=False), h_r, h_s, encoder_mask]
    elif experiment in ["residualv28","residualv29"]:
      map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}, training=False), source["domain"]]
    else:
      map_input_fn = lambda ids: model.labels_inputter({"ids": ids}, training=False)
    decoded = model.decoder.dynamic_decode(
        map_input_fn,
        tf.fill([batch_size], START_OF_SENTENCE_ID),
        end_id=END_OF_SENTENCE_ID,
        initial_state=decoder_state,
        decoding_strategy=decoding_strategy,
        maximum_iterations=250)
    target_lengths = decoded.lengths
    target_tokens = ids_to_tokens.lookup(tf.cast(decoded.ids, tf.int64)) 
    return target_tokens, target_lengths
  # Runs the training loop.
  import time
  start = time.time()  
  dev_iterators = [iter(dev_dataset) for dev_dataset in dev_datasets]
  train_iterators = [iter(train_dataset) for train_dataset in train_datasets]

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _number_examples = []
  step = optimizer.iterations.numpy()
  
  excluded_params = []
  for var in model.trainable_variables:
    if tf.shape(var)[-1].numpy()==31266 or tf.shape(var)[0].numpy()==31266:
      print(var.name)
      excluded_params.append(var.name)
  if step <= 1:
    initializer = config.get("initializer","default")
    if initializer == "default":
      print("Initializing variables by tensorflow default")      
    elif initializer == "variance_scaling":
      print("Initializing variables by tf.variance_scaling")
      initial_value = []
      for v in model.trainable_variables:
        shape = tf.shape(v).numpy()
        initial_value.append(variance_scaling_initialier(shape, scale=1.0, mode="fan_avg", distribution="uniform"))
      weight_reset(initial_value)   
  else:
    print("current domain_logits", config.get("domain_logits",[0.0]*len(domain)))
    domain_logits.assign(config.get("domain_logits",[0.0]*len(domain)))
    # compute domain rewards
    rewards = [0.0] * len(domain)
    snapshots = [v.value() for v in model.trainable_variables]
    saved_step = optimizer.iterations.numpy()
    #######
    current_probs = tf.nn.softmax(domain_logits).numpy()
    print("current_probs: ", current_probs)
    #######
    
  domain_num = len(domain)
  excluded_reward_acc = np.zeros((domain_num, domain_num))
  excluded_norm_acc = np.zeros((domain_num, domain_num))
  included_reward_acc = np.zeros((domain_num, domain_num))
  included_norm_acc = np.zeros((domain_num, domain_num))
  reward_acc = np.zeros((domain_num, domain_num, 10))
  output_file_1 = "/gpfsdswork/projects/rech/sfz/utt84zy/DAFE/output_1"
  output_file_2 = "/gpfsdswork/projects/rech/sfz/utt84zy/DAFE/output_2"
  ref_file = "/gpfsdswork/projects/rech/sfz/utt84zy/DAFE/ref"
  loss_diff = []
  rewards_acc = []
  with _summary_writer.as_default(): 
      for it in range(5):#while True:
        try:
          # compute domain rewards
          rewards = [0.0] * len(domain)
          snapshots = [v.value() for v in model.trainable_variables]
          saved_step = optimizer.iterations.numpy()
          #######
          current_probs = tf.nn.softmax(domain_logits).numpy()
          print("current_probs: ", current_probs)
          ####### Prepare dev batch
          dev_batches = []
          for j, dev_iter in enumerate(dev_iterators):
            dev_batches_domain_i = []
            batch_count = 0
            for _ in range(config.get("dev_batch_per_run_num",10)):
              src, tgt = next(dev_iter)
              batch_count += sum([tf.shape(val)[0].numpy() for val in src["domain"].values])
              dev_batches_domain_i.append((src,tgt))
            dev_batches.append(dev_batches_domain_i)
            print("number of dev batches of domain %s: %d"%(config.get("eval_src")[j], batch_count))
          #######        
          total_reward = 0
          count = 0
          loss_t = 0
          loss_t_1 = 0
          for i, train_iter in enumerate(train_iterators):
            _reward = 0.0
            weight_reset(snapshots)
            with strategy.scope():
              ####### loss of dev batch at theta_t
              for j, dev_iter in enumerate(dev_iterators):
                loss_ = 0
                for src, tgt in dev_batches[j]:
                  loss_per_device = strategy.experimental_run_v2(_compute_loss, args=(src, tgt))
                  loss_ += strategy.reduce(tf.distribute.ReduceOp.MEAN, loss_per_device, None)
                print("average loss at theta_t on %s: %f"%(config.get("eval_src")[j], loss_/len(dev_batches[j])))
                loss_t = loss_/len(dev_batches[j])
              ##### compute theta_t+1
              for _ in range(config.get("train_batch_per_run_num",10)): 
                for _ in range(config.get("train_batch_step_accum",10)):
                  src, tgt = next(train_iterators[i])
                  #print("training domain: ", [d[0].numpy() for d in src["domain"].values])
                  strategy.experimental_run_v2(_accumulate_train_gradients, args=(src, tgt))
                strategy.experimental_run_v2(lambda: train_gradient_accumulator(sub_gradient_accumulator.gradients))
                strategy.experimental_run_v2(_apply_dev_train_gradients)
              strategy.experimental_run_v2(sub_gradient_accumulator.reset)

            ##### accumulate gradient over dev set of k tgt domains at theta_t+1
            with strategy.scope():
              for j, dev_iter in enumerate(dev_iterators):
                _sum = 0.0
                _dev_norm = 0.0
                _tr_norm = 0.0
                _sum_1 = 0.0
                _sum_2 = 0.0
                _dev_norm_1 = 0.0
                _tr_norm_1 = 0.0
                _dev_norm_2 = 0.0
                _tr_norm_2 = 0.0
                loss_ = 0
                for src, tgt in dev_batches[j]:
                  loss_per_device = strategy.experimental_run_v2(_compute_loss, args=(src, tgt))
                  loss_ += strategy.reduce(tf.distribute.ReduceOp.MEAN, loss_per_device, None)
                print("average loss at theta_t+1 on %s: %f"%(config.get("eval_src")[j], loss_/len(dev_batches[j])))
                loss_t_1 = loss_/len(dev_batches[j])
                for src, tgt in dev_batches[j]:
                  strategy.experimental_run_v2(_accumulate_dev_gradients, args=(src, tgt))
                strategy.experimental_run_v2(lambda: dev_gradient_accumulator(sub_gradient_accumulator.gradients))
                strategy.experimental_run_v2(sub_gradient_accumulator.reset)         
                for dev_grad, tr_grad, var, snapshot in zip(dev_gradient_accumulator._gradients, train_gradient_accumulator._gradients, model.trainable_variables, snapshots):
                  #tr_grad = snapshot - var
                  _sum += tf.reduce_sum(dev_grad * tr_grad)
                  _dev_norm += tf.reduce_sum(dev_grad * dev_grad)
                  _tr_norm += tf.reduce_sum(tr_grad * tr_grad)
                if config.get("cosine_reward",True):
                  _reward_ij = _sum / (tf.sqrt(_dev_norm * _tr_norm) + 1e-10) * domain_importances[j]
                else:
                  _reward_ij = _sum * learning_rate(saved_step) * domain_importances[j]
                _reward += _reward_ij
                loss_diff.append(- loss_t_1 + loss_t)
                rewards_acc.append(_reward_ij)
                print("reward of training set %d to dev set %d: %f"%(i,j,_reward_ij))
                # reset dev gradient accumulations to zero
                strategy.experimental_run_v2(dev_gradient_accumulator.reset)
                #print(dev_gradient_accumulator.gradients[0])
              # reset train dev gradient accumulations to zero
              strategy.experimental_run_v2(train_gradient_accumulator.reset)
            rewards[i] = _reward.numpy()
            # reset model parameters
            weight_reset(snapshots)
            optimizer.iterations.assign(saved_step)
          #######
          weight_reset(snapshots)
          optimizer.iterations.assign(saved_step)
          #######
        except tf.errors.OutOfRangeError:
          print("average reward: ", total_reward/count)

  print(loss_diff)
  print(rewards_acc)
  print(np.cov(loss_diff, rewards_acc))

def train_IW_v0(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          adapter_optimizer=None,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  redistribute_every = config.get("redistribute_every",2000)
  if config.get("use_meta_optimizer",False):
    inner_optimizer = tf.keras.optimizers.SGD(config.get("meta_train_lr",0.001))
  else:
    inner_optimizer = optimizer
  #####
  if checkpoint_path is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
    checkpoint.restore(checkpoint_path)
  else:
    if checkpoint_manager.latest_checkpoint is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
      checkpoint.restore(checkpoint_manager.latest_checkpoint)    
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  eval_domain = config.get("eval_domain")
  ###### early stopping criterion
  current_max_eval_bleu = 0.0
  descending_streak = 0
  ######
  if not config.get("domain_importances",None):
    domain_importances = [1.0/len(eval_domain)]*len(eval_domain)
  else:
    domain_importances = config.get("domain_importances")
  print("There are %d in-domain corpora"%len(source_file))
  ###############
  print("cosine_reward: ",config.get("cosine_reward",True))
  ###############

  print("maximum_length", maximum_length)
  train_datasets_p = [] 
  datasets_size = [count_lines(src) for src in source_file]
  picking_prob = [data_size/sum(datasets_size) for data_size in datasets_size]
  temperature=config.get("temperature",1.0)
  picking_prob = [w ** temperature for w in picking_prob]
  picking_prob = [w/sum(picking_prob) for w in picking_prob]
  if config.get("picking_prob",None):
    picking_prob = config.get("picking_prob",None)
  print("initial domain picking probability: ", picking_prob)
  for i,src,tgt in zip(domain, source_file, target_file):
    train_datasets_p.append(model.examples_inputter.make_training_dataset(src, tgt,
            batch_size=batch_train_size,
            batch_type=batch_type,
            domain=i,
            single_pass=False,
            shuffle_buffer_size=shuffle_buffer_size,
            length_bucket_width=1,  # Bucketize sequences by the same length for efficiency.
            maximum_features_length=maximum_length,
            maximum_labels_length=maximum_length))
  train_dataset = tf.data.experimental.sample_from_datasets(train_datasets_p, weights=None)
  with strategy.scope():
    base_dataset = train_dataset
    train_dataset = strategy.experimental_distribute_datasets_from_function(
          lambda _: base_dataset)  
  #############
  train_datasets = [create_training_dataset(strategy, model, [domain], [source_file], [target_file], batch_train_size//2, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=None, temperature=config.get("temperature",1.0))
                                            for domain, source_file, target_file in zip(config.get("domain"), config.get("src"), config.get("tgt"))]

  dev_datasets = []
  for d, source_file, target_file in zip(config.get("eval_domain"), config.get("eval_src"), config.get("eval_ref")):
    dev_dataset = model.examples_inputter.make_training_dataset(source_file, target_file,
              batch_size=batch_train_size//2,
              batch_type="tokens",
              domain=d,
              single_pass=False,
              shuffle_buffer_size=shuffle_buffer_size,
              length_bucket_width=config.get("length_bucket_width",1),  # Bucketize sequences by the same length for efficiency.
              maximum_features_length=None,
              maximum_labels_length=None)
    with strategy.scope():
      base_dataset_ = dev_dataset
      dev_dataset = strategy.experimental_distribute_datasets_from_function(
          lambda _: base_dataset_)
    dev_datasets.append(dev_dataset)
  #############
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
    sub_gradient_accumulator = optimizer_util.GradientAccumulator()
    dev_gradient_accumulator = optimizer_util.GradientAccumulator()
    train_gradient_accumulator = optimizer_util.GradientAccumulator()
    domain_rewards = tf.Variable([0.0]*len(domain), trainable=False, aggregation=tf.compat.v1.VariableAggregation.MEAN, synchronization=tf.VariableSynchronization.AUTO)
    d_logits_grad_accumulator = optimizer_util.GradientAccumulator()
    
  print("actor_parameterization: ",config.get("actor_parameterization","softmax"))
  if config.get("actor_parameterization","softmax") =="softmax":
    domain_logits = tf.Variable([0.0]*len(domain), trainable=True)
  elif config.get("actor_parameterization","softmax") =="linear":
    domain_logits = tf.Variable(picking_prob, trainable=True)

  domain_weights = tf.Variable(picking_prob, trainable=False)
  grad_domain_logits_accum = tf.Variable(tf.zeros_like(domain_logits), trainable=False)
  sampler_optimizer = tf.keras.optimizers.SGD(learning_rate=config.get("sampler_optim_lr",0.01)) 
  sampler_vars = [domain_logits]
  print("init domain_logits: ", domain_logits)
  print("domain_rewards: ", domain_rewards)
  print("domain_importances: ", domain_importances)
  
  @tf.function
  def _grad_sampler_accum():
    if config.get("actor_parameterization","softmax") =="softmax":
      loss = - tf.reduce_sum(tf.stop_gradient(tf.nn.softmax(domain_logits)) * tf.nn.log_softmax(domain_logits) * domain_rewards)
    elif config.get("actor_parameterization","softmax") =="linear":
      loss = - tf.reduce_sum(domain_logits * domain_rewards)
    if config.get("sampler_entropy_constraint",False):
      print("sampler_entropy_constraint_weight",config.get("sampler_entropy_constraint_weight",1e-3))
      loss +=  tf.reduce_sum(config.get("sampler_entropy_constraint_weight",1e-3) * tf.nn.log_softmax(domain_logits) * tf.nn.softmax(domain_logits))
    grad = sampler_optimizer.get_gradients(loss,[domain_logits])
    grad_domain_logits_accum.assign_add(grad[0])
    return tf.reduce_sum(tf.stop_gradient(tf.nn.softmax(domain_logits)) * domain_rewards)

  @tf.function
  def _sampler_step_1():
    sampler_optimizer.apply_gradients([(grad_domain_logits_accum, domain_logits)])
    if config.get("actor_parameterization","softmax") =="linear":
      #domain_logits.assign(domain_logits - tf.reduce_min(domain_logits))
      domain_logits.assign(tf.clip_by_value(domain_logits, clip_value_min=0.0, clip_value_max=10.0))
      domain_logits.assign(domain_logits/tf.reduce_sum(domain_logits))
    grad_domain_logits_accum.assign(tf.zeros_like(domain_logits))

  def update_sampling_distribution(logits):
    logits = logits.numpy()
    for i, l in enumerate(logits):
        if logits[i] < 0:
            logits[i] = 0
    if sum(logits) == 0:
        logits = [0.1 for _ in range(len(logits))]
    p = np.array(logits) / sum(logits)
    print("new domain probs")
    print(p)
    return p

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)
    _domain = source["domain"][0]
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    
    variables = model.trainable_variables
    #tf.cond(tf.math.equal(tf.math.floormod(optimizer.iterations,redistribute_every),0), lambda: tf.print("domain_weights:",domain_weights), lambda: _)
    gradients = optimizer.get_gradients(training_loss*domain_weights[_domain], variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples, _domain

  def _accumulate_dev_train_gradients(source, target):
    with tf.GradientTape() as tape:
      variables = model.trainable_variables    
      tape.watch(variables)
      outputs, _ = model(
          source,
          labels=target,
          training=True,
          step=optimizer.iterations)
      loss = model.compute_loss(outputs, target, training=True)

      if isinstance(loss, tuple):
        training_loss = loss[0] / loss[1]
        reported_loss = loss[0] / loss[2]
      else:
        training_loss, reported_loss = loss, loss

      gradients = tape.gradient(training_loss, variables)
      sub_gradient_accumulator(gradients)
      return training_loss
  
  def _reset_dev_train_gradients():
    dev_gradient_accumulator.reset() # for dev_gradient_accumulator in dev_gradient_accumulators]
    [train_gradient_accumulator.reset() for train_gradient_accumulator in train_gradient_accumulators]

  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
  
  def _apply_dev_train_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(sub_gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(sub_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    inner_optimizer.apply_gradients(grads_and_vars)
    sub_gradient_accumulator.reset()
 
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples, per_replica_domain = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
      _domain = per_replica_domain
    return loss, num_examples, _domain
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  @tf.function
  def _dev_train_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_dev_train_gradients)
  
  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  dev_iterators = [iter(dev_dataset) for dev_dataset in dev_datasets]
  train_iterators = [iter(train_dataset) for train_dataset in train_datasets]
 
  _, _, _ = next(train_data_flow)

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _number_examples = []
  step = optimizer.iterations.numpy()
  ########
  excluded_params = []
  for var in model.trainable_variables:
    if tf.shape(var)[-1].numpy()==31266 or tf.shape(var)[0].numpy()==31266:
      print(var.name)
      excluded_params.append(var.name)
  ########
  if config.get("reset_step",None):
    print("start from %d-th step"%config.get("reset_step",150000))
    optimizer.iterations.assign(config.get("reset_step",150000))
  
  if step <= 1:
    initializer = config.get("initializer","default")
    if initializer == "default":
      print("Initializing variables by tensorflow default")      
    elif initializer == "variance_scaling":
      print("Initializing variables by tf.variance_scaling")
      initial_value = []
      for v in model.trainable_variables:
        shape = tf.shape(v).numpy()
        initial_value.append(variance_scaling_initialier(shape, scale=1.0, mode="fan_avg", distribution="uniform"))
      weight_reset(initial_value)   
  else:
    print("current domain_logits", config.get("domain_logits",[0.0]*len(domain)))
    domain_logits.assign(config.get("domain_logits",[0.0]*len(domain)))
    if config.get("actor_parameterization","softmax") =="softmax":
      probs = tf.nn.softmax(domain_logits)
    elif config.get("actor_parameterization","softmax") =="linear":
      probs = domain_logits
    new_picking_prob = update_sampling_distribution(probs)
    domain_weights.assign(new_picking_prob)
    print("domain_weights: ", domain_weights)
  if config.get("continual_learning", False):
    print("Continual Learning needs to load from old model")
    assert config.get("checkpoint_path") != None
    checkpoint_path = config.get("checkpoint_path")
    load_and_update_if_needed_from_ckpt(config["model_dir"],   
                        checkpoint_path,                        
                        trackables={"model":model},
                        vocab_update=True,
                        model_key="model")

  score_type = config.get("score_type","MultiBLEU")
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  ref_eval_concat = file_concatenate(config["eval_ref"],"ref_eval_concat",dir_name=os.path.join(config["model_dir"],"eval"))
  domain_counts = [0.0] * len(domain)
  with _summary_writer.as_default():
    while True:
      ####Training batch
      loss, num_examples, _domain = next(train_data_flow)    
      _loss.append(loss.numpy())
      _number_examples.append(num_examples.numpy())
      _step()  
      step = optimizer.iterations.numpy()
      for v in _domain.values:
        domain_counts[int(v.numpy())] +=1

      if step % redistribute_every == 0 and step > config.get("warm_start",5000):
        # compute domain rewards
        rewards = [0.0] * len(domain)
        snapshots = [v.value() for v in model.trainable_variables]
        saved_step = optimizer.iterations.numpy()
        #######
        if config.get("actor_parameterization","softmax") =="softmax":
          current_probs = tf.nn.softmax(domain_logits).numpy()
        elif config.get("actor_parameterization","softmax") =="linear":
          current_probs = domain_logits.numpy()
        print("current_probs: ", current_probs)
        ####### Prepare dev batch
        dev_batches = []
        for j, dev_iter in enumerate(dev_iterators):
          dev_batches_domain_i = []
          for _ in range(config.get("dev_batch_per_run_num",10)):
            src, tgt = next(dev_iter)
            dev_batches_domain_i.append((src,tgt))
          dev_batches.append(dev_batches_domain_i)
        #######        
        for i, train_iter in enumerate(train_iterators):
          _reward = 0.0
          weight_reset(snapshots)
          with strategy.scope():
            ##### compute theta_t+1
            for _ in range(config.get("train_batch_per_run_num",10)): 
              src, tgt = next(train_iterators[i])
              strategy.experimental_run_v2(_accumulate_dev_train_gradients, args=(src, tgt))
              train_gradient_accumulator(sub_gradient_accumulator.gradients)
              strategy.experimental_run_v2(_apply_dev_train_gradients)
            strategy.experimental_run_v2(sub_gradient_accumulator.reset)
          ##### accumulate gradient over dev set of k tgt domains at theta_t+1
          with strategy.scope():
            for j, dev_iter in enumerate(dev_iterators):
              _sum = 0.0
              _dev_norm = 0.0
              _tr_norm = 0.0
              #count = 0
              for src, tgt in dev_batches[j]:
                strategy.experimental_run_v2(_accumulate_dev_train_gradients, args=(src, tgt))
              dev_gradient_accumulator(sub_gradient_accumulator.gradients)
              strategy.experimental_run_v2(sub_gradient_accumulator.reset)         
              for dev_grad, tr_grad, var, snapshot in zip(dev_gradient_accumulator._gradients, train_gradient_accumulator._gradients, model.trainable_variables, snapshots):
                if var.name not in excluded_params: #sum([substring not in var.name for substring in config.get("param_to_exclude_from_reward",["hello"])])>0: #True:#"ADAP_" not in var.name:
                  #tr_grad = var.value() - snapshot
                  _sum += tf.reduce_sum(dev_grad * tr_grad)
                  _dev_norm += tf.reduce_sum(dev_grad * dev_grad)
                  _tr_norm += tf.reduce_sum(tr_grad * tr_grad)
                  #count +=1
              #print("number_of_parameters_in_reward: %d"%(count))
              if config.get("cosine_reward",True):
                _reward += _sum / (tf.sqrt(_dev_norm * _tr_norm) + 1e-10) * domain_importances[j]
              else:
                _reward += _sum * domain_importances[j] #_sum * learning_rate(saved_step) * domain_importances[j]
              # reset dev gradient accumulations to zero
              strategy.experimental_run_v2(dev_gradient_accumulator.reset)
              #print(dev_gradient_accumulator.gradients[0])
            # reset train dev gradient accumulations to zero
            strategy.experimental_run_v2(train_gradient_accumulator.reset)
            #print(sub_gradient_accumulator.gradients[0])
            #print(train_gradient_accumulator.gradients[0])
          #_reward /= len(domain)
          rewards[i] = _reward.numpy()
          # reset model parameters
          weight_reset(snapshots)
          optimizer.iterations.assign(saved_step)
        domain_rewards.assign(tf.constant(rewards))
        if not config.get("cosine_reward",True):
          domain_rewards.assign(tf.clip_by_value(domain_rewards, clip_value_min=-1.0, clip_value_max=1.0))
        # compute new domain distribution
        print("domain rewards", domain_rewards)
        for _ in range(config.get("domain_sampler_optim_step", 30)):
          _ = _grad_sampler_accum()
          _sampler_step_1()
          
        print("domain_logits: ", domain_logits.numpy())
        if config.get("actor_parameterization","softmax") =="softmax":
          probs = tf.nn.softmax(domain_logits)
        elif config.get("actor_parameterization","softmax") =="linear":
          probs = domain_logits
        new_picking_prob = update_sampling_distribution(probs)
        tf.summary.experimental.set_step(saved_step)
        domain_weights.assign(new_picking_prob)
        print("domain_weights: ", domain_weights)
        for i in range(len(domain)):
          tf.summary.scalar("reward_%d"%i, rewards[i], description="reward of using training set %d"%(i))
          tf.summary.scalar("domain_prob_%d"%i, new_picking_prob[i], description="probability of using training set %d"%(i))
        tf.summary.flush()
        #######
        weight_reset(snapshots)
        optimizer.iterations.assign(saved_step)
        #######

      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
          "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
          step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        output_files = []
        new_bleu = 0.0
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
          output_files.append(output_file)
          new_bleu += score * domain_importances[i]
        ##### BLEU on concat dev set.
        output_file_concat = file_concatenate(output_files,"output_file_concat.%s"%os.path.basename(checkpoint_path))
        score = scorer(ref_eval_concat, output_file_concat)
        print("score of model %s on concat dev set: "%checkpoint_manager.latest_checkpoint, score)
        tf.summary.scalar("concat_eval_score", score, description="BLEU on concat dev set")
        #############################
        if new_bleu >= current_max_eval_bleu:
          current_max_eval_bleu = new_bleu
          descending_streak = 0
        else:
          descending_streak += 1
      tf.summary.flush()
      if config.get("early_stopping",True) and descending_streak >= 5:
        break
      if step > train_steps:
        break

def train_domain_mixing_residual(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    checkpoint_path = checkpoint_manager.latest_checkpoint
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  #####
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config["domain"]
  
  print("There are %d in-domain corpora"%len(source_file))

  train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=True,picking_prob=config.get("picking_prob",None))
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    non_adv_gradient_accumulator = optimizer_util.GradientAccumulator()  
    adv_gradient_accumulator = optimizer_util.GradientAccumulator()
    adv_optimizer = tf.keras.optimizers.SGD(config.get("adv_lr",0.1))

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    
    loss = model.compute_loss(outputs, target, training=True)  
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss

    if config.get("ADAP_activity_regularizing",False):
      layer_activity_regularization_loss_scale = config.get("layer_activity_regularization_loss_scale",0.001)
      output_activity_regularization_loss_scale = config.get("output_activity_regularization_loss_scale",0.001)
      d_classification_gate_loss_scale = config.get("d_classification_gate_loss_scale",0.01)
      d_classifier_activity_regularization_loss_scale = config.get("d_classifier_activity_regularization_loss_scale",1.0)
      d_classifier_weight_regularization_losses_scale = config.get("d_classifier_weight_regularization_losses_scale",1.0)
      print("layer_activity_regularization_loss_scale: ", layer_activity_regularization_loss_scale)
      print("output_activity_regularization_loss_scale: ", output_activity_regularization_loss_scale)
      print("d_classification_gate_loss_scale: ", d_classification_gate_loss_scale)
      print("d_classifier_weight_regularization_losses_scale: ", d_classifier_weight_regularization_losses_scale)
      if isinstance(layer_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        layer_activity_regularization_loss_scale = tf.constant(layer_activity_regularization_loss_scale)
        layer_activity_regularization_loss_scale = tf.nn.embedding_lookup(layer_activity_regularization_loss_scale, domain)

      if isinstance(output_activity_regularization_loss_scale, list):
        domain = source["domain"][0]
        output_activity_regularization_loss_scale = tf.constant(output_activity_regularization_loss_scale)
        output_activity_regularization_loss_scale = tf.nn.embedding_lookup(output_activity_regularization_loss_scale, domain)
      regularization_losses = model.losses
      print("model_name_scope", model.name_scope())
      print(regularization_losses)
      layer_activity_regularization_losses = []
      output_activity_regularization_losses = []
      d_classification_gate_losses = []
      d_classifier_activity_regularization_losses = []
      d_classifier_weight_regularization_losses = []
      for loss_ in regularization_losses:
        if "multi_adap__dense" in loss_.name:
          output_activity_regularization_losses.append(loss_)
        elif "ADAP_gate" in loss_.name: #and "ActivityRegularizer" not in loss_.name and "Regularizer" not in loss_.name
          if "ActivityRegularizer" in loss_.name:
            d_classifier_activity_regularization_losses.append(loss_)
          elif "Regularizer" in loss_.name:
            d_classifier_weight_regularization_losses.append(loss_)
          else:
            d_classification_gate_losses.append(loss_)
        elif "ADAP_" in loss_.name:
          layer_activity_regularization_losses.append(loss_)

      print("There are %d adaptation regularization loss on hidden layers____"%len(layer_activity_regularization_losses))
      print("There are %d adaptation regularization loss on output layer_____"%len(output_activity_regularization_losses))
      print("There are %d adaptation regularization loss on domain classification gate_____"%len(d_classification_gate_losses))
      print("There are %d d_classifier_activity_regularization_losses"%len(d_classifier_activity_regularization_losses))
      print("There are %d d_classifier_weight_regularization_losses"%len(d_classifier_weight_regularization_losses))
      if (len(layer_activity_regularization_losses)>0) and layer_activity_regularization_loss_scale>0:
        training_loss += layer_activity_regularization_loss_scale * tf.add_n(layer_activity_regularization_losses)

      if len(output_activity_regularization_losses)>0 and output_activity_regularization_loss_scale>0:
        training_loss += output_activity_regularization_loss_scale * tf.add_n(output_activity_regularization_losses)

      if len(d_classification_gate_losses)>0 and d_classification_gate_loss_scale>0:
        training_loss += d_classification_gate_loss_scale * tf.add_n(d_classification_gate_losses) / importance_weights[domain]

      if len(d_classifier_activity_regularization_losses)>0 and d_classifier_activity_regularization_loss_scale>0:
        training_loss += d_classifier_activity_regularization_loss_scale * tf.add_n(d_classifier_activity_regularization_losses)

      if len(d_classifier_weight_regularization_losses)>0 and d_classifier_weight_regularization_losses_scale>0:
        training_loss += d_classifier_weight_regularization_losses_scale * tf.add_n(d_classifier_weight_regularization_losses)

    domain_classification_logits = model(
        source,
        labels=target,
        training=True,
        adapter_activate=False,
        return_domain_classification_logits=True,
        step=optimizer.iterations)
    encoder_classification_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(source["domain"], domain_classification_logits))
    
    if config["adv_training"]:
      print("adv_training")
      total_loss = training_loss + 0.5 * tf.reduce_mean(tf.nn.log_softmax(domain_classification_logits) * tf.nn.softmax(domain_classification_logits))
    else:
      total_loss = training_loss + encoder_classification_loss
    non_adv_vars = [v for v in model.trainable_variables if "On_top_encoder_domain_classification" not in v.name]
    adv_vars = [v for v in model.trainable_variables if "On_top_encoder_domain_classification" in v.name] 
    #####
    reported_loss = training_loss
    print("var numb: ", len(non_adv_vars))
    for v in non_adv_vars:
      print(v.name)
    gradients = optimizer.get_gradients(total_loss, non_adv_vars)
    non_adv_gradient_accumulator(gradients)
    #####
    print("adv_var_numb: ", len(adv_vars))
    for v in adv_vars:
      print(v.name)
    gradients = adv_optimizer.get_gradients(encoder_classification_loss, adv_vars)
    adv_gradient_accumulator(gradients)
    #####
    num_examples = tf.reduce_sum(target["length"])
    return reported_loss, encoder_classification_loss, num_examples

  def _apply_gradients():
    non_adv_vars = [v for v in model.trainable_variables if "On_top_encoder_domain_classification" not in v.name]
    grads_and_vars = []
    for gradient, variable in zip(non_adv_gradient_accumulator.gradients, non_adv_vars):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(non_adv_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    non_adv_gradient_accumulator.reset()

  def _apply_adv_gradients():
    adv_vars = [v for v in model.trainable_variables if "On_top_encoder_domain_classification" in v.name]  
    grads_and_vars = []
    for gradient, variable in zip(adv_gradient_accumulator.gradients, adv_vars):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(adv_gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    adv_optimizer.apply_gradients(grads_and_vars)
    adv_gradient_accumulator.reset()

  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_encoder_classification_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)     
      encoder_classification_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_encoder_classification_loss, None)   
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, encoder_classification_loss, num_examples
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)
  
  def _adv_step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_adv_gradients)

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())

  ### Running one step to compile graph
  _, _, _ = next(train_data_flow)

  ### Initialize weights or update if needed for Continual Learning
  if config.get("continual_learning", False):
    assert config.get("checkpoint_path") != None
    checkpoint_path = config.get("checkpoint_path")
    load_and_update_if_needed_from_ckpt(config["model_dir"],   
                        checkpoint_path,
                        trackables={"model":model},
                        model_key="model")
                        
  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  _loss = []  
  _encoder_classification_loss = []
  _number_examples = []

  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        loss, encoder_classification_loss, num_examples = next(train_data_flow)    
        _loss.append(loss)
        _encoder_classification_loss.append(encoder_classification_loss)
        _number_examples.append(num_examples)
      _step()  
      _adv_step()
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; Encoder_classification_loss = %f, number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_encoder_classification_loss), np.sum(_number_examples), elapsed)
        _loss = []  
        _encoder_classification_loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
      tf.summary.flush()
      if step > train_steps:
        break

def CL_marine(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          adapter_optimizer=None,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  else:
    if checkpoint_path is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
      checkpoint.restore(checkpoint_path)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  ###### early stopping criterion
  current_max_eval_bleu = 0.0
  descending_streak = 0
  ######
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  
  print("There are %d in-domain corpora"%len(source_file))
  phase_id = 0
  train_dataset = create_training_dataset(strategy, model, domain, source_file[:phase_id+1], target_file[:phase_id+1], batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))
  
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples
 
  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  _, _ = next(train_data_flow)

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _number_examples = []
  step = optimizer.iterations.numpy()

  score_type = config.get("score_type","MultiBLEU")
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  ref_eval_concat = file_concatenate(config["eval_ref"],"ref_eval_concat",dir_name=os.path.join(config["model_dir"],"eval"))
  phase_step = config.get("phase_step",14000)
  with _summary_writer.as_default():
    while True:
      #####Training batch
      loss, num_examples = next(train_data_flow)    
      _loss.append(loss.numpy())
      _number_examples.append(num_examples.numpy())
      _step()  
      step = optimizer.iterations.numpy()

      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        output_files = []
        new_bleu = 0.0
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
          output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
          score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
          tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
          output_files.append(output_file)
        ##### BLEU on concat dev set.
        output_file_concat = file_concatenate(output_files,"output_file_concat.%s"%os.path.basename(checkpoint_path))
        score = scorer(ref_eval_concat, output_file_concat)
        print("score of model %s on concat dev set: "%checkpoint_manager.latest_checkpoint, score)
        new_bleu = score
        tf.summary.scalar("concat_eval_score", score, description="BLEU on concat dev set")
        #############################
        if new_bleu >= current_max_eval_bleu:
          current_max_eval_bleu = new_bleu
          descending_streak = 0
        else:
          descending_streak += 1
      if descending_streak >= 5:
        break
      tf.summary.flush()
      if step > train_steps:
        break

      if step % phase_step == 0:
        del train_dataset
        del train_data_flow
        phase_id +=1
        print("entering phase %d"%phase_id)
        train_dataset = create_training_dataset(strategy, model, domain, source_file[:phase_id+1], target_file[:phase_id+1], batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))
        @dataset_util.function_on_next(train_dataset)
        def _train_forward(next_fn):    
          with strategy.scope():
            per_replica_source, per_replica_target = next_fn()
            per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
                _accumulate_gradients, args=(per_replica_source, per_replica_target))
            # TODO: these reductions could be delayed until _step is called.
            loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
            num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
          return loss, num_examples
                
        train_data_flow = iter(_train_forward())

def priming_train(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          adapter_optimizer=None,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100):

  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  else:
    if checkpoint_path is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
      checkpoint.restore(checkpoint_path)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  ###### early stopping criterion
  current_max_eval_bleu = 0.0
  descending_streak = 0
  ######
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  pre_file = config["pre"]
  
  print("There are %d in-domain corpora"%len(source_file))
  
  train_dataset = create_priming_training_dataset(strategy, model, source_file, target_file, pre_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))

  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
    
  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    for var in variables:
      print(var.name)
    
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples
  
  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples

  @dataset_util.function_on_next(train_dataset)
  def _train_iteration(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_xsource, per_replica_target, per_replica_xtarget = next_fn()
      return per_replica_source, per_replica_xsource, per_replica_target, per_replica_xtarget
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  _, _ = next(train_data_flow)

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _number_examples = []
  score_type = config.get("score_type","MultiBLEU")
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()

  with _summary_writer.as_default():
    while True:
      #####Training batch
      #for _ in range(int(config.get("accumulation_step",1))):
      loss, num_examples = next(train_data_flow)    
      _loss.append(loss.numpy())
      _number_examples.append(num_examples.numpy())
      _step()  
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
          "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
          step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        output_files = []
        new_bleu = 0.0
        output_file = os.path.join(config["model_dir"],"eval",os.path.basename(config["eval_src"]) + ".trans." + os.path.basename(checkpoint_path))
        score = priming_translate([config["eval_src"], config["eval_pre"]], config["eval_tgt"], model, checkpoint_manager, checkpoint, 0, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
        new_bleu = score
        #############################
        if new_bleu >= current_max_eval_bleu:
          current_max_eval_bleu = new_bleu
          descending_streak = 0
        else:
          descending_streak += 1
      if descending_streak >= 5:
        break
      tf.summary.flush()
      if step > train_steps:
        break

def priming_train_chasing(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          adapter_optimizer=None,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100):

  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  else:
    if checkpoint_path is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
      checkpoint.restore(checkpoint_path)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  ###### early stopping criterion
  current_max_eval_bleu = 0.0
  descending_streak = 0
  ######
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_pre_file = config["src_pre"]
  source_hide_file = config["src_hide"]
  target_file = config["tgt"]
  chasing_alpha = tf.Variable(config.get("chasing_alpha",0.0005),trainable=False)
  chasing_alpha_step = config.get("chasing_alpha_step",30000)
  print("There are %d in-domain corpora"%len(source_pre_file))
  
  train_dataset = create_priming_training_dataset(strategy, model, source_pre_file, target_file, source_hide_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))

  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
    
  def _accumulate_gradients(source, target):
    hide_outputs, _, pre_decoder_outputs, hide_decoder_outputs = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    
    loss = model.compute_loss(hide_outputs, target, training=True)
    max_time = tf.shape(pre_decoder_outputs)[1]
    labels_lengths = model.labels_inputter.get_length(target)
    weights = tf.sequence_mask(
      labels_lengths, maxlen=max_time, dtype=pre_decoder_outputs.dtype)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    
    training_loss += tf.reduce_sum(tf.reduce_sum((pre_decoder_outputs - tf.stop_gradient(hide_decoder_outputs))**2, -1) * weights) * chasing_alpha

    variables = model.trainable_variables
    print("var numb: ", len(variables))
    #for var in variables:
    #  print(var.name)
    
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples
  
  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples

  @dataset_util.function_on_next(train_dataset)
  def _train_iteration(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_xsource, per_replica_target, per_replica_xtarget = next_fn()
      return per_replica_source, per_replica_xsource, per_replica_target, per_replica_xtarget
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  _, _ = next(train_data_flow)

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _number_examples = []
  score_type = config.get("score_type","MultiBLEU")
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()

  with _summary_writer.as_default():
    while True:
      #####Training batch
      #for _ in range(int(config.get("accumulation_step",1))):
      loss, num_examples = next(train_data_flow)    
      _loss.append(loss.numpy())
      _number_examples.append(num_examples.numpy())
      _step()  
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
          "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
          step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        output_files = []
        new_bleu = 0.0
        output_file = os.path.join(config["model_dir"],"eval",os.path.basename(config["eval_pre"]) + ".trans." + os.path.basename(checkpoint_path))
        score = priming_translate_v1([config["eval_pre"], config["eval_hide"]], config["eval_tgt"], model, checkpoint_manager, checkpoint, 0, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
        new_bleu = score
        #############################
        if new_bleu >= current_max_eval_bleu:
          current_max_eval_bleu = new_bleu
          descending_streak = 0
        else:
          descending_streak += 1
      if step % chasing_alpha_step == 0:
        print("update chasing_alpha")
        chasing_alpha.assign(chasing_alpha*2)
        print("new chasing_alpha: %f"%chasing_alpha.numpy())
      if descending_streak >= 5:
        break
      tf.summary.flush()
      if step > train_steps:
        break

def priming_train_adversarial(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          adapter_optimizer=None,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100):

  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  else:
    if checkpoint_path is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
      checkpoint.restore(checkpoint_path)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  ###### early stopping criterion
  current_max_eval_bleu = 0.0
  descending_streak = 0
  ######
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_pre_file = config["src_pre"]
  source_hide_file = config["src_hide"]
  target_file = config["tgt"]
  chasing_alpha = tf.Variable(config.get("chasing_alpha",0.0005),trainable=False)
  chasing_alpha_step = config.get("chasing_alpha_step",30000)
  print("There are %d in-domain corpora"%len(source_pre_file))
  
  train_dataset = create_priming_training_dataset(strategy, model, source_pre_file, target_file, source_hide_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))

  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
    
  def _accumulate_gradients(source, target):
    hide_outputs, _, pre_decoder_outputs, hide_decoder_outputs = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    
    loss = model.compute_loss(hide_outputs, target, training=True)
    max_time = tf.shape(pre_decoder_outputs)[1]
    labels_lengths = model.labels_inputter.get_length(target)
    weights = tf.sequence_mask(
      labels_lengths, maxlen=max_time, dtype=pre_decoder_outputs.dtype)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    
    training_loss += tf.reduce_sum(tf.reduce_sum((pre_decoder_outputs - tf.stop_gradient(hide_decoder_outputs))**2, -1) * weights) * chasing_alpha

    variables = model.trainable_variables
    print("var numb: ", len(variables))
    #for var in variables:
    #  print(var.name)
    
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples
  
  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples

  @dataset_util.function_on_next(train_dataset)
  def _train_iteration(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_xsource, per_replica_target, per_replica_xtarget = next_fn()
      return per_replica_source, per_replica_xsource, per_replica_target, per_replica_xtarget
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  _, _ = next(train_data_flow)

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _number_examples = []
  score_type = config.get("score_type","MultiBLEU")
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()

  with _summary_writer.as_default():
    while True:
      #####Training batch
      #for _ in range(int(config.get("accumulation_step",1))):
      loss, num_examples = next(train_data_flow)    
      _loss.append(loss.numpy())
      _number_examples.append(num_examples.numpy())
      _step()  
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
          "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
          step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        output_files = []
        new_bleu = 0.0
        output_file = os.path.join(config["model_dir"],"eval",os.path.basename(config["eval_pre"]) + ".trans." + os.path.basename(checkpoint_path))
        score = priming_translate_v1([config["eval_pre"], config["eval_hide"]], config["eval_tgt"], model, checkpoint_manager, checkpoint, 0, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
        new_bleu = score
        #############################
        if new_bleu >= current_max_eval_bleu:
          current_max_eval_bleu = new_bleu
          descending_streak = 0
        else:
          descending_streak += 1
      if step % chasing_alpha_step == 0:
        print("update chasing_alpha")
        chasing_alpha.assign(chasing_alpha*2)
        print("new chasing_alpha: %f"%chasing_alpha.numpy())
      if descending_streak >= 5:
        break
      tf.summary.flush()
      if step > train_steps:
        break

def multilingual_train(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          adapter_optimizer=None,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100):

  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  else:
    if checkpoint_path is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
      checkpoint.restore(checkpoint_path)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  ###### early stopping criterion
  current_max_eval_bleu = 0.0
  descending_streak = 0
  ######
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  pre_file = config["pre"]
  
  print("There are %d in-domain corpora"%len(source_file))
  
  train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))

  #####
  optimizer = tfa.optimizers.LazyAdam(0.0015)
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator()  
    
  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    
    variables = model.trainable_variables
    print("var numb: ", len(variables))
    for var in variables:
      print(var.name)
    
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, num_examples
  
  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
 
  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.experimental_run_v2(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, num_examples

  @dataset_util.function_on_next(train_dataset)
  def _train_iteration(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_xsource, per_replica_target, per_replica_xtarget = next_fn()
      return per_replica_source, per_replica_xsource, per_replica_target, per_replica_xtarget
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.experimental_run_v2(_apply_gradients)

  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  _, _ = next(train_data_flow)

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _number_examples = []
  score_type = config.get("score_type","MultiBLEU")
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()

  with _summary_writer.as_default():
    while True:
      #####Training batch
      loss, num_examples = next(train_data_flow)    
      _loss.append(loss.numpy())
      _number_examples.append(num_examples.numpy())
      _step()  
      step = optimizer.iterations.numpy()
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
          "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
          step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        output_files = []
        new_bleu = 0.0
        output_file = os.path.join(config["model_dir"],"eval",os.path.basename(config["eval_src"]) + ".trans." + os.path.basename(checkpoint_path))
        #score = 
        #new_bleu = score
        #############################
        if new_bleu >= current_max_eval_bleu:
          current_max_eval_bleu = new_bleu
          descending_streak = 0
        else:
          descending_streak += 1
      if descending_streak >= 5:
        break
      tf.summary.flush()
      if step > train_steps:
        break

def train_elbo_sparse_layer(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          adapter_optimizer=None,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  else:
    if checkpoint_path is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
      checkpoint.restore(checkpoint_path)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  ###### early stopping criterion
  current_max_eval_bleu = 0.0
  descending_streak = 0
  ######
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  
  print("There are %d in-domain corpora"%len(source_file))
  
  train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))
  
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator() 
  gumbel_temperature = tf.Variable(1.0,trainable=False)
  
  kl_term_coeff = config.get("kl_coeff",1.0)

  def _accumulate_gradients(source, target):
    outputs, _, kl_term = model(
        source,
        gumbel_temperature=gumbel_temperature,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss
    if config.get("multi_domain", True):
      _domain = source["domain"][0]
    else:
      _domain = 0

    variables = model.trainable_variables
    print("var numb: ", len(variables))
    
    gradients = optimizer.get_gradients(training_loss + kl_term_coeff * kl_term, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, kl_term, num_examples, _domain
     
  def _apply_gradients():
    variables = model.trainable_variables
    model_vars = []
    classifier_vars = []
    for var in variables:
      if "ADAP_gate/dense" in var.name:
        classifier_vars.append(var)
      else:
        model_vars.append(var)
    variables = model_vars + classifier_vars
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()

  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_kl_loss, per_replica_num_examples, per_replica_domain = strategy.run(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
      kl_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_kl_loss, None)
      _domain = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_domain, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
    return loss, kl_loss, _domain, num_examples
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.run(_apply_gradients)

  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  _, _, _, _ = next(train_data_flow)

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _kl_loss = []
  _number_examples = []
  step = optimizer.iterations.numpy()
  if config.get("reset_step",None):
    print("start from %d-th step"%config.get("reset_step",150000))
    optimizer.iterations.assign(config.get("reset_step",150000))
  if step <= 1:
    initializer = config.get("initializer","default")
    if initializer == "default":
      print("Initializing variables by tensorflow default")      
    elif initializer == "variance_scaling":
      print("Initializing variables by tf.variance_scaling")
      initial_value = []
      for v in model.trainable_variables:
        shape = tf.shape(v).numpy()
        initial_value.append(variance_scaling_initialier(shape, scale=1.0, mode="fan_avg", distribution="uniform"))
      weight_reset(initial_value)       
  
  score_type = config.get("score_type","MultiBLEU")
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  ref_eval_concat = file_concatenate(config["eval_ref"],"ref_eval_concat",dir_name=os.path.join(config["model_dir"],"eval"))
  gumbel_temperature_decay = config.get("gumbel_temperature_decay",1000)
  r = config.get("r_coeff",1e-4)
  min_temperature = config.get("min_temperature",0.5)
  print("dropout_rate",config.get("dropout_rate"))
  print("min_temperature",min_temperature)
  print("gumbel_temperature_decay",gumbel_temperature_decay)
  print("r_coeff",r)
  step = optimizer.iterations.numpy()
  gumbel_temperature.assign(tf.cast(tf.math.maximum(min_temperature, tf.math.exp(-r*step)),tf.float32))
  print("gumbel_temperature: ",gumbel_temperature)
  with _summary_writer.as_default():
    while True:
      #####Training batch
      loss, kl_loss, _domain, num_examples = next(train_data_flow)    
      _loss.append(loss.numpy())
      _kl_loss.append(kl_loss.numpy())
      _number_examples.append(num_examples.numpy())
      _step()  
      step = optimizer.iterations.numpy()
      
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; KL_loss = %f, gumbel_temperature = %f, number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_kl_loss), gumbel_temperature, np.sum(_number_examples), elapsed)
        _loss = []
        _kl_loss = []
        _number_examples = []
        start = time.time()
      if step % gumbel_temperature_decay==0:
        gumbel_temperature.assign(tf.cast(tf.math.maximum(min_temperature, tf.math.exp(-r*step)),tf.float32))
        #print("gumbel_temperature: ",gumbel_temperature)
      if step % save_every == 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        output_files = []
        new_bleu = 0.0
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
            output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
            score = translate_sparse_layer(src, ref, model, checkpoint_manager, checkpoint, i, output_file, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
            tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
            output_files.append(output_file)
        ##### BLEU on concat dev set.
        output_file_concat = file_concatenate(output_files,"output_file_concat.%s"%os.path.basename(checkpoint_path))
        score = scorer(ref_eval_concat, output_file_concat)
        print("score of model %s on concat dev set: "%checkpoint_manager.latest_checkpoint, score)
        new_bleu = score
        tf.summary.scalar("concat_eval_score", score, description="BLEU on concat dev set")
        #############################
        if new_bleu >= current_max_eval_bleu:
          current_max_eval_bleu = new_bleu
          descending_streak = 0
        else:
          descending_streak += 1
      if descending_streak >= 5:
        break
      tf.summary.flush()
      if step > train_steps:
        break

def translate_sparse_layer(source_file,
              reference,
              model,
              checkpoint_manager,
              checkpoint,              
              domain,
              output_file,
              length_penalty,
              is_noisy=1,
              gumbel_temperature = 0.2,
              checkpoint_path=None,
              probs_file=None,
              experiment="ldr",
              score_type="MultiBLEU",
              batch_size=5,
              beam_size=5):
  
  tfa.options.disable_custom_kernel()
  # Create the inference dataset.
  if checkpoint_path == None:
    checkpoint_path = checkpoint_manager.latest_checkpoint
  tf.get_logger().info("Evaluating model %s", checkpoint_path)
  print("In domain %d"%domain)
  checkpoint.restore(checkpoint_path)
  dataset = model.examples_inputter.make_inference_dataset(source_file, batch_size, domain, is_noisy=is_noisy)
  iterator = iter(dataset)

  # Create the mapping for target ids to tokens.
  ids_to_tokens = model.labels_inputter.ids_to_tokens

  domain_one_logits = tf.nn.embedding_lookup(model.domain_one_logits,domain)
  domain_zero_logits = tf.nn.embedding_lookup(model.domain_zero_logits,domain)
  
  unit_selection_logits = tf.transpose(tf.concat([tf.expand_dims(domain_zero_logits,0),tf.expand_dims(domain_one_logits,0)],0))


  import tensorflow_probability as tfp
  tfd = tfp.distributions
  gumbel_dist = tfd.Gumbel(loc=0.,scale=1.)
  gumbel_one = gumbel_dist.sample([model.num_domain_unit_group])
  gumbel_zero = gumbel_dist.sample([model.num_domain_unit_group])

  print("gumbel_temperature",gumbel_temperature)
  prob_one = tf.math.exp((domain_one_logits + gumbel_one)/gumbel_temperature)
  prob_zero = tf.math.exp((domain_zero_logits + gumbel_zero)/gumbel_temperature)
  #tf.print("prob_one",prob_one,summarize=-1)
  #tf.print("prob_zero",prob_zero,summarize=-1)
  total_prob = prob_one + prob_zero
  
  #tf.print("total_prob",total_prob,summarize=-1)

  prob_one = prob_one/total_prob
  prob_zero = prob_zero/total_prob

  domain_dropout_mask_ = tf.concat([tf.ones(model.num_shared_units),tf.cast(tf.reshape(tf.transpose(tf.tile(tf.expand_dims(prob_one,0),[model.unit_group_size,1])),[-1]),tf.float32)],-1)
  tf.print("domain_one_logits",domain_one_logits,domain_one_logits + gumbel_one,summarize=-1)
  tf.print("domain_zero_logits",domain_zero_logits,domain_zero_logits + gumbel_zero,summarize=-1)
  tf.print("domain_dropout_mask_",domain_dropout_mask_,summarize=-1)
  domain_dropout_mask = tf.concat([tf.ones(model.num_shared_units),tf.cast(tf.reshape(tf.transpose(tf.tile(tf.expand_dims(tf.math.argmax(unit_selection_logits,1),0),[model.unit_group_size,1])),[-1]),tf.float32)],-1)
  tf.print("dropout_mask:",domain_dropout_mask,summarize=-1)
  tf.print("dropout_logits",unit_selection_logits,summarize=-1)
  tf.print("domain", domain, tf.math.argmax(unit_selection_logits,1), sep="#", summarize=-1)
  @tf.function
  def predict_next():    
    source = next(iterator)
    source_length = source["length"]
    batch_size = tf.shape(source_length)[0]
    source_inputs = model.features_inputter(source)
    encoder_outputs, _, _ = model.encoder([source_inputs, source["domain"], domain_dropout_mask], source_length, training=False, internal_node_printing=True)
    
    # Prepare the decoding strategy.
    if beam_size > 1:
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      source_length = tfa.seq2seq.tile_batch(source_length, beam_size)
      decoding_strategy = onmt.utils.BeamSearch(beam_size, length_penalty=length_penalty)
    else:
      decoding_strategy = onmt.utils.GreedySearch()

    # Run dynamic decoding.
    decoder_state = model.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=source_length)
    map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}, training=False), tf.dtypes.cast(tf.fill(tf.expand_dims(tf.shape(ids)[0],0), domain), tf.int64), domain_dropout_mask]
    
    decoded = model.decoder.dynamic_decode(
        map_input_fn,
        tf.fill([batch_size], START_OF_SENTENCE_ID),
        end_id=END_OF_SENTENCE_ID,
        initial_state=decoder_state,
        decoding_strategy=decoding_strategy,
        maximum_iterations=250)
    target_lengths = decoded.lengths
    target_tokens = ids_to_tokens.lookup(tf.cast(decoded.ids, tf.int64))
    return target_tokens, target_lengths

  # Iterates on the dataset.
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  print("output file: ", output_file)
  with open(output_file, "w") as output_:
    while True:    
      try:
        batch_tokens, batch_length = predict_next()
        for tokens, length in zip(batch_tokens.numpy(), batch_length.numpy()):
          sentence = b" ".join(tokens[0][:length[0]])
          print_bytes(sentence, output_)
          #print_bytes(sentence)
      except tf.errors.OutOfRangeError:
        break
  if reference!=None:
    print("score of model %s on test set %s: "%(checkpoint_manager.latest_checkpoint, source_file), scorer(reference, output_file))
    score = scorer(reference, output_file)
    if score is None:
      return 0.0
    else:
      return score

def train_elbo_topK_sparse_layer(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          adapter_optimizer=None,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 

  import tensorflow_probability as tfp
  import scipy
  from scipy import optimize
  tfd = tfp.distributions
  gumbel_dist = tfd.Gumbel(loc=0.,scale=1.)
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  else:
    if checkpoint_path is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
      checkpoint.restore(checkpoint_path)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  ###### early stopping criterion
  current_max_eval_bleu = 0.0
  descending_streak = 0
  ######
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  
  print("There are %d in-domain corpora"%len(source_file))
  
  train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))
  
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator() 
    gradient_group_allocation_accumulator = optimizer_util.GradientAccumulator()
    latent_logit_optimizer = tfa.optimizers.LazyAdam(config.get("latent_logit_lr",0.01))

  temperature = tf.Variable(0.2,trainable=False)
  
  kl_term_coeff = config.get("kl_coeff",1.0)
  K = config.get("domain_group_allocation_num",int( (1-config.get("dropout_rate",0.5)) * config.get("num_domain_unit_group",32)))
  print("kl_term_coeff",kl_term_coeff)
  print("topK: ", K)

  def print_tensor(soft_mask):
    def print():
      tf.print("residue",soft_mask,summarize=-1)
      return tf.constant(1)
    return print
  
  def do_nothing():
    return tf.constant(1)

  def _accumulate_gradients(source, target):
    domain = source["domain"][0]
    gumbel_sample = gumbel_dist.sample([model.num_domain_unit_group])
    #with tf.GradientTape(persistent=True) as g:
    latent_group_allocation_logit_ = tf.nn.embedding_lookup(model.latent_group_allocation_logit,domain)
    domain_allocation_probs = tf.math.softmax(latent_group_allocation_logit_)
    f = lambda x: tf.reduce_sum(tf.math.sigmoid((gumbel_sample+latent_group_allocation_logit_+x)/temperature)) - K
    temp_x = tfp.math.find_root_chandrupatla(f, low=-100, high=100, position_tolerance=1e-08,value_tolerance=0.0, max_iterations=100, stopping_policy_fn=tf.reduce_all,validate_args=False, name='find_root_chandrupatla').estimated_root
    residue = tf.reduce_sum(tf.math.sigmoid((gumbel_sample+latent_group_allocation_logit_+temp_x)/temperature)) - K
    soft_mask_logits = (gumbel_sample+latent_group_allocation_logit_+temp_x)/temperature
    #tf.print("soft_mask_logits",soft_mask_logits,summarize=-1)
    soft_mask = tf.math.sigmoid(soft_mask_logits)
    
    #tf.cond( tf.math.equal(tf.math.floormod(optimizer.iterations,100),0), true_fn = print_tensor(residue), false_fn = do_nothing)
    #tf.print("soft_mask", soft_mask, "domain_allocation_probs",domain_allocation_probs,summarize=-1)
    soft_mask_total = tf.concat([tf.ones(model.num_shared_units),tf.cast(tf.repeat(soft_mask,model.unit_group_size),tf.float32)],-1)
    kl_term = - tf.reduce_mean(tf.math.log(domain_allocation_probs))

    outputs, _ = model(
        source,
        domain_dropout_mask=soft_mask_total,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss

    if config.get("multi_domain", True):
      _domain = source["domain"][0]
    else:
      _domain = 0

    variables = model.trainable_variables
    model_variables = []
    latent_group_allocation_logit = None
    for v in variables:
      if "latent_group_allocation_logit" in v.name:
        latent_group_allocation_logit = v
      else:
        model_variables.append(v)
    print("var numb: ", len(variables))
    
    gradients = optimizer.get_gradients(training_loss, model_variables)
    gradient_soft_mask = optimizer.get_gradients(training_loss,[soft_mask])
    
    deltaL_deltaM = gradient_soft_mask[0] # in R^n_g
    #tf.print("gradient_soft_mask",gradient_soft_mask[0],summarize=-1)

    delta_sigmoid = tf.math.square(tf.math.sigmoid((gumbel_sample+latent_group_allocation_logit_+temp_x)/temperature))/tf.math.exp((gumbel_sample+latent_group_allocation_logit_+temp_x)/temperature)
    deltaresidue_deltalogit1 = delta_sigmoid
    M1 = tf.linalg.diag(delta_sigmoid)
    #deltaSoftMax_deltaLogit = g.jacobian(domain_allocation_probs,latent_group_allocation_logit)
    #tf.print("deltaSoftMax_deltaLogit",deltaSoftMax_deltaLogit,summarize=-1)
    #deltaSoftMax_deltaLogit_1 = tf.tile(tf.expand_dims(domain_allocation_probs,1),[1,model.num_domain_unit_group]) * (tf.linalg.diag(tf.ones(model.num_domain_unit_group))-tf.tile(tf.expand_dims(domain_allocation_probs,0),[model.num_domain_unit_group,1]))
    #tf.print("deltaSoftMax_deltaLogit_1",deltaSoftMax_deltaLogit_1,summarize=-1)
    #deltaresidue_deltalogit = g.gradient(residue,latent_group_allocation_logit)
    #deltaresidue_deltatempx = tf.gradients(residue,temp_x)[0] / temperature
    deltaresidue_deltatempx1 = tf.reduce_sum(delta_sigmoid)
    #tf.print("deltaresidue_deltatempx1",deltaresidue_deltatempx1)
    
    deltaTempx_deltaLogit = - tf.tile(tf.expand_dims(deltaresidue_deltalogit1 / deltaresidue_deltatempx1,0),[model.num_domain_unit_group,1])
    #tf.print("deltaresidue_deltalogit", deltaresidue_deltalogit1, "deltaresidue_deltatempx", deltaresidue_deltatempx1, "deltaTempx_deltaLogit", deltaTempx_deltaLogit, summarize=-1)
    #deltaM_deltaLogit = tf.linalg.matmul(M1, tf.eye(model.num_domain_unit_group) + deltaTempx_deltaLogit) #, M1, transpose_a=True, transpose_b=True)
    deltaM_deltaLogit = tf.linalg.matmul(tf.eye(model.num_domain_unit_group) + deltaTempx_deltaLogit, M1, transpose_a=True, transpose_b=True)
    deltaL_deltaLogit = tf.linalg.matmul(tf.expand_dims(deltaL_deltaM,0),deltaM_deltaLogit)
    group_allocation_gradient = optimizer.get_gradients(kl_term * kl_term_coeff, latent_group_allocation_logit)
    group_allocation_gradient[0] = tf.clip_by_norm(tf.tensor_scatter_nd_add(group_allocation_gradient[0],tf.expand_dims(group_allocation_gradient[0].indices,1),deltaL_deltaLogit),1.0)
    #tf.print("group_allocation_gradient",group_allocation_gradient,"domain",_domain,summarize=-1)
    #tf.print("deltaL_deltaLogit",deltaL_deltaLogit,summarize=-1)
    #M3 = tf.linalg.matmul( tf.tile(tf.expand_dims(domain_allocation_probs,1),[1,model.num_domain_unit_group]) * (tf.tile(tf.expand_dims(domain_allocation_probs,0),[model.num_domain_unit_group,1]) * tf.linalg.diag(-tf.ones(model.num_domain_unit_group)) + 1)
    #M4 = 
    #M2 = 1/temperature*(M3 + M4) 
    #tf.tile(tf.expand_dims(tf.hessians(soft_mask,latent_group_allocation_logit) / tf.hessians(soft_mask,temp_x),0),[model.num_domain_unit_group,1])
    #tf.linalg.diag(tf.math.square(tf.math.sigmoid((gumbel_sample+domain_allocation_probs)/temperature+temp_x)))
    #gradient_softmask_domain_allocation_logits = 1/temperature * tf.linalg.matmul( tf.tile(tf.expand_dims(domain_allocation_probs,1),[1,model.num_domain_unit_group]) * (tf.tile(tf.expand_dims(domain_allocation_probs,0),[model.num_domain_unit_group,1]) * tf.linalg.diag(-tf.ones(model.num_domain_unit_group)) + 1) - tf.tile(tf.expand_dims(tf.hessians(soft_mask,latent_group_allocation_logit) / tf.hessians(soft_mask,temp_x),0),[model.num_domain_unit_group,1]) , left_matrix, transpose_a=True, transpose_b=True)
    #tf.print("gradient_soft_mask",gradient_soft_mask[0],summarize=-1)
    #gradients_domain_allocation_logits = tf.linalg.matmul(gradient_soft_mask[])
    gradient_accumulator(gradients)
    gradient_group_allocation_accumulator(group_allocation_gradient)
    num_examples = tf.reduce_sum(target["length"])
    #tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))    
    return reported_loss, kl_term, num_examples, _domain, residue
     
  def _apply_gradients():
    variables = model.trainable_variables
    model_variables = []
    latent_group_allocation_logit = None
    for v in variables:
      if "latent_group_allocation_logit" in v.name:
        latent_group_allocation_logit = v
      else:
        model_variables.append(v)
    grads_and_vars = []

    for gradient, variable in zip(gradient_accumulator.gradients, model_variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    grads_and_vars.append((gradient_group_allocation_accumulator.gradients[0] / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32)), latent_group_allocation_logit))
    #latent_logit_optimizer.apply_gradients([(gradient_group_allocation_accumulator.gradients[0] / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32)), latent_group_allocation_logit)])
    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()
    gradient_group_allocation_accumulator.reset()

  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_kl_loss, per_replica_num_examples, per_replica_domain, per_replica_residue = strategy.run(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
      kl_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_kl_loss, None)
      _domain = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_domain, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
      residue = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_residue, None)

    return loss, kl_loss, _domain, num_examples, residue
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.run(_apply_gradients)

  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  _, _, _, _, _ = next(train_data_flow)

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _kl_loss = []
  _number_examples = []
  _residue = []
  step = optimizer.iterations.numpy()
  if config.get("reset_step",None):
    print("start from %d-th step"%config.get("reset_step",150000))
    optimizer.iterations.assign(config.get("reset_step",150000))
  
  if step <= 1:
    initializer = config.get("initializer","default")
    if initializer == "default":
      print("Initializing variables by tensorflow default")      
    elif initializer == "variance_scaling":
      print("Initializing variables by tf.variance_scaling")
      initial_value = []
      for v in model.trainable_variables:
        shape = tf.shape(v).numpy()
        initial_value.append(variance_scaling_initialier(shape, scale=1.0, mode="fan_avg", distribution="uniform"))
      weight_reset(initial_value)       
  
  score_type = config.get("score_type","MultiBLEU")
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  ref_eval_concat = file_concatenate(config["eval_ref"],"ref_eval_concat",dir_name=os.path.join(config["model_dir"],"eval"))
  gumbel_temperature_decay = config.get("gumbel_temperature_decay",1000)
  r = config.get("r_coeff",1e-4)
  min_temperature = config.get("min_temperature",0.5)
  start_temperature = config.get("start_temperature",0.5)
  print("dropout_rate",config.get("dropout_rate"))
  print("min_temperature",min_temperature)
  print("gumbel_temperature_decay",gumbel_temperature_decay)
  print("r_coeff",r)
  step = optimizer.iterations.numpy()
  temperature.assign(tf.cast(tf.math.maximum(min_temperature, start_temperature * tf.math.exp(-r*step)),tf.float32))
  print("temperature: ",temperature)
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        loss, kl_loss, _domain, num_examples, residue = next(train_data_flow)    
        _loss.append(loss.numpy())
        _kl_loss.append(kl_loss.numpy())
        _number_examples.append(num_examples.numpy())
        _residue.append(residue.numpy())
      _step()  
      step = optimizer.iterations.numpy()
      
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; KL_loss = %f, temperature = %f, number_examples = %d, residue = %f, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_kl_loss), temperature, np.sum(_number_examples), np.mean(_residue), elapsed)
        _loss = []
        _kl_loss = []
        _number_examples = []
        _residue = []
        start = time.time()
      if step % gumbel_temperature_decay==0:
        temperature.assign(tf.cast(tf.math.maximum(min_temperature, start_temperature * tf.math.exp(-r*step)),tf.float32))
        #print("gumbel_temperature: ",gumbel_temperature)
      if step % save_every == 0 and step > 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % config.get("latent_logit_print_every",2000)==0:
        tf.print("latent_group_allocation_logit",model.latent_group_allocation_logit,summarize=-1)
      if step % eval_every == 0 and step > 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        output_files = []
        new_bleu = 0.0
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
            output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
            score = translate_topK_sparse_layer(src, ref, model, checkpoint_manager, checkpoint, i, output_file, topK=K, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
            tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
            output_files.append(output_file)
        ##### BLEU on concat dev set.
        output_file_concat = file_concatenate(output_files,"output_file_concat.%s"%os.path.basename(checkpoint_path))
        score = scorer(ref_eval_concat, output_file_concat)
        print("score of model %s on concat dev set: "%checkpoint_manager.latest_checkpoint, score)
        new_bleu = score
        tf.summary.scalar("concat_eval_score", score, description="BLEU on concat dev set")
        #############################
        if new_bleu >= current_max_eval_bleu:
          current_max_eval_bleu = new_bleu
          descending_streak = 0
        else:
          descending_streak += 1
      if descending_streak >= 5:
        break
      tf.summary.flush()
      if step > train_steps:
        break

def translate_topK_sparse_layer(source_file,
              reference,
              model,
              checkpoint_manager,
              checkpoint,              
              domain,
              output_file,
              length_penalty,
              topK=1,
              is_noisy=1,
              gumbel_temperature = 0.2,
              checkpoint_path=None,
              probs_file=None,
              experiment="ldr",
              score_type="MultiBLEU",
              batch_size=5,
              beam_size=5):
  
  tfa.options.disable_custom_kernel()
  # Create the inference dataset.
  if checkpoint_path == None:
    checkpoint_path = checkpoint_manager.latest_checkpoint
  tf.get_logger().info("Evaluating model %s", checkpoint_path)
  print("In domain %d"%domain)
  checkpoint.restore(checkpoint_path)
  dataset = model.examples_inputter.make_inference_dataset(source_file, batch_size, domain, is_noisy=is_noisy)
  iterator = iter(dataset)

  # Create the mapping for target ids to tokens.
  ids_to_tokens = model.labels_inputter.ids_to_tokens

  topK_ = tf.math.top_k(tf.nn.embedding_lookup(model.latent_group_allocation_logit,domain),k=topK).indices.numpy()
  group_allocation = np.zeros(model.num_domain_unit_group)
  for i in topK_:
    group_allocation[i] = 1

  tf.print("group_allocation:",group_allocation,"domain:",domain,summarize=-1)

  group_allocation = tf.repeat(tf.Variable(group_allocation,dtype=tf.float32),model.unit_group_size)

  domain_dropout_mask = tf.concat([tf.ones(model.num_shared_units),group_allocation],-1)
  

  @tf.function
  def predict_next():    
    source = next(iterator)
    source_length = source["length"]
    batch_size = tf.shape(source_length)[0]
    source_inputs = model.features_inputter(source)
    encoder_outputs, _, _ = model.encoder([source_inputs, source["domain"], domain_dropout_mask], source_length, training=False, internal_node_printing=True)
    
    # Prepare the decoding strategy.
    if beam_size > 1:
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      source_length = tfa.seq2seq.tile_batch(source_length, beam_size)
      decoding_strategy = onmt.utils.BeamSearch(beam_size, length_penalty=length_penalty)
    else:
      decoding_strategy = onmt.utils.GreedySearch()

    # Run dynamic decoding.
    decoder_state = model.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=source_length)
    map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}, training=False), tf.dtypes.cast(tf.fill(tf.expand_dims(tf.shape(ids)[0],0), domain), tf.int64), domain_dropout_mask]
    
    decoded = model.decoder.dynamic_decode(
        map_input_fn,
        tf.fill([batch_size], START_OF_SENTENCE_ID),
        end_id=END_OF_SENTENCE_ID,
        initial_state=decoder_state,
        decoding_strategy=decoding_strategy,
        maximum_iterations=250)
    target_lengths = decoded.lengths
    target_tokens = ids_to_tokens.lookup(tf.cast(decoded.ids, tf.int64))
    return target_tokens, target_lengths

  # Iterates on the dataset.
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  print("output file: ", output_file)
  with open(output_file, "w") as output_:
    while True:    
      try:
        batch_tokens, batch_length = predict_next()
        for tokens, length in zip(batch_tokens.numpy(), batch_length.numpy()):
          sentence = b" ".join(tokens[0][:length[0]])
          print_bytes(sentence, output_)
          #print_bytes(sentence)
      except tf.errors.OutOfRangeError:
        break
  if reference!=None:
    print("score of model %s on test set %s: "%(checkpoint_manager.latest_checkpoint, source_file), scorer(reference, output_file))
    score = scorer(reference, output_file)
    if score is None:
      return 0.0
    else:
      return score

def train_tf_25(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          adapter_optimizer=None,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 

  import tensorflow_probability as tfp
  import scipy
  from scipy import optimize
  tfd = tfp.distributions
  gumbel_dist = tfd.Gumbel(loc=0.,scale=1.)
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  else:
    if checkpoint_path is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
      checkpoint.restore(checkpoint_path)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  ###### early stopping criterion
  current_max_eval_bleu = 0.0
  descending_streak = 0
  ######
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  
  print("There are %d in-domain corpora"%len(source_file))
  
  train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))
  
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator() 

  def _accumulate_gradients(source, target):
    outputs, _ = model(
        source,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss

    variables = model.trainable_variables
    
    gradients = optimizer.get_gradients(training_loss, variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
    return reported_loss, num_examples
     
  def _apply_gradients():
    variables = model.trainable_variables
    grads_and_vars = []

    for gradient, variable in zip(gradient_accumulator.gradients, variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))

    optimizer.apply_gradients(grads_and_vars)
    gradient_accumulator.reset()

  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.run(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)

    return loss, num_examples
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.run(_apply_gradients)

  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  _, _, = next(train_data_flow)

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _number_examples = []
  step = optimizer.iterations.numpy()
  if config.get("reset_step",None):
    print("start from %d-th step"%config.get("reset_step",150000))
    optimizer.iterations.assign(config.get("reset_step",150000))
  
  if step <= 1:
    initializer = config.get("initializer","default")
    if initializer == "default":
      print("Initializing variables by tensorflow default")      
    elif initializer == "variance_scaling":
      print("Initializing variables by tf.variance_scaling")
      initial_value = []
      for v in model.trainable_variables:
        shape = tf.shape(v).numpy()
        initial_value.append(variance_scaling_initialier(shape, scale=1.0, mode="fan_avg", distribution="uniform"))
      weight_reset(initial_value)       
  
  score_type = config.get("score_type","MultiBLEU")
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  ref_eval_concat = file_concatenate(config["eval_ref"],"ref_eval_concat",dir_name=os.path.join(config["model_dir"],"eval"))
    
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        loss, num_examples = next(train_data_flow)    
        _loss.append(loss.numpy())
        _number_examples.append(num_examples.numpy())
      _step()  
      step = optimizer.iterations.numpy()
      
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d; after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        _loss = []
        _number_examples = []
        start = time.time()
      
      if step % save_every == 0 and step > 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % config.get("latent_logit_print_every",2000)==0:
        tf.print("latent_group_allocation_logit",model.latent_group_allocation_logit,summarize=-1)
      if step % eval_every == 0 and step > 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        output_files = []
        new_bleu = 0.0
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
            output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
            score = translate(src, ref, model, checkpoint_manager, checkpoint, i, output_file, topK=K, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
            tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
            output_files.append(output_file)
        ##### BLEU on concat dev set.
        output_file_concat = file_concatenate(output_files,"output_file_concat.%s"%os.path.basename(checkpoint_path))
        score = scorer(ref_eval_concat, output_file_concat)
        print("score of model %s on concat dev set: "%checkpoint_manager.latest_checkpoint, score)
        new_bleu = score
        tf.summary.scalar("concat_eval_score", score, description="BLEU on concat dev set")
        #############################
        if new_bleu >= current_max_eval_bleu:
          current_max_eval_bleu = new_bleu
          descending_streak = 0
        else:
          descending_streak += 1
      if descending_streak >= 5:
        break
      tf.summary.flush()
      if step > train_steps:
        break

def train_elbo_topK_sparse_layer_multi_layer(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          adapter_optimizer=None,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 

  import tensorflow_probability as tfp
  import scipy
  from scipy import optimize
  tfd = tfp.distributions
  gumbel_dist = tfd.Gumbel(loc=0.,scale=1.)
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  else:
    if checkpoint_path is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
      checkpoint.restore(checkpoint_path)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  ###### early stopping criterion
  current_max_eval_bleu = 0.0
  descending_streak = 0
  ######
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  
  print("There are %d in-domain corpora"%len(source_file))
  
  train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))
  
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator() 
    gradient_group_allocation_accumulator = optimizer_util.GradientAccumulator()
    latent_logit_optimizer = tfa.optimizers.LazyAdam(config.get("latent_logit_lr",0.01))

  temperature = tf.Variable(0.2,trainable=False)
  
  kl_term_coeff = config.get("kl_coeff",1.0)
  K = config.get("domain_group_allocation_num",int( (1-config.get("dropout_rate",0.5)) * config.get("num_domain_unit_group",32)))
  print("kl_term_coeff",kl_term_coeff)
  print("topK: ", K)

  my_matrix = np.zeros((model.num_domain_unit_group),)

  def _accumulate_gradients(source, target):
    domain = source["domain"][0]
    kl_loss_per_layer = []
    soft_mask_total_per_layer = []
    soft_mask_logits_per_layer = []
    delta_sigmoid_per_layer = []
    residue_per_layer = []
    for i in range(model.encoder.num_layers + model.decoder.num_layers + 1):
      gumbel_sample = gumbel_dist.sample([model.num_domain_unit_group])
      latent_group_allocation_logit_ = tf.nn.embedding_lookup(model.latent_group_allocation_logit_per_layer[i],domain)
      domain_allocation_probs = tf.math.softmax(latent_group_allocation_logit_)
      kl_loss_per_layer.append(- tf.reduce_mean(tf.math.log(domain_allocation_probs)))
      f = lambda x: tf.reduce_sum(tf.math.sigmoid((gumbel_sample+latent_group_allocation_logit_+x)/temperature)) - K
      temp_x = tfp.math.find_root_chandrupatla(f, low=-100, high=100, position_tolerance=1e-08,value_tolerance=0.0, max_iterations=100, stopping_policy_fn=tf.reduce_all,validate_args=False, name='find_root_chandrupatla').estimated_root
      residue_per_layer.append(tf.reduce_sum(tf.math.sigmoid((gumbel_sample+latent_group_allocation_logit_+temp_x)/temperature)) - K)
      soft_mask_logits = (gumbel_sample+latent_group_allocation_logit_+temp_x)/temperature
      soft_mask_logits_per_layer.append(soft_mask_logits)
      #tf.print("soft_mask_logits",soft_mask_logits,summarize=-1)
      soft_mask = tf.math.sigmoid(soft_mask_logits)
      soft_mask_total_per_layer.append(tf.concat([tf.ones(model.num_shared_units),tf.cast(tf.repeat(tf.reduce_sum(tf.one_hot(tf.math.top_k(tf.nn.embedding_lookup(model.latent_group_allocation_logit_per_layer[i],domain),k=K).indices, depth=model.num_domain_unit_group),0),model.unit_group_size),tf.float32)],-1))

      #tf.print("soft_mask", soft_mask, "domain_allocation_probs",domain_allocation_probs,summarize=-1)
      #soft_mask_total_per_layer.append(tf.concat([tf.ones(model.num_shared_units),tf.cast(tf.repeat(soft_mask, model.unit_group_size),tf.float32)],-1))
      delta_sigmoid_per_layer.append(tf.math.square(tf.math.sigmoid((gumbel_sample+latent_group_allocation_logit_+temp_x)/temperature))/tf.math.exp((gumbel_sample+latent_group_allocation_logit_+temp_x)/temperature))
    
    # for i, mask_per_layer in enumerate(soft_mask_total_per_layer):
    #   tf.print(mask_per_layer, "domain: ", domain, "layer: ", i, summarize=-1)

    outputs, _ = model(
        source,
        domain_dropout_mask=soft_mask_total_per_layer,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss

    if config.get("multi_domain", True):
      _domain = source["domain"][0]
    else:
      _domain = 0

    variables = model.trainable_variables
    model_variables = []
    for v in variables:
      if not "latent_group_allocation_logit_per_layer" in v.name:
        model_variables.append(v)
      else:
        continue
    print("var numb: ", len(variables))
    
    gradients = optimizer.get_gradients(training_loss, model_variables)
    #deltaL_deltaM = optimizer.get_gradients(training_loss, soft_mask_logits_per_layer)
    deltaL_deltaM = optimizer.get_gradients(training_loss, soft_mask_total_per_layer)
    
    #optimizer.get_gradients(training_loss,soft_mask_logits_per_layer)
    group_allocation_gradient_per_layer = []
    for i in range(model.encoder.num_layers + model.decoder.num_layers+1):
      delta_sigmoid = delta_sigmoid_per_layer[i]
      deltaresidue_deltalogit1 = delta_sigmoid
      M1 = tf.linalg.diag(delta_sigmoid)
      deltaresidue_deltatempx1 = tf.reduce_sum(delta_sigmoid)
      #tf.print("deltaresidue_deltatempx1",deltaresidue_deltatempx1)      
      deltaTempx_deltaLogit = - tf.tile(tf.expand_dims(deltaresidue_deltalogit1 / deltaresidue_deltatempx1,0),[model.num_domain_unit_group,1])
      #tf.print("deltaresidue_deltalogit", deltaresidue_deltalogit1, "deltaresidue_deltatempx", deltaresidue_deltatempx1, "deltaTempx_deltaLogit", deltaTempx_deltaLogit, summarize=-1)
      deltaM_deltaLogit = tf.eye(model.num_domain_unit_group) + deltaTempx_deltaLogit
      deltaL_deltaLogit = tf.linalg.matmul(tf.expand_dims(deltaL_deltaM[i],0),tf.repeat(deltaM_deltaLogit,repeats=model.unit_group_size,axis=0))
      group_allocation_gradient = optimizer.get_gradients(kl_loss_per_layer[i] * kl_term_coeff, model.latent_group_allocation_logit_per_layer[i])
      group_allocation_gradient[0] = tf.clip_by_norm(tf.tensor_scatter_nd_add(group_allocation_gradient[0],tf.expand_dims(group_allocation_gradient[0].indices,1),deltaL_deltaLogit),1.0)
      group_allocation_gradient_per_layer.append(group_allocation_gradient[0])
    gradient_accumulator(gradients)
    gradient_group_allocation_accumulator(group_allocation_gradient_per_layer)
    num_examples = tf.reduce_sum(target["length"])
    #tf.print("reported_loss: ",reported_loss)
    #tf.print("KL loss: ",tf.math.add_n(kl_loss_per_layer)/len(kl_loss_per_layer))
    return reported_loss, tf.math.add_n(kl_loss_per_layer)/len(kl_loss_per_layer), num_examples, _domain, tf.math.add_n(residue_per_layer)/len(residue_per_layer)
     
  def _apply_gradients():
    variables = model.trainable_variables
    model_variables = []
    for v in variables:
      if not "latent_group_allocation_logit_per_layer" in v.name:
        model_variables.append(v)
      else:
        continue

    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, model_variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)

    grads_and_vars = []
    for gradient, variable in zip(gradient_group_allocation_accumulator.gradients, model.latent_group_allocation_logit_per_layer):
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    latent_logit_optimizer.apply_gradients(grads_and_vars)

    gradient_accumulator.reset()
    gradient_group_allocation_accumulator.reset()

  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_kl_loss, per_replica_num_examples, per_replica_domain, per_replica_residue = strategy.run(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
      kl_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_kl_loss, None)
      _domain = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_domain, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
      residue = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_residue, None)

    return loss, kl_loss, _domain, num_examples, residue
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.run(_apply_gradients)

  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  _, _, _, _, _ = next(train_data_flow)

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _kl_loss = []
  _number_examples = []
  _residue = []
  step = optimizer.iterations.numpy()
  if config.get("reset_step",None):
    print("start from %d-th step"%config.get("reset_step",150000))
    optimizer.iterations.assign(config.get("reset_step",150000))
  
  if step <= 1:
    initializer = config.get("initializer","default")
    if initializer == "default":
      print("Initializing variables by tensorflow default")      
    elif initializer == "variance_scaling":
      print("Initializing variables by tf.variance_scaling")
      initial_value = []
      for v in model.trainable_variables:
        shape = tf.shape(v).numpy()
        initial_value.append(variance_scaling_initialier(shape, scale=1.0, mode="fan_avg", distribution="uniform"))
      weight_reset(initial_value)       
  
  score_type = config.get("score_type","MultiBLEU")
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  ref_eval_concat = file_concatenate(config["eval_ref"],"ref_eval_concat",dir_name=os.path.join(config["model_dir"],"eval"))
  gumbel_temperature_decay = config.get("gumbel_temperature_decay",1000)
  r = config.get("r_coeff",1e-4)
  min_temperature = config.get("min_temperature",0.5)
  start_temperature = config.get("start_temperature",0.5)
  print("dropout_rate",config.get("dropout_rate"))
  print("min_temperature",min_temperature)
  print("gumbel_temperature_decay",gumbel_temperature_decay)
  print("r_coeff",r)
  step = optimizer.iterations.numpy()
  temperature.assign(tf.cast(tf.math.maximum(min_temperature, start_temperature * tf.math.exp(-r*step)),tf.float32))
  print("temperature: ",temperature)
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        loss, kl_loss, _domain, num_examples, residue = next(train_data_flow)    
        _loss.append(loss.numpy())
        _kl_loss.append(kl_loss.numpy())
        _number_examples.append(num_examples.numpy())
        _residue.append(residue.numpy())
      _step()  
      step = optimizer.iterations.numpy()
      
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; KL_loss = %f, temperature = %f, number_examples = %d, residue = %f, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_kl_loss), temperature, np.sum(_number_examples), np.mean(_residue), elapsed)
        tf.summary.experimental.set_step(step)
        tf.summary.scalar("CE_loss", np.mean(_loss), description="training loss")
        tf.summary.flush()
        _loss = []
        _kl_loss = []
        _number_examples = []
        _residue = []
        start = time.time()
      if step % gumbel_temperature_decay==0:
        temperature.assign(tf.cast(tf.math.maximum(min_temperature, start_temperature * tf.math.exp(-r*step)),tf.float32))
        #print("gumbel_temperature: ",gumbel_temperature)
      if step % save_every == 0 and step > 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % config.get("latent_logit_print_every",2000)==0:
        tf.print("latent_group_allocation_logit",model.latent_group_allocation_logit_per_layer[-1],summarize=-1)
      if step % eval_every == 0 and step > 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        output_files = []
        new_bleu = 0.0
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
            output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
            score = translate_topK_sparse_layer_multi_layer(src, ref, model, checkpoint_manager, checkpoint, i, output_file, topK=K, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
            tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
            output_files.append(output_file)
        ##### BLEU on concat dev set.
        output_file_concat = file_concatenate(output_files,"output_file_concat.%s"%os.path.basename(checkpoint_path))
        score = scorer(ref_eval_concat, output_file_concat)
        print("score of model %s on concat dev set: "%checkpoint_manager.latest_checkpoint, score)
        new_bleu = score
        tf.summary.scalar("concat_eval_score", score, description="BLEU on concat dev set")
        #############################
        if new_bleu >= current_max_eval_bleu:
          current_max_eval_bleu = new_bleu
          descending_streak = 0
        else:
          descending_streak += 1
      if descending_streak >= 5:
        break
      tf.summary.flush()
      if step > train_steps:
        break

def finetune_elbo_topK_sparse_layer_multi_layer(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          adapter_optimizer=None,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 

  import tensorflow_probability as tfp
  import scipy
  from scipy import optimize
  tfd = tfp.distributions
  gumbel_dist = tfd.Gumbel(loc=0.,scale=1.)
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  else:
    if checkpoint_path is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
      checkpoint.restore(checkpoint_path)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  ###### early stopping criterion
  current_max_eval_bleu = 0.0
  descending_streak = 0
  ######
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  
  print("There are %d in-domain corpora"%len(source_file))
  
  train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))
  
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator() 

  temperature = tf.Variable(0.2,trainable=False)
  
  kl_term_coeff = config.get("kl_coeff",1.0)
  K = config.get("domain_group_allocation_num",int( (1-config.get("dropout_rate",0.5)) * config.get("num_domain_unit_group",32)))
  print("kl_term_coeff",kl_term_coeff)
  print("topK: ", K)

  my_matrix = np.zeros((model.num_domain_unit_group),)

  def _accumulate_gradients(source, target):
    domain = source["domain"][0]
    kl_loss_per_layer = []
    soft_mask_total_per_layer = []
    soft_mask_logits_per_layer = []
    delta_sigmoid_per_layer = []
    residue_per_layer = []
    for i in range(model.encoder.num_layers + model.decoder.num_layers + 1):
      latent_group_allocation_logit_ = tf.nn.embedding_lookup(model.latent_group_allocation_logit_per_layer[i],domain)
      soft_mask_total_per_layer.append(tf.concat([tf.ones(model.num_shared_units),tf.cast(tf.repeat(tf.reduce_sum(tf.one_hot(tf.math.top_k(tf.nn.embedding_lookup(model.latent_group_allocation_logit_per_layer[i],domain),k=K).indices, depth=model.num_domain_unit_group),0),model.unit_group_size),tf.float32)],-1))


    outputs, _ = model(
        source,
        domain_dropout_mask=soft_mask_total_per_layer,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss

    variables = model.trainable_variables
    model_variables = []
    for v in variables:
      if not "latent_group_allocation_logit_per_layer" in v.name:
        model_variables.append(v)
      else:
        continue
    print("var numb: ", len(variables))
    
    gradients = optimizer.get_gradients(training_loss, model_variables)
    gradient_accumulator(gradients)
    num_examples = tf.reduce_sum(target["length"])
       
    return reported_loss, num_examples
     
  def _apply_gradients():
    variables = model.trainable_variables
    model_variables = []
    for v in variables:
      if not "latent_group_allocation_logit_per_layer" in v.name:
        model_variables.append(v)
      else:
        continue

    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, model_variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)

    gradient_accumulator.reset()

  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_num_examples = strategy.run(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)

    return loss, num_examples
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.run(_apply_gradients)

  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  _, _ = next(train_data_flow)

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _kl_loss = []
  _number_examples = []
  _residue = []
  step = optimizer.iterations.numpy()
  if config.get("reset_step",None):
    print("start from %d-th step"%config.get("reset_step",150000))
    optimizer.iterations.assign(config.get("reset_step",150000))
  
  if step <= 1:
    initializer = config.get("initializer","default")
    if initializer == "default":
      print("Initializing variables by tensorflow default")      
    elif initializer == "variance_scaling":
      print("Initializing variables by tf.variance_scaling")
      initial_value = []
      for v in model.trainable_variables:
        shape = tf.shape(v).numpy()
        initial_value.append(variance_scaling_initialier(shape, scale=1.0, mode="fan_avg", distribution="uniform"))
      weight_reset(initial_value)       
  
  score_type = config.get("score_type","MultiBLEU")
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  ref_eval_concat = file_concatenate(config["eval_ref"],"ref_eval_concat",dir_name=os.path.join(config["model_dir"],"eval"))
  gumbel_temperature_decay = config.get("gumbel_temperature_decay",1000)
  r = config.get("r_coeff",1e-4)
  min_temperature = config.get("min_temperature",0.5)
  start_temperature = config.get("start_temperature",0.5)
  print("dropout_rate",config.get("dropout_rate"))
  print("min_temperature",min_temperature)
  print("gumbel_temperature_decay",gumbel_temperature_decay)
  print("r_coeff",r)
  step = optimizer.iterations.numpy()
  temperature.assign(tf.cast(tf.math.maximum(min_temperature, start_temperature * tf.math.exp(-r*step)),tf.float32))
  print("temperature: ",temperature)
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        loss, num_examples = next(train_data_flow)    
        _loss.append(loss.numpy())
        _number_examples.append(num_examples.numpy())
      _step()  
      step = optimizer.iterations.numpy()
      
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; number_examples = %d, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.sum(_number_examples), elapsed)
        tf.summary.experimental.set_step(step)
        tf.summary.scalar("CE_loss", np.mean(_loss), description="training loss")
        tf.summary.flush()
        _loss = []
        _number_examples = []
        start = time.time()
      if step % save_every == 0 and step > 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % eval_every == 0 and step > 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        output_files = []
        new_bleu = 0.0
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
            output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
            score = translate_topK_sparse_layer_multi_layer(src, ref, model, checkpoint_manager, checkpoint, i, output_file, topK=K, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
            tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
            output_files.append(output_file)
        ##### BLEU on concat dev set.
        output_file_concat = file_concatenate(output_files,"output_file_concat.%s"%os.path.basename(checkpoint_path))
        score = scorer(ref_eval_concat, output_file_concat)
        print("score of model %s on concat dev set: "%checkpoint_manager.latest_checkpoint, score)
        new_bleu = score
        tf.summary.scalar("concat_eval_score", score, description="BLEU on concat dev set")
        #############################
        if new_bleu >= current_max_eval_bleu:
          current_max_eval_bleu = new_bleu
          descending_streak = 0
        else:
          descending_streak += 1
      if descending_streak >= 5:
        break
      tf.summary.flush()
      if step > train_steps:
        break

def translate_topK_sparse_layer_multi_layer(source_file,
              reference,
              model,
              checkpoint_manager,
              checkpoint,              
              domain,
              output_file,
              length_penalty,
              topK=1,
              is_noisy=1,
              gumbel_temperature = 0.2,
              checkpoint_path=None,
              probs_file=None,
              experiment="ldr",
              score_type="MultiBLEU",
              batch_size=5,
              beam_size=5):
  
  tfa.options.disable_custom_kernel()
  # Create the inference dataset.
  if checkpoint_path == None:
    checkpoint_path = checkpoint_manager.latest_checkpoint
  tf.get_logger().info("Evaluating model %s", checkpoint_path)
  print("In domain %d"%domain)
  checkpoint.restore(checkpoint_path)
  dataset = model.examples_inputter.make_inference_dataset(source_file, batch_size, domain, is_noisy=is_noisy)
  iterator = iter(dataset)

  # Create the mapping for target ids to tokens.
  ids_to_tokens = model.labels_inputter.ids_to_tokens

  domain_dropout_mask = []

  for i in range(model.mask_num):
    topK_ = tf.math.top_k(tf.nn.embedding_lookup(model.latent_group_allocation_logit_per_layer[i],domain),k=topK).indices.numpy()
    group_allocation = np.zeros(model.num_domain_unit_group)
    for j in topK_:
      group_allocation[j] = 1

    tf.print("group_allocation:",group_allocation,"domain:",domain,"layer:",i,summarize=-1)

    group_allocation = tf.repeat(tf.Variable(group_allocation,dtype=tf.float32),model.unit_group_size)

    domain_dropout_mask.append(tf.concat([tf.ones(model.num_shared_units),group_allocation],-1))  

  @tf.function
  def predict_next():    
    source = next(iterator)
    source_length = source["length"]
    batch_size = tf.shape(source_length)[0]
    source_inputs = model.features_inputter(source)
    encoder_outputs, _, _ = model.encoder([source_inputs, source["domain"], domain_dropout_mask[:model.encoder.num_layers+1]], source_length, training=False, internal_node_printing=True)
    
    # Prepare the decoding strategy.
    if beam_size > 1:
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      source_length = tfa.seq2seq.tile_batch(source_length, beam_size)
      decoding_strategy = onmt.utils.BeamSearch(beam_size, length_penalty=length_penalty)
    else:
      decoding_strategy = onmt.utils.GreedySearch()

    # Run dynamic decoding.
    decoder_state = model.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=source_length)
    map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}, training=False), tf.dtypes.cast(tf.fill(tf.expand_dims(tf.shape(ids)[0],0), domain), tf.int64), domain_dropout_mask[model.encoder.num_layers+1:]]
    
    decoded = model.decoder.dynamic_decode(
        map_input_fn,
        tf.fill([batch_size], START_OF_SENTENCE_ID),
        end_id=END_OF_SENTENCE_ID,
        initial_state=decoder_state,
        decoding_strategy=decoding_strategy,
        maximum_iterations=250)
    target_lengths = decoded.lengths
    target_tokens = ids_to_tokens.lookup(tf.cast(decoded.ids, tf.int64))
    return target_tokens, target_lengths

  # Iterates on the dataset.
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  print("output file: ", output_file)
  with open(output_file, "w") as output_:
    while True:    
      try:
        batch_tokens, batch_length = predict_next()
        for tokens, length in zip(batch_tokens.numpy(), batch_length.numpy()):
          sentence = b" ".join(tokens[0][:length[0]])
          print_bytes(sentence, output_)
          #print_bytes(sentence)
      except tf.errors.OutOfRangeError:
        break
  if reference!=None:
    print("score of model %s on test set %s: "%(checkpoint_manager.latest_checkpoint, source_file), scorer(reference, output_file))
    score = scorer(reference, output_file)
    if score is None:
      return 0.0
    else:
      return score

def train_elbo_topK_sparse_layer_multi_layer_v1(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          adapter_optimizer=None,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 

  import tensorflow_probability as tfp
  import scipy
  from scipy import optimize
  tfd = tfp.distributions
  gumbel_dist = tfd.Gumbel(loc=0.,scale=1.)
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  else:
    if checkpoint_path is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
      checkpoint.restore(checkpoint_path)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  ###### early stopping criterion
  current_max_eval_bleu = 0.0
  descending_streak = 0
  ######
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  
  print("There are %d in-domain corpora"%len(source_file))
  
  train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))
  
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator() 
    gradient_group_allocation_accumulator = optimizer_util.GradientAccumulator()
    latent_logit_optimizer = tfa.optimizers.LazyAdam(config.get("latent_logit_lr",0.01))

  temperature = tf.Variable(0.2,trainable=False)
  
  kl_term_coeff = config.get("kl_coeff",1.0)
  K = config.get("domain_group_allocation_num",int( (1-config.get("dropout_rate",0.5)) * config.get("num_domain_unit_group",32)))
  print("kl_term_coeff",kl_term_coeff)
  print("topK: ", K)

  def _accumulate_gradients(source, target):
    domain = source["domain"][0]
    kl_loss_per_layer = []
    soft_mask_total_per_layer = []
    soft_mask_logits_per_layer = []
    delta_sigmoid_per_layer = []
    residue_per_layer = []
    for i in range(model.encoder.num_layers + model.decoder.num_layers + 2):
      gumbel_sample = gumbel_dist.sample([model.num_domain_unit_group])
      latent_group_allocation_logit_ = tf.nn.embedding_lookup(model.latent_group_allocation_logit_per_layer[i],domain)
      domain_allocation_probs = tf.math.softmax(latent_group_allocation_logit_)
      kl_loss_per_layer.append(- tf.reduce_mean(tf.math.log(domain_allocation_probs)))
      f = lambda x: tf.reduce_sum(tf.math.sigmoid((gumbel_sample+latent_group_allocation_logit_+x)/temperature)) - K
      temp_x = tfp.math.find_root_chandrupatla(f, low=-100, high=100, position_tolerance=1e-08,value_tolerance=0.0, max_iterations=100, stopping_policy_fn=tf.reduce_all,validate_args=False, name='find_root_chandrupatla').estimated_root
      residue_per_layer.append(tf.reduce_sum(tf.math.sigmoid((gumbel_sample+latent_group_allocation_logit_+temp_x)/temperature)) - K)
      soft_mask_logits = (gumbel_sample+latent_group_allocation_logit_+temp_x)/temperature
      soft_mask_logits_per_layer.append(soft_mask_logits)
      #tf.print("soft_mask_logits",soft_mask_logits,summarize=-1)
      soft_mask = tf.math.sigmoid(soft_mask_logits)
      #tf.print("soft_mask", soft_mask, "domain_allocation_probs",domain_allocation_probs,summarize=-1)
      soft_mask_total_per_layer.append(tf.concat([tf.ones(model.num_shared_units),tf.cast(tf.repeat(soft_mask, model.unit_group_size),tf.float32)],-1))
      delta_sigmoid_per_layer.append(tf.math.square(tf.math.sigmoid((gumbel_sample+latent_group_allocation_logit_+temp_x)/temperature))/tf.math.exp((gumbel_sample+latent_group_allocation_logit_+temp_x)/temperature))
    
    # for i, mask_per_layer in enumerate(soft_mask_total_per_layer):
    #   tf.print(mask_per_layer, "domain: ", domain, "layer: ", i, summarize=-1)

    outputs, _ = model(
        source,
        domain_dropout_mask=soft_mask_total_per_layer,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss

    if config.get("multi_domain", True):
      _domain = source["domain"][0]
    else:
      _domain = 0

    variables = model.trainable_variables
    model_variables = []
    for v in variables:
      if not "latent_group_allocation_logit_per_layer" in v.name:
        model_variables.append(v)
      else:
        continue
    print("var numb: ", len(variables))
    
    gradients = optimizer.get_gradients(training_loss, model_variables)
    deltaL_deltaM = optimizer.get_gradients(training_loss, soft_mask_logits_per_layer)
    #optimizer.get_gradients(training_loss,soft_mask_logits_per_layer)
    group_allocation_gradient_per_layer = []
    for i in range(model.encoder.num_layers + model.decoder.num_layers+2):
      delta_sigmoid = delta_sigmoid_per_layer[i]
      deltaresidue_deltalogit1 = delta_sigmoid
      M1 = tf.linalg.diag(delta_sigmoid)
      deltaresidue_deltatempx1 = tf.reduce_sum(delta_sigmoid)
      #tf.print("deltaresidue_deltatempx1",deltaresidue_deltatempx1)      
      deltaTempx_deltaLogit = - tf.tile(tf.expand_dims(deltaresidue_deltalogit1 / deltaresidue_deltatempx1,0),[model.num_domain_unit_group,1])
      #tf.print("deltaresidue_deltalogit", deltaresidue_deltalogit1, "deltaresidue_deltatempx", deltaresidue_deltatempx1, "deltaTempx_deltaLogit", deltaTempx_deltaLogit, summarize=-1)
      deltaM_deltaLogit = tf.eye(model.num_domain_unit_group) + deltaTempx_deltaLogit
      deltaL_deltaLogit = tf.linalg.matmul(tf.expand_dims(deltaL_deltaM[i],0),deltaM_deltaLogit)
      group_allocation_gradient = optimizer.get_gradients(kl_loss_per_layer[i] * kl_term_coeff, model.latent_group_allocation_logit_per_layer[i])
      group_allocation_gradient[0] = tf.clip_by_norm(tf.tensor_scatter_nd_add(group_allocation_gradient[0],tf.expand_dims(group_allocation_gradient[0].indices,1),deltaL_deltaLogit),1.0)
      group_allocation_gradient_per_layer.append(group_allocation_gradient[0])
    gradient_accumulator(gradients)
    gradient_group_allocation_accumulator(group_allocation_gradient_per_layer)
    num_examples = tf.reduce_sum(target["length"])
       
    return reported_loss, tf.math.add_n(kl_loss_per_layer)/len(kl_loss_per_layer), num_examples, _domain, tf.math.add_n(residue_per_layer)/len(residue_per_layer)
     
  def _apply_gradients():
    variables = model.trainable_variables
    model_variables = []
    for v in variables:
      if not "latent_group_allocation_logit_per_layer" in v.name:
        model_variables.append(v)
      else:
        continue

    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, model_variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)

    grads_and_vars = []
    for gradient, variable in zip(gradient_group_allocation_accumulator.gradients, model.latent_group_allocation_logit_per_layer):
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    latent_logit_optimizer.apply_gradients(grads_and_vars)

    gradient_accumulator.reset()
    gradient_group_allocation_accumulator.reset()

  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_kl_loss, per_replica_num_examples, per_replica_domain, per_replica_residue = strategy.run(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
      kl_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_kl_loss, None)
      _domain = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_domain, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
      residue = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_residue, None)

    return loss, kl_loss, _domain, num_examples, residue
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.run(_apply_gradients)

  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  _, _, _, _, _ = next(train_data_flow)

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _kl_loss = []
  _number_examples = []
  _residue = []
  step = optimizer.iterations.numpy()
  
  score_type = config.get("score_type","MultiBLEU")
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  ref_eval_concat = file_concatenate(config["eval_ref"],"ref_eval_concat",dir_name=os.path.join(config["model_dir"],"eval"))
  gumbel_temperature_decay = config.get("gumbel_temperature_decay",1000)
  r = config.get("r_coeff",1e-4)
  min_temperature = config.get("min_temperature",0.5)
  start_temperature = config.get("start_temperature",0.5)
  print("dropout_rate",config.get("dropout_rate"))
  print("min_temperature",min_temperature)
  print("gumbel_temperature_decay",gumbel_temperature_decay)
  print("r_coeff",r)
  step = optimizer.iterations.numpy()
  temperature.assign(tf.cast(tf.math.maximum(min_temperature, start_temperature * tf.math.exp(-r*step)),tf.float32))
  print("temperature: ",temperature)
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        loss, kl_loss, _domain, num_examples, residue = next(train_data_flow)    
        _loss.append(loss.numpy())
        _kl_loss.append(kl_loss.numpy())
        _number_examples.append(num_examples.numpy())
        _residue.append(residue.numpy())
      _step()  
      step = optimizer.iterations.numpy()
      
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; KL_loss = %f, temperature = %f, number_examples = %d, residue = %f, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_kl_loss), temperature, np.sum(_number_examples), np.mean(_residue), elapsed)
        _loss = []
        _kl_loss = []
        _number_examples = []
        _residue = []
        start = time.time()
      if step % gumbel_temperature_decay==0:
        temperature.assign(tf.cast(tf.math.maximum(min_temperature, start_temperature * tf.math.exp(-r*step)),tf.float32))
        #print("gumbel_temperature: ",gumbel_temperature)
      if step % save_every == 0 and step > 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % config.get("latent_logit_print_every",2000)==0:
        tf.print("latent_group_allocation_logit",model.latent_group_allocation_logit_per_layer[-1],summarize=-1)
      if step % eval_every == 0 and step > 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        output_files = []
        new_bleu = 0.0
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
            output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
            score = translate_topK_sparse_layer_multi_layer(src, ref, model, checkpoint_manager, checkpoint, i, output_file, topK=K, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
            tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
            output_files.append(output_file)
        ##### BLEU on concat dev set.
        output_file_concat = file_concatenate(output_files,"output_file_concat.%s"%os.path.basename(checkpoint_path))
        score = scorer(ref_eval_concat, output_file_concat)
        print("score of model %s on concat dev set: "%checkpoint_manager.latest_checkpoint, score)
        new_bleu = score
        tf.summary.scalar("concat_eval_score", score, description="BLEU on concat dev set")
        #############################
        if new_bleu >= current_max_eval_bleu:
          current_max_eval_bleu = new_bleu
          descending_streak = 0
        else:
          descending_streak += 1
      if descending_streak >= 5:
        break
      tf.summary.flush()
      if step > train_steps:
        break

def fewshot_elbo_topK_sparse_layer_multi_layer(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          adapter_optimizer=None,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 

  import tensorflow_probability as tfp
  import scipy
  from scipy import optimize
  tfd = tfp.distributions
  gumbel_dist = tfd.Gumbel(loc=0.,scale=1.)
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  else:
    if checkpoint_path is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
      checkpoint.restore(checkpoint_path)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  ###### early stopping criterion
  current_max_eval_bleu = 0.0
  descending_streak = 0
  ######
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  
  print("There are %d in-domain corpora"%len(source_file))
  
  train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))
  
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    #gradient_accumulator = optimizer_util.GradientAccumulator() 
    gradient_group_allocation_accumulator = optimizer_util.GradientAccumulator()
    latent_logit_optimizer = tfa.optimizers.LazyAdam(config.get("latent_logit_lr",0.01))

  temperature = tf.Variable(0.2,trainable=False)
  
  kl_term_coeff = config.get("kl_coeff",1.0)
  K = config.get("domain_group_allocation_num",int( (1-config.get("dropout_rate",0.5)) * config.get("num_domain_unit_group",32)))
  print("kl_term_coeff",kl_term_coeff)
  print("topK: ", K)

  def _accumulate_gradients(source, target):
    domain = source["domain"][0]
    kl_loss_per_layer = []
    soft_mask_total_per_layer = []
    soft_mask_logits_per_layer = []
    delta_sigmoid_per_layer = []
    residue_per_layer = []
    for i in range(model.encoder.num_layers + model.decoder.num_layers + 1):
      gumbel_sample = gumbel_dist.sample([model.num_domain_unit_group])
      latent_group_allocation_logit_ = tf.nn.embedding_lookup(model.latent_group_allocation_logit_per_layer[i],domain)
      domain_allocation_probs = tf.math.softmax(latent_group_allocation_logit_)
      kl_loss_per_layer.append(- tf.reduce_mean(tf.math.log(domain_allocation_probs)))
      f = lambda x: tf.reduce_sum(tf.math.sigmoid((gumbel_sample+latent_group_allocation_logit_+x)/temperature)) - K
      temp_x = tfp.math.find_root_chandrupatla(f, low=-100, high=100, position_tolerance=1e-08,value_tolerance=0.0, max_iterations=100, stopping_policy_fn=tf.reduce_all,validate_args=False, name='find_root_chandrupatla').estimated_root
      residue_per_layer.append(tf.reduce_sum(tf.math.sigmoid((gumbel_sample+latent_group_allocation_logit_+temp_x)/temperature)) - K)
      soft_mask_logits = (gumbel_sample+latent_group_allocation_logit_+temp_x)/temperature
      soft_mask_logits_per_layer.append(soft_mask_logits)
      #tf.print("soft_mask_logits",soft_mask_logits,summarize=-1)
      soft_mask = tf.math.sigmoid(soft_mask_logits)
      #tf.print("soft_mask", soft_mask, "domain_allocation_probs",domain_allocation_probs,summarize=-1)
      soft_mask_total_per_layer.append(tf.concat([tf.ones(model.num_shared_units),tf.cast(tf.repeat(soft_mask, model.unit_group_size),tf.float32)],-1))
      delta_sigmoid_per_layer.append(tf.math.square(tf.math.sigmoid((gumbel_sample+latent_group_allocation_logit_+temp_x)/temperature))/tf.math.exp((gumbel_sample+latent_group_allocation_logit_+temp_x)/temperature))
    
    # for i, mask_per_layer in enumerate(soft_mask_total_per_layer):
    #   tf.print(mask_per_layer, "domain: ", domain, "layer: ", i, summarize=-1)

    outputs, _ = model(
        source,
        domain_dropout_mask=soft_mask_total_per_layer,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss

    if config.get("multi_domain", True):
      _domain = source["domain"][0]
    else:
      _domain = 0

    variables = model.trainable_variables
    model_variables = []
    for v in variables:
      if not "latent_group_allocation_logit_per_layer" in v.name:
        model_variables.append(v)
      else:
        continue
    print("var numb: ", len(variables))
    
    #gradients = optimizer.get_gradients(training_loss, model_variables)
    deltaL_deltaM = optimizer.get_gradients(training_loss, soft_mask_logits_per_layer)
    #optimizer.get_gradients(training_loss,soft_mask_logits_per_layer)
    group_allocation_gradient_per_layer = []
    for i in range(model.encoder.num_layers + model.decoder.num_layers+1):
      delta_sigmoid = delta_sigmoid_per_layer[i]
      deltaresidue_deltalogit1 = delta_sigmoid
      M1 = tf.linalg.diag(delta_sigmoid)
      deltaresidue_deltatempx1 = tf.reduce_sum(delta_sigmoid)
      #tf.print("deltaresidue_deltatempx1",deltaresidue_deltatempx1)      
      deltaTempx_deltaLogit = - tf.tile(tf.expand_dims(deltaresidue_deltalogit1 / deltaresidue_deltatempx1,0),[model.num_domain_unit_group,1])
      #tf.print("deltaresidue_deltalogit", deltaresidue_deltalogit1, "deltaresidue_deltatempx", deltaresidue_deltatempx1, "deltaTempx_deltaLogit", deltaTempx_deltaLogit, summarize=-1)
      deltaM_deltaLogit = tf.eye(model.num_domain_unit_group) + deltaTempx_deltaLogit
      deltaL_deltaLogit = tf.linalg.matmul(tf.expand_dims(deltaL_deltaM[i],0),deltaM_deltaLogit)
      group_allocation_gradient = optimizer.get_gradients(kl_loss_per_layer[i] * kl_term_coeff, model.latent_group_allocation_logit_per_layer[i])
      group_allocation_gradient[0] = tf.clip_by_norm(tf.tensor_scatter_nd_add(group_allocation_gradient[0],tf.expand_dims(group_allocation_gradient[0].indices,1),deltaL_deltaLogit),1.0)
      group_allocation_gradient_per_layer.append(group_allocation_gradient[0])
    #gradient_accumulator(gradients)
    gradient_group_allocation_accumulator(group_allocation_gradient_per_layer)
    num_examples = tf.reduce_sum(target["length"])
       
    return reported_loss, tf.math.add_n(kl_loss_per_layer)/len(kl_loss_per_layer), num_examples, _domain, tf.math.add_n(residue_per_layer)/len(residue_per_layer)
     
  def _apply_gradients():
    variables = model.trainable_variables
    model_variables = []
    for v in variables:
      if not "latent_group_allocation_logit_per_layer" in v.name:
        model_variables.append(v)
      else:
        continue

    # grads_and_vars = []
    # for gradient, variable in zip(gradient_accumulator.gradients, model_variables):
    #   # optimizer.apply_gradients will sum the gradients accross replicas.
    #   scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
    #   grads_and_vars.append((scaled_gradient, variable))
    # optimizer.apply_gradients(grads_and_vars)

    grads_and_vars = []
    for gradient, variable in zip(gradient_group_allocation_accumulator.gradients, model.latent_group_allocation_logit_per_layer):
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_group_allocation_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    latent_logit_optimizer.apply_gradients(grads_and_vars)

    #gradient_accumulator.reset()
    gradient_group_allocation_accumulator.reset()

  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_kl_loss, per_replica_num_examples, per_replica_domain, per_replica_residue = strategy.run(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
      kl_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_kl_loss, None)
      _domain = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_domain, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
      residue = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_residue, None)

    return loss, kl_loss, _domain, num_examples, residue
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.run(_apply_gradients)

  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  _, _, _, _, _ = next(train_data_flow)

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _kl_loss = []
  _number_examples = []
  _residue = []
  step = optimizer.iterations.numpy()
  latent_logit_optimizer.iterations.assign(step)   
  
  score_type = config.get("score_type","MultiBLEU")
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  ref_eval_concat = file_concatenate(config["eval_ref"],"ref_eval_concat",dir_name=os.path.join(config["model_dir"],"eval"))
  gumbel_temperature_decay = config.get("gumbel_temperature_decay",1000)
  r = config.get("r_coeff",1e-4)
  min_temperature = config.get("min_temperature",0.5)
  start_temperature = config.get("start_temperature",0.5)
  print("dropout_rate",config.get("dropout_rate"))
  print("min_temperature",min_temperature)
  print("gumbel_temperature_decay",gumbel_temperature_decay)
  print("r_coeff",r)
  step = latent_logit_optimizer.iterations.numpy()
  temperature.assign(tf.cast(tf.math.maximum(min_temperature, start_temperature * tf.math.exp(-r*step)),tf.float32))
  print("temperature: ",temperature)
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        loss, kl_loss, _domain, num_examples, residue = next(train_data_flow)    
        _loss.append(loss.numpy())
        _kl_loss.append(kl_loss.numpy())
        _number_examples.append(num_examples.numpy())
        _residue.append(residue.numpy())
      _step()  
      step = latent_logit_optimizer.iterations.numpy()
      
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; KL_loss = %f, temperature = %f, number_examples = %d, residue = %f, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_kl_loss), temperature, np.sum(_number_examples), np.mean(_residue), elapsed)
        _loss = []
        _kl_loss = []
        _number_examples = []
        _residue = []
        start = time.time()
      if step % gumbel_temperature_decay==0:
        temperature.assign(tf.cast(tf.math.maximum(min_temperature, start_temperature * tf.math.exp(-r*step)),tf.float32))
        #print("gumbel_temperature: ",gumbel_temperature)
      if step % save_every == 0 and step > 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % config.get("latent_logit_print_every",2000)==0:
        tf.print("latent_group_allocation_logit",model.latent_group_allocation_logit_per_layer[-1],summarize=-1)
      if step % eval_every == 0 and step > 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        output_files = []
        new_bleu = 0.0
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
            output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
            score = translate_topK_sparse_layer_multi_layer(src, ref, model, checkpoint_manager, checkpoint, i, output_file, topK=K, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
            tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
            output_files.append(output_file)
        ##### BLEU on concat dev set.
        output_file_concat = file_concatenate(output_files,"output_file_concat.%s"%os.path.basename(checkpoint_path))
        score = scorer(ref_eval_concat, output_file_concat)
        print("score of model %s on concat dev set: "%checkpoint_manager.latest_checkpoint, score)
        new_bleu = score
        tf.summary.scalar("concat_eval_score", score, description="BLEU on concat dev set")
        #############################
        if new_bleu >= current_max_eval_bleu:
          current_max_eval_bleu = new_bleu
          descending_streak = 0
        else:
          descending_streak += 1
      if descending_streak >= 5:
        break
      tf.summary.flush()
      if step > train_steps:
        break

def train_elbo_hierarchical_topK_sparse_layer_multi_layer(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          adapter_optimizer=None,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 

  import tensorflow_probability as tfp
  import scipy
  from scipy import optimize
  tfd = tfp.distributions
  gumbel_dist = tfd.Gumbel(loc=0.,scale=1.)
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  else:
    if checkpoint_path is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
      checkpoint.restore(checkpoint_path)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  ###### early stopping criterion
  current_max_eval_bleu = 0.0
  descending_streak = 0
  ######
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  
  print("There are %d in-domain corpora"%len(source_file))
  
  train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))
  
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator() 
    gradient_group_allocation_accumulator = optimizer_util.GradientAccumulator()
    latent_logit_optimizer = tfa.optimizers.LazyAdam(config.get("latent_logit_lr",0.01))

  temperature = tf.Variable(0.2,trainable=False)
  topK_temperature = tf.Variable(0.2,trainable=False)
  kl_term_coeff = config.get("kl_coeff",1.0)
  kl_topK_term_coeff = config.get("kl_topK_coeff",1.0)
  #K = config.get("domain_group_allocation_num",int( (1-config.get("dropout_rate",0.5)) * config.get("num_domain_unit_group",32)))
  print("kl_term_coeff",kl_term_coeff)
  print("kl_topK_term_coeff",kl_topK_term_coeff)
  #print("topK: ", K)

  def _accumulate_gradients(source, target):
    domain = source["domain"][0]
    kl_loss_per_layer = []
    kl_topK_loss_per_layer = []
    soft_mask_total_per_layer = []
    soft_mask_logits_per_layer = []
    delta_sigmoid_per_layer = []
    delta_softmax_topK_per_layer = []
    residue_per_layer = []
    
    for i in range(model.encoder.num_layers + model.decoder.num_layers):
      gumbel_sample = gumbel_dist.sample([model.num_domain_unit_group])
      gumbel_topK_sample = gumbel_dist.sample([model.num_domain_unit_group-1])
      #Logits
      latent_group_allocation_logit_ = tf.nn.embedding_lookup(model.latent_group_allocation_logit_per_layer[i],domain)
      latent_topk_logit_ = tf.nn.embedding_lookup(model.latent_topk_logit_per_layer[i],domain)

      #Dropping probs
      domain_allocation_probs = tf.math.softmax(latent_group_allocation_logit_)
      domain_topk_probs = tf.math.softmax(latent_topk_logit_)
      
      # Rate upper bound
      kl_loss_per_layer.append(- tf.reduce_mean(domain_allocation_probs * tf.math.log(domain_allocation_probs)))
      kl_topK_loss_per_layer.append(- tf.reduce_mean(domain_topk_probs * tf.math.log(domain_topk_probs)))

      # Solve KKT constraints
      K = tf.reduce_sum(tf.math.softmax((latent_topk_logit_+gumbel_topK_sample)/topK_temperature) * (tf.range(model.num_domain_unit_group-1)+1))
      f = lambda x: tf.reduce_sum(tf.math.sigmoid((gumbel_sample+latent_group_allocation_logit_+x)/temperature)) - K
      temp_x = tfp.math.find_root_chandrupatla(f, low=-100, high=100, position_tolerance=1e-08,value_tolerance=0.0, max_iterations=100, stopping_policy_fn=tf.reduce_all,validate_args=False, name='find_root_chandrupatla').estimated_root
      residue_per_layer.append(tf.reduce_sum(tf.math.sigmoid((gumbel_sample+latent_group_allocation_logit_+temp_x)/temperature)) - K)
      soft_mask_logits = (gumbel_sample+latent_group_allocation_logit_+temp_x)/temperature
      soft_mask_logits_per_layer.append(soft_mask_logits)
      #tf.print("soft_mask_logits",soft_mask_logits,summarize=-1)
      soft_mask = tf.math.sigmoid(soft_mask_logits)
      #tf.print("soft_mask", soft_mask, "domain_allocation_probs",domain_allocation_probs,summarize=-1)
      soft_mask_total_per_layer.append(tf.concat([tf.ones(model.num_shared_units),tf.cast(tf.repeat(soft_mask, model.unit_group_size),tf.float32)],-1))
      delta_sigmoid_per_layer.append(tf.math.square(tf.math.sigmoid((gumbel_sample+latent_group_allocation_logit_+temp_x)/temperature))/tf.math.exp((gumbel_sample+latent_group_allocation_logit_+temp_x)/temperature))
      delta_softmax_topK_per_layer.append(tf.tile(tf.reshape(domain_topk_probs,(-1,1)),[1,model.num_domain_unit_group-1]) * (tf.eyes(model.num_domain_unit_group-1) - tf.tile(tf.reshape(domain_topk_probs,(1,-1)),[model.num_domain_unit_group-1,1])))
    # for i, mask_per_layer in enumerate(soft_mask_total_per_layer):
    #   tf.print(mask_per_layer, "domain: ", domain, "layer: ", i, summarize=-1)

    outputs, _ = model(
        source,
        domain_dropout_mask=soft_mask_total_per_layer,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss

    if config.get("multi_domain", True):
      _domain = source["domain"][0]
    else:
      _domain = 0

    variables = model.trainable_variables
    model_variables = []
    for v in variables:
      if not "latent_group_allocation_logit_per_layer" in v.name:
        model_variables.append(v)
      else:
        continue
    print("var numb: ", len(variables))
    
    gradients = optimizer.get_gradients(training_loss, model_variables)
    deltaL_deltaM = optimizer.get_gradients(training_loss, soft_mask_logits_per_layer)
    #optimizer.get_gradients(training_loss,soft_mask_logits_per_layer)
    group_allocation_gradient_per_layer = []
    for i in range(model.encoder.num_layers + model.decoder.num_layers):
      delta_sigmoid = delta_sigmoid_per_layer[i]
      delta_softmax_topk = delta_softmax_topK_per_layer[i]
      deltaresidue_deltalogit1 = delta_sigmoid
      M1 = tf.linalg.diag(delta_sigmoid)
      deltaresidue_deltatempx1 = tf.reduce_sum(delta_sigmoid)
      #tf.print("deltaresidue_deltatempx1",deltaresidue_deltatempx1)      
      deltaTempx_deltaLogit = - tf.tile(tf.expand_dims(deltaresidue_deltalogit1 / deltaresidue_deltatempx1,0),[model.num_domain_unit_group,1])
      #tf.print("deltaresidue_deltalogit", deltaresidue_deltalogit1, "deltaresidue_deltatempx", deltaresidue_deltatempx1, "deltaTempx_deltaLogit", deltaTempx_deltaLogit, summarize=-1)
      deltaM_deltaLogit = tf.eye(model.num_domain_unit_group) + deltaTempx_deltaLogit
      deltaTempx_deltaTopKLogit = tf.linalg.matmul(tf.range(model.num_domain_unit_group-1),delta_softmax_topk) / deltaresidue_deltatempx1
      deltaL_deltaLogit = tf.linalg.matmul(tf.expand_dims(deltaL_deltaM[i],0),deltaM_deltaLogit) # move temperature constant to step size
      group_allocation_gradient = optimizer.get_gradients(kl_loss_per_layer[i] * kl_term_coeff, model.latent_group_allocation_logit_per_layer[i])
      group_allocation_gradient[0] = tf.clip_by_norm(tf.tensor_scatter_nd_add(group_allocation_gradient[0],tf.expand_dims(group_allocation_gradient[0].indices,1),deltaL_deltaLogit),1.0)
      group_allocation_gradient_per_layer.append(group_allocation_gradient[0])
    gradient_accumulator(gradients)
    gradient_group_allocation_accumulator(group_allocation_gradient_per_layer)
    num_examples = tf.reduce_sum(target["length"])
       
    return reported_loss, tf.math.add_n(kl_loss_per_layer)/len(kl_loss_per_layer), num_examples, _domain, tf.math.add_n(residue_per_layer)/len(residue_per_layer)
     
  def _apply_gradients():
    variables = model.trainable_variables
    model_variables = []
    for v in variables:
      if not "latent_group_allocation_logit_per_layer" in v.name:
        model_variables.append(v)
      else:
        continue

    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, model_variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)

    grads_and_vars = []
    for gradient, variable in zip(gradient_group_allocation_accumulator.gradients, model.latent_group_allocation_logit_per_layer):
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    latent_logit_optimizer.apply_gradients(grads_and_vars)

    gradient_accumulator.reset()
    gradient_group_allocation_accumulator.reset()

  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_kl_loss, per_replica_num_examples, per_replica_domain, per_replica_residue = strategy.run(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
      kl_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_kl_loss, None)
      _domain = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_domain, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
      residue = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_residue, None)

    return loss, kl_loss, _domain, num_examples, residue
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.run(_apply_gradients)

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  _, _, _, _, _ = next(train_data_flow)

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _kl_loss = []
  _number_examples = []
  _residue = []
  step = optimizer.iterations.numpy()
    
  score_type = config.get("score_type","MultiBLEU")
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  ref_eval_concat = file_concatenate(config["eval_ref"],"ref_eval_concat",dir_name=os.path.join(config["model_dir"],"eval"))
  gumbel_temperature_decay = config.get("gumbel_temperature_decay",1000)
  r = config.get("r_coeff",1e-4)
  min_temperature = config.get("min_temperature",0.5)
  start_temperature = config.get("start_temperature",0.5)
  print("dropout_rate",config.get("dropout_rate"))
  print("min_temperature",min_temperature)
  print("gumbel_temperature_decay",gumbel_temperature_decay)
  print("r_coeff",r)
  step = optimizer.iterations.numpy()
  temperature.assign(tf.cast(tf.math.maximum(min_temperature, start_temperature * tf.math.exp(-r*step)),tf.float32))
  print("temperature: ",temperature)
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        loss, kl_loss, _domain, num_examples, residue = next(train_data_flow)    
        _loss.append(loss.numpy())
        _kl_loss.append(kl_loss.numpy())
        _number_examples.append(num_examples.numpy())
        _residue.append(residue.numpy())
      _step()  
      step = optimizer.iterations.numpy()
      
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; KL_loss = %f, temperature = %f, number_examples = %d, residue = %f, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_kl_loss), temperature, np.sum(_number_examples), np.mean(_residue), elapsed)
        tf.summary.experimental.set_step(step)
        tf.summary.scalar("CE_loss", np.mean(_loss), description="training loss")
        tf.summary.flush()
        _loss = []
        _kl_loss = []
        _number_examples = []
        _residue = []
        start = time.time()
      if step % gumbel_temperature_decay==0:
        temperature.assign(tf.cast(tf.math.maximum(min_temperature, start_temperature * tf.math.exp(-r*step)),tf.float32))
        #print("gumbel_temperature: ",gumbel_temperature)
      if step % save_every == 0 and step > 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % config.get("latent_logit_print_every",2000)==0:
        tf.print("latent_group_allocation_logit",model.latent_group_allocation_logit_per_layer[-1],summarize=-1)
      if step % eval_every == 0 and step > 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        output_files = []
        new_bleu = 0.0
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
            output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
            score = translate_topK_sparse_layer_multi_layer(src, ref, model, checkpoint_manager, checkpoint, i, output_file, topK=K, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
            tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
            output_files.append(output_file)
        ##### BLEU on concat dev set.
        output_file_concat = file_concatenate(output_files,"output_file_concat.%s"%os.path.basename(checkpoint_path))
        score = scorer(ref_eval_concat, output_file_concat)
        print("score of model %s on concat dev set: "%checkpoint_manager.latest_checkpoint, score)
        new_bleu = score
        tf.summary.scalar("concat_eval_score", score, description="BLEU on concat dev set")
        #############################
        if new_bleu >= current_max_eval_bleu:
          current_max_eval_bleu = new_bleu
          descending_streak = 0
        else:
          descending_streak += 1
      if descending_streak >= 5:
        break
      tf.summary.flush()
      if step > train_steps:
        break

def train_elbo_Instance_Aware_topK_sparse_layer_multi_layer(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          adapter_optimizer=None,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 

  import tensorflow_probability as tfp
  import scipy
  from scipy import optimize
  tfd = tfp.distributions
  gumbel_dist = tfd.Gumbel(loc=0.,scale=1.)
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  else:
    if checkpoint_path is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
      checkpoint.restore(checkpoint_path)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  ###### early stopping criterion
  current_max_eval_bleu = 0.0
  descending_streak = 0
  ######
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  
  print("There are %d in-domain corpora"%len(source_file))
  
  train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))
  
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator() 
    gradient_group_allocation_accumulator = optimizer_util.GradientAccumulator()
    latent_logit_optimizer = tfa.optimizers.LazyAdam(config.get("latent_logit_lr",0.01))

  temperature = tf.Variable(0.2,trainable=False)
  
  kl_term_coeff = config.get("kl_coeff",1.0)
  K = config.get("domain_group_allocation_num",int( (1-config.get("dropout_rate",0.5)) * config.get("num_domain_unit_group",32)))
  print("kl_term_coeff",kl_term_coeff)
  print("topK: ", K)

  my_matrix = np.zeros((model.num_domain_unit_group),)

  def _accumulate_gradients(source, target):
    domain = source["domain"][0]
    kl_loss_per_layer = []
    soft_mask_total_per_layer = []
    soft_mask_logits_per_layer = []
    delta_sigmoid_per_layer = []
    residue_per_layer = []
    latent_group_allocation_logit = []
    #####
    source_length = model.features_inputter.get_length(source)
    source_inputs = model.features_inputter(source, training=True)

    meta_encoding, sequence_length, _ = model.meta_encoder(
        source_inputs, sequence_length=source_length, training=True)
    padding_mask = model.meta_encoder.build_mask(source_inputs, sequence_length=sequence_length)
    padding_mask = tf.expand_dims(padding_mask, 2)
    meta_encoding = meta_encoding * padding_mask
    batch_size = tf.shape(padding_mask)[0]
    with tf.GradientTape() as tape:
      for i in range(model.encoder.num_layers + model.decoder.num_layers):
        gumbel_sample = gumbel_dist.sample([batch_size,model.num_domain_unit_group])
        latent_group_allocation_logit_ = model.mask_generators[i](tf.reduce_sum(meta_encoding)/tf.expand_dims(sequence_length,1))
        latent_group_allocation_logit.append(latent_group_allocation_logit_)
        domain_allocation_probs = tf.math.softmax(latent_group_allocation_logit_,-1)
        kl_loss_per_layer.append(- tf.reduce_mean(tf.reduce_sum(tf.math.log(domain_allocation_probs)*domain_allocation_probs,-1)))
        #f = lambda x: tf.reduce_sum(tf.math.sigmoid((gumbel_sample+latent_group_allocation_logit_+x)/temperature),-1) - K
        f = lambda x: tf.expand_dims(tf.reduce_sum(tf.math.sigmoid((gumbel_sample+latent_group_allocation_logit_+x)/temperature),-1) - K,1)
        temp_x = tfp.math.find_root_chandrupatla(f, low=-100, high=100, position_tolerance=1e-08,value_tolerance=0.0, max_iterations=100, stopping_policy_fn=tf.reduce_all,validate_args=False, name='find_root_chandrupatla').estimated_root
        residue_per_layer.append(tf.reduce_sum(tf.reduce_sum(tf.math.sigmoid((gumbel_sample+latent_group_allocation_logit_+temp_x)/temperature),-1) - K))
        soft_mask_logits = (gumbel_sample+latent_group_allocation_logit_+temp_x)/temperature
        soft_mask_logits_per_layer.append(soft_mask_logits)
        #tf.print("soft_mask_logits",soft_mask_logits,summarize=-1)
        soft_mask = tf.math.sigmoid(soft_mask_logits)
        soft_mask_total_per_layer.append(tf.cast(tf.repeat(tf.reduce_sum(tf.one_hot(tf.math.top_k(latent_group_allocation_logit_,k=K).indices, depth=model.num_domain_unit_group),1),model.unit_group_size,axis=-1),tf.float32))
        tf.print("soft_mask", soft_mask, "domain_allocation_probs",domain_allocation_probs,summarize=-1)
        #soft_mask_total_per_layer.append(tf.concat([tf.ones(model.num_shared_units),tf.cast(tf.repeat(soft_mask, model.unit_group_size),tf.float32)],-1))
        delta_sigmoid_per_layer.append(tf.math.square(tf.math.sigmoid((gumbel_sample+latent_group_allocation_logit_+temp_x)/temperature))/tf.math.exp((gumbel_sample+latent_group_allocation_logit_+temp_x)/temperature))
      
    # for i, mask_per_layer in enumerate(soft_mask_total_per_layer):
    #   tf.print(mask_per_layer, "domain: ", domain, "layer: ", i, summarize=-1)

    outputs, _ = model(
        source,
        domain_dropout_mask=soft_mask_total_per_layer,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
    else:
      training_loss, reported_loss = loss, loss

    if config.get("multi_domain", True):
      _domain = source["domain"][0]
    else:
      _domain = 0

    variables = model.trainable_variables
    model_variables = []
    for v in variables:
      if not "mask_generator" in v.name:
        model_variables.append(v)
      else:
        continue
    print("var numb: ", len(variables))
    
    gradients = optimizer.get_gradients(training_loss, model_variables)
    #deltaL_deltaM = optimizer.get_gradients(training_loss, soft_mask_logits_per_layer)
    deltaL_deltaM = optimizer.get_gradients(training_loss, soft_mask_total_per_layer)
    
    #optimizer.get_gradients(training_loss,soft_mask_logits_per_layer)
    group_allocation_gradient_per_layer = []
    for i in range(model.encoder.num_layers + model.decoder.num_layers):
      delta_sigmoid = delta_sigmoid_per_layer[i]
      deltaresidue_deltalogit1 = delta_sigmoid
      M1 = tf.linalg.diag(delta_sigmoid)
      deltaresidue_deltatempx1 = tf.reduce_sum(delta_sigmoid,-1)
      #tf.print("deltaresidue_deltatempx1",deltaresidue_deltatempx1)      
      deltaTempx_deltaLogit = - tf.tile(tf.expand_dims(deltaresidue_deltalogit1 / deltaresidue_deltatempx1,1),[1,model.num_domain_unit_group,1])
      #tf.print("deltaresidue_deltalogit", deltaresidue_deltalogit1, "deltaresidue_deltatempx", deltaresidue_deltatempx1, "deltaTempx_deltaLogit", deltaTempx_deltaLogit, summarize=-1)
      deltaM_deltaLogit = tf.eye(model.num_domain_unit_group,0) + deltaTempx_deltaLogit
      deltaL_deltaLogit = tf.linalg.matmul(tf.expand_dims(deltaL_deltaM[i],0),tf.repeat(deltaM_deltaLogit,repeats=model.unit_group_size,axis=0))
      deltaKL_deltaLogit = optimizer.get_gradients(kl_loss_per_layer[i] * kl_term_coeff, latent_group_allocation_logit[i])
      deltaLogit = deltaKL_deltaLogit + deltaL_deltaLogit
      print("deltaLogit: ",deltaLogit)
      jacobians = tape.jacobian(latent_group_allocation_logit[i],model.mask_generators[i].trainable_variables)
      for jacobian in jacobians:
        if jacobian.shape.rank==4:
          group_allocation_gradient_per_layer.append(tf.clip_by_norm(tf.einsum('bj,bjkh->kh',deltaLogit,jacobian),1.0))
        elif jacobian.shape.rank==3:
          group_allocation_gradient_per_layer.append(tf.clip_by_norm(tf.einsum('bj,bjk->k',deltaLogit,jacobian),1.0))
      

    gradient_accumulator(gradients)
    gradient_group_allocation_accumulator(group_allocation_gradient_per_layer)
    num_examples = tf.reduce_sum(target["length"])
       
    return reported_loss, tf.math.add_n(kl_loss_per_layer)/len(kl_loss_per_layer), num_examples, _domain, tf.math.add_n(residue_per_layer)/len(residue_per_layer)
     
  def _apply_gradients():
    variables = model.trainable_variables
    model_variables = []
    for v in variables:
      if not "latent_group_allocation_logit_per_layer" in v.name:
        model_variables.append(v)
      else:
        continue

    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, model_variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)

    grads_and_vars = []
    mask_generator_vars = []
    for i in range(model.encoder.num_layers + model.decoder.num_layers):
      for var in model.mask_generators[i]:
        mask_generator_vars.append(var)
      
    for gradient, variable in zip(gradient_group_allocation_accumulator.gradients, mask_generator_vars):
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    latent_logit_optimizer.apply_gradients(grads_and_vars)

    gradient_accumulator.reset()
    gradient_group_allocation_accumulator.reset()

  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_kl_loss, per_replica_num_examples, per_replica_domain, per_replica_residue = strategy.run(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
      kl_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_kl_loss, None)
      _domain = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_domain, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
      residue = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_residue, None)

    return loss, kl_loss, _domain, num_examples, residue
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.run(_apply_gradients)

  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  _, _, _, _, _ = next(train_data_flow)

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _kl_loss = []
  _number_examples = []
  _residue = []
  step = optimizer.iterations.numpy()
  if config.get("reset_step",None):
    print("start from %d-th step"%config.get("reset_step",150000))
    optimizer.iterations.assign(config.get("reset_step",150000))
  
  if step <= 1:
    initializer = config.get("initializer","default")
    if initializer == "default":
      print("Initializing variables by tensorflow default")      
    elif initializer == "variance_scaling":
      print("Initializing variables by tf.variance_scaling")
      initial_value = []
      for v in model.trainable_variables:
        shape = tf.shape(v).numpy()
        initial_value.append(variance_scaling_initialier(shape, scale=1.0, mode="fan_avg", distribution="uniform"))
      weight_reset(initial_value)       
  
  score_type = config.get("score_type","MultiBLEU")
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  ref_eval_concat = file_concatenate(config["eval_ref"],"ref_eval_concat",dir_name=os.path.join(config["model_dir"],"eval"))
  gumbel_temperature_decay = config.get("gumbel_temperature_decay",1000)
  r = config.get("r_coeff",1e-4)
  min_temperature = config.get("min_temperature",0.5)
  start_temperature = config.get("start_temperature",0.5)
  print("dropout_rate",config.get("dropout_rate"))
  print("min_temperature",min_temperature)
  print("gumbel_temperature_decay",gumbel_temperature_decay)
  print("r_coeff",r)
  step = optimizer.iterations.numpy()
  temperature.assign(tf.cast(tf.math.maximum(min_temperature, start_temperature * tf.math.exp(-r*step)),tf.float32))
  print("temperature: ",temperature)
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        loss, kl_loss, _domain, num_examples, residue = next(train_data_flow)    
        _loss.append(loss.numpy())
        _kl_loss.append(kl_loss.numpy())
        _number_examples.append(num_examples.numpy())
        _residue.append(residue.numpy())
      _step()  
      step = optimizer.iterations.numpy()
      
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; KL_loss = %f, temperature = %f, number_examples = %d, residue = %f, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_kl_loss), temperature, np.sum(_number_examples), np.mean(_residue), elapsed)
        tf.summary.experimental.set_step(step)
        tf.summary.scalar("CE_loss", np.mean(_loss), description="training loss")
        tf.summary.flush()
        _loss = []
        _kl_loss = []
        _number_examples = []
        _residue = []
        start = time.time()
      if step % gumbel_temperature_decay==0:
        temperature.assign(tf.cast(tf.math.maximum(min_temperature, start_temperature * tf.math.exp(-r*step)),tf.float32))
        #print("gumbel_temperature: ",gumbel_temperature)
      if step % save_every == 0 and step > 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % config.get("latent_logit_print_every",2000)==0:
        tf.print("latent_group_allocation_logit",model.latent_group_allocation_logit_per_layer[-1],summarize=-1)
      if step % eval_every == 0 and step > 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        output_files = []
        new_bleu = 0.0
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
            output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
            score = translate_topK_sparse_layer_multi_layer(src, ref, model, checkpoint_manager, checkpoint, i, output_file, topK=K, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
            tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
            output_files.append(output_file)
        ##### BLEU on concat dev set.
        output_file_concat = file_concatenate(output_files,"output_file_concat.%s"%os.path.basename(checkpoint_path))
        score = scorer(ref_eval_concat, output_file_concat)
        print("score of model %s on concat dev set: "%checkpoint_manager.latest_checkpoint, score)
        new_bleu = score
        tf.summary.scalar("concat_eval_score", score, description="BLEU on concat dev set")
        #############################
        if new_bleu >= current_max_eval_bleu:
          current_max_eval_bleu = new_bleu
          descending_streak = 0
        else:
          descending_streak += 1
      if descending_streak >= 5:
        break
      tf.summary.flush()
      if step > train_steps:
        break

def translate_Instance_Aware_topK_sparse_layer_multi_layer(source_file,
              reference,
              model,
              checkpoint_manager,
              checkpoint,              
              domain,
              output_file,
              length_penalty,
              topK=1,
              is_noisy=1,
              gumbel_temperature = 0.2,
              checkpoint_path=None,
              probs_file=None,
              experiment="ldr",
              score_type="MultiBLEU",
              batch_size=5,
              beam_size=5):
  
  tfa.options.disable_custom_kernel()
  # Create the inference dataset.
  if checkpoint_path == None:
    checkpoint_path = checkpoint_manager.latest_checkpoint
  tf.get_logger().info("Evaluating model %s", checkpoint_path)
  print("In domain %d"%domain)
  checkpoint.restore(checkpoint_path)
  dataset = model.examples_inputter.make_inference_dataset(source_file, batch_size, domain, is_noisy=is_noisy)
  iterator = iter(dataset)

  # Create the mapping for target ids to tokens.
  ids_to_tokens = model.labels_inputter.ids_to_tokens

  @tf.function
  def predict_next():    
    source = next(iterator)
    source_length = source["length"]
    batch_size = tf.shape(source_length)[0]
    source_inputs = model.features_inputter(source)
    
    meta_encoding, sequence_length, _ = model.meta_encoder(
        source_inputs, sequence_length=source_length, training=True)
    padding_mask = model.meta_encoder.build_mask(source_inputs, sequence_length=sequence_length)
    padding_mask = tf.expand_dims(padding_mask, 2)
    meta_encoding = meta_encoding * padding_mask
    batch_size = tf.shape(padding_mask)[0]
    dropout_mask = []
    for i in range(model.encoder.num_layers + model.decoder.num_layers):
      latent_group_allocation_logit_ = model.mask_generators[i](tf.reduce_sum(meta_encoding)/tf.expand_dims(sequence_length,1))
      soft_mask = tf.math.sigmoid(soft_mask_logits)
      dropout_mask.append(tf.cast(tf.repeat(tf.reduce_sum(tf.one_hot(tf.math.top_k(latent_group_allocation_logit_,k=K).indices, depth=model.num_domain_unit_group),1),model.unit_group_size,axis=-1),tf.float32))

    encoder_outputs, _, _ = model.encoder([source_inputs, source["domain"], dropout_mask[:model.encoder.num_layers]], source_length, training=False, internal_node_printing=True)

    # Prepare the decoding strategy.
    if beam_size > 1:
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      source_length = tfa.seq2seq.tile_batch(source_length, beam_size)
      decoding_strategy = onmt.utils.BeamSearch(beam_size, length_penalty=length_penalty)
    else:
      decoding_strategy = onmt.utils.GreedySearch()

    # Run dynamic decoding.
    decoder_state = model.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=source_length)
    map_input_fn = lambda ids: [model.labels_inputter({"ids": ids}, training=False), tf.dtypes.cast(tf.fill(tf.expand_dims(tf.shape(ids)[0],0), domain), tf.int64), dropout_mask[model.encoder.num_layers:]]
    
    decoded = model.decoder.dynamic_decode(
        map_input_fn,
        tf.fill([batch_size], START_OF_SENTENCE_ID),
        end_id=END_OF_SENTENCE_ID,
        initial_state=decoder_state,
        decoding_strategy=decoding_strategy,
        maximum_iterations=250)
    target_lengths = decoded.lengths
    target_tokens = ids_to_tokens.lookup(tf.cast(decoded.ids, tf.int64))
    return target_tokens, target_lengths

  # Iterates on the dataset.
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  print("output file: ", output_file)
  with open(output_file, "w") as output_:
    while True:    
      try:
        batch_tokens, batch_length = predict_next()
        for tokens, length in zip(batch_tokens.numpy(), batch_length.numpy()):
          sentence = b" ".join(tokens[0][:length[0]])
          print_bytes(sentence, output_)
          #print_bytes(sentence)
      except tf.errors.OutOfRangeError:
        break
  if reference!=None:
    print("score of model %s on test set %s: "%(checkpoint_manager.latest_checkpoint, source_file), scorer(reference, output_file))
    score = scorer(reference, output_file)
    if score is None:
      return 0.0
    else:
      return score

def train_elbo_multilingual_topK_sparse_layer_multi_layer(config,
          optimizer,          
          learning_rate,
          model,  
          strategy,  
          checkpoint_manager,
          checkpoint,
          adapter_optimizer=None,
          checkpoint_path=None,
          maximum_length=80,
          batch_size = 2048,
          batch_type = "tokens",
          experiment="residual",
          shuffle_buffer_size=-1,  # Uniform shuffle.
          train_steps=200000,
          save_every=5000,
          eval_every=15000,
          report_every=100): 

  import tensorflow_probability as tfp
  import scipy
  from scipy import optimize
  tfd = tfp.distributions
  assert config.get("multilingual",False)==True
  gumbel_dist = tfd.Gumbel(loc=0.,scale=1.)
  if config.get("train_steps",None)!=None:
    train_steps = config.get("train_steps")
  if config.get("batch_type",None)!=None:
    batch_type = config.get("batch_type")
  #####
  if checkpoint_manager.latest_checkpoint is not None:
    tf.get_logger().info("Restoring parameters from %s", checkpoint_manager.latest_checkpoint)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
  else:
    if checkpoint_path is not None:
      tf.get_logger().info("Restoring parameters from %s", checkpoint_path)
      checkpoint.restore(checkpoint_path)
  #####
  _summary_writer = tf.summary.create_file_writer(config["model_dir"])
  ###### early stopping criterion
  current_max_eval_bleu = 0.0
  descending_streak = 0
  ######
  batch_train_size = config["batch_train_size"]  
  batch_type = batch_type
  source_file = config["src"]
  target_file = config["tgt"]
  domain = config.get("domain",None)
  
  print("There are %d in-domain corpora"%len(source_file))
  
  train_dataset = create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, 
                                            maximum_length, length_bucket_width=config.get("length_bucket_width",1), 
                                            multi_domain=config.get("multi_domain", True), picking_prob=config.get("picking_prob",None), temperature=config.get("temperature",1.0))
  
  #####
  with strategy.scope():
    model.create_variables(optimizer=optimizer)
    gradient_accumulator = optimizer_util.GradientAccumulator() 
    gradient_group_allocation_accumulator = optimizer_util.GradientAccumulator()
    latent_logit_optimizer = tfa.optimizers.LazyAdam(config.get("latent_logit_lr",0.01))

  temperature = tf.Variable(0.2,trainable=False)
  
  kl_term_coeff = config.get("kl_coeff",1.0)
  K = config.get("domain_group_allocation_num",int( (1-config.get("dropout_rate",0.5)) * config.get("num_domain_unit_group",32)))
  print("kl_term_coeff",kl_term_coeff)
  print("topK: ", K)

  my_matrix = np.zeros((model.num_domain_unit_group),)
  num_languages = config.get("num_languages",None)
  assert num_languages>0, "This is multilingual training, please declare number of languages in config file"

  def _accumulate_gradients(source, target):
    domain = source["domain"][0]//(num_languages-1)
    kl_loss_per_layer = []
    soft_mask_total_per_layer = []
    soft_mask_logits_per_layer = []
    delta_sigmoid_per_layer = []
    residue_per_layer = []
    for i in range(model.encoder.num_layers + model.decoder.num_layers + 1):
      gumbel_sample = gumbel_dist.sample([model.num_domain_unit_group])
      latent_group_allocation_logit_ = tf.nn.embedding_lookup(model.latent_group_allocation_logit_per_layer[i],domain)
      domain_allocation_probs = tf.math.softmax(latent_group_allocation_logit_)
      kl_loss_per_layer.append(- tf.reduce_mean(tf.math.log(domain_allocation_probs)))
      f = lambda x: tf.reduce_sum(tf.math.sigmoid((gumbel_sample+latent_group_allocation_logit_+x)/temperature)) - K
      temp_x = tfp.math.find_root_chandrupatla(f, low=-100, high=100, position_tolerance=1e-08,value_tolerance=0.0, max_iterations=10, stopping_policy_fn=tf.reduce_all,validate_args=False, name='find_root_chandrupatla').estimated_root
      residue_per_layer.append(tf.reduce_sum(tf.math.sigmoid((gumbel_sample+latent_group_allocation_logit_+temp_x)/temperature)) - K)
      soft_mask_logits = (gumbel_sample+latent_group_allocation_logit_+temp_x)/temperature
      soft_mask_logits_per_layer.append(soft_mask_logits)
      #tf.print("soft_mask_logits",soft_mask_logits,summarize=-1)
      soft_mask = tf.math.sigmoid(soft_mask_logits)
      soft_mask_total_per_layer.append(tf.concat([tf.ones(model.num_shared_units),tf.cast(tf.repeat(tf.reduce_sum(tf.one_hot(tf.math.top_k(tf.nn.embedding_lookup(model.latent_group_allocation_logit_per_layer[i],domain),k=K).indices, depth=model.num_domain_unit_group),0),model.unit_group_size),tf.float32)],-1))

      #tf.print("soft_mask", soft_mask, "domain_allocation_probs",domain_allocation_probs,summarize=-1)
      #soft_mask_total_per_layer.append(tf.concat([tf.ones(model.num_shared_units),tf.cast(tf.repeat(soft_mask, model.unit_group_size),tf.float32)],-1))
      delta_sigmoid_per_layer.append(tf.math.square(tf.math.sigmoid((gumbel_sample+latent_group_allocation_logit_+temp_x)/temperature))/tf.math.exp((gumbel_sample+latent_group_allocation_logit_+temp_x)/temperature))
    
    # for i, mask_per_layer in enumerate(soft_mask_total_per_layer):
    #   tf.print(mask_per_layer, "domain: ", domain, "layer: ", i, summarize=-1)

    outputs, _ = model(
        source,
        domain_dropout_mask=soft_mask_total_per_layer,
        labels=target,
        training=True,
        step=optimizer.iterations)
    loss = model.compute_loss(outputs, target, training=True)

    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2]
      #tf.print("token num: ", loss[1], loss[2])
    else:
      training_loss, reported_loss = loss, loss

    if config.get("multi_domain", True):
      _domain = source["domain"][0]
    else:
      _domain = 0

    variables = model.trainable_variables
    model_variables = []
    for v in variables:
      if not "latent_group_allocation_logit_per_layer" in v.name:
        model_variables.append(v)
      else:
        continue
    print("var numb: ", len(variables))
    
    gradients = optimizer.get_gradients(training_loss, model_variables)
    #deltaL_deltaM = optimizer.get_gradients(training_loss, soft_mask_logits_per_layer)
    deltaL_deltaM = optimizer.get_gradients(training_loss, soft_mask_total_per_layer)
    
    #optimizer.get_gradients(training_loss,soft_mask_logits_per_layer)
    group_allocation_gradient_per_layer = []
    for i in range(model.encoder.num_layers + model.decoder.num_layers+1):
      delta_sigmoid = delta_sigmoid_per_layer[i]
      deltaresidue_deltalogit1 = delta_sigmoid
      M1 = tf.linalg.diag(delta_sigmoid)
      deltaresidue_deltatempx1 = tf.reduce_sum(delta_sigmoid)
      #tf.print("deltaresidue_deltatempx1",deltaresidue_deltatempx1)      
      deltaTempx_deltaLogit = - tf.tile(tf.expand_dims(deltaresidue_deltalogit1 / deltaresidue_deltatempx1,0),[model.num_domain_unit_group,1])
      #tf.print("deltaresidue_deltalogit", deltaresidue_deltalogit1, "deltaresidue_deltatempx", deltaresidue_deltatempx1, "deltaTempx_deltaLogit", deltaTempx_deltaLogit, summarize=-1)
      deltaM_deltaLogit = tf.eye(model.num_domain_unit_group) + deltaTempx_deltaLogit
      deltaL_deltaLogit = tf.linalg.matmul(tf.expand_dims(deltaL_deltaM[i],0),tf.repeat(deltaM_deltaLogit,repeats=model.unit_group_size,axis=0))
      group_allocation_gradient = optimizer.get_gradients(kl_loss_per_layer[i] * kl_term_coeff, model.latent_group_allocation_logit_per_layer[i])
      group_allocation_gradient[0] = tf.clip_by_norm(tf.tensor_scatter_nd_add(group_allocation_gradient[0],tf.expand_dims(group_allocation_gradient[0].indices,1),deltaL_deltaLogit),1.0)
      group_allocation_gradient_per_layer.append(group_allocation_gradient[0])
    gradient_accumulator(gradients)
    gradient_group_allocation_accumulator(group_allocation_gradient_per_layer)
    num_examples = tf.reduce_sum(target["length"])
    #tf.print("reported_loss: ",reported_loss)
    #tf.print("KL loss: ",tf.math.add_n(kl_loss_per_layer)/len(kl_loss_per_layer))
    return reported_loss, tf.math.add_n(kl_loss_per_layer)/len(kl_loss_per_layer), num_examples, _domain, tf.math.add_n(residue_per_layer)/len(residue_per_layer)
     
  def _apply_gradients():
    variables = model.trainable_variables
    model_variables = []
    for v in variables:
      if not "latent_group_allocation_logit_per_layer" in v.name:
        model_variables.append(v)
      else:
        continue
    # tf.print("gradient_accumulator.step: ",gradient_accumulator.step)
    # tf.print("strategy.num_replicas_in_sync",strategy.num_replicas_in_sync)
    grads_and_vars = []
    for gradient, variable in zip(gradient_accumulator.gradients, model_variables):
      # optimizer.apply_gradients will sum the gradients accross replicas.
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    optimizer.apply_gradients(grads_and_vars)

    grads_and_vars = []
    for gradient, variable in zip(gradient_group_allocation_accumulator.gradients, model.latent_group_allocation_logit_per_layer):
      scaled_gradient = gradient / (strategy.num_replicas_in_sync * tf.cast(gradient_accumulator.step, tf.float32))
      grads_and_vars.append((scaled_gradient, variable))
    latent_logit_optimizer.apply_gradients(grads_and_vars)

    gradient_accumulator.reset()
    gradient_group_allocation_accumulator.reset()

  @dataset_util.function_on_next(train_dataset)
  def _train_forward(next_fn):    
    with strategy.scope():
      per_replica_source, per_replica_target = next_fn()
      per_replica_loss, per_replica_kl_loss, per_replica_num_examples, per_replica_domain, per_replica_residue = strategy.run(
          _accumulate_gradients, args=(per_replica_source, per_replica_target))
      # TODO: these reductions could be delayed until _step is called.
      loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)
      kl_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_kl_loss, None)
      _domain = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_domain, None)      
      num_examples = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_num_examples, None)
      residue = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_residue, None)

    return loss, kl_loss, _domain, num_examples, residue
  
  @tf.function
  def _step():
    with strategy.scope():
      strategy.run(_apply_gradients)

  def _set_weight(v, w):
    v.assign(tf.cast(w,v.dtype))

  @tf.function
  def weight_reset(snapshots):
    with strategy.scope():
      for snap, var in zip(snapshots, model.trainable_variables):
        strategy.extended.update(var, _set_weight, args=(snap, ))

  # Runs the training loop.
  import time
  start = time.time()  
  train_data_flow = iter(_train_forward())
  _, _, _, _, _ = next(train_data_flow)

  print("number of replicas: %d"%strategy.num_replicas_in_sync)
  print("accumulation step", config.get("accumulation_step",1))
  _loss = []  
  _kl_loss = []
  _number_examples = []
  _residue = []
  step = optimizer.iterations.numpy()
  if config.get("reset_step",None):
    print("start from %d-th step"%config.get("reset_step",150000))
    optimizer.iterations.assign(config.get("reset_step",150000))
  
  if step <= 1:
    initializer = config.get("initializer","default")
    if initializer == "default":
      print("Initializing variables by tensorflow default")      
    elif initializer == "variance_scaling":
      print("Initializing variables by tf.variance_scaling")
      initial_value = []
      for v in model.trainable_variables:
        shape = tf.shape(v).numpy()
        initial_value.append(variance_scaling_initialier(shape, scale=1.0, mode="fan_avg", distribution="uniform"))
      weight_reset(initial_value)       
  
  score_type = config.get("score_type","MultiBLEU")
  if score_type == "sacreBLEU":
    print("using sacreBLEU")
    scorer = BLEUScorer()
  elif score_type == "MultiBLEU":
    print("using MultiBLEU")
    scorer = MultiBLEUScorer()
  ref_eval_concat = file_concatenate(config["eval_ref"],"ref_eval_concat",dir_name=os.path.join(config["model_dir"],"eval"))
  gumbel_temperature_decay = config.get("gumbel_temperature_decay",1000)
  r = config.get("r_coeff",1e-4)
  min_temperature = config.get("min_temperature",0.5)
  start_temperature = config.get("start_temperature",0.5)
  print("dropout_rate",config.get("dropout_rate"))
  print("min_temperature",min_temperature)
  print("gumbel_temperature_decay",gumbel_temperature_decay)
  print("r_coeff",r)
  step = optimizer.iterations.numpy()
  temperature.assign(tf.cast(tf.math.maximum(min_temperature, start_temperature * tf.math.exp(-r*step)),tf.float32))
  print("temperature: ",temperature)
  with _summary_writer.as_default():
    while True:
      #####Training batch
      for _ in range(int(config.get("accumulation_step",1))):
        loss, kl_loss, _domain, num_examples, residue = next(train_data_flow)    
        _loss.append(loss.numpy())
        _kl_loss.append(kl_loss.numpy())
        _number_examples.append(num_examples.numpy())
        _residue.append(residue.numpy())
      _step()  
      step = optimizer.iterations.numpy()
      
      if step % report_every == 0:
        elapsed = time.time() - start
        tf.get_logger().info(
            "Step = %d ; Learning rate = %f ; Loss = %f; KL_loss = %f, temperature = %f, number_examples = %d, residue = %f, after %f seconds",
            step, learning_rate(step), np.mean(_loss), np.mean(_kl_loss), temperature, np.sum(_number_examples), np.mean(_residue), elapsed)
        tf.summary.experimental.set_step(step)
        tf.summary.scalar("CE_loss", np.mean(_loss), description="training loss")
        tf.summary.flush()
        _loss = []
        _kl_loss = []
        _number_examples = []
        _residue = []
        start = time.time()
      if step % gumbel_temperature_decay==0:
        temperature.assign(tf.cast(tf.math.maximum(min_temperature, start_temperature * tf.math.exp(-r*step)),tf.float32))
        #print("gumbel_temperature: ",gumbel_temperature)
      if step % save_every == 0 and step > 0:
        tf.get_logger().info("Saving checkpoint for step %d", step)
        checkpoint_manager.save(checkpoint_number=step)
      if step % config.get("latent_logit_print_every",2000)==0:
        tf.print("latent_group_allocation_logit",model.latent_group_allocation_logit_per_layer[-1],summarize=-1)
      if step % eval_every == 0 and step > 0:
        checkpoint_path = checkpoint_manager.latest_checkpoint
        tf.summary.experimental.set_step(step)
        output_files = []
        new_bleu = 0.0
        for src,ref,i in zip(config["eval_src"],config["eval_ref"],config["eval_domain"]):
            output_file = os.path.join(config["model_dir"],"eval",os.path.basename(src) + ".trans." + os.path.basename(checkpoint_path))
            score = translate_topK_sparse_layer_multi_layer(src, ref, model, checkpoint_manager, checkpoint, i//num_languages, output_file, topK=K, length_penalty=config.get("length_penalty",0.6), experiment=experiment)
            tf.summary.scalar("eval_score_%d"%i, score, description="BLEU on test set %s"%src)
            output_files.append(output_file)
        ##### BLEU on concat dev set.
        output_file_concat = file_concatenate(output_files,"output_file_concat.%s"%os.path.basename(checkpoint_path))
        score = scorer(ref_eval_concat, output_file_concat)
        print("score of model %s on concat dev set: "%checkpoint_manager.latest_checkpoint, score)
        new_bleu = score
        tf.summary.scalar("concat_eval_score", score, description="BLEU on concat dev set")
        #############################
        if new_bleu >= current_max_eval_bleu:
          current_max_eval_bleu = new_bleu
          descending_streak = 0
        else:
          descending_streak += 1
      if descending_streak >= 5:
        break
      tf.summary.flush()
      if step > train_steps:
        break
      














































































  














