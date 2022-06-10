# -*- coding: utf-8 -*-

"""Standard sequence-to-sequence model."""
import sys
from numpy import source
sys.path.append("/gpfsdswork/projects/rech/sfz/utt84zy/anaconda3/envs/huggingface/lib/python3.7/site-packages")

import six

import tensorflow as tf
import tensorflow_addons as tfa

from opennmt import constants
from opennmt import inputters
from opennmt import layers
from opennmt.models.sequence_to_sequence import SequenceToSequence
from opennmt.data import noise
from opennmt.data import text
from opennmt.data import vocab
from opennmt.layers import reducer
from opennmt.models import model
from opennmt.utils import decoding
from opennmt.utils import losses
from opennmt.data import dataset as dataset_util
from opennmt.utils.misc import print_bytes, format_translation_output, merge_dict, shape_list
from opennmt.decoders import decoder as decoder_util
from opennmt.models.sequence_to_sequence import EmbeddingsSharingLevel, SequenceToSequence, SequenceToSequenceInputter, replace_unknown_target, _add_noise
from utils.my_inputter import My_inputter, Priming_SequenceToSequenceInputter, Multi_domain_SequenceToSequenceInputter, Multi_domain_SequenceToSequenceInputter_withprob, Multi_domain_SequenceToSequenceInputter_DRO
from utils.utils_ import make_domain_mask, masking
from opennmt.layers import common
from layers.layers import Classification_layer
from opennmt.utils.losses import _softmax_cross_entropy
from utils.my_inputter import My_inputter
from encoders.self_attention_encoder import Multi_domain_SelfAttentionEncoder_v1, Multi_domain_SelfAttentionEncoder_v16, Multi_domain_SelfAttentionEncoder_v2, Multi_domain_SelfAttentionEncoder_v12, Multi_domain_SelfAttentionEncoder_v15
from layers.common import Dense

class Multi_domain_SequenceToSequence(model.SequenceGenerator):

  """A sequence to sequence model."""

  def __init__(self,
               source_inputter,
               target_inputter,
               encoder,
               decoder,
               num_domains=6,
               num_units=512,
               share_embeddings=EmbeddingsSharingLevel.NONE):

    if not isinstance(target_inputter, inputters.WordEmbedder) and not isinstance(target_inputter, My_inputter):
      raise TypeError("Target inputter must be a WordEmbedder or My_inputter")
    if EmbeddingsSharingLevel.share_input_embeddings(share_embeddings):
      if isinstance(source_inputter, inputters.ParallelInputter):
        source_inputters = source_inputter.inputters
      else:
        source_inputters = [source_inputter]
      for inputter in source_inputters:
        if not isinstance(inputter, inputters.WordEmbedder):
          raise TypeError("Sharing embeddings requires all inputters to be a "
                          "WordEmbedder")

    examples_inputter = Multi_domain_SequenceToSequenceInputter(
        source_inputter,
        target_inputter,
        share_parameters=EmbeddingsSharingLevel.share_input_embeddings(share_embeddings))
    super(Multi_domain_SequenceToSequence, self).__init__(examples_inputter)
    self.encoder = encoder
    self.decoder = decoder
    self.share_embeddings = share_embeddings
    self.classification_layer = Classification_layer(num_units, domain_numb=num_domains, name="On_top_encoder_domain_classification")
 
  def auto_config(self, num_replicas=1):
    config = super(Multi_domain_SequenceToSequence, self).auto_config(num_replicas=num_replicas)
    return merge_dict(config, {
        "params": {
            "beam_width": 5
        },
        "train": {
            "sample_buffer_size": -1,
            "max_step": 200000
        },
        "infer": {
            "batch_size": 32,
            "length_bucket_width": 5
        }
    })

  def initialize(self, data_config, params=None):
    super(Multi_domain_SequenceToSequence, self).initialize(data_config, params=params)
    if self.params.get("contrastive_learning"):
      noiser = noise.WordNoiser(
          noises=[noise.WordOmission(1)],
          subword_token=self.params.get("decoding_subword_token", "￭"),
          is_spacer=self.params.get("decoding_subword_token_is_spacer"))
      self.labels_inputter.set_noise(noiser, in_place=False)

  def build(self, input_shape):
    super(Multi_domain_SequenceToSequence, self).build(input_shape)
    output_layer = None
    if EmbeddingsSharingLevel.share_target_embeddings(self.share_embeddings):
      output_layer = layers.Dense(
          self.labels_inputter.vocabulary_size,
          weight=self.labels_inputter.embedding,
          transpose=True)
    self.decoder.initialize(
        vocab_size=self.labels_inputter.vocabulary_size,
        output_layer=output_layer)

  def call(self, features, labels=None, training=None, step=None, internal_node_printing=False, return_domain_classification_logits=False, return_embedding=False, adapter_activate=True, inference=True):
    # Encode the source.
    assert isinstance(self.features_inputter, My_inputter)
    assert isinstance(self.labels_inputter, My_inputter)    
    source_length = self.features_inputter.get_length(features)
    source_inputs = self.features_inputter(features, training=training)
    if isinstance(self.encoder, Multi_domain_SelfAttentionEncoder_v1) or isinstance(self.encoder, Multi_domain_SelfAttentionEncoder_v2) or isinstance(self.encoder, Multi_domain_SelfAttentionEncoder_v12) or isinstance(self.encoder, Multi_domain_SelfAttentionEncoder_v15) or isinstance(self.encoder, Multi_domain_SelfAttentionEncoder_v16):
      encoder_outputs, encoder_state, encoder_sequence_length = self.encoder(
        [source_inputs, features["domain"], features["is_noisy"]], sequence_length=source_length, training=training, adapter_activate=adapter_activate, internal_node_printing=internal_node_printing)
    else:
      encoder_outputs, encoder_state, encoder_sequence_length = self.encoder(
        [source_inputs, features["domain"], features["is_noisy"]], sequence_length=source_length, training=training)

    if return_domain_classification_logits:
      _, domain_classification_logits = self.classification_layer(encoder_outputs, encoder_sequence_length, training=training)

    outputs = None
    predictions = None

    # When a target is provided, compute the decoder outputs for it.
    if adapter_activate:
      if labels is not None:
        outputs, target_inputs = self._decode_target(
            labels,
            encoder_outputs,
            encoder_state,
            encoder_sequence_length,
            step=step,
            training=training)

      # When not in training, also compute the model predictions.
      if not training and inference:
        predictions = self._dynamic_decode(
            features,
            encoder_outputs,
            encoder_state,
            encoder_sequence_length)
      if return_embedding:
        return outputs, predictions, source_inputs, target_inputs
      else:
        return outputs, predictions
    else:
      return domain_classification_logits

  def adv_call(self, features, labels=None, training=None, step=None):
    # Encode the source.
    assert isinstance(self.features_inputter, My_inputter)
    assert isinstance(self.labels_inputter, My_inputter)    
    source_length = self.features_inputter.get_length(features)
    source_inputs = self.features_inputter(features, training=training)
    encoder_outputs, encoder_state, encoder_sequence_length = self.encoder.adv_call(
        [source_inputs, features["domain"], features["is_noisy"]], sequence_length=source_length, training=training)
    
    outputs = None
    predictions = None

    # When a target is provided, compute the decoder outputs for it.
    if labels is not None:
      outputs = self._adv_decode_target(
          labels,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length,
          step=step,
          training=training)
    
    return outputs, predictions

  def forward_fn(self, features, args_dict, labels=None, training=None, step=None):
    # Encode the source.
    training=True
    assert labels!=None
    assert isinstance(self.features_inputter, My_inputter)
    assert isinstance(self.labels_inputter, My_inputter)
    source_length = self.features_inputter.get_length(features)
    source_inputs = self.features_inputter.forward_fn(features, args_dict, training=training)
    encoder_outputs, encoder_state, encoder_sequence_length = self.encoder.forward_fn(
        [source_inputs, features["domain"], features["is_noisy"]], args_dict, sequence_length=source_length, training=training)

    outputs = None
    predictions = None

    # When a target is provided, compute the decoder outputs for it.
    if labels is not None:
      outputs = self._decode_target_forward_fn(
          labels,
          args_dict,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length,
          step=step,
          training=training)

    return outputs, predictions

  def _adv_decode_target(self,
                     labels,
                     encoder_outputs,
                     encoder_state,
                     encoder_sequence_length,
                     step=None,
                     training=None):
    params = self.params
    target_inputs = self.labels_inputter(labels, training=training)
    input_fn = lambda ids: [self.labels_inputter({"ids": ids}, training=training), labels["domain"]]

    sampling_probability = None
    if training:
      sampling_probability = decoder_util.get_sampling_probability(
          step,
          read_probability=params.get("scheduled_sampling_read_probability"),
          schedule_type=params.get("scheduled_sampling_type"),
          k=params.get("scheduled_sampling_k"))

    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    logits, _, attention = self.decoder.adv_forward(
        [target_inputs, labels["domain"]],
        sequence_length=self.labels_inputter.get_length(labels),
              initial_state=initial_state,
              memory=encoder_outputs,
              memory_sequence_length=encoder_sequence_length,
              input_fn=input_fn,
              sampling_probability=sampling_probability,
              training=training)
    outputs = dict(logits=logits, attention=attention)

    return outputs

  def _decode_target(self,
                     labels,
                     encoder_outputs,
                     encoder_state,
                     encoder_sequence_length,
                     step=None,
                     training=None,
                     internal_node_printing=False):
    params = self.params
    target_inputs = self.labels_inputter(labels, training=training)
    input_fn = lambda ids: [self.labels_inputter({"ids": ids}, training=training), labels["domain"]]

    sampling_probability = None
    if training:
      sampling_probability = decoder_util.get_sampling_probability(
          step,
          read_probability=params.get("scheduled_sampling_read_probability"),
          schedule_type=params.get("scheduled_sampling_type"),
          k=params.get("scheduled_sampling_k"))

    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    logits, _, attention = self.decoder(
        [target_inputs, labels["domain"]],
        self.labels_inputter.get_length(labels),
        state=initial_state,
        input_fn=input_fn,
        sampling_probability=sampling_probability,
        training=training)
    outputs = dict(logits=logits, attention=attention)

    noisy_ids = labels.get("noisy_ids")
    if noisy_ids is not None and params.get("contrastive_learning"):
      # In case of contrastive learning, also forward the erroneous
      # translation to compute its log likelihood later.
      noisy_inputs = self.labels_inputter({"ids": noisy_ids}, training=training)
      noisy_logits, _, _ = self.decoder(
          noisy_inputs,
          labels["noisy_length"],
          state=initial_state,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
      outputs["noisy_logits"] = noisy_logits
    return outputs, target_inputs

  def _decode_target_forward_fn(self,
                     labels,
                     args_dict,
                     encoder_outputs,
                     encoder_state,
                     encoder_sequence_length,
                     step=None,
                     training=None):
    params = self.params
    target_inputs = self.labels_inputter.forward_fn(labels, args_dict, training=training)
    input_fn = lambda ids: [self.labels_inputter.forward_fn({"ids": ids}, args_dict, training=training), labels["domain"]]

    sampling_probability = None
    if training:
      sampling_probability = decoder_util.get_sampling_probability(
          step,
          read_probability=params.get("scheduled_sampling_read_probability"),
          schedule_type=params.get("scheduled_sampling_type"),
          k=params.get("scheduled_sampling_k"))

    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    
    logits, _, attention = self.decoder.forward_fn(
        [target_inputs, labels["domain"]],
        args_dict,
        self.labels_inputter.get_length(labels),
        initial_state=initial_state,
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        input_fn=input_fn,
        sampling_probability=sampling_probability,
        training=training)
        
    outputs = dict(logits=logits, attention=attention)
    
    return outputs
 
  def _dynamic_decode(self, features, encoder_outputs, encoder_state, encoder_sequence_length):
    params = self.params
    batch_size = tf.shape(tf.nest.flatten(encoder_outputs)[0])[0]
    start_ids = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
    beam_size = params.get("beam_width", 1)

    if beam_size > 1:
      # Tile encoder outputs to prepare for beam search.
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      encoder_sequence_length = tfa.seq2seq.tile_batch(encoder_sequence_length, beam_size)
      if encoder_state is not None:
        encoder_state = tfa.seq2seq.tile_batch(encoder_state, beam_size)

    # Dynamically decodes from the encoder outputs.
    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    sampled_ids, sampled_length, log_probs, alignment, _ = self.decoder.dynamic_decode(
        lambda ids: [self.labels_inputter({"ids": ids}), features["domain"]],
        start_ids,
        initial_state=initial_state,
        decoding_strategy=decoding.DecodingStrategy.from_params(params),
        sampler=decoding.Sampler.from_params(params),
        maximum_iterations=params.get("maximum_decoding_length", 250),
        minimum_iterations=params.get("minimum_decoding_length", 0))
    target_tokens = self.labels_inputter.ids_to_tokens.lookup(tf.cast(sampled_ids, tf.int64))

    # Maybe replace unknown targets by the source tokens with the highest attention weight.
    if params.get("replace_unknown_target", False):
      if alignment is None:
        raise TypeError("replace_unknown_target is not compatible with decoders "
                        "that don't return alignment history")
      if not isinstance(self.features_inputter, inputters.WordEmbedder):
        raise TypeError("replace_unknown_target is only defined when the source "
                        "inputter is a WordEmbedder")
      source_tokens = features["tokens"]
      if beam_size > 1:
        source_tokens = tfa.seq2seq.tile_batch(source_tokens, beam_size)
      # Merge batch and beam dimensions.
      original_shape = tf.shape(target_tokens)
      target_tokens = tf.reshape(target_tokens, [-1, original_shape[-1]])
      align_shape = shape_list(alignment)
      attention = tf.reshape(
          alignment, [align_shape[0] * align_shape[1], align_shape[2], align_shape[3]])
      # We don't have attention for </s> but ensure that the attention time dimension matches
      # the tokens time dimension.
      attention = reducer.align_in_time(attention, tf.shape(target_tokens)[1])
      replaced_target_tokens = replace_unknown_target(target_tokens, source_tokens, attention)
      target_tokens = tf.reshape(replaced_target_tokens, original_shape)

    # Maybe add noise to the predictions.
    decoding_noise = params.get("decoding_noise")
    if decoding_noise:
      target_tokens, sampled_length = _add_noise(
          target_tokens,
          sampled_length,
          decoding_noise,
          params.get("decoding_subword_token", "￭"),
          params.get("decoding_subword_token_is_spacer"))
      alignment = None  # Invalidate alignments.

    predictions = {
        "tokens": target_tokens,
        "length": sampled_length,
        "log_probs": log_probs
    }
    if alignment is not None:
      predictions["alignment"] = alignment

    # Maybe restrict the number of returned hypotheses based on the user parameter.
    num_hypotheses = params.get("num_hypotheses", 1)
    if num_hypotheses > 0:
      if num_hypotheses > beam_size:
        raise ValueError("n_best cannot be greater than beam_width")
      for key, value in six.iteritems(predictions):
        predictions[key] = value[:, :num_hypotheses]
    return predictions

  def compute_loss(self, outputs, labels, training=True):
    params = self.params
    if not isinstance(outputs, dict):
      outputs = dict(logits=outputs)
    logits = outputs["logits"]
    noisy_logits = outputs.get("noisy_logits")
    attention = outputs.get("attention")
    if noisy_logits is not None and params.get("contrastive_learning"):
      return losses.max_margin_loss(
          logits,
          labels["ids_out"],
          labels["length"],
          noisy_logits,
          labels["noisy_ids_out"],
          labels["noisy_length"],
          eta=params.get("max_margin_eta", 0.1))
    labels_lengths = self.labels_inputter.get_length(labels)
    #print("average_loss_in_time", params.get("average_loss_in_time", False))
    loss, loss_normalizer, loss_token_normalizer = losses.cross_entropy_sequence_loss(
        logits,
        labels["ids_out"],
        labels_lengths,
        label_smoothing=params.get("label_smoothing", 0.0),
        average_in_time=params.get("average_loss_in_time", False),
        training=training)
    if training:
      gold_alignments = labels.get("alignment")
      guided_alignment_type = params.get("guided_alignment_type")
      if gold_alignments is not None and guided_alignment_type is not None:
        if attention is None:
          tf.get_logger().warning("This model did not return attention vectors; "
                                  "guided alignment will not be applied")
        else:
          loss += losses.guided_alignment_cost(
              attention[:, :-1],  # Do not constrain last timestep.
              gold_alignments,
              sequence_length=labels_lengths - 1,
              cost_type=guided_alignment_type,
              weight=params.get("guided_alignment_weight", 1))
    return loss, loss_normalizer, loss_token_normalizer

  def compute_individual_loss(self, outputs, labels, training=True):
    if not isinstance(outputs, dict):
      outputs = dict(logits=outputs)
    logits = outputs["logits"]
    labels_lengths = self.labels_inputter.get_length(labels)
  
    max_time = tf.shape(logits)[1]

    cross_entropy = _softmax_cross_entropy(logits, labels["ids_out"], 0.1, training)
    weights = tf.sequence_mask(
        labels_lengths, maxlen=max_time, dtype=cross_entropy.dtype)
    loss = tf.reduce_sum(cross_entropy * weights,1)
    loss_token_normalizer = tf.reduce_sum(weights,1)
    
    return loss/loss_token_normalizer
  
  def print_prediction(self, prediction, params=None, stream=None):
    if params is None:
      params = {}
    num_hypotheses = len(prediction["tokens"])
    for i in range(num_hypotheses):
      target_length = prediction["length"][i]
      tokens = prediction["tokens"][i][:target_length]
      sentence = self.labels_inputter.tokenizer.detokenize(tokens)
      score = None
      attention = None
      alignment_type = None
      if params.get("with_scores"):
        score = prediction["log_probs"][i]
      if params.get("with_alignments"):
        attention = prediction["alignment"][i][:target_length]
        alignment_type = params["with_alignments"]
      sentence = format_translation_output(
          sentence,
          score=score,
          attention=attention,
          alignment_type=alignment_type)
      print_bytes(tf.compat.as_bytes(sentence), stream=stream)

  def classification_on_top_encoder(self, features, training=None):

    source_length = self.features_inputter.get_length(features)
    source_inputs = self.features_inputter(features, training=training)
    encoder_outputs, _, encoder_sequence_length = self.encoder(
        [source_inputs, features["domain"]], sequence_length=source_length, training=training)
    e, logits = self.classification_layer(encoder_outputs, encoder_sequence_length, training=training)
    return e, logits

  def sentence_encode(self, features, training=False):
    source_length = self.features_inputter.get_length(features)
    source_inputs = self.features_inputter(features, training=training)
    mask = self.encoder.build_mask(source_inputs, source_length, dtype=tf.float32)
    mask = tf.expand_dims(mask,2)
    return tf.reduce_mean(source_inputs * tf.broadcast_to(mask, tf.shape(source_inputs)),1)

  def transfer_weights(self, new_model, new_optimizer=None, optimizer=None, ignore_weights=None):
    updated_variables = []

    def _map_variables(inputter_fn, vars_fn):
      mapping, _ = vocab.get_mapping(
          inputter_fn(self).vocabulary_file,
          inputter_fn(new_model).vocabulary_file)
      vars_a, vocab_axes = vars_fn(self)
      vars_b, _ = vars_fn(new_model)
      for var_a, var_b, vocab_axis in zip(vars_a, vars_b, vocab_axes):
        if new_optimizer is not None and optimizer is not None:
          variables = vocab.update_variable_and_slots(
              var_a,
              var_b,
              optimizer,
              new_optimizer,
              mapping,
              vocab_axis=vocab_axis)
        else:
          variables = [vocab.update_variable(var_a, var_b, mapping, vocab_axis=vocab_axis)]
        updated_variables.extend(variables)
      return vars_b

class LDR_SequenceToSequence_v1(model.SequenceGenerator):
  """A sequence to sequence model."""

  def __init__(self,
               source_inputter,
               target_inputter,
               encoder,
               decoder,
               num_units=512,
               num_domains=6,
               share_embeddings=EmbeddingsSharingLevel.NONE):
    """
    if not isinstance(target_inputter, inputters.WordEmbedder):
      raise TypeError("Target inputter must be a WordEmbedder")
    """
    if EmbeddingsSharingLevel.share_input_embeddings(share_embeddings):
      if isinstance(source_inputter, inputters.ParallelInputter):
        source_inputters = source_inputter.inputters
      else:
        source_inputters = [source_inputter]
      for inputter in source_inputters:
        if not isinstance(inputter, inputters.WordEmbedder):
          raise TypeError("Sharing embeddings requires all inputters to be a "
                          "WordEmbedder")

    examples_inputter = Multi_domain_SequenceToSequenceInputter(
        source_inputter,
        target_inputter,
        share_parameters=EmbeddingsSharingLevel.share_input_embeddings(share_embeddings))
    super(LDR_SequenceToSequence_v1, self).__init__(examples_inputter)
    self.encoder = encoder
    self.decoder = decoder
    self.share_embeddings = share_embeddings
    self.classification_layer = Classification_layer(num_units, domain_numb=num_domains, name="On_top_encoder_domain_classification")
    
  def auto_config(self, num_replicas=1):
    config = super(LDR_SequenceToSequence_v1, self).auto_config(num_replicas=num_replicas)
    return merge_dict(config, {
        "params": {
            "beam_width": 5
        },
        "train": {
            "sample_buffer_size": -1,
            "max_step": 200000
        },
        "infer": {
            "batch_size": 32,
            "length_bucket_width": 5
        }
    })

  def initialize(self, data_config, params=None):
    super(LDR_SequenceToSequence_v1, self).initialize(data_config, params=params)
    if self.params.get("contrastive_learning"):
      # Use the simplest and most effective CL_one from the paper.
      # https://www.aclweb.org/anthology/P19-1623
      noiser = noise.WordNoiser(
          noises=[noise.WordOmission(1)],
          subword_token=self.params.get("decoding_subword_token", "￭"),
          is_spacer=self.params.get("decoding_subword_token_is_spacer"))
      self.labels_inputter.set_noise(noiser, in_place=False)

  def build(self, input_shape):
    super(LDR_SequenceToSequence_v1, self).build(input_shape)
    output_layer = None
    if EmbeddingsSharingLevel.share_target_embeddings(self.share_embeddings):
      output_layer = layers.Dense(
          self.labels_inputter.vocabulary_size,
          weight=self.labels_inputter.embedding,
          transpose=True)
    self.decoder.initialize(
        vocab_size=self.labels_inputter.vocabulary_size,
        output_layer=output_layer)

  def call(self, features, labels=None, training=None, step=None):
    # Encode the source.
    source_length = self.features_inputter.get_length(features)
    source_inputs = self.features_inputter(features, training=training)
    encoder_outputs, encoder_state, encoder_sequence_length = self.encoder(
        source_inputs, sequence_length=source_length, training=training)
    _, domain_classification_logits = self.classification_layer(encoder_outputs, encoder_sequence_length, training=training)

    outputs = None
    predictions = None

    # When a target is provided, compute the decoder outputs for it.
    if labels is not None:
      outputs = self._decode_target(
          labels,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length,
          step=step,
          training=training)
      outputs = dict(logits=outputs["logits"], attention=outputs["attention"], domain_classification_logits=domain_classification_logits)

    # When not in training, also compute the model predictions.
    if not training:
      predictions = self._dynamic_decode(
          features,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length)

    return outputs, predictions

  def _decode_target(self,
                     labels,
                     encoder_outputs,
                     encoder_state,
                     encoder_sequence_length,
                     step=None,
                     training=None):
    params = self.params
    target_inputs = self.labels_inputter(labels, training=training)
    input_fn = lambda ids: self.labels_inputter({"ids": ids}, training=training)

    sampling_probability = None
    if training:
      sampling_probability = decoder_util.get_sampling_probability(
          step,
          read_probability=params.get("scheduled_sampling_read_probability"),
          schedule_type=params.get("scheduled_sampling_type"),
          k=params.get("scheduled_sampling_k"))

    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    logits, _, attention = self.decoder(
        target_inputs,
        self.labels_inputter.get_length(labels),
        state=initial_state,
        input_fn=input_fn,
        sampling_probability=sampling_probability,
        training=training)
    outputs = dict(logits=logits, attention=attention)

    noisy_ids = labels.get("noisy_ids")
    if noisy_ids is not None and params.get("contrastive_learning"):
      # In case of contrastive learning, also forward the erroneous
      # translation to compute its log likelihood later.
      noisy_inputs = self.labels_inputter({"ids": noisy_ids}, training=training)
      noisy_logits, _, _ = self.decoder(
          noisy_inputs,
          labels["noisy_length"],
          state=initial_state,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
      outputs["noisy_logits"] = noisy_logits
    return outputs

  def _dynamic_decode(self, features, encoder_outputs, encoder_state, encoder_sequence_length):
    params = self.params
    batch_size = tf.shape(tf.nest.flatten(encoder_outputs)[0])[0]
    start_ids = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
    beam_size = params.get("beam_width", 1)

    if beam_size > 1:
      # Tile encoder outputs to prepare for beam search.
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      encoder_sequence_length = tfa.seq2seq.tile_batch(encoder_sequence_length, beam_size)
      if encoder_state is not None:
        encoder_state = tfa.seq2seq.tile_batch(encoder_state, beam_size)

    # Dynamically decodes from the encoder outputs.
    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    sampled_ids, sampled_length, log_probs, alignment, _ = self.decoder.dynamic_decode(
        self.labels_inputter,
        start_ids,
        initial_state=initial_state,
        decoding_strategy=decoding.DecodingStrategy.from_params(params),
        sampler=decoding.Sampler.from_params(params),
        maximum_iterations=params.get("maximum_decoding_length", 250),
        minimum_iterations=params.get("minimum_decoding_length", 0))
    target_tokens = self.labels_inputter.ids_to_tokens.lookup(tf.cast(sampled_ids, tf.int64))

    # Maybe replace unknown targets by the source tokens with the highest attention weight.
    if params.get("replace_unknown_target", False):
      if alignment is None:
        raise TypeError("replace_unknown_target is not compatible with decoders "
                        "that don't return alignment history")
      if not isinstance(self.features_inputter, inputters.WordEmbedder):
        raise TypeError("replace_unknown_target is only defined when the source "
                        "inputter is a WordEmbedder")
      source_tokens = features["tokens"]
      if beam_size > 1:
        source_tokens = tfa.seq2seq.tile_batch(source_tokens, beam_size)
      # Merge batch and beam dimensions.
      original_shape = tf.shape(target_tokens)
      target_tokens = tf.reshape(target_tokens, [-1, original_shape[-1]])
      align_shape = shape_list(alignment)
      attention = tf.reshape(
          alignment, [align_shape[0] * align_shape[1], align_shape[2], align_shape[3]])
      # We don't have attention for </s> but ensure that the attention time dimension matches
      # the tokens time dimension.
      attention = reducer.align_in_time(attention, tf.shape(target_tokens)[1])
      replaced_target_tokens = replace_unknown_target(target_tokens, source_tokens, attention)
      target_tokens = tf.reshape(replaced_target_tokens, original_shape)

    # Maybe add noise to the predictions.
    decoding_noise = params.get("decoding_noise")
    if decoding_noise:
      target_tokens, sampled_length = _add_noise(
          target_tokens,
          sampled_length,
          decoding_noise,
          params.get("decoding_subword_token", "￭"),
          params.get("decoding_subword_token_is_spacer"))
      alignment = None  # Invalidate alignments.

    predictions = {
        "tokens": target_tokens,
        "length": sampled_length,
        "log_probs": log_probs
    }
    if alignment is not None:
      predictions["alignment"] = alignment

    # Maybe restrict the number of returned hypotheses based on the user parameter.
    num_hypotheses = params.get("num_hypotheses", 1)
    if num_hypotheses > 0:
      if num_hypotheses > beam_size:
        raise ValueError("n_best cannot be greater than beam_width")
      for key, value in six.iteritems(predictions):
        predictions[key] = value[:, :num_hypotheses]
    return predictions

  def compute_loss(self, outputs, labels, training=True):
    params = self.params
    if not isinstance(outputs, dict):
      outputs = dict(logits=outputs)
    logits = outputs["logits"]
    noisy_logits = outputs.get("noisy_logits")
    attention = outputs.get("attention")
    if noisy_logits is not None and params.get("contrastive_learning"):
      return losses.max_margin_loss(
          logits,
          labels["ids_out"],
          labels["length"],
          noisy_logits,
          labels["noisy_ids_out"],
          labels["noisy_length"],
          eta=params.get("max_margin_eta", 0.1))
    labels_lengths = self.labels_inputter.get_length(labels)
    print("label_smoothing: ",params.get("label_smoothing", 0.0))
    print("average_loss_in_time: ", params.get("average_loss_in_time", False))
    loss, loss_normalizer, loss_token_normalizer = losses.cross_entropy_sequence_loss(
        logits,
        labels["ids_out"],
        labels_lengths,
        label_smoothing=params.get("label_smoothing", 0.0),
        average_in_time=params.get("average_loss_in_time", False),
        training=training)
    if training:
      gold_alignments = labels.get("alignment")
      guided_alignment_type = params.get("guided_alignment_type")
      if gold_alignments is not None and guided_alignment_type is not None:
        if attention is None:
          tf.get_logger().warning("This model did not return attention vectors; "
                                  "guided alignment will not be applied")
        else:
          loss += losses.guided_alignment_cost(
              attention[:, :-1],  # Do not constrain last timestep.
              gold_alignments,
              sequence_length=labels_lengths - 1,
              cost_type=guided_alignment_type,
              weight=params.get("guided_alignment_weight", 1))
    return loss, loss_normalizer, loss_token_normalizer

  def print_prediction(self, prediction, params=None, stream=None):
    if params is None:
      params = {}
    num_hypotheses = len(prediction["tokens"])
    for i in range(num_hypotheses):
      target_length = prediction["length"][i]
      tokens = prediction["tokens"][i][:target_length]
      sentence = self.labels_inputter.tokenizer.detokenize(tokens)
      score = None
      attention = None
      alignment_type = None
      if params.get("with_scores"):
        score = prediction["log_probs"][i]
      if params.get("with_alignments"):
        attention = prediction["alignment"][i][:target_length]
        alignment_type = params["with_alignments"]
      sentence = format_translation_output(
          sentence,
          score=score,
          attention=attention,
          alignment_type=alignment_type)
      print_bytes(tf.compat.as_bytes(sentence), stream=stream)

  def transfer_weights(self, new_model, new_optimizer=None, optimizer=None, ignore_weights=None):
    updated_variables = []

    def _map_variables(inputter_fn, vars_fn):
      mapping, _ = vocab.get_mapping(
          inputter_fn(self).vocabulary_file,
          inputter_fn(new_model).vocabulary_file)
      vars_a, vocab_axes = vars_fn(self)
      vars_b, _ = vars_fn(new_model)
      for var_a, var_b, vocab_axis in zip(vars_a, vars_b, vocab_axes):
        if new_optimizer is not None and optimizer is not None:
          variables = vocab.update_variable_and_slots(
              var_a,
              var_b,
              optimizer,
              new_optimizer,
              mapping,
              vocab_axis=vocab_axis)
        else:
          variables = [vocab.update_variable(var_a, var_b, mapping, vocab_axis=vocab_axis)]
        updated_variables.extend(variables)
      return vars_b

    _map_variables(
        lambda model: model.features_inputter,
        lambda model: ([model.features_inputter.embedding], [0]))
    _map_variables(
        lambda model: model.labels_inputter,
        lambda model: ([
            model.labels_inputter.embedding,
            model.decoder.output_layer.kernel,
            model.decoder.output_layer.bias], [0, 1, 0]))

    return super(LDR_SequenceToSequence_v1, self).transfer_weights(
        new_model,
        new_optimizer=new_optimizer,
        optimizer=optimizer,
        ignore_weights=updated_variables)

class LDR_SequenceToSequence(model.SequenceGenerator):
  """A sequence to sequence model."""

  def __init__(self,
               source_inputter,
               target_inputter,
               encoder,
               decoder,
               num_units=512,
               num_domains=6,
               share_embeddings=EmbeddingsSharingLevel.NONE):
    """
    if not isinstance(target_inputter, inputters.WordEmbedder):
      raise TypeError("Target inputter must be a WordEmbedder")
    """
    if EmbeddingsSharingLevel.share_input_embeddings(share_embeddings):
      if isinstance(source_inputter, inputters.ParallelInputter):
        source_inputters = source_inputter.inputters
      else:
        source_inputters = [source_inputter]
      for inputter in source_inputters:
        if not isinstance(inputter, inputters.WordEmbedder):
          raise TypeError("Sharing embeddings requires all inputters to be a "
                          "WordEmbedder")

    examples_inputter = Multi_domain_SequenceToSequenceInputter(
        source_inputter,
        target_inputter,
        share_parameters=EmbeddingsSharingLevel.share_input_embeddings(share_embeddings))
    super(LDR_SequenceToSequence, self).__init__(examples_inputter)
    self.encoder = encoder
    self.decoder = decoder
    self.share_embeddings = share_embeddings
    
  def auto_config(self, num_replicas=1):
    config = super(LDR_SequenceToSequence, self).auto_config(num_replicas=num_replicas)
    return merge_dict(config, {
        "params": {
            "beam_width": 5
        },
        "train": {
            "sample_buffer_size": -1,
            "max_step": 200000
        },
        "infer": {
            "batch_size": 32,
            "length_bucket_width": 5
        }
    })

  def initialize(self, data_config, params=None):
    super(LDR_SequenceToSequence, self).initialize(data_config, params=params)
    if self.params.get("contrastive_learning"):
      # Use the simplest and most effective CL_one from the paper.
      # https://www.aclweb.org/anthology/P19-1623
      noiser = noise.WordNoiser(
          noises=[noise.WordOmission(1)],
          subword_token=self.params.get("decoding_subword_token", "￭"),
          is_spacer=self.params.get("decoding_subword_token_is_spacer"))
      self.labels_inputter.set_noise(noiser, in_place=False)

  def build(self, input_shape):
    super(LDR_SequenceToSequence, self).build(input_shape)
    output_layer = None
    if EmbeddingsSharingLevel.share_target_embeddings(self.share_embeddings):
      output_layer = layers.Dense(
          self.labels_inputter.vocabulary_size,
          weight=self.labels_inputter.embedding,
          transpose=True)
    self.decoder.initialize(
        vocab_size=self.labels_inputter.vocabulary_size,
        output_layer=output_layer)

  def call(self, features, labels=None, training=None, step=None):
    # Encode the source.
    source_length = self.features_inputter.get_length(features)
    source_inputs = self.features_inputter(features, training=training)
    encoder_outputs, encoder_state, encoder_sequence_length = self.encoder(
        source_inputs, sequence_length=source_length, training=training)

    outputs = None
    predictions = None

    # When a target is provided, compute the decoder outputs for it.
    if labels is not None:
      outputs = self._decode_target(
          labels,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length,
          step=step,
          training=training)

    # When not in training, also compute the model predictions.
    if not training:
      predictions = self._dynamic_decode(
          features,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length)

    return outputs, predictions

  def _decode_target(self,
                     labels,
                     encoder_outputs,
                     encoder_state,
                     encoder_sequence_length,
                     step=None,
                     training=None):
    params = self.params
    target_inputs = self.labels_inputter(labels, training=training)
    input_fn = lambda ids: self.labels_inputter({"ids": ids}, training=training)

    sampling_probability = None
    if training:
      sampling_probability = decoder_util.get_sampling_probability(
          step,
          read_probability=params.get("scheduled_sampling_read_probability"),
          schedule_type=params.get("scheduled_sampling_type"),
          k=params.get("scheduled_sampling_k"))

    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    logits, _, attention = self.decoder(
        target_inputs,
        self.labels_inputter.get_length(labels),
        state=initial_state,
        input_fn=input_fn,
        sampling_probability=sampling_probability,
        training=training)
    outputs = dict(logits=logits, attention=attention)

    noisy_ids = labels.get("noisy_ids")
    if noisy_ids is not None and params.get("contrastive_learning"):
      # In case of contrastive learning, also forward the erroneous
      # translation to compute its log likelihood later.
      noisy_inputs = self.labels_inputter({"ids": noisy_ids}, training=training)
      noisy_logits, _, _ = self.decoder(
          noisy_inputs,
          labels["noisy_length"],
          state=initial_state,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
      outputs["noisy_logits"] = noisy_logits
    return outputs

  def _dynamic_decode(self, features, encoder_outputs, encoder_state, encoder_sequence_length):
    params = self.params
    batch_size = tf.shape(tf.nest.flatten(encoder_outputs)[0])[0]
    start_ids = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
    beam_size = params.get("beam_width", 1)

    if beam_size > 1:
      # Tile encoder outputs to prepare for beam search.
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      encoder_sequence_length = tfa.seq2seq.tile_batch(encoder_sequence_length, beam_size)
      if encoder_state is not None:
        encoder_state = tfa.seq2seq.tile_batch(encoder_state, beam_size)

    # Dynamically decodes from the encoder outputs.
    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    sampled_ids, sampled_length, log_probs, alignment, _ = self.decoder.dynamic_decode(
        self.labels_inputter,
        start_ids,
        initial_state=initial_state,
        decoding_strategy=decoding.DecodingStrategy.from_params(params),
        sampler=decoding.Sampler.from_params(params),
        maximum_iterations=params.get("maximum_decoding_length", 250),
        minimum_iterations=params.get("minimum_decoding_length", 0))
    target_tokens = self.labels_inputter.ids_to_tokens.lookup(tf.cast(sampled_ids, tf.int64))

    # Maybe replace unknown targets by the source tokens with the highest attention weight.
    if params.get("replace_unknown_target", False):
      if alignment is None:
        raise TypeError("replace_unknown_target is not compatible with decoders "
                        "that don't return alignment history")
      if not isinstance(self.features_inputter, inputters.WordEmbedder):
        raise TypeError("replace_unknown_target is only defined when the source "
                        "inputter is a WordEmbedder")
      source_tokens = features["tokens"]
      if beam_size > 1:
        source_tokens = tfa.seq2seq.tile_batch(source_tokens, beam_size)
      # Merge batch and beam dimensions.
      original_shape = tf.shape(target_tokens)
      target_tokens = tf.reshape(target_tokens, [-1, original_shape[-1]])
      align_shape = shape_list(alignment)
      attention = tf.reshape(
          alignment, [align_shape[0] * align_shape[1], align_shape[2], align_shape[3]])
      # We don't have attention for </s> but ensure that the attention time dimension matches
      # the tokens time dimension.
      attention = reducer.align_in_time(attention, tf.shape(target_tokens)[1])
      replaced_target_tokens = replace_unknown_target(target_tokens, source_tokens, attention)
      target_tokens = tf.reshape(replaced_target_tokens, original_shape)

    # Maybe add noise to the predictions.
    decoding_noise = params.get("decoding_noise")
    if decoding_noise:
      target_tokens, sampled_length = _add_noise(
          target_tokens,
          sampled_length,
          decoding_noise,
          params.get("decoding_subword_token", "￭"),
          params.get("decoding_subword_token_is_spacer"))
      alignment = None  # Invalidate alignments.

    predictions = {
        "tokens": target_tokens,
        "length": sampled_length,
        "log_probs": log_probs
    }
    if alignment is not None:
      predictions["alignment"] = alignment

    # Maybe restrict the number of returned hypotheses based on the user parameter.
    num_hypotheses = params.get("num_hypotheses", 1)
    if num_hypotheses > 0:
      if num_hypotheses > beam_size:
        raise ValueError("n_best cannot be greater than beam_width")
      for key, value in six.iteritems(predictions):
        predictions[key] = value[:, :num_hypotheses]
    return predictions

  def compute_loss(self, outputs, labels, training=True):
    params = self.params
    if not isinstance(outputs, dict):
      outputs = dict(logits=outputs)
    logits = outputs["logits"]
    noisy_logits = outputs.get("noisy_logits")
    attention = outputs.get("attention")
    if noisy_logits is not None and params.get("contrastive_learning"):
      return losses.max_margin_loss(
          logits,
          labels["ids_out"],
          labels["length"],
          noisy_logits,
          labels["noisy_ids_out"],
          labels["noisy_length"],
          eta=params.get("max_margin_eta", 0.1))
    labels_lengths = self.labels_inputter.get_length(labels)
    print("label_smoothing: ",params.get("label_smoothing", 0.0))
    print("average_loss_in_time: ", params.get("average_loss_in_time", False))
    loss, loss_normalizer, loss_token_normalizer = losses.cross_entropy_sequence_loss(
        logits,
        labels["ids_out"],
        labels_lengths,
        label_smoothing=params.get("label_smoothing", 0.0),
        average_in_time=params.get("average_loss_in_time", False),
        training=training)
    if training:
      gold_alignments = labels.get("alignment")
      guided_alignment_type = params.get("guided_alignment_type")
      if gold_alignments is not None and guided_alignment_type is not None:
        if attention is None:
          tf.get_logger().warning("This model did not return attention vectors; "
                                  "guided alignment will not be applied")
        else:
          loss += losses.guided_alignment_cost(
              attention[:, :-1],  # Do not constrain last timestep.
              gold_alignments,
              sequence_length=labels_lengths - 1,
              cost_type=guided_alignment_type,
              weight=params.get("guided_alignment_weight", 1))
    return loss, loss_normalizer, loss_token_normalizer

  def print_prediction(self, prediction, params=None, stream=None):
    if params is None:
      params = {}
    num_hypotheses = len(prediction["tokens"])
    for i in range(num_hypotheses):
      target_length = prediction["length"][i]
      tokens = prediction["tokens"][i][:target_length]
      sentence = self.labels_inputter.tokenizer.detokenize(tokens)
      score = None
      attention = None
      alignment_type = None
      if params.get("with_scores"):
        score = prediction["log_probs"][i]
      if params.get("with_alignments"):
        attention = prediction["alignment"][i][:target_length]
        alignment_type = params["with_alignments"]
      sentence = format_translation_output(
          sentence,
          score=score,
          attention=attention,
          alignment_type=alignment_type)
      print_bytes(tf.compat.as_bytes(sentence), stream=stream)

  def transfer_weights(self, new_model, new_optimizer=None, optimizer=None, ignore_weights=None):
    updated_variables = []

    def _map_variables(inputter_fn, vars_fn):
      mapping, _ = vocab.get_mapping(
          inputter_fn(self).vocabulary_file,
          inputter_fn(new_model).vocabulary_file)
      vars_a, vocab_axes = vars_fn(self)
      vars_b, _ = vars_fn(new_model)
      for var_a, var_b, vocab_axis in zip(vars_a, vars_b, vocab_axes):
        if new_optimizer is not None and optimizer is not None:
          variables = vocab.update_variable_and_slots(
              var_a,
              var_b,
              optimizer,
              new_optimizer,
              mapping,
              vocab_axis=vocab_axis)
        else:
          variables = [vocab.update_variable(var_a, var_b, mapping, vocab_axis=vocab_axis)]
        updated_variables.extend(variables)
      return vars_b

    _map_variables(
        lambda model: model.features_inputter,
        lambda model: ([model.features_inputter.embedding], [0]))
    _map_variables(
        lambda model: model.labels_inputter,
        lambda model: ([
            model.labels_inputter.embedding,
            model.decoder.output_layer.kernel,
            model.decoder.output_layer.bias], [0, 1, 0]))

    return super(LDR_SequenceToSequence, self).transfer_weights(
        new_model,
        new_optimizer=new_optimizer,
        optimizer=optimizer,
        ignore_weights=updated_variables)

class SequenceToSequence_WDC(model.SequenceGenerator):
  """A sequence to sequence model."""

  def __init__(self,
               source_inputter,
               target_inputter,
               encoder,
               decoder,
               num_domains=6,
               num_units=512,
               share_embeddings=EmbeddingsSharingLevel.NONE):
    
    if not isinstance(target_inputter, inputters.WordEmbedder) and not isinstance(target_inputter, My_inputter):
      raise TypeError("Target inputter must be a WordEmbedder or My_inputter")
    if EmbeddingsSharingLevel.share_input_embeddings(share_embeddings):
      if isinstance(source_inputter, inputters.ParallelInputter):
        source_inputters = source_inputter.inputters
      else:
        source_inputters = [source_inputter]
      for inputter in source_inputters:
        if not isinstance(inputter, inputters.WordEmbedder):
          raise TypeError("Sharing embeddings requires all inputters to be a "
                          "WordEmbedder")

    examples_inputter = Multi_domain_SequenceToSequenceInputter(
        source_inputter,
        target_inputter,
        share_parameters=EmbeddingsSharingLevel.share_input_embeddings(share_embeddings))
    super(SequenceToSequence_WDC, self).__init__(examples_inputter)

    self.encoder = encoder
    self.decoder = decoder
    self.share_embeddings = share_embeddings
    self.classification_layer = Classification_layer(num_units, domain_numb=num_domains, name="On_top_encoder_domain_classification")
    self.classification_decoder_layer = Classification_layer(num_units, domain_numb=num_domains, name="On_top_decoder_domain_classification")
    self.adv_classification_layer = Classification_layer(num_units, domain_numb=num_domains, name="ADV_on_top_encoder_domain_classification")
    self.share_gate = layers.Dense(num_units, use_bias=True, activation=tf.nn.sigmoid)
    self.specific_gate = layers.Dense(num_units, use_bias=True, activation=tf.nn.sigmoid)

  def auto_config(self, num_replicas=1):
    config = super(SequenceToSequence_WDC, self).auto_config(num_replicas=num_replicas)
    return merge_dict(config, {
        "params": {
            "beam_width": 4
        },
        "train": {
            "sample_buffer_size": -1,
            "max_step": 500000
        },
        "infer": {
            "batch_size": 32,
            "length_bucket_width": 5
        }
    })

  def initialize(self, data_config, params=None):
    super(SequenceToSequence_WDC, self).initialize(data_config, params=params)
    if self.params.get("contrastive_learning"):
      # Use the simplest and most effective CL_one from the paper.
      # https://www.aclweb.org/anthology/P19-1623
      noiser = noise.WordNoiser(
          noises=[noise.WordOmission(1)],
          subword_token=self.params.get("decoding_subword_token", "￭"),
          is_spacer=self.params.get("decoding_subword_token_is_spacer"))
      self.labels_inputter.set_noise(noiser, in_place=False)

  def build(self, input_shape):
    super(SequenceToSequence_WDC, self).build(input_shape)
    output_layer = None
    if EmbeddingsSharingLevel.share_target_embeddings(self.share_embeddings):
      output_layer = layers.Dense(
          self.labels_inputter.vocabulary_size,
          weight=self.labels_inputter.embedding,
          transpose=True)
    self.decoder.initialize(
        vocab_size=self.labels_inputter.vocabulary_size,
        output_layer=output_layer)

  def classification_on_top_encoder(self, features, training=None):

    source_length = self.features_inputter.get_length(features)
    source_inputs = self.features_inputter(features, training=training)
    encoder_outputs, _, encoder_sequence_length = self.encoder(
        source_inputs, sequence_length=source_length, training=training)
    _, outputs = self.classification_layer(encoder_outputs, encoder_sequence_length, training=training)
    _, adv_outputs = self.adv_classification_layer(encoder_outputs, encoder_sequence_length, training=training)
    return outputs, adv_outputs   

  def call(self, features, labels=None, training=None, step=None):
    # Encode the source.
    source_length = self.features_inputter.get_length(features)
    source_inputs = self.features_inputter(features, training=training)
    encoder_outputs, encoder_state, encoder_sequence_length = self.encoder(
        source_inputs, sequence_length=source_length, training=training)
    outputs = None
    predictions = None
    e_r, logits_r = self.classification_layer(encoder_outputs, encoder_sequence_length, training=training)
    e_s, logits_s = self.adv_classification_layer(encoder_outputs, encoder_sequence_length, training=training)
    g_s = self.share_gate(tf.concat([tf.tile(tf.expand_dims(e_s,1),[1,tf.shape(encoder_outputs)[1],1]),encoder_outputs],-1))
    g_r = self.specific_gate(tf.concat([tf.tile(tf.expand_dims(e_r,1),[1,tf.shape(encoder_outputs)[1],1]),encoder_outputs],-1))
    h_r = g_r * encoder_outputs
    h_s = g_s * encoder_outputs
    encoder_mask = self.encoder.build_mask(source_inputs, sequence_length=encoder_sequence_length)
    # When a target is provided, compute the decoder outputs for it.
    if labels is not None:
      outputs = self._decode_target(
          labels,
          encoder_outputs,
          h_r,
          h_s,
          encoder_mask,
          encoder_state,
          encoder_sequence_length,
          step=step,
          training=training)
      #outputs = {"outputs": outputs, "classification_logits": (logits_r, logits_s)}
      outputs = dict(logits=outputs["logits"], state=outputs["state"], attention=outputs["attention"], classification_logits_r=logits_r,classification_logits_s=logits_s)

    # When not in training, also compute the model predictions.
    if not training:
      predictions = self._dynamic_decode(
          features,
          encoder_outputs,
          h_r,
          h_s,
          encoder_mask,
          encoder_state,
          encoder_sequence_length)
    
    return outputs, predictions

  def _decode_target(self,
                     labels,
                     encoder_outputs,
                     h_r,
                     h_s,
                     encoder_mask,
                     encoder_state,
                     encoder_sequence_length,
                     step=None,
                     training=None):
    params = self.params
    target_inputs = self.labels_inputter(labels, training=training)
    input_fn = lambda ids: [self.labels_inputter({"ids": ids}, training=training), h_r, h_s, encoder_mask]

    sampling_probability = None
    if training:
      sampling_probability = decoder_util.get_sampling_probability(
          step,
          read_probability=params.get("scheduled_sampling_read_probability"),
          schedule_type=params.get("scheduled_sampling_type"),
          k=params.get("scheduled_sampling_k"))

    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    logits, state, attention = self.decoder(
        [target_inputs, h_r, h_s, encoder_mask],
        self.labels_inputter.get_length(labels),
        state=initial_state,
        input_fn=input_fn,
        sampling_probability=sampling_probability,
        training=training)
    labels_lengths = self.labels_inputter.get_length(labels)
    _, decoder_classification_outputs = self.classification_decoder_layer(state, labels_lengths, training=training)
    outputs = dict(logits=logits, state=decoder_classification_outputs, attention=attention)

    return outputs

  def _dynamic_decode(self, features, encoder_outputs, h_r, h_s, encoder_mask, encoder_state, encoder_sequence_length):
    params = self.params
    batch_size = tf.shape(tf.nest.flatten(encoder_outputs)[0])[0]
    start_ids = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
    beam_size = params.get("beam_width", 1)

    if beam_size > 1:
      # Tile encoder outputs to prepare for beam search.
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      encoder_sequence_length = tfa.seq2seq.tile_batch(encoder_sequence_length, beam_size)
      if encoder_state is not None:
        encoder_state = tfa.seq2seq.tile_batch(encoder_state, beam_size)

    # Dynamically decodes from the encoder outputs.
    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    sampled_ids, sampled_length, log_probs, alignment, _ = self.decoder.dynamic_decode(
        lambda ids: [self.labels_inputter({"ids": ids}), tfa.seq2seq.tile_batch(h_r, beam_size), tfa.seq2seq.tile_batch(h_s, beam_size), tfa.seq2seq.tile_batch(encoder_mask, beam_size)],
        start_ids,
        initial_state=initial_state,
        decoding_strategy=decoding.DecodingStrategy.from_params(params),
        sampler=decoding.Sampler.from_params(params),
        maximum_iterations=params.get("maximum_decoding_length", 250),
        minimum_iterations=params.get("minimum_decoding_length", 0))
    target_tokens = self.labels_inputter.ids_to_tokens.lookup(tf.cast(sampled_ids, tf.int64))

    # Maybe replace unknown targets by the source tokens with the highest attention weight.
    if params.get("replace_unknown_target", False):
      if alignment is None:
        raise TypeError("replace_unknown_target is not compatible with decoders "
                        "that don't return alignment history")
      if not isinstance(self.features_inputter, inputters.WordEmbedder):
        raise TypeError("replace_unknown_target is only defined when the source "
                        "inputter is a WordEmbedder")
      source_tokens = features["tokens"]
      if beam_size > 1:
        source_tokens = tfa.seq2seq.tile_batch(source_tokens, beam_size)
      # Merge batch and beam dimensions.
      original_shape = tf.shape(target_tokens)
      target_tokens = tf.reshape(target_tokens, [-1, original_shape[-1]])
      align_shape = shape_list(alignment)
      attention = tf.reshape(
          alignment, [align_shape[0] * align_shape[1], align_shape[2], align_shape[3]])
      # We don't have attention for </s> but ensure that the attention time dimension matches
      # the tokens time dimension.
      attention = reducer.align_in_time(attention, tf.shape(target_tokens)[1])
      replaced_target_tokens = replace_unknown_target(target_tokens, source_tokens, attention)
      target_tokens = tf.reshape(replaced_target_tokens, original_shape)

    # Maybe add noise to the predictions.
    decoding_noise = params.get("decoding_noise")
    if decoding_noise:
      target_tokens, sampled_length = _add_noise(
          target_tokens,
          sampled_length,
          decoding_noise,
          params.get("decoding_subword_token", "￭"),
          params.get("decoding_subword_token_is_spacer"))
      alignment = None  # Invalidate alignments.

    predictions = {
        "tokens": target_tokens,
        "length": sampled_length,
        "log_probs": log_probs
    }
    if alignment is not None:
      predictions["alignment"] = alignment

    # Maybe restrict the number of returned hypotheses based on the user parameter.
    num_hypotheses = params.get("num_hypotheses", 1)
    if num_hypotheses > 0:
      if num_hypotheses > beam_size:
        raise ValueError("n_best cannot be greater than beam_width")
      for key, value in six.iteritems(predictions):
        predictions[key] = value[:, :num_hypotheses]
    return predictions

  def compute_loss(self, outputs, labels, training=True):
    params = self.params
    if not isinstance(outputs, dict):
      outputs = dict(logits=outputs)
    logits = outputs["logits"]
    noisy_logits = outputs.get("noisy_logits")
    attention = outputs.get("attention")
    if noisy_logits is not None and params.get("contrastive_learning"):
      return losses.max_margin_loss(
          logits,
          labels["ids_out"],
          labels["length"],
          noisy_logits,
          labels["noisy_ids_out"],
          labels["noisy_length"],
          eta=params.get("max_margin_eta", 0.1))
    labels_lengths = self.labels_inputter.get_length(labels)
    loss, loss_normalizer, loss_token_normalizer = losses.cross_entropy_sequence_loss(
        logits,
        labels["ids_out"],
        labels_lengths,
        label_smoothing=params.get("label_smoothing", 0.0),
        average_in_time=params.get("average_loss_in_time", False),
        training=training)
    if training:
      gold_alignments = labels.get("alignment")
      guided_alignment_type = params.get("guided_alignment_type")
      if gold_alignments is not None and guided_alignment_type is not None:
        if attention is None:
          tf.get_logger().warning("This model did not return attention vectors; "
                                  "guided alignment will not be applied")
        else:
          loss += losses.guided_alignment_cost(
              attention[:, :-1],  # Do not constrain last timestep.
              gold_alignments,
              sequence_length=labels_lengths - 1,
              cost_type=guided_alignment_type,
              weight=params.get("guided_alignment_weight", 1))
    return loss, loss_normalizer, loss_token_normalizer

  def print_prediction(self, prediction, params=None, stream=None):
    if params is None:
      params = {}
    num_hypotheses = len(prediction["tokens"])
    for i in range(num_hypotheses):
      target_length = prediction["length"][i]
      tokens = prediction["tokens"][i][:target_length]
      sentence = self.labels_inputter.tokenizer.detokenize(tokens)
      score = None
      attention = None
      alignment_type = None
      if params.get("with_scores"):
        score = prediction["log_probs"][i]
      if params.get("with_alignments"):
        attention = prediction["alignment"][i][:target_length]
        alignment_type = params["with_alignments"]
      sentence = format_translation_output(
          sentence,
          score=score,
          attention=attention,
          alignment_type=alignment_type)
      print_bytes(tf.compat.as_bytes(sentence), stream=stream)

  def transfer_weights(self, new_model, new_optimizer=None, optimizer=None, ignore_weights=None):
    updated_variables = []

    def _map_variables(inputter_fn, vars_fn):
      mapping, _ = vocab.get_mapping(
          inputter_fn(self).vocabulary_file,
          inputter_fn(new_model).vocabulary_file)
      vars_a, vocab_axes = vars_fn(self)
      vars_b, _ = vars_fn(new_model)
      for var_a, var_b, vocab_axis in zip(vars_a, vars_b, vocab_axes):
        if new_optimizer is not None and optimizer is not None:
          variables = vocab.update_variable_and_slots(
              var_a,
              var_b,
              optimizer,
              new_optimizer,
              mapping,
              vocab_axis=vocab_axis)
        else:
          variables = [vocab.update_variable(var_a, var_b, mapping, vocab_axis=vocab_axis)]
        updated_variables.extend(variables)
      return vars_b

    _map_variables(
        lambda model: model.features_inputter,
        lambda model: ([model.features_inputter.embedding], [0]))
    _map_variables(
        lambda model: model.labels_inputter,
        lambda model: ([
            model.labels_inputter.embedding,
            model.decoder.output_layer.kernel,
            model.decoder.output_layer.bias], [0, 1, 0]))

    return super(SequenceToSequence_WDC, self).transfer_weights(
        new_model,
        new_optimizer=new_optimizer,
        optimizer=optimizer,
        ignore_weights=updated_variables)

class Multi_domain_SequenceToSequence_v2(model.SequenceGenerator):

  """A sequence to sequence model."""

  def __init__(self,
               source_inputter,
               target_inputter,
               lm_encoder,
               encoder,
               decoder,
               share_embeddings=EmbeddingsSharingLevel.NONE):

    if not isinstance(target_inputter, inputters.WordEmbedder):
      raise TypeError("Target inputter must be a WordEmbedder")
    if EmbeddingsSharingLevel.share_input_embeddings(share_embeddings):
      if isinstance(source_inputter, inputters.ParallelInputter):
        source_inputters = source_inputter.inputters
      else:
        source_inputters = [source_inputter]
      for inputter in source_inputters:
        if not isinstance(inputter, inputters.WordEmbedder):
          raise TypeError("Sharing embeddings requires all inputters to be a "
                          "WordEmbedder")

    examples_inputter = Multi_domain_SequenceToSequenceInputter(
        source_inputter,
        target_inputter,
        share_parameters=EmbeddingsSharingLevel.share_input_embeddings(share_embeddings))
    super(Multi_domain_SequenceToSequence_v2, self).__init__(examples_inputter)
    self.contextualized_wemb = lm_encoder
    self.encoder = encoder
    self.decoder = decoder
    self.share_embeddings = share_embeddings

  def auto_config(self, num_replicas=1):
    config = super(Multi_domain_SequenceToSequence_v2, self).auto_config(num_replicas=num_replicas)
    return merge_dict(config, {
        "params": {
            "beam_width": 5
        },
        "train": {
            "sample_buffer_size": -1,
            "max_step": 200000
        },
        "infer": {
            "batch_size": 32,
            "length_bucket_width": 5
        }
    })

  def initialize(self, data_config, params=None):
    super(Multi_domain_SequenceToSequence_v2, self).initialize(data_config, params=params)
    if self.params.get("contrastive_learning"):
      noiser = noise.WordNoiser(
          noises=[noise.WordOmission(1)],
          subword_token=self.params.get("decoding_subword_token", "￭"),
          is_spacer=self.params.get("decoding_subword_token_is_spacer"))
      self.labels_inputter.set_noise(noiser, in_place=False)

  def build(self, input_shape):
    super(Multi_domain_SequenceToSequence_v2, self).build(input_shape)
    output_layer = None
    if EmbeddingsSharingLevel.share_target_embeddings(self.share_embeddings):
      output_layer = layers.Dense(
          self.labels_inputter.vocabulary_size,
          weight=self.labels_inputter.embedding,
          transpose=True)
    self.decoder.initialize(
        vocab_size=self.labels_inputter.vocabulary_size,
        output_layer=output_layer)

  def call(self, features, labels=None, training=None, step=None):
    # Encode the source.
    assert isinstance(self.features_inputter, My_inputter)
    assert isinstance(self.labels_inputter, My_inputter)
    source_length = self.features_inputter.get_length(features)
    source_inputs = self.features_inputter(features, training=training)
    source_inputs = self.contextualized_wemb(source_inputs)
    encoder_outputs, encoder_state, encoder_sequence_length = self.encoder(
        [source_inputs, features["domain"]], sequence_length=source_length, training=training)

    outputs = None
    predictions = None

    # When a target is provided, compute the decoder outputs for it.
    if labels is not None:
      outputs = self._decode_target(
          labels,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length,
          step=step,
          training=training)

    # When not in training, also compute the model predictions.
    if not training:
      predictions = self._dynamic_decode(
          features,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length)

    return outputs, predictions

  def forward_fn(self, features, args_dict, labels=None, training=None, step=None):
    # Encode the source.
    training=True
    assert labels!=None
    assert isinstance(self.features_inputter, My_inputter)
    assert isinstance(self.labels_inputter, My_inputter)
    source_length = self.features_inputter.get_length(features)
    source_inputs = self.features_inputter.forward_fn(features, args_dict, training=training)
    source_inputs = self.contextualized_wemb.forward_fn(source_inputs)
    encoder_outputs, encoder_state, encoder_sequence_length = self.encoder.forward_fn(
        [source_inputs, features["domain"]], args_dict, sequence_length=source_length, training=training)

    outputs = None
    predictions = None

    # When a target is provided, compute the decoder outputs for it.
    if labels is not None:
      outputs = self._decode_target_forward_fn(
          labels,
          args_dict,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length,
          step=step,
          training=training)

    return outputs, predictions

  def _decode_target(self,
                     labels,
                     encoder_outputs,
                     encoder_state,
                     encoder_sequence_length,
                     step=None,
                     training=None):
    params = self.params
    target_inputs = self.labels_inputter(labels, training=training)
    input_fn = lambda ids: [self.labels_inputter({"ids": ids}, training=training), labels["domain"]]

    sampling_probability = None
    if training:
      sampling_probability = decoder_util.get_sampling_probability(
          step,
          read_probability=params.get("scheduled_sampling_read_probability"),
          schedule_type=params.get("scheduled_sampling_type"),
          k=params.get("scheduled_sampling_k"))

    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    logits, _, attention = self.decoder(
        [target_inputs, labels["domain"]],
        self.labels_inputter.get_length(labels),
        state=initial_state,
        input_fn=input_fn,
        sampling_probability=sampling_probability,
        training=training)
    outputs = dict(logits=logits, attention=attention)

    noisy_ids = labels.get("noisy_ids")
    if noisy_ids is not None and params.get("contrastive_learning"):
      # In case of contrastive learning, also forward the erroneous
      # translation to compute its log likelihood later.
      noisy_inputs = self.labels_inputter({"ids": noisy_ids}, training=training)
      noisy_logits, _, _ = self.decoder(
          noisy_inputs,
          labels["noisy_length"],
          state=initial_state,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
      outputs["noisy_logits"] = noisy_logits
    return outputs

  def _decode_target_forward_fn(self,
                     labels,
                     args_dict,
                     encoder_outputs,
                     encoder_state,
                     encoder_sequence_length,
                     step=None,
                     training=None):
    params = self.params
    target_inputs = self.labels_inputter.forward_fn(labels, args_dict, training=training)
    input_fn = lambda ids: [self.labels_inputter.forward_fn({"ids": ids}, args_dict, training=training), labels["domain"]]

    sampling_probability = None
    if training:
      sampling_probability = decoder_util.get_sampling_probability(
          step,
          read_probability=params.get("scheduled_sampling_read_probability"),
          schedule_type=params.get("scheduled_sampling_type"),
          k=params.get("scheduled_sampling_k"))

    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    
    logits, _, attention = self.decoder.forward_fn(
        [target_inputs, labels["domain"]],
        args_dict,
        self.labels_inputter.get_length(labels),
        initial_state=initial_state,
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        input_fn=input_fn,
        sampling_probability=sampling_probability,
        training=training)
        
    outputs = dict(logits=logits, attention=attention)
    """
    noisy_ids = labels.get("noisy_ids")
    if noisy_ids is not None and params.get("contrastive_learning"):
      # In case of contrastive learning, also forward the erroneous
      # translation to compute its log likelihood later.
      noisy_inputs = self.labels_inputter({"ids": noisy_ids}, training=training)
      noisy_logits, _, _ = self.decoder.forward_fn(
          noisy_inputs,
          args_dict,
          labels["noisy_length"],
          state=initial_state,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
      outputs["noisy_logits"] = noisy_logits
      """
    return outputs
 
  def _dynamic_decode(self, features, encoder_outputs, encoder_state, encoder_sequence_length):
    params = self.params
    batch_size = tf.shape(tf.nest.flatten(encoder_outputs)[0])[0]
    start_ids = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
    beam_size = params.get("beam_width", 1)

    if beam_size > 1:
      # Tile encoder outputs to prepare for beam search.
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      encoder_sequence_length = tfa.seq2seq.tile_batch(encoder_sequence_length, beam_size)
      if encoder_state is not None:
        encoder_state = tfa.seq2seq.tile_batch(encoder_state, beam_size)

    # Dynamically decodes from the encoder outputs.
    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    sampled_ids, sampled_length, log_probs, alignment, _ = self.decoder.dynamic_decode(
        lambda ids: [self.labels_inputter({"ids": ids}), features["domain"]],
        start_ids,
        initial_state=initial_state,
        decoding_strategy=decoding.DecodingStrategy.from_params(params),
        sampler=decoding.Sampler.from_params(params),
        maximum_iterations=params.get("maximum_decoding_length", 250),
        minimum_iterations=params.get("minimum_decoding_length", 0))
    target_tokens = self.labels_inputter.ids_to_tokens.lookup(tf.cast(sampled_ids, tf.int64))

    # Maybe replace unknown targets by the source tokens with the highest attention weight.
    if params.get("replace_unknown_target", False):
      if alignment is None:
        raise TypeError("replace_unknown_target is not compatible with decoders "
                        "that don't return alignment history")
      if not isinstance(self.features_inputter, inputters.WordEmbedder):
        raise TypeError("replace_unknown_target is only defined when the source "
                        "inputter is a WordEmbedder")
      source_tokens = features["tokens"]
      if beam_size > 1:
        source_tokens = tfa.seq2seq.tile_batch(source_tokens, beam_size)
      # Merge batch and beam dimensions.
      original_shape = tf.shape(target_tokens)
      target_tokens = tf.reshape(target_tokens, [-1, original_shape[-1]])
      align_shape = shape_list(alignment)
      attention = tf.reshape(
          alignment, [align_shape[0] * align_shape[1], align_shape[2], align_shape[3]])
      # We don't have attention for </s> but ensure that the attention time dimension matches
      # the tokens time dimension.
      attention = reducer.align_in_time(attention, tf.shape(target_tokens)[1])
      replaced_target_tokens = replace_unknown_target(target_tokens, source_tokens, attention)
      target_tokens = tf.reshape(replaced_target_tokens, original_shape)

    # Maybe add noise to the predictions.
    decoding_noise = params.get("decoding_noise")
    if decoding_noise:
      target_tokens, sampled_length = _add_noise(
          target_tokens,
          sampled_length,
          decoding_noise,
          params.get("decoding_subword_token", "￭"),
          params.get("decoding_subword_token_is_spacer"))
      alignment = None  # Invalidate alignments.

    predictions = {
        "tokens": target_tokens,
        "length": sampled_length,
        "log_probs": log_probs
    }
    if alignment is not None:
      predictions["alignment"] = alignment

    # Maybe restrict the number of returned hypotheses based on the user parameter.
    num_hypotheses = params.get("num_hypotheses", 1)
    if num_hypotheses > 0:
      if num_hypotheses > beam_size:
        raise ValueError("n_best cannot be greater than beam_width")
      for key, value in six.iteritems(predictions):
        predictions[key] = value[:, :num_hypotheses]
    return predictions

  def compute_loss(self, outputs, labels, training=True):
    params = self.params
    if not isinstance(outputs, dict):
      outputs = dict(logits=outputs)
    logits = outputs["logits"]
    noisy_logits = outputs.get("noisy_logits")
    attention = outputs.get("attention")
    if noisy_logits is not None and params.get("contrastive_learning"):
      return losses.max_margin_loss(
          logits,
          labels["ids_out"],
          labels["length"],
          noisy_logits,
          labels["noisy_ids_out"],
          labels["noisy_length"],
          eta=params.get("max_margin_eta", 0.1))
    labels_lengths = self.labels_inputter.get_length(labels)
    loss, loss_normalizer, loss_token_normalizer = losses.cross_entropy_sequence_loss(
        logits,
        labels["ids_out"],
        labels_lengths,
        label_smoothing=params.get("label_smoothing", 0.0),
        average_in_time=params.get("average_loss_in_time", False),
        training=training)
    if training:
      gold_alignments = labels.get("alignment")
      guided_alignment_type = params.get("guided_alignment_type")
      if gold_alignments is not None and guided_alignment_type is not None:
        if attention is None:
          tf.get_logger().warning("This model did not return attention vectors; "
                                  "guided alignment will not be applied")
        else:
          loss += losses.guided_alignment_cost(
              attention[:, :-1],  # Do not constrain last timestep.
              gold_alignments,
              sequence_length=labels_lengths - 1,
              cost_type=guided_alignment_type,
              weight=params.get("guided_alignment_weight", 1))
    return loss, loss_normalizer, loss_token_normalizer
  
  def transfer_weights(self, new_model, new_optimizer=None, optimizer=None, ignore_weights=None):
    updated_variables = []

    def _map_variables(inputter_fn, vars_fn):
      mapping, _ = vocab.get_mapping(
          inputter_fn(self).vocabulary_file,
          inputter_fn(new_model).vocabulary_file)
      vars_a, vocab_axes = vars_fn(self)
      vars_b, _ = vars_fn(new_model)
      for var_a, var_b, vocab_axis in zip(vars_a, vars_b, vocab_axes):
        if new_optimizer is not None and optimizer is not None:
          variables = vocab.update_variable_and_slots(
              var_a,
              var_b,
              optimizer,
              new_optimizer,
              mapping,
              vocab_axis=vocab_axis)
        else:
          variables = [vocab.update_variable(var_a, var_b, mapping, vocab_axis=vocab_axis)]
        updated_variables.extend(variables)
      return vars_b

  def print_prediction(self, prediction, params=None, stream=None):
    if params is None:
      params = {}
    num_hypotheses = len(prediction["tokens"])
    for i in range(num_hypotheses):
      target_length = prediction["length"][i]
      tokens = prediction["tokens"][i][:target_length]
      sentence = self.labels_inputter.tokenizer.detokenize(tokens)
      score = None
      attention = None
      alignment_type = None
      if params.get("with_scores"):
        score = prediction["log_probs"][i]
      if params.get("with_alignments"):
        attention = prediction["alignment"][i][:target_length]
        alignment_type = params["with_alignments"]
      sentence = format_translation_output(
          sentence,
          score=score,
          attention=attention,
          alignment_type=alignment_type)
      print_bytes(tf.compat.as_bytes(sentence), stream=stream)

class SequenceToSequence_with_dprob(model.SequenceGenerator):

  """A sequence to sequence model."""

  def __init__(self,
               source_inputter,
               target_inputter,
               probs_inputter,
               encoder,
               decoder,
               num_domains=2,
               num_units=512,
               share_embeddings=EmbeddingsSharingLevel.NONE):

    if not isinstance(target_inputter, inputters.WordEmbedder) and not isinstance(target_inputter, My_inputter):
      raise TypeError("Target inputter must be a WordEmbedder or My_inputter")
    if EmbeddingsSharingLevel.share_input_embeddings(share_embeddings):
      if isinstance(source_inputter, inputters.ParallelInputter):
        source_inputters = source_inputter.inputters
      else:
        source_inputters = [source_inputter]
      for inputter in source_inputters:
        if not isinstance(inputter, inputters.WordEmbedder):
          raise TypeError("Sharing embeddings requires all inputters to be a "
                          "WordEmbedder")

    examples_inputter = Multi_domain_SequenceToSequenceInputter_withprob(
        source_inputter,
        target_inputter,
        probs_inputter,
        share_parameters=EmbeddingsSharingLevel.share_input_embeddings(share_embeddings))
    super(SequenceToSequence_with_dprob, self).__init__(examples_inputter)
    self.encoder = encoder
    self.decoder = decoder
    self.share_embeddings = share_embeddings
 
  def auto_config(self, num_replicas=1):
    config = super(SequenceToSequence_with_dprob, self).auto_config(num_replicas=num_replicas)
    return merge_dict(config, {
        "params": {
            "beam_width": 5
        },
        "train": {
            "sample_buffer_size": -1,
            "max_step": 200000
        },
        "infer": {
            "batch_size": 32,
            "length_bucket_width": 5
        }
    })

  def initialize(self, data_config, params=None):
    super(SequenceToSequence_with_dprob, self).initialize(data_config, params=params)
    if self.params.get("contrastive_learning"):
      noiser = noise.WordNoiser(
          noises=[noise.WordOmission(1)],
          subword_token=self.params.get("decoding_subword_token", "￭"),
          is_spacer=self.params.get("decoding_subword_token_is_spacer"))
      self.labels_inputter.set_noise(noiser, in_place=False)

  def build(self, input_shape):
    super(SequenceToSequence_with_dprob, self).build(input_shape)
    output_layer = None
    if EmbeddingsSharingLevel.share_target_embeddings(self.share_embeddings):
      output_layer = layers.Dense(
          self.labels_inputter.vocabulary_size,
          weight=self.labels_inputter.embedding,
          transpose=True)
    self.decoder.initialize(
        vocab_size=self.labels_inputter.vocabulary_size,
        output_layer=output_layer)

  def call(self, features, labels=None, training=None, step=None, internal_node_printing=False, inference=True):
    # Encode the source.
    assert isinstance(self.features_inputter, My_inputter)
    assert isinstance(self.labels_inputter, My_inputter)    
    source_length = self.features_inputter.get_length(features)
    source_inputs = self.features_inputter(features, training=training)
    if isinstance(self.encoder, Multi_domain_SelfAttentionEncoder_v1) or isinstance(self.encoder, Multi_domain_SelfAttentionEncoder_v2) or isinstance(self.encoder, Multi_domain_SelfAttentionEncoder_v12) or isinstance(self.encoder, Multi_domain_SelfAttentionEncoder_v15):
      encoder_outputs, encoder_state, encoder_sequence_length = self.encoder(
        [source_inputs, features["domain"]], sequence_length=source_length, training=training, internal_node_printing=internal_node_printing)
    else:
      encoder_outputs, encoder_state, encoder_sequence_length = self.encoder(
        [source_inputs, features["domain"]], sequence_length=source_length, training=training)

    outputs = None
    predictions = None

    # When a target is provided, compute the decoder outputs for it.
    if labels is not None:
      outputs = self._decode_target(
          labels,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length,
          step=step,
          training=training)

    # When not in training, also compute the model predictions.
    if not training and inference:
      predictions = self._dynamic_decode(
          features,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length)
    
    return outputs, predictions

  def _decode_target(self,
                     labels,
                     encoder_outputs,
                     encoder_state,
                     encoder_sequence_length,
                     step=None,
                     training=None,
                     internal_node_printing=False):
    params = self.params
    target_inputs = self.labels_inputter(labels, training=training)
    #print("labels: ", labels)
    #print("target_inputs",target_inputs)
    input_fn = lambda ids: [self.labels_inputter({"ids": ids}, training=training), labels["domain"]]

    sampling_probability = None
    if training:
      sampling_probability = decoder_util.get_sampling_probability(
          step,
          read_probability=params.get("scheduled_sampling_read_probability"),
          schedule_type=params.get("scheduled_sampling_type"),
          k=params.get("scheduled_sampling_k"))

    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)

    logits, _, attention = self.decoder(
        [target_inputs, labels["domain"]],
        self.labels_inputter.get_length(labels),
        state=initial_state,
        input_fn=input_fn,
        sampling_probability=sampling_probability,
        training=training)
    outputs = dict(logits=logits, attention=attention)
    
    return outputs
 
  def _dynamic_decode(self, features, encoder_outputs, encoder_state, encoder_sequence_length):
    params = self.params
    batch_size = tf.shape(tf.nest.flatten(encoder_outputs)[0])[0]
    start_ids = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
    beam_size = params.get("beam_width", 1)

    if beam_size > 1:
      # Tile encoder outputs to prepare for beam search.
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      encoder_sequence_length = tfa.seq2seq.tile_batch(encoder_sequence_length, beam_size)
      if encoder_state is not None:
        encoder_state = tfa.seq2seq.tile_batch(encoder_state, beam_size)

    # Dynamically decodes from the encoder outputs.
    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    sampled_ids, sampled_length, log_probs, alignment, _ = self.decoder.dynamic_decode(
        lambda ids: [self.labels_inputter({"ids": ids}), features["domain"]],
        start_ids,
        initial_state=initial_state,
        decoding_strategy=decoding.DecodingStrategy.from_params(params),
        sampler=decoding.Sampler.from_params(params),
        maximum_iterations=params.get("maximum_decoding_length", 250),
        minimum_iterations=params.get("minimum_decoding_length", 0))
    target_tokens = self.labels_inputter.ids_to_tokens.lookup(tf.cast(sampled_ids, tf.int64))

    # Maybe replace unknown targets by the source tokens with the highest attention weight.
    if params.get("replace_unknown_target", False):
      if alignment is None:
        raise TypeError("replace_unknown_target is not compatible with decoders "
                        "that don't return alignment history")
      if not isinstance(self.features_inputter, inputters.WordEmbedder):
        raise TypeError("replace_unknown_target is only defined when the source "
                        "inputter is a WordEmbedder")
      source_tokens = features["tokens"]
      if beam_size > 1:
        source_tokens = tfa.seq2seq.tile_batch(source_tokens, beam_size)
      # Merge batch and beam dimensions.
      original_shape = tf.shape(target_tokens)
      target_tokens = tf.reshape(target_tokens, [-1, original_shape[-1]])
      align_shape = shape_list(alignment)
      attention = tf.reshape(
          alignment, [align_shape[0] * align_shape[1], align_shape[2], align_shape[3]])
      # We don't have attention for </s> but ensure that the attention time dimension matches
      # the tokens time dimension.
      attention = reducer.align_in_time(attention, tf.shape(target_tokens)[1])
      replaced_target_tokens = replace_unknown_target(target_tokens, source_tokens, attention)
      target_tokens = tf.reshape(replaced_target_tokens, original_shape)

    # Maybe add noise to the predictions.
    decoding_noise = params.get("decoding_noise")
    if decoding_noise:
      target_tokens, sampled_length = _add_noise(
          target_tokens,
          sampled_length,
          decoding_noise,
          params.get("decoding_subword_token", "￭"),
          params.get("decoding_subword_token_is_spacer"))
      alignment = None  # Invalidate alignments.

    predictions = {
        "tokens": target_tokens,
        "length": sampled_length,
        "log_probs": log_probs
    }
    if alignment is not None:
      predictions["alignment"] = alignment

    # Maybe restrict the number of returned hypotheses based on the user parameter.
    num_hypotheses = params.get("num_hypotheses", 1)
    if num_hypotheses > 0:
      if num_hypotheses > beam_size:
        raise ValueError("n_best cannot be greater than beam_width")
      for key, value in six.iteritems(predictions):
        predictions[key] = value[:, :num_hypotheses]
    return predictions

  def compute_loss(self, outputs, labels, training=True):
    params = self.params
    if not isinstance(outputs, dict):
      outputs = dict(logits=outputs)
    logits = outputs["logits"]
    noisy_logits = outputs.get("noisy_logits")
    attention = outputs.get("attention")
    if noisy_logits is not None and params.get("contrastive_learning"):
      return losses.max_margin_loss(
          logits,
          labels["ids_out"],
          labels["length"],
          noisy_logits,
          labels["noisy_ids_out"],
          labels["noisy_length"],
          eta=params.get("max_margin_eta", 0.1))
    labels_lengths = self.labels_inputter.get_length(labels)
    print("average_loss_in_time", params.get("average_loss_in_time", False))
    loss, loss_normalizer, loss_token_normalizer = losses.cross_entropy_sequence_loss(
        logits,
        labels["ids_out"],
        labels_lengths,
        label_smoothing=params.get("label_smoothing", 0.0),
        average_in_time=params.get("average_loss_in_time", False),
        training=training)
    if training:
      gold_alignments = labels.get("alignment")
      guided_alignment_type = params.get("guided_alignment_type")
      if gold_alignments is not None and guided_alignment_type is not None:
        if attention is None:
          tf.get_logger().warning("This model did not return attention vectors; "
                                  "guided alignment will not be applied")
        else:
          loss += losses.guided_alignment_cost(
              attention[:, :-1],  # Do not constrain last timestep.
              gold_alignments,
              sequence_length=labels_lengths - 1,
              cost_type=guided_alignment_type,
              weight=params.get("guided_alignment_weight", 1))
    return loss, loss_normalizer, loss_token_normalizer
  
  def print_prediction(self, prediction, params=None, stream=None):
    if params is None:
      params = {}
    num_hypotheses = len(prediction["tokens"])
    for i in range(num_hypotheses):
      target_length = prediction["length"][i]
      tokens = prediction["tokens"][i][:target_length]
      sentence = self.labels_inputter.tokenizer.detokenize(tokens)
      score = None
      attention = None
      alignment_type = None
      if params.get("with_scores"):
        score = prediction["log_probs"][i]
      if params.get("with_alignments"):
        attention = prediction["alignment"][i][:target_length]
        alignment_type = params["with_alignments"]
      sentence = format_translation_output(
          sentence,
          score=score,
          attention=attention,
          alignment_type=alignment_type)
      print_bytes(tf.compat.as_bytes(sentence), stream=stream)

  def transfer_weights(self, new_model, new_optimizer=None, optimizer=None, ignore_weights=None):
    updated_variables = []

    def _map_variables(inputter_fn, vars_fn):
      mapping, _ = vocab.get_mapping(
          inputter_fn(self).vocabulary_file,
          inputter_fn(new_model).vocabulary_file)
      vars_a, vocab_axes = vars_fn(self)
      vars_b, _ = vars_fn(new_model)
      for var_a, var_b, vocab_axis in zip(vars_a, vars_b, vocab_axes):
        if new_optimizer is not None and optimizer is not None:
          variables = vocab.update_variable_and_slots(
              var_a,
              var_b,
              optimizer,
              new_optimizer,
              mapping,
              vocab_axis=vocab_axis)
        else:
          variables = [vocab.update_variable(var_a, var_b, mapping, vocab_axis=vocab_axis)]
        updated_variables.extend(variables)
      return vars_b

  def create_variables(self, optimizer=None):
    """Creates the model variables by running it once.

    Args:
      optimizer: If set, also create the optimizer variables.
    """
    if self.built:
      return

    # Create input features from the input signatures. We remove the leading
    # batch dimension as sometimes assumed by make_features methods and set
    # unspecified dimensions to 1.
    features = tf.nest.map_structure(
        lambda spec: tf.fill(
            [dim or 1 for dim in spec.shape.as_list()[1:]],
            tf.constant("" if spec.dtype is tf.string else 1, dtype=spec.dtype)),
        self.examples_inputter.input_signature())
    print(features)
    features = self.examples_inputter.make_features(features=features)

    # Add the batch dimension back before calling the model.
    features, labels = tf.nest.map_structure(lambda x: tf.expand_dims(x, 0), features)
    _ = self(features, labels=labels, training=True, step=0)

    if optimizer is not None:
      _ = optimizer.iterations
      optimizer._create_hypers()  # pylint: disable=protected-access
      optimizer._create_slots(self.trainable_variables)  # pylint: disable=protected-access

class Multi_domain_SequenceToSequence_DRO(model.SequenceGenerator):

  """A sequence to sequence model."""

  def __init__(self,
               source_inputter,
               target_inputter,
               probs_inputter,
               encoder,
               decoder,
               num_domains=2,
               num_units=512,
               share_embeddings=EmbeddingsSharingLevel.NONE):

    if not isinstance(target_inputter, inputters.WordEmbedder) and not isinstance(target_inputter, My_inputter):
      raise TypeError("Target inputter must be a WordEmbedder or My_inputter")
    if EmbeddingsSharingLevel.share_input_embeddings(share_embeddings):
      if isinstance(source_inputter, inputters.ParallelInputter):
        source_inputters = source_inputter.inputters
      else:
        source_inputters = [source_inputter]
      for inputter in source_inputters:
        if not isinstance(inputter, inputters.WordEmbedder):
          raise TypeError("Sharing embeddings requires all inputters to be a "
                          "WordEmbedder")

    examples_inputter = Multi_domain_SequenceToSequenceInputter_DRO(
        source_inputter,
        target_inputter,
        probs_inputter,
        share_parameters=EmbeddingsSharingLevel.share_input_embeddings(share_embeddings))
    super(Multi_domain_SequenceToSequence_DRO, self).__init__(examples_inputter)
    self.encoder = encoder
    self.decoder = decoder
    self.share_embeddings = share_embeddings
 
  def auto_config(self, num_replicas=1):
    config = super(Multi_domain_SequenceToSequence_DRO, self).auto_config(num_replicas=num_replicas)
    return merge_dict(config, {
        "params": {
            "beam_width": 5
        },
        "train": {
            "sample_buffer_size": -1,
            "max_step": 200000
        },
        "infer": {
            "batch_size": 32,
            "length_bucket_width": 5
        }
    })

  def initialize(self, data_config, params=None):
    super(Multi_domain_SequenceToSequence_DRO, self).initialize(data_config, params=params)
    if self.params.get("contrastive_learning"):
      noiser = noise.WordNoiser(
          noises=[noise.WordOmission(1)],
          subword_token=self.params.get("decoding_subword_token", "￭"),
          is_spacer=self.params.get("decoding_subword_token_is_spacer"))
      self.labels_inputter.set_noise(noiser, in_place=False)

  def build(self, input_shape):
    super(Multi_domain_SequenceToSequence_DRO, self).build(input_shape)
    output_layer = None
    if EmbeddingsSharingLevel.share_target_embeddings(self.share_embeddings):
      output_layer = layers.Dense(
          self.labels_inputter.vocabulary_size,
          weight=self.labels_inputter.embedding,
          transpose=True)
    self.decoder.initialize(
        vocab_size=self.labels_inputter.vocabulary_size,
        output_layer=output_layer)

  def call(self, features, labels=None, training=None, step=None, internal_node_printing=False):
    # Encode the source.
    assert isinstance(self.features_inputter, My_inputter)
    assert isinstance(self.labels_inputter, My_inputter)    
    source_length = self.features_inputter.get_length(features)
    source_inputs = self.features_inputter(features, training=training)
    if isinstance(self.encoder, Multi_domain_SelfAttentionEncoder_v1) or isinstance(self.encoder, Multi_domain_SelfAttentionEncoder_v2) or isinstance(self.encoder, Multi_domain_SelfAttentionEncoder_v12) or isinstance(self.encoder, Multi_domain_SelfAttentionEncoder_v15):
      encoder_outputs, encoder_state, encoder_sequence_length = self.encoder(
        [source_inputs, features["domain"]], sequence_length=source_length, training=training, internal_node_printing=internal_node_printing)
    else:
      encoder_outputs, encoder_state, encoder_sequence_length = self.encoder(
        [source_inputs, features["domain"]], sequence_length=source_length, training=training)

    #_, domain_classification_logits = self.classification_layer(encoder_outputs, encoder_sequence_length, training=training)

    outputs = None
    predictions = None

    # When a target is provided, compute the decoder outputs for it.
    if labels is not None:
      outputs = self._decode_target(
          labels,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length,
          step=step,
          training=training)
      #outputs = dict(logits=outputs["logits"], attention=outputs["attention"], domain_classification_logits=domain_classification_logits)

    # When not in training, also compute the model predictions.
    if not training:
      predictions = self._dynamic_decode(
          features,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length)
    
    return outputs, predictions
  
  def _decode_target(self,
                     labels,
                     encoder_outputs,
                     encoder_state,
                     encoder_sequence_length,
                     step=None,
                     training=None,
                     internal_node_printing=False):
    params = self.params
    target_inputs = self.labels_inputter(labels, training=training)
    input_fn = lambda ids: [self.labels_inputter({"ids": ids}, training=training), labels["domain"]]

    sampling_probability = None
    if training:
      sampling_probability = decoder_util.get_sampling_probability(
          step,
          read_probability=params.get("scheduled_sampling_read_probability"),
          schedule_type=params.get("scheduled_sampling_type"),
          k=params.get("scheduled_sampling_k"))

    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    logits, _, attention = self.decoder(
        [target_inputs, labels["domain"]],
        self.labels_inputter.get_length(labels),
        state=initial_state,
        input_fn=input_fn,
        sampling_probability=sampling_probability,
        training=training)
    outputs = dict(logits=logits, attention=attention)

    noisy_ids = labels.get("noisy_ids")
    if noisy_ids is not None and params.get("contrastive_learning"):
      # In case of contrastive learning, also forward the erroneous
      # translation to compute its log likelihood later.
      noisy_inputs = self.labels_inputter({"ids": noisy_ids}, training=training)
      noisy_logits, _, _ = self.decoder(
          noisy_inputs,
          labels["noisy_length"],
          state=initial_state,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
      outputs["noisy_logits"] = noisy_logits
    return outputs
 
  def _dynamic_decode(self, features, encoder_outputs, encoder_state, encoder_sequence_length):
    params = self.params
    batch_size = tf.shape(tf.nest.flatten(encoder_outputs)[0])[0]
    start_ids = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
    beam_size = params.get("beam_width", 1)

    if beam_size > 1:
      # Tile encoder outputs to prepare for beam search.
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      encoder_sequence_length = tfa.seq2seq.tile_batch(encoder_sequence_length, beam_size)
      if encoder_state is not None:
        encoder_state = tfa.seq2seq.tile_batch(encoder_state, beam_size)

    # Dynamically decodes from the encoder outputs.
    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    sampled_ids, sampled_length, log_probs, alignment, _ = self.decoder.dynamic_decode(
        lambda ids: [self.labels_inputter({"ids": ids}), features["domain"]],
        start_ids,
        initial_state=initial_state,
        decoding_strategy=decoding.DecodingStrategy.from_params(params),
        sampler=decoding.Sampler.from_params(params),
        maximum_iterations=params.get("maximum_decoding_length", 250),
        minimum_iterations=params.get("minimum_decoding_length", 0))
    target_tokens = self.labels_inputter.ids_to_tokens.lookup(tf.cast(sampled_ids, tf.int64))

    # Maybe replace unknown targets by the source tokens with the highest attention weight.
    if params.get("replace_unknown_target", False):
      if alignment is None:
        raise TypeError("replace_unknown_target is not compatible with decoders "
                        "that don't return alignment history")
      if not isinstance(self.features_inputter, inputters.WordEmbedder):
        raise TypeError("replace_unknown_target is only defined when the source "
                        "inputter is a WordEmbedder")
      source_tokens = features["tokens"]
      if beam_size > 1:
        source_tokens = tfa.seq2seq.tile_batch(source_tokens, beam_size)
      # Merge batch and beam dimensions.
      original_shape = tf.shape(target_tokens)
      target_tokens = tf.reshape(target_tokens, [-1, original_shape[-1]])
      align_shape = shape_list(alignment)
      attention = tf.reshape(
          alignment, [align_shape[0] * align_shape[1], align_shape[2], align_shape[3]])
      # We don't have attention for </s> but ensure that the attention time dimension matches
      # the tokens time dimension.
      attention = reducer.align_in_time(attention, tf.shape(target_tokens)[1])
      replaced_target_tokens = replace_unknown_target(target_tokens, source_tokens, attention)
      target_tokens = tf.reshape(replaced_target_tokens, original_shape)

    # Maybe add noise to the predictions.
    decoding_noise = params.get("decoding_noise")
    if decoding_noise:
      target_tokens, sampled_length = _add_noise(
          target_tokens,
          sampled_length,
          decoding_noise,
          params.get("decoding_subword_token", "￭"),
          params.get("decoding_subword_token_is_spacer"))
      alignment = None  # Invalidate alignments.

    predictions = {
        "tokens": target_tokens,
        "length": sampled_length,
        "log_probs": log_probs
    }
    if alignment is not None:
      predictions["alignment"] = alignment

    # Maybe restrict the number of returned hypotheses based on the user parameter.
    num_hypotheses = params.get("num_hypotheses", 1)
    if num_hypotheses > 0:
      if num_hypotheses > beam_size:
        raise ValueError("n_best cannot be greater than beam_width")
      for key, value in six.iteritems(predictions):
        predictions[key] = value[:, :num_hypotheses]
    return predictions

  def compute_loss(self, outputs, labels, training=True):
    params = self.params
    if not isinstance(outputs, dict):
      outputs = dict(logits=outputs)
    logits = outputs["logits"]
    noisy_logits = outputs.get("noisy_logits")
    attention = outputs.get("attention")
    if noisy_logits is not None and params.get("contrastive_learning"):
      return losses.max_margin_loss(
          logits,
          labels["ids_out"],
          labels["length"],
          noisy_logits,
          labels["noisy_ids_out"],
          labels["noisy_length"],
          eta=params.get("max_margin_eta", 0.1))
    labels_lengths = self.labels_inputter.get_length(labels)
    print("average_loss_in_time", params.get("average_loss_in_time", False))
    loss, loss_normalizer, loss_token_normalizer = losses.cross_entropy_sequence_loss(
        logits,
        labels["ids_out"],
        labels_lengths,
        label_smoothing=params.get("label_smoothing", 0.0),
        average_in_time=params.get("average_loss_in_time", False),
        training=training)
    if training:
      gold_alignments = labels.get("alignment")
      guided_alignment_type = params.get("guided_alignment_type")
      if gold_alignments is not None and guided_alignment_type is not None:
        if attention is None:
          tf.get_logger().warning("This model did not return attention vectors; "
                                  "guided alignment will not be applied")
        else:
          loss += losses.guided_alignment_cost(
              attention[:, :-1],  # Do not constrain last timestep.
              gold_alignments,
              sequence_length=labels_lengths - 1,
              cost_type=guided_alignment_type,
              weight=params.get("guided_alignment_weight", 1))
    return loss, loss_normalizer, loss_token_normalizer
  
  def print_prediction(self, prediction, params=None, stream=None):
    if params is None:
      params = {}
    num_hypotheses = len(prediction["tokens"])
    for i in range(num_hypotheses):
      target_length = prediction["length"][i]
      tokens = prediction["tokens"][i][:target_length]
      sentence = self.labels_inputter.tokenizer.detokenize(tokens)
      score = None
      attention = None
      alignment_type = None
      if params.get("with_scores"):
        score = prediction["log_probs"][i]
      if params.get("with_alignments"):
        attention = prediction["alignment"][i][:target_length]
        alignment_type = params["with_alignments"]
      sentence = format_translation_output(
          sentence,
          score=score,
          attention=attention,
          alignment_type=alignment_type)
      print_bytes(tf.compat.as_bytes(sentence), stream=stream)

  def transfer_weights(self, new_model, new_optimizer=None, optimizer=None, ignore_weights=None):
    updated_variables = []

    def _map_variables(inputter_fn, vars_fn):
      mapping, _ = vocab.get_mapping(
          inputter_fn(self).vocabulary_file,
          inputter_fn(new_model).vocabulary_file)
      vars_a, vocab_axes = vars_fn(self)
      vars_b, _ = vars_fn(new_model)
      for var_a, var_b, vocab_axis in zip(vars_a, vars_b, vocab_axes):
        if new_optimizer is not None and optimizer is not None:
          variables = vocab.update_variable_and_slots(
              var_a,
              var_b,
              optimizer,
              new_optimizer,
              mapping,
              vocab_axis=vocab_axis)
        else:
          variables = [vocab.update_variable(var_a, var_b, mapping, vocab_axis=vocab_axis)]
        updated_variables.extend(variables)
      return vars_b

class Priming_SequenceToSequence(model.SequenceGenerator):
  """A sequence to sequence model."""

  def __init__(self,
               source_inputter,
               target_inputter,
               encoder,
               pre_encoder,
               decoder,
               version=1,
               share_embeddings=EmbeddingsSharingLevel.NONE):
    if not isinstance(target_inputter, inputters.WordEmbedder):
      raise TypeError("Target inputter must be a WordEmbedder")
    if EmbeddingsSharingLevel.share_input_embeddings(share_embeddings):
      if isinstance(source_inputter, inputters.ParallelInputter):
        source_inputters = source_inputter.inputters
      else:
        source_inputters = [source_inputter]
      for inputter in source_inputters:
        if not isinstance(inputter, inputters.WordEmbedder):
          raise TypeError("Sharing embeddings requires all inputters to be a "
                          "WordEmbedder")

    examples_inputter = Priming_SequenceToSequenceInputter(
        source_inputter,
        target_inputter,
        share_parameters=EmbeddingsSharingLevel.share_input_embeddings(share_embeddings))
    super(Priming_SequenceToSequence, self).__init__(examples_inputter)
    self.encoder = encoder
    self.pre_encoder = pre_encoder
    self.decoder = decoder
    self.share_embeddings = share_embeddings
    self.version = version
    
    if self.version==5:
      print("share encoder")
      self.pre_encoder = self.encoder
  
  def auto_config(self, num_replicas=1):
    config = super(Priming_SequenceToSequence, self).auto_config(num_replicas=num_replicas)
    return merge_dict(config, {
        "params": {
            "beam_width": 4
        },
        "train": {
            "sample_buffer_size": -1,
            "max_step": 500000
        },
        "infer": {
            "batch_size": 32,
            "length_bucket_width": 5
        }
    })

  def initialize(self, data_config, params=None):
    super(Priming_SequenceToSequence, self).initialize(data_config, params=params)
    if self.params.get("contrastive_learning"):
      # Use the simplest and most effective CL_one from the paper.
      # https://www.aclweb.org/anthology/P19-1623
      noiser = noise.WordNoiser(
          noises=[noise.WordOmission(1)],
          subword_token=self.params.get("decoding_subword_token", "￭"),
          is_spacer=self.params.get("decoding_subword_token_is_spacer"))
      self.labels_inputter.set_noise(noiser, in_place=False)

  def build(self, input_shape):
    super(Priming_SequenceToSequence, self).build(input_shape)
    output_layer = None
    if EmbeddingsSharingLevel.share_target_embeddings(self.share_embeddings):
      output_layer = layers.Dense(
          self.labels_inputter.vocabulary_size,
          weight=self.labels_inputter.embedding,
          transpose=True)
    self.decoder.initialize(
        vocab_size=self.labels_inputter.vocabulary_size,
        output_layer=output_layer)

  def create_variables(self, optimizer=None):
    """Creates the model variables by running it once.

    Args:
      optimizer: If set, also create the optimizer variables.
    """
    if self.built:
      return

    # Create input features from the input signatures. We remove the leading
    # batch dimension as sometimes assumed by make_features methods and set
    # unspecified dimensions to 1.
    features = tf.nest.map_structure(
        lambda spec: tf.fill(
            [dim or 1 for dim in spec.shape.as_list()[1:]],
            tf.constant("" if spec.dtype is tf.string else 1, dtype=spec.dtype)),
        self.examples_inputter.input_signature())
    #print("features: ",features)
    features = self.examples_inputter.make_features(features=features)
    #print("features: ",features)
    # Add the batch dimension back before calling the model.
    features, labels = tf.nest.map_structure(lambda x: tf.expand_dims(x, 0), features)
    _ = self(features, labels=labels, training=True, step=0)

    if optimizer is not None:
      _ = optimizer.iterations
      optimizer._create_hypers()  # pylint: disable=protected-access
      optimizer._create_slots(self.trainable_variables)  # pylint: disable=protected-access
  
  def call(self, features, labels=None, training=None, step=None):
    # Encode the source.
    source_length = self.features_inputter.get_length(features)
    source_inputs = self.features_inputter(features, training=training)

    source_length, pre_length = source_length
    source_inputs, pre_inputs = source_inputs

    encoder_outputs, encoder_state, encoder_sequence_length = self.encoder(
      source_inputs, sequence_length=source_length, training=training)
    pre_encoder_outputs, pre_encoder_state, pre_encoder_sequence_length = self.pre_encoder(
      pre_inputs, sequence_length=pre_length, training=training)
    
    outputs = None
    predictions = None

    # When a target is provided, compute the decoder outputs for it.
    if labels is not None:
      outputs = self._decode_target(
          labels,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length,
          pre_encoder_outputs,
          pre_encoder_state,
          pre_encoder_sequence_length,
          step=step,
          training=training)

    # When not in training, also compute the model predictions.
    if not training:
      predictions = self._dynamic_decode(
          features,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length,
          pre_encoder_outputs,
          pre_encoder_state,
          pre_encoder_sequence_length)

    return outputs, predictions

  def _decode_target(self,
                     labels,
                     encoder_outputs,
                     encoder_state,
                     encoder_sequence_length,
                     pre_encoder_outputs,
                     pre_encoder_state,
                     pre_encoder_sequence_length,
                     step=None,
                     training=None):
    params = self.params
    target_inputs = self.labels_inputter(labels, training=training)
    input_fn = lambda ids: self.labels_inputter({"ids": ids}, training=training)

    sampling_probability = None
    if training:
      sampling_probability = decoder_util.get_sampling_probability(
          step,
          read_probability=params.get("scheduled_sampling_read_probability"),
          schedule_type=params.get("scheduled_sampling_type"),
          k=params.get("scheduled_sampling_k"))



    if self.version in [1,5]:
      initial_state = self.decoder.initial_state(
        memory=tf.concat([encoder_outputs, pre_encoder_outputs], axis=1),
        memory_sequence_length= [encoder_sequence_length,pre_encoder_sequence_length],
        initial_state= None)

    logits, _, attention = self.decoder(
        target_inputs,
        self.labels_inputter.get_length(labels),
        state=initial_state,
        input_fn=input_fn,
        sampling_probability=sampling_probability,
        training=training)
    outputs = dict(logits=logits, attention=attention)

    return outputs

  def _dynamic_decode(self, features, encoder_outputs, encoder_state, encoder_sequence_length, pre_encoder_outputs,
          pre_encoder_state,
          pre_encoder_sequence_length):

    params = self.params
    batch_size = tf.shape(tf.nest.flatten(encoder_outputs)[0])[0]
    start_ids = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
    beam_size = params.get("beam_width", 1)

    if beam_size > 1:
      # Tile encoder outputs to prepare for beam search.
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      encoder_sequence_length = tfa.seq2seq.tile_batch(encoder_sequence_length, beam_size)
      if encoder_state is not None:
        encoder_state = tfa.seq2seq.tile_batch(encoder_state, beam_size)
      
      pre_encoder_outputs = tfa.seq2seq.tile_batch(pre_encoder_outputs, beam_size)
      pre_encoder_sequence_length = tfa.seq2seq.tile_batch(pre_encoder_sequence_length, beam_size)
      if encoder_state is not None:
        pre_encoder_state = tfa.seq2seq.tile_batch(pre_encoder_state, beam_size)
      
    # Dynamically decodes from the encoder outputs.
    if self.version in [1,5]:
      initial_state = self.decoder.initial_state(
        memory=tf.concat([encoder_outputs, pre_encoder_outputs], axis=1),
        memory_sequence_length= [encoder_sequence_length, pre_encoder_sequence_length],
        initial_state= None)

    sampled_ids, sampled_length, log_probs, alignment, _ = self.decoder.dynamic_decode(
        self.labels_inputter,
        start_ids,
        initial_state=initial_state,
        decoding_strategy=decoding.DecodingStrategy.from_params(params),
        sampler=decoding.Sampler.from_params(params),
        maximum_iterations=params.get("maximum_decoding_length", 250),
        minimum_iterations=params.get("minimum_decoding_length", 0))
    target_tokens = self.labels_inputter.ids_to_tokens.lookup(tf.cast(sampled_ids, tf.int64))

    predictions = {
        "tokens": target_tokens,
        "length": sampled_length,
        "log_probs": log_probs
    }
    if alignment is not None:
      predictions["alignment"] = alignment

    # Maybe restrict the number of returned hypotheses based on the user parameter.
    num_hypotheses = params.get("num_hypotheses", 1)
    if num_hypotheses > 0:
      if num_hypotheses > beam_size:
        raise ValueError("n_best cannot be greater than beam_width")
      for key, value in six.iteritems(predictions):
        predictions[key] = value[:, :num_hypotheses]
    return predictions

  def compute_loss(self, outputs, labels, training=True):
    params = self.params
    if not isinstance(outputs, dict):
      outputs = dict(logits=outputs)
    logits = outputs["logits"]
    noisy_logits = outputs.get("noisy_logits")
    attention = outputs.get("attention")
    if noisy_logits is not None and params.get("contrastive_learning"):
      return losses.max_margin_loss(
          logits,
          labels["ids_out"],
          labels["length"],
          noisy_logits,
          labels["noisy_ids_out"],
          labels["noisy_length"],
          eta=params.get("max_margin_eta", 0.1))
    labels_lengths = self.labels_inputter.get_length(labels)
    loss, loss_normalizer, loss_token_normalizer = losses.cross_entropy_sequence_loss(
        logits,
        labels["ids_out"],
        labels_lengths,
        label_smoothing=params.get("label_smoothing", 0.0),
        average_in_time=params.get("average_loss_in_time", False),
        training=training)
    if training:
      gold_alignments = labels.get("alignment")
      guided_alignment_type = params.get("guided_alignment_type")
      if gold_alignments is not None and guided_alignment_type is not None:
        if attention is None:
          tf.get_logger().warning("This model did not return attention vectors; "
                                  "guided alignment will not be applied")
        else:
          loss += losses.guided_alignment_cost(
              attention[:, :-1],  # Do not constrain last timestep.
              gold_alignments,
              sequence_length=labels_lengths - 1,
              cost_type=guided_alignment_type,
              weight=params.get("guided_alignment_weight", 1))
    return loss, loss_normalizer, loss_token_normalizer

  def print_prediction(self, prediction, params=None, stream=None):
    if params is None:
      params = {}
    num_hypotheses = len(prediction["tokens"])
    for i in range(num_hypotheses):
      target_length = prediction["length"][i]
      tokens = prediction["tokens"][i][:target_length]
      sentence = self.labels_inputter.tokenizer.detokenize(tokens)
      score = None
      attention = None
      alignment_type = None
      if params.get("with_scores"):
        score = prediction["log_probs"][i]
      if params.get("with_alignments"):
        attention = prediction["alignment"][i][:target_length]
        alignment_type = params["with_alignments"]
      sentence = format_translation_output(
          sentence,
          score=score,
          attention=attention,
          alignment_type=alignment_type)
      print_bytes(tf.compat.as_bytes(sentence), stream=stream)

  def transfer_weights(self, new_model, new_optimizer=None, optimizer=None, ignore_weights=None):
    updated_variables = []

    def _map_variables(inputter_fn, vars_fn):
      mapping, _ = vocab.get_mapping(
          inputter_fn(self).vocabulary_file,
          inputter_fn(new_model).vocabulary_file)
      vars_a, vocab_axes = vars_fn(self)
      vars_b, _ = vars_fn(new_model)
      for var_a, var_b, vocab_axis in zip(vars_a, vars_b, vocab_axes):
        if new_optimizer is not None and optimizer is not None:
          variables = vocab.update_variable_and_slots(
              var_a,
              var_b,
              optimizer,
              new_optimizer,
              mapping,
              vocab_axis=vocab_axis)
        else:
          variables = [vocab.update_variable(var_a, var_b, mapping, vocab_axis=vocab_axis)]
        updated_variables.extend(variables)
      return vars_b

    _map_variables(
        lambda model: model.features_inputter,
        lambda model: ([model.features_inputter.embedding], [0]))
    _map_variables(
        lambda model: model.labels_inputter,
        lambda model: ([
            model.labels_inputter.embedding,
            model.decoder.output_layer.kernel,
            model.decoder.output_layer.bias], [0, 1, 0]))

    return super(SequenceToSequence, self).transfer_weights(
        new_model,
        new_optimizer=new_optimizer,
        optimizer=optimizer,
        ignore_weights=updated_variables)

class Priming_SequenceToSequence_v1(model.SequenceGenerator):
  
  def __init__(self,
               source_inputter,
               target_inputter,
               encoder,
               decoder,
               share_embeddings=EmbeddingsSharingLevel.NONE):
    
    if not isinstance(target_inputter, inputters.WordEmbedder):
      raise TypeError("Target inputter must be a WordEmbedder")
    if EmbeddingsSharingLevel.share_input_embeddings(share_embeddings):
      if isinstance(source_inputter, inputters.ParallelInputter):
        source_inputters = source_inputter.inputters
      else:
        source_inputters = [source_inputter]
      for inputter in source_inputters:
        if not isinstance(inputter, inputters.WordEmbedder):
          raise TypeError("Sharing embeddings requires all inputters to be a "
                          "WordEmbedder")

    examples_inputter = Priming_SequenceToSequenceInputter(
        source_inputter,
        target_inputter,
        share_parameters=EmbeddingsSharingLevel.share_input_embeddings(share_embeddings))
    super(Priming_SequenceToSequence_v1, self).__init__(examples_inputter)
    self.encoder = encoder
    self.decoder = decoder
    self.share_embeddings = share_embeddings

  def auto_config(self, num_replicas=1):
    config = super(Priming_SequenceToSequence_v1, self).auto_config(num_replicas=num_replicas)
    return merge_dict(config, {
        "params": {
            "beam_width": 4
        },
        "train": {
            "sample_buffer_size": -1,
            "max_step": 500000
        },
        "infer": {
            "batch_size": 32,
            "length_bucket_width": 5
        }
    })

  def initialize(self, data_config, params=None):
    super(Priming_SequenceToSequence_v1, self).initialize(data_config, params=params)
    if self.params.get("contrastive_learning"):
      # Use the simplest and most effective CL_one from the paper.
      # https://www.aclweb.org/anthology/P19-1623
      noiser = noise.WordNoiser(
          noises=[noise.WordOmission(1)],
          subword_token=self.params.get("decoding_subword_token", "￭"),
          is_spacer=self.params.get("decoding_subword_token_is_spacer"))
      self.labels_inputter.set_noise(noiser, in_place=False)

  def build(self, input_shape):
    super(Priming_SequenceToSequence_v1, self).build(input_shape)
    output_layer = None
    if EmbeddingsSharingLevel.share_target_embeddings(self.share_embeddings):
      output_layer = layers.Dense(
          self.labels_inputter.vocabulary_size,
          weight=self.labels_inputter.embedding,
          transpose=True)
    self.decoder.initialize(
        vocab_size=self.labels_inputter.vocabulary_size,
        output_layer=output_layer)

  def call(self, features, labels=None, training=None, step=None):
    # Encode the source.
    source_length = self.features_inputter.get_length(features)
    source_inputs = self.features_inputter(features, training=training)

    source_pre_length, source_hide_length = source_length
    source_pre_inputs, source_hide_inputs = source_inputs

    encoder_hide_outputs, encoder_hide_state, encoder_hide_sequence_length = self.encoder(
        source_hide_inputs, sequence_length=source_hide_length, training=training)

    encoder_pre_outputs, encoder_pre_state, encoder_pre_sequence_length = self.encoder(
        source_pre_inputs, sequence_length=source_pre_length, training=training)

    hide_outputs = None
    predictions = None
    pre_decoder_outputs = None
    hide_decoder_outputs = None
    # When a target is provided, compute the decoder outputs for it.
    if labels is not None:
      pre_outputs, pre_decoder_outputs = self._decode_target(
          labels,
          encoder_pre_outputs,
          encoder_pre_state,
          encoder_pre_sequence_length,
          step=step,
          training=training)

      hide_outputs, hide_decoder_outputs = self._decode_target(
          labels,
          encoder_hide_outputs,
          encoder_hide_state,
          encoder_hide_sequence_length,
          step=step,
          training=training)

    # When not in training, also compute the model predictions.
    if not training:
      predictions = self._dynamic_decode(
          features,
          encoder_pre_outputs,
          encoder_pre_state,
          encoder_pre_sequence_length)

    return hide_outputs, predictions, pre_decoder_outputs, hide_decoder_outputs

  def _decode_target(self,
                     labels,
                     encoder_outputs,
                     encoder_state,
                     encoder_sequence_length,
                     step=None,
                     training=None):
    params = self.params
    target_inputs = self.labels_inputter(labels, training=training)
    input_fn = lambda ids: self.labels_inputter({"ids": ids}, training=training)

    sampling_probability = None
    if training:
      sampling_probability = decoder_util.get_sampling_probability(
          step,
          read_probability=params.get("scheduled_sampling_read_probability"),
          schedule_type=params.get("scheduled_sampling_type"),
          k=params.get("scheduled_sampling_k"))

    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    logits, _, attention = self.decoder(
        target_inputs,
        self.labels_inputter.get_length(labels),
        state=initial_state,
        input_fn=input_fn,
        sampling_probability=sampling_probability,
        training=training)
    logits, decoder_outputs = logits
    outputs = dict(logits=logits, attention=attention)

    return outputs, decoder_outputs

  def _dynamic_decode(self, features, encoder_outputs, encoder_state, encoder_sequence_length):
    params = self.params
    batch_size = tf.shape(tf.nest.flatten(encoder_outputs)[0])[0]
    start_ids = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
    beam_size = params.get("beam_width", 1)

    if beam_size > 1:
      # Tile encoder outputs to prepare for beam search.
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      encoder_sequence_length = tfa.seq2seq.tile_batch(encoder_sequence_length, beam_size)
      if encoder_state is not None:
        encoder_state = tfa.seq2seq.tile_batch(encoder_state, beam_size)

    # Dynamically decodes from the encoder outputs.
    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    sampled_ids, sampled_length, log_probs, alignment, _ = self.decoder.dynamic_decode(
        self.labels_inputter,
        start_ids,
        initial_state=initial_state,
        decoding_strategy=decoding.DecodingStrategy.from_params(params),
        sampler=decoding.Sampler.from_params(params),
        maximum_iterations=params.get("maximum_decoding_length", 250),
        minimum_iterations=params.get("minimum_decoding_length", 0))
    target_tokens = self.labels_inputter.ids_to_tokens.lookup(tf.cast(sampled_ids, tf.int64))

    # Maybe replace unknown targets by the source tokens with the highest attention weight.
    if params.get("replace_unknown_target", False):
      if alignment is None:
        raise TypeError("replace_unknown_target is not compatible with decoders "
                        "that don't return alignment history")
      if not isinstance(self.features_inputter, inputters.WordEmbedder):
        raise TypeError("replace_unknown_target is only defined when the source "
                        "inputter is a WordEmbedder")
      source_tokens = features["tokens"]
      if beam_size > 1:
        source_tokens = tfa.seq2seq.tile_batch(source_tokens, beam_size)
      # Merge batch and beam dimensions.
      original_shape = tf.shape(target_tokens)
      target_tokens = tf.reshape(target_tokens, [-1, original_shape[-1]])
      align_shape = shape_list(alignment)
      attention = tf.reshape(
          alignment, [align_shape[0] * align_shape[1], align_shape[2], align_shape[3]])
      # We don't have attention for </s> but ensure that the attention time dimension matches
      # the tokens time dimension.
      attention = reducer.align_in_time(attention, tf.shape(target_tokens)[1])
      replaced_target_tokens = replace_unknown_target(target_tokens, source_tokens, attention)
      target_tokens = tf.reshape(replaced_target_tokens, original_shape)

    # Maybe add noise to the predictions.
    decoding_noise = params.get("decoding_noise")
    if decoding_noise:
      target_tokens, sampled_length = _add_noise(
          target_tokens,
          sampled_length,
          decoding_noise,
          params.get("decoding_subword_token", "￭"),
          params.get("decoding_subword_token_is_spacer"))
      alignment = None  # Invalidate alignments.

    predictions = {
        "tokens": target_tokens,
        "length": sampled_length,
        "log_probs": log_probs
    }
    if alignment is not None:
      predictions["alignment"] = alignment

    # Maybe restrict the number of returned hypotheses based on the user parameter.
    num_hypotheses = params.get("num_hypotheses", 1)
    if num_hypotheses > 0:
      if num_hypotheses > beam_size:
        raise ValueError("n_best cannot be greater than beam_width")
      for key, value in six.iteritems(predictions):
        predictions[key] = value[:, :num_hypotheses]
    return predictions

  def compute_loss(self, outputs, labels, training=True):
    params = self.params
    if not isinstance(outputs, dict):
      outputs = dict(logits=outputs)
    logits = outputs["logits"]
    noisy_logits = outputs.get("noisy_logits")
    attention = outputs.get("attention")
    if noisy_logits is not None and params.get("contrastive_learning"):
      return losses.max_margin_loss(
          logits,
          labels["ids_out"],
          labels["length"],
          noisy_logits,
          labels["noisy_ids_out"],
          labels["noisy_length"],
          eta=params.get("max_margin_eta", 0.1))
    labels_lengths = self.labels_inputter.get_length(labels)
    loss, loss_normalizer, loss_token_normalizer = losses.cross_entropy_sequence_loss(
        logits,
        labels["ids_out"],
        labels_lengths,
        label_smoothing=params.get("label_smoothing", 0.0),
        average_in_time=params.get("average_loss_in_time", False),
        training=training)
    if training:
      gold_alignments = labels.get("alignment")
      guided_alignment_type = params.get("guided_alignment_type")
      if gold_alignments is not None and guided_alignment_type is not None:
        if attention is None:
          tf.get_logger().warning("This model did not return attention vectors; "
                                  "guided alignment will not be applied")
        else:
          loss += losses.guided_alignment_cost(
              attention[:, :-1],  # Do not constrain last timestep.
              gold_alignments,
              sequence_length=labels_lengths - 1,
              cost_type=guided_alignment_type,
              weight=params.get("guided_alignment_weight", 1))
    return loss, loss_normalizer, loss_token_normalizer

  def print_prediction(self, prediction, params=None, stream=None):
    if params is None:
      params = {}
    num_hypotheses = len(prediction["tokens"])
    for i in range(num_hypotheses):
      target_length = prediction["length"][i]
      tokens = prediction["tokens"][i][:target_length]
      sentence = self.labels_inputter.tokenizer.detokenize(tokens)
      score = None
      attention = None
      alignment_type = None
      if params.get("with_scores"):
        score = prediction["log_probs"][i]
      if params.get("with_alignments"):
        attention = prediction["alignment"][i][:target_length]
        alignment_type = params["with_alignments"]
      sentence = format_translation_output(
          sentence,
          score=score,
          attention=attention,
          alignment_type=alignment_type)
      print_bytes(tf.compat.as_bytes(sentence), stream=stream)

  def transfer_weights(self, new_model, new_optimizer=None, optimizer=None, ignore_weights=None):
    updated_variables = []

    def _map_variables(inputter_fn, vars_fn):
      mapping, _ = vocab.get_mapping(
          inputter_fn(self).vocabulary_file,
          inputter_fn(new_model).vocabulary_file)
      vars_a, vocab_axes = vars_fn(self)
      vars_b, _ = vars_fn(new_model)
      for var_a, var_b, vocab_axis in zip(vars_a, vars_b, vocab_axes):
        if new_optimizer is not None and optimizer is not None:
          variables = vocab.update_variable_and_slots(
              var_a,
              var_b,
              optimizer,
              new_optimizer,
              mapping,
              vocab_axis=vocab_axis)
        else:
          variables = [vocab.update_variable(var_a, var_b, mapping, vocab_axis=vocab_axis)]
        updated_variables.extend(variables)
      return vars_b

    _map_variables(
        lambda model: model.features_inputter,
        lambda model: ([model.features_inputter.embedding], [0]))
    _map_variables(
        lambda model: model.labels_inputter,
        lambda model: ([
            model.labels_inputter.embedding,
            model.decoder.output_layer.kernel,
            model.decoder.output_layer.bias], [0, 1, 0]))

    return super(SequenceToSequence, self).transfer_weights(
        new_model,
        new_optimizer=new_optimizer,
        optimizer=optimizer,
        ignore_weights=updated_variables)
  
class Multi_domain_SequenceToSequence_sparse(model.SequenceGenerator):

  """A sequence to sequence model."""

  def __init__(self,
               source_inputter,
               target_inputter,
               encoder,
               decoder,
               version=1,
               num_domains=6,
               dropout_rate=0.2,
               num_domain_unit_group=12,
               unit_group_size=16,
               num_shared_units=480,
               num_units=512,
               share_embeddings=EmbeddingsSharingLevel.NONE):

    if not isinstance(target_inputter, inputters.WordEmbedder) and not isinstance(target_inputter, My_inputter):
      raise TypeError("Target inputter must be a WordEmbedder or My_inputter")
    if EmbeddingsSharingLevel.share_input_embeddings(share_embeddings):
      if isinstance(source_inputter, inputters.ParallelInputter):
        source_inputters = source_inputter.inputters
      else:
        source_inputters = [source_inputter]
      for inputter in source_inputters:
        if not isinstance(inputter, inputters.WordEmbedder):
          raise TypeError("Sharing embeddings requires all inputters to be a "
                          "WordEmbedder")

    examples_inputter = Multi_domain_SequenceToSequenceInputter(
        source_inputter,
        target_inputter,
        share_parameters=EmbeddingsSharingLevel.share_input_embeddings(share_embeddings))
    super(Multi_domain_SequenceToSequence_sparse, self).__init__(examples_inputter)
    self.encoder = encoder
    self.decoder = decoder
    self.share_embeddings = share_embeddings
    self.num_domains = num_domains
    self.num_domain_unit_group=num_domain_unit_group
    self.unit_group_size = unit_group_size
    self.dropout_rate = dropout_rate
    self.num_units = num_units
    self.num_shared_units = num_shared_units
    assert num_shared_units + unit_group_size * num_domain_unit_group == num_units
    self.version = version
  def auto_config(self, num_replicas=1):
    config = super(Multi_domain_SequenceToSequence_sparse, self).auto_config(num_replicas=num_replicas)
    return merge_dict(config, {
        "params": {
            "beam_width": 5
        },
        "train": {
            "sample_buffer_size": -1,
            "max_step": 200000
        },
        "infer": {
            "batch_size": 32,
            "length_bucket_width": 5
        }
    })

  def initialize(self, data_config, params=None):
    super(Multi_domain_SequenceToSequence_sparse, self).initialize(data_config, params=params)
    if self.params.get("contrastive_learning"):
      noiser = noise.WordNoiser(
          noises=[noise.WordOmission(1)],
          subword_token=self.params.get("decoding_subword_token", "￭"),
          is_spacer=self.params.get("decoding_subword_token_is_spacer"))
      self.labels_inputter.set_noise(noiser, in_place=False)

  def build(self, input_shape):
    super(Multi_domain_SequenceToSequence_sparse, self).build(input_shape)
    output_layer = None
    if EmbeddingsSharingLevel.share_target_embeddings(self.share_embeddings):
      output_layer = layers.Dense(
          self.labels_inputter.vocabulary_size,
          weight=self.labels_inputter.embedding,
          transpose=True)
    self.decoder.initialize(
        vocab_size=self.labels_inputter.vocabulary_size,
        output_layer=output_layer)

    self.domain_one_logits = self.add_weight("domain_one_logits", shape=[self.num_domains, self.num_domain_unit_group])
    self.domain_zero_logits = self.add_weight("domain_zero_logits", shape=[self.num_domains, self.num_domain_unit_group])

  def call(self, features, gumbel_temperature=0.5, labels=None, training=None, step=None, internal_node_printing=False, return_domain_classification_logits=False, return_embedding=False, adapter_activate=True, inference=True):
    # Encode the source.
    assert isinstance(self.features_inputter, My_inputter)
    assert isinstance(self.labels_inputter, My_inputter)    
    source_length = self.features_inputter.get_length(features)
    source_inputs = self.features_inputter(features, training=training)
    
    domain = features["domain"][0]
    domain_one_logits = tf.nn.embedding_lookup(self.domain_one_logits,domain)
    domain_zero_logits = tf.nn.embedding_lookup(self.domain_zero_logits,domain)
    
    unit_selection_logits = tf.transpose(tf.concat([tf.expand_dims(domain_zero_logits,0),tf.expand_dims(domain_one_logits,0)],0))
    #tf.print("domain_one_logits",domain_one_logits,summarize=-1)
    #tf.print("domain_zero_logits",domain_zero_logits,summarize=-1)
    #tf.print("unit_selection_logits",unit_selection_logits,summarize=-1)
    #domain_dropout_mask = 
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    gumbel_dist = tfd.Gumbel(loc=0.,scale=1.)
    if self.version == 1:
      print("version: ",self.version)
      gumbel_one = gumbel_dist.sample([self.num_domain_unit_group])
      gumbel_zero = gumbel_dist.sample([self.num_domain_unit_group])

      prob_one = tf.math.exp((domain_one_logits+gumbel_one)/gumbel_temperature)
      prob_zero = tf.math.exp((domain_zero_logits+gumbel_zero)/gumbel_temperature)
      #tf.print("prob_one",prob_one,summarize=-1)
      #tf.print("prob_zero",prob_zero,summarize=-1)
      total_prob = prob_one + prob_zero
      
      #tf.print("total_prob",total_prob,summarize=-1)

      prob_one = prob_one/total_prob
      prob_zero = prob_zero/total_prob

      #tf.print("prob_one_f",prob_one,summarize=-1)
      #tf.print("prob_zero_f",prob_zero,summarize=-1)

      KL_term = None
      dropout_rate = self.dropout_rate
      print("dropout_rate",dropout_rate)
      if training:
        KL_term = - tf.reduce_sum((1-dropout_rate) * tf.math.log(prob_one) + dropout_rate * tf.math.log(prob_zero))
      
      if training:
        domain_dropout_mask = tf.concat([tf.ones(self.num_shared_units),tf.cast(tf.reshape(tf.transpose(tf.tile(tf.expand_dims(prob_one,0),[self.unit_group_size,1])),[-1]),tf.float32)],-1)
        #tf.print("domain_dropout_mask",domain_dropout_mask,summarize=-1)
        #tf.print("gumbel_temperature",gumbel_temperature,summarize=-1)
      else:
        domain_dropout_mask = tf.concat([tf.ones(self.num_shared_units),tf.cast(tf.reshape(tf.transpose(tf.tile(tf.expand_dims(tf.math.argmax(unit_selection_logits,1),0),[self.unit_group_size,1])),[-1]),tf.float32)],-1)

    elif self.version == 2:
      print("version: ",self.version)
      gumbel_one = gumbel_dist.sample([tf.shape(source_inputs)[0],1,self.num_domain_unit_group])
      gumbel_zero = gumbel_dist.sample([tf.shape(source_inputs)[0],1,self.num_domain_unit_group])

      domain_one_logits = tf.tile(tf.expand_dims(tf.expand_dims(domain_one_logits,0),0),[tf.shape(source_inputs)[0],1,1])
      #tf.print(domain_one_logits,summarize=-1)
      domain_zero_logits = tf.tile(tf.expand_dims(tf.expand_dims(domain_zero_logits,0),0),[tf.shape(source_inputs)[0],1,1])
      #tf.print(domain_one_logits,summarize=-1)

      prob_one = tf.math.exp((domain_one_logits+gumbel_one)/gumbel_temperature)
      prob_zero = tf.math.exp((domain_zero_logits+gumbel_zero)/gumbel_temperature)

      total_prob = prob_one + prob_zero
      
      prob_one = prob_one/total_prob
      prob_zero = prob_zero/total_prob

      KL_term = None
      dropout_rate = self.dropout_rate
      print("dropout_rate",dropout_rate)
      if training:
        KL_term = - tf.reduce_sum(tf.reduce_mean((1-dropout_rate) * tf.math.log(prob_one) + dropout_rate * tf.math.log(prob_zero),0))
      
      if training:
        domain_dropout_mask = tf.concat([tf.ones([tf.shape(source_inputs)[0],1,self.num_shared_units]), tf.repeat(prob_one,self.unit_group_size,-1)],-1)
        #tf.print("domain_dropout_mask",domain_dropout_mask,summarize=-1)
        #tf.print("gumbel_temperature",gumbel_temperature,summarize=-1)
      else:
        domain_dropout_mask = tf.concat([tf.ones(self.num_shared_units), tf.cast(tf.repeat(tf.math.argmax(unit_selection_logits,1),self.unit_group_size,-1),tf.float32)],-1)

    encoder_outputs, encoder_state, encoder_sequence_length = self.encoder(
        [source_inputs, features["domain"], domain_dropout_mask], sequence_length=source_length, training=training)
    
    outputs = None
    predictions = None

    # When a target is provided, compute the decoder outputs for it.
    
    if labels is not None:
      outputs = self._decode_target(
          labels,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length,
          domain_dropout_mask,
          step=step,
          training=training)

    # When not in training, also compute the model predictions.
    if not training:
      predictions = self._dynamic_decode(
          features,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length,
          domain_dropout_mask)
    
    return outputs, predictions, KL_term
  
  def _decode_target(self,
                     labels,
                     encoder_outputs,
                     encoder_state,
                     encoder_sequence_length,
                     domain_dropout_mask,
                     step=None,
                     training=None,
                     internal_node_printing=False):
    params = self.params
    target_inputs = self.labels_inputter(labels, training=training)
    input_fn = lambda ids: [self.labels_inputter({"ids": ids}, training=training), labels["domain"]]

    sampling_probability = None
    if training:
      sampling_probability = decoder_util.get_sampling_probability(
          step,
          read_probability=params.get("scheduled_sampling_read_probability"),
          schedule_type=params.get("scheduled_sampling_type"),
          k=params.get("scheduled_sampling_k"))

    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    logits, _, attention = self.decoder(
        [target_inputs, labels["domain"], domain_dropout_mask],
        self.labels_inputter.get_length(labels),
        state=initial_state,
        input_fn=input_fn,
        sampling_probability=sampling_probability,
        training=training)
    outputs = dict(logits=logits, attention=attention)

    return outputs
  
  def _dynamic_decode(self, features, encoder_outputs, encoder_state, encoder_sequence_length, domain_dropout_mask):
    params = self.params
    batch_size = tf.shape(tf.nest.flatten(encoder_outputs)[0])[0]
    start_ids = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
    beam_size = params.get("beam_width", 1)

    if beam_size > 1:
      # Tile encoder outputs to prepare for beam search.
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      encoder_sequence_length = tfa.seq2seq.tile_batch(encoder_sequence_length, beam_size)
      if encoder_state is not None:
        encoder_state = tfa.seq2seq.tile_batch(encoder_state, beam_size)

    # Dynamically decodes from the encoder outputs.
    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    sampled_ids, sampled_length, log_probs, alignment, _ = self.decoder.dynamic_decode(
        lambda ids: [self.labels_inputter({"ids": ids}), features["domain"], domain_dropout_mask],
        start_ids,
        initial_state=initial_state,
        decoding_strategy=decoding.DecodingStrategy.from_params(params),
        sampler=decoding.Sampler.from_params(params),
        maximum_iterations=params.get("maximum_decoding_length", 250),
        minimum_iterations=params.get("minimum_decoding_length", 0))
    target_tokens = self.labels_inputter.ids_to_tokens.lookup(tf.cast(sampled_ids, tf.int64))

    predictions = {
        "tokens": target_tokens,
        "length": sampled_length,
        "log_probs": log_probs
    }
    
    # Maybe restrict the number of returned hypotheses based on the user parameter.
    num_hypotheses = params.get("num_hypotheses", 1)
    if num_hypotheses > 0:
      if num_hypotheses > beam_size:
        raise ValueError("n_best cannot be greater than beam_width")
      for key, value in six.iteritems(predictions):
        predictions[key] = value[:, :num_hypotheses]
    return predictions

  def compute_loss(self, outputs, labels, training=True):
    params = self.params
    if not isinstance(outputs, dict):
      outputs = dict(logits=outputs)
    logits = outputs["logits"]
    labels_lengths = self.labels_inputter.get_length(labels)

    loss, loss_normalizer, loss_token_normalizer = losses.cross_entropy_sequence_loss(
        logits,
        labels["ids_out"],
        labels_lengths,
        label_smoothing=params.get("label_smoothing", 0.0),
        average_in_time=params.get("average_loss_in_time", False),
        training=training)
    
    return loss, loss_normalizer, loss_token_normalizer
  
  def print_prediction(self, prediction, params=None, stream=None):
    if params is None:
      params = {}
    num_hypotheses = len(prediction["tokens"])
    for i in range(num_hypotheses):
      target_length = prediction["length"][i]
      tokens = prediction["tokens"][i][:target_length]
      sentence = self.labels_inputter.tokenizer.detokenize(tokens)
      score = None
      attention = None
      alignment_type = None
      if params.get("with_scores"):
        score = prediction["log_probs"][i]
      if params.get("with_alignments"):
        attention = prediction["alignment"][i][:target_length]
        alignment_type = params["with_alignments"]
      sentence = format_translation_output(
          sentence,
          score=score,
          attention=attention,
          alignment_type=alignment_type)
      print_bytes(tf.compat.as_bytes(sentence), stream=stream)

  def transfer_weights(self, new_model, new_optimizer=None, optimizer=None, ignore_weights=None):
    updated_variables = []

    def _map_variables(inputter_fn, vars_fn):
      mapping, _ = vocab.get_mapping(
          inputter_fn(self).vocabulary_file,
          inputter_fn(new_model).vocabulary_file)
      vars_a, vocab_axes = vars_fn(self)
      vars_b, _ = vars_fn(new_model)
      for var_a, var_b, vocab_axis in zip(vars_a, vars_b, vocab_axes):
        if new_optimizer is not None and optimizer is not None:
          variables = vocab.update_variable_and_slots(
              var_a,
              var_b,
              optimizer,
              new_optimizer,
              mapping,
              vocab_axis=vocab_axis)
        else:
          variables = [vocab.update_variable(var_a, var_b, mapping, vocab_axis=vocab_axis)]
        updated_variables.extend(variables)
      return vars_b

def _shift_target_sequence(labels, prefix=""):
  labels_ids = labels["%sids" % prefix]
  bos = tf.constant([constants.START_OF_SENTENCE_ID], dtype=labels_ids.dtype)
  eos = tf.constant([constants.END_OF_SENTENCE_ID], dtype=labels_ids.dtype)
  labels["%sids" % prefix] = tf.concat([bos, labels_ids], axis=0)
  labels["%sids_out" % prefix] = tf.concat([labels_ids, eos], axis=0)
  labels["%slength" % prefix] += 1

def align_tokens_from_attention(tokens, attention):
  """Returns aligned tokens from the attention.

  Args:
    tokens: The tokens on which the attention is applied as a string
      ``tf.Tensor`` of shape :math:`[B, T_s]`.
    attention: The attention vector of shape :math:`[B, T_t, T_s]`.

  Returns:
    The aligned tokens as a string ``tf.Tensor`` of shape :math:`[B, T_t]`.
  """
  alignment = tf.argmax(attention, axis=-1, output_type=tf.int32)
  return tf.gather(tokens, alignment, axis=1, batch_dims=1)

def replace_unknown_target(target_tokens,
                           source_tokens,
                           attention,
                           unknown_token=constants.UNKNOWN_TOKEN):
  """Replaces all target unknown tokens by the source token with the highest
  attention.

  Args:
    target_tokens: A a string ``tf.Tensor`` of shape :math:`[B, T_t]`.
    source_tokens: A a string ``tf.Tensor`` of shape :math:`[B, T_s]`.
    attention: The attention vector of shape :math:`[B, T_t, T_s]`.
    unknown_token: The target token to replace.

  Returns:
    A string ``tf.Tensor`` with the same shape and type as :obj:`target_tokens`
    but will all instances of :obj:`unknown_token` replaced by the aligned source
    token.
  """
  aligned_source_tokens = align_tokens_from_attention(source_tokens, attention)
  return tf.where(
      tf.equal(target_tokens, unknown_token),
      x=aligned_source_tokens,
      y=target_tokens)

def _add_noise(tokens, lengths, params, subword_token, is_spacer=None):
  if not isinstance(params, list):
    raise ValueError("Expected a list of noise modules")
  noises = []
  for module in params:
    noise_type, args = six.next(six.iteritems(module))
    if not isinstance(args, list):
      args = [args]
    noise_type = noise_type.lower()
    if noise_type == "dropout":
      noise_class = noise.WordDropout
    elif noise_type == "replacement":
      noise_class = noise.WordReplacement
    elif noise_type == "permutation":
      noise_class = noise.WordPermutation
    else:
      raise ValueError("Invalid noise type: %s" % noise_type)
    noises.append(noise_class(*args))
  noiser = noise.WordNoiser(noises=noises, subword_token=subword_token, is_spacer=is_spacer)
  return noiser(tokens, lengths, keep_shape=True)

class Multi_domain_SequenceToSequence_TopK_sparse(model.SequenceGenerator):

  """A sequence to sequence model."""

  def __init__(self,
               source_inputter,
               target_inputter,
               encoder,
               decoder,
               version=1,
               num_domains=6,
               dropout_rate=0.2,
               num_domain_unit_group=12,
               unit_group_size=16,
               num_shared_units=480,
               num_units=512,
               share_embeddings=EmbeddingsSharingLevel.NONE):

    if not isinstance(target_inputter, inputters.WordEmbedder) and not isinstance(target_inputter, My_inputter):
      raise TypeError("Target inputter must be a WordEmbedder or My_inputter")
    if EmbeddingsSharingLevel.share_input_embeddings(share_embeddings):
      if isinstance(source_inputter, inputters.ParallelInputter):
        source_inputters = source_inputter.inputters
      else:
        source_inputters = [source_inputter]
      for inputter in source_inputters:
        if not isinstance(inputter, inputters.WordEmbedder):
          raise TypeError("Sharing embeddings requires all inputters to be a "
                          "WordEmbedder")

    examples_inputter = Multi_domain_SequenceToSequenceInputter(
        source_inputter,
        target_inputter,
        share_parameters=EmbeddingsSharingLevel.share_input_embeddings(share_embeddings))
    super(Multi_domain_SequenceToSequence_TopK_sparse, self).__init__(examples_inputter)
    self.encoder = encoder
    self.decoder = decoder
    self.share_embeddings = share_embeddings
    self.num_domains = num_domains
    self.num_domain_unit_group=num_domain_unit_group
    self.unit_group_size = unit_group_size
    self.dropout_rate = dropout_rate
    self.num_units = num_units
    self.num_shared_units = num_shared_units
    assert num_shared_units + unit_group_size * num_domain_unit_group == num_units
    self.version = version
  
  def auto_config(self, num_replicas=1):
    config = super(Multi_domain_SequenceToSequence_TopK_sparse, self).auto_config(num_replicas=num_replicas)
    return merge_dict(config, {
        "params": {
            "beam_width": 5
        },
        "train": {
            "sample_buffer_size": -1,
            "max_step": 200000
        },
        "infer": {
            "batch_size": 32,
            "length_bucket_width": 5
        }
    })

  def initialize(self, data_config, params=None):
    super(Multi_domain_SequenceToSequence_TopK_sparse, self).initialize(data_config, params=params)
    if self.params.get("contrastive_learning"):
      noiser = noise.WordNoiser(
          noises=[noise.WordOmission(1)],
          subword_token=self.params.get("decoding_subword_token", "￭"),
          is_spacer=self.params.get("decoding_subword_token_is_spacer"))
      self.labels_inputter.set_noise(noiser, in_place=False)

  def create_variables(self, optimizer=None):
    """Creates the model variables by running it once.

    Args:
      optimizer: If set, also create the optimizer variables.
    """
    if self.built:
      return

    # Create input features from the input signatures. We remove the leading
    # batch dimension as sometimes assumed by make_features methods and set
    # unspecified dimensions to 1.
    features = tf.nest.map_structure(
        lambda spec: tf.fill(
            [dim or 1 for dim in spec.shape.as_list()[1:]],
            tf.constant("" if spec.dtype is tf.string else 1, dtype=spec.dtype)),
        self.examples_inputter.input_signature())
    features = self.examples_inputter.make_features(features=features)

    # Add the batch dimension back before calling the model.
    features, labels = tf.nest.map_structure(lambda x: tf.expand_dims(x, 0), features)
    _ = self(features, domain_dropout_mask=tf.ones(self.num_units), labels=labels, training=True, step=0)

    if optimizer is not None:
      _ = optimizer.iterations
      optimizer._create_hypers()  # pylint: disable=protected-access
      optimizer._create_slots(self.trainable_variables)  # pylint: disable=protected-access

  def build(self, input_shape):
    super(Multi_domain_SequenceToSequence_TopK_sparse, self).build(input_shape)
    output_layer = None
    if EmbeddingsSharingLevel.share_target_embeddings(self.share_embeddings):
      output_layer = layers.Dense(
          self.labels_inputter.vocabulary_size,
          weight=self.labels_inputter.embedding,
          transpose=True)
    self.decoder.initialize(
        vocab_size=self.labels_inputter.vocabulary_size,
        output_layer=output_layer)

    self.latent_group_allocation_logit = self.add_weight("latent_group_allocation_logit", initializer=tf.keras.initializers.Zeros(), shape=[self.num_domains, self.num_domain_unit_group])
    self.soft_mask = tf.ones(self.num_units)

  def call(self, features, domain_dropout_mask=None, labels=None, training=None, step=None, internal_node_printing=False, return_domain_classification_logits=False, return_embedding=False, adapter_activate=True, inference=True):
    # Encode the source.
    assert isinstance(self.features_inputter, My_inputter)
    assert isinstance(self.labels_inputter, My_inputter)    
    assert domain_dropout_mask != None
    
    source_length = self.features_inputter.get_length(features)
    source_inputs = self.features_inputter(features, training=training)

    encoder_outputs, encoder_state, encoder_sequence_length = self.encoder(
        [source_inputs, features["domain"], domain_dropout_mask], sequence_length=source_length, training=training)
    
    outputs = None
    predictions = None

    # When a target is provided, compute the decoder outputs for it.
    
    if labels is not None:
      outputs = self._decode_target(
          labels,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length,
          domain_dropout_mask,
          step=step,
          training=training)

    # When not in training, also compute the model predictions.
    if not training:
      predictions = self._dynamic_decode(
          features,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length,
          domain_dropout_mask)
    
    return outputs, predictions
  
  def _decode_target(self,
                     labels,
                     encoder_outputs,
                     encoder_state,
                     encoder_sequence_length,
                     domain_dropout_mask,
                     step=None,
                     training=None,
                     internal_node_printing=False):
    params = self.params
    target_inputs = self.labels_inputter(labels, training=training)
    input_fn = lambda ids: [self.labels_inputter({"ids": ids}, training=training), labels["domain"]]

    sampling_probability = None
    if training:
      sampling_probability = decoder_util.get_sampling_probability(
          step,
          read_probability=params.get("scheduled_sampling_read_probability"),
          schedule_type=params.get("scheduled_sampling_type"),
          k=params.get("scheduled_sampling_k"))

    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    logits, _, attention = self.decoder(
        [target_inputs, labels["domain"], domain_dropout_mask],
        self.labels_inputter.get_length(labels),
        state=initial_state,
        input_fn=input_fn,
        sampling_probability=sampling_probability,
        training=training)
    outputs = dict(logits=logits, attention=attention)

    return outputs
  
  def _dynamic_decode(self, features, encoder_outputs, encoder_state, encoder_sequence_length, domain_dropout_mask):
    params = self.params
    batch_size = tf.shape(tf.nest.flatten(encoder_outputs)[0])[0]
    start_ids = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
    beam_size = params.get("beam_width", 1)

    if beam_size > 1:
      # Tile encoder outputs to prepare for beam search.
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      encoder_sequence_length = tfa.seq2seq.tile_batch(encoder_sequence_length, beam_size)
      if encoder_state is not None:
        encoder_state = tfa.seq2seq.tile_batch(encoder_state, beam_size)

    # Dynamically decodes from the encoder outputs.
    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    sampled_ids, sampled_length, log_probs, alignment, _ = self.decoder.dynamic_decode(
        lambda ids: [self.labels_inputter({"ids": ids}), features["domain"], domain_dropout_mask],
        start_ids,
        initial_state=initial_state,
        decoding_strategy=decoding.DecodingStrategy.from_params(params),
        sampler=decoding.Sampler.from_params(params),
        maximum_iterations=params.get("maximum_decoding_length", 250),
        minimum_iterations=params.get("minimum_decoding_length", 0))
    target_tokens = self.labels_inputter.ids_to_tokens.lookup(tf.cast(sampled_ids, tf.int64))

    predictions = {
        "tokens": target_tokens,
        "length": sampled_length,
        "log_probs": log_probs
    }
    
    # Maybe restrict the number of returned hypotheses based on the user parameter.
    num_hypotheses = params.get("num_hypotheses", 1)
    if num_hypotheses > 0:
      if num_hypotheses > beam_size:
        raise ValueError("n_best cannot be greater than beam_width")
      for key, value in six.iteritems(predictions):
        predictions[key] = value[:, :num_hypotheses]
    return predictions

  def compute_loss(self, outputs, labels, training=True):
    params = self.params
    if not isinstance(outputs, dict):
      outputs = dict(logits=outputs)
    logits = outputs["logits"]
    labels_lengths = self.labels_inputter.get_length(labels)

    loss, loss_normalizer, loss_token_normalizer = losses.cross_entropy_sequence_loss(
        logits,
        labels["ids_out"],
        labels_lengths,
        label_smoothing=params.get("label_smoothing", 0.0),
        average_in_time=params.get("average_loss_in_time", False),
        training=training)
    
    return loss, loss_normalizer, loss_token_normalizer
  
  def print_prediction(self, prediction, params=None, stream=None):
    if params is None:
      params = {}
    num_hypotheses = len(prediction["tokens"])
    for i in range(num_hypotheses):
      target_length = prediction["length"][i]
      tokens = prediction["tokens"][i][:target_length]
      sentence = self.labels_inputter.tokenizer.detokenize(tokens)
      score = None
      attention = None
      alignment_type = None
      if params.get("with_scores"):
        score = prediction["log_probs"][i]
      if params.get("with_alignments"):
        attention = prediction["alignment"][i][:target_length]
        alignment_type = params["with_alignments"]
      sentence = format_translation_output(
          sentence,
          score=score,
          attention=attention,
          alignment_type=alignment_type)
      print_bytes(tf.compat.as_bytes(sentence), stream=stream)

  def transfer_weights(self, new_model, new_optimizer=None, optimizer=None, ignore_weights=None):
    updated_variables = []

    def _map_variables(inputter_fn, vars_fn):
      mapping, _ = vocab.get_mapping(
          inputter_fn(self).vocabulary_file,
          inputter_fn(new_model).vocabulary_file)
      vars_a, vocab_axes = vars_fn(self)
      vars_b, _ = vars_fn(new_model)
      for var_a, var_b, vocab_axis in zip(vars_a, vars_b, vocab_axes):
        if new_optimizer is not None and optimizer is not None:
          variables = vocab.update_variable_and_slots(
              var_a,
              var_b,
              optimizer,
              new_optimizer,
              mapping,
              vocab_axis=vocab_axis)
        else:
          variables = [vocab.update_variable(var_a, var_b, mapping, vocab_axis=vocab_axis)]
        updated_variables.extend(variables)
      return vars_b

class Multi_domain_SequenceToSequence_TopK_sparse_multi_layer(model.SequenceGenerator):

  """A sequence to sequence model."""

  def __init__(self,
               source_inputter,
               target_inputter,
               encoder,
               decoder,
               version=1,
               num_domains=6,
               dropout_rate=0.2,
               num_domain_unit_group=12,
               unit_group_size=16,
               num_shared_units=480,
               num_units=512,
               share_embeddings=EmbeddingsSharingLevel.NONE):

    if not isinstance(target_inputter, inputters.WordEmbedder) and not isinstance(target_inputter, My_inputter):
      raise TypeError("Target inputter must be a WordEmbedder or My_inputter")
    if EmbeddingsSharingLevel.share_input_embeddings(share_embeddings):
      if isinstance(source_inputter, inputters.ParallelInputter):
        source_inputters = source_inputter.inputters
      else:
        source_inputters = [source_inputter]
      for inputter in source_inputters:
        if not isinstance(inputter, inputters.WordEmbedder):
          raise TypeError("Sharing embeddings requires all inputters to be a "
                          "WordEmbedder")

    examples_inputter = Multi_domain_SequenceToSequenceInputter(
        source_inputter,
        target_inputter,
        share_parameters=EmbeddingsSharingLevel.share_input_embeddings(share_embeddings))
    super(Multi_domain_SequenceToSequence_TopK_sparse_multi_layer, self).__init__(examples_inputter)
    self.encoder = encoder
    self.decoder = decoder
    self.share_embeddings = share_embeddings
    self.num_domains = num_domains
    self.num_domain_unit_group=num_domain_unit_group
    self.unit_group_size = unit_group_size
    self.dropout_rate = dropout_rate
    self.num_units = num_units
    self.num_shared_units = num_shared_units
    assert num_shared_units + unit_group_size * num_domain_unit_group == num_units
    self.version = version
    self.mask_num = encoder.num_layers + decoder.num_layers + 1
  
  def auto_config(self, num_replicas=1):
    config = super(Multi_domain_SequenceToSequence_TopK_sparse_multi_layer, self).auto_config(num_replicas=num_replicas)
    return merge_dict(config, {
        "params": {
            "beam_width": 5
        },
        "train": {
            "sample_buffer_size": -1,
            "max_step": 200000
        },
        "infer": {
            "batch_size": 32,
            "length_bucket_width": 5
        }
    })

  def initialize(self, data_config, params=None):
    super(Multi_domain_SequenceToSequence_TopK_sparse_multi_layer, self).initialize(data_config, params=params)
    if self.params.get("contrastive_learning"):
      noiser = noise.WordNoiser(
          noises=[noise.WordOmission(1)],
          subword_token=self.params.get("decoding_subword_token", "￭"),
          is_spacer=self.params.get("decoding_subword_token_is_spacer"))
      self.labels_inputter.set_noise(noiser, in_place=False)

  def create_variables(self, optimizer=None):
    """Creates the model variables by running it once.

    Args:
      optimizer: If set, also create the optimizer variables.
    """
    if self.built:
      return

    # Create input features from the input signatures. We remove the leading
    # batch dimension as sometimes assumed by make_features methods and set
    # unspecified dimensions to 1.
    features = tf.nest.map_structure(
        lambda spec: tf.fill(
            [dim or 1 for dim in spec.shape.as_list()[1:]],
            tf.constant("" if spec.dtype is tf.string else 1, dtype=spec.dtype)),
        self.examples_inputter.input_signature())
    features = self.examples_inputter.make_features(features=features)

    # Add the batch dimension back before calling the model.
    features, labels = tf.nest.map_structure(lambda x: tf.expand_dims(x, 0), features)
    _ = self(features, domain_dropout_mask=[tf.ones(self.num_units)]*(self.encoder.num_layers+self.decoder.num_layers+1), labels=labels, training=True, step=0)

    if optimizer is not None:
      _ = optimizer.iterations
      optimizer._create_hypers()  # pylint: disable=protected-access
      optimizer._create_slots(self.trainable_variables)  # pylint: disable=protected-access

  def build(self, input_shape):
    super(Multi_domain_SequenceToSequence_TopK_sparse_multi_layer, self).build(input_shape)
    output_layer = None
    if EmbeddingsSharingLevel.share_target_embeddings(self.share_embeddings):
      output_layer = layers.Dense(
          self.labels_inputter.vocabulary_size,
          weight=self.labels_inputter.embedding,
          transpose=True)
    self.decoder.initialize(
        vocab_size=self.labels_inputter.vocabulary_size,
        output_layer=output_layer)

    self.latent_group_allocation_logit_per_layer = [self.add_weight("latent_group_allocation_logit_per_layer_%d"%i, initializer=tf.keras.initializers.Zeros(), shape=[self.num_domains, self.num_domain_unit_group]) for i in range(self.encoder.num_layers+self.decoder.num_layers+1)]

  def call(self, features, domain_dropout_mask=None, labels=None, training=None, step=None, internal_node_printing=False, return_domain_classification_logits=False, return_embedding=False, adapter_activate=True, inference=True):
    # Encode the source.
    assert isinstance(self.features_inputter, My_inputter)
    assert isinstance(self.labels_inputter, My_inputter)    
    assert domain_dropout_mask != None
    
    source_length = self.features_inputter.get_length(features)
    source_inputs = self.features_inputter(features, training=training)

    encoder_outputs, encoder_state, encoder_sequence_length = self.encoder(
        [source_inputs, features["domain"], domain_dropout_mask[:self.encoder.num_layers+1]], sequence_length=source_length, training=training)
    
    outputs = None
    predictions = None

    # When a target is provided, compute the decoder outputs for it.
    
    if labels is not None:
      outputs = self._decode_target(
          labels,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length,
          domain_dropout_mask[self.encoder.num_layers+1:],
          step=step,
          training=training)

    # When not in training, also compute the model predictions.
    if not training:
      predictions = self._dynamic_decode(
          features,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length,
          domain_dropout_mask[self.encoder.num_layers+1:])
    
    return outputs, predictions
  
  def _decode_target(self,
                     labels,
                     encoder_outputs,
                     encoder_state,
                     encoder_sequence_length,
                     domain_dropout_mask,
                     step=None,
                     training=None,
                     internal_node_printing=False):
    params = self.params
    target_inputs = self.labels_inputter(labels, training=training)
    input_fn = lambda ids: [self.labels_inputter({"ids": ids}, training=training), labels["domain"]]

    sampling_probability = None
    if training:
      sampling_probability = decoder_util.get_sampling_probability(
          step,
          read_probability=params.get("scheduled_sampling_read_probability"),
          schedule_type=params.get("scheduled_sampling_type"),
          k=params.get("scheduled_sampling_k"))

    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    logits, _, attention = self.decoder(
        [target_inputs, labels["domain"], domain_dropout_mask],
        self.labels_inputter.get_length(labels),
        state=initial_state,
        input_fn=input_fn,
        sampling_probability=sampling_probability,
        training=training)
    outputs = dict(logits=logits, attention=attention)

    return outputs
  
  def _dynamic_decode(self, features, encoder_outputs, encoder_state, encoder_sequence_length, domain_dropout_mask):
    params = self.params
    batch_size = tf.shape(tf.nest.flatten(encoder_outputs)[0])[0]
    start_ids = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
    beam_size = params.get("beam_width", 1)

    if beam_size > 1:
      # Tile encoder outputs to prepare for beam search.
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      encoder_sequence_length = tfa.seq2seq.tile_batch(encoder_sequence_length, beam_size)
      if encoder_state is not None:
        encoder_state = tfa.seq2seq.tile_batch(encoder_state, beam_size)

    # Dynamically decodes from the encoder outputs.
    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    sampled_ids, sampled_length, log_probs, alignment, _ = self.decoder.dynamic_decode(
        lambda ids: [self.labels_inputter({"ids": ids}), features["domain"], domain_dropout_mask],
        start_ids,
        initial_state=initial_state,
        decoding_strategy=decoding.DecodingStrategy.from_params(params),
        sampler=decoding.Sampler.from_params(params),
        maximum_iterations=params.get("maximum_decoding_length", 250),
        minimum_iterations=params.get("minimum_decoding_length", 0))
    target_tokens = self.labels_inputter.ids_to_tokens.lookup(tf.cast(sampled_ids, tf.int64))

    predictions = {
        "tokens": target_tokens,
        "length": sampled_length,
        "log_probs": log_probs
    }
    
    # Maybe restrict the number of returned hypotheses based on the user parameter.
    num_hypotheses = params.get("num_hypotheses", 1)
    if num_hypotheses > 0:
      if num_hypotheses > beam_size:
        raise ValueError("n_best cannot be greater than beam_width")
      for key, value in six.iteritems(predictions):
        predictions[key] = value[:, :num_hypotheses]
    return predictions

  def compute_loss(self, outputs, labels, training=True):
    params = self.params
    if not isinstance(outputs, dict):
      outputs = dict(logits=outputs)
    logits = outputs["logits"]
    labels_lengths = self.labels_inputter.get_length(labels)

    loss, loss_normalizer, loss_token_normalizer = losses.cross_entropy_sequence_loss(
        logits,
        labels["ids_out"],
        labels_lengths,
        label_smoothing=params.get("label_smoothing", 0.0),
        average_in_time=params.get("average_loss_in_time", False),
        training=training)
    
    return loss, loss_normalizer, loss_token_normalizer
  
  def print_prediction(self, prediction, params=None, stream=None):
    if params is None:
      params = {}
    num_hypotheses = len(prediction["tokens"])
    for i in range(num_hypotheses):
      target_length = prediction["length"][i]
      tokens = prediction["tokens"][i][:target_length]
      sentence = self.labels_inputter.tokenizer.detokenize(tokens)
      score = None
      attention = None
      alignment_type = None
      if params.get("with_scores"):
        score = prediction["log_probs"][i]
      if params.get("with_alignments"):
        attention = prediction["alignment"][i][:target_length]
        alignment_type = params["with_alignments"]
      sentence = format_translation_output(
          sentence,
          score=score,
          attention=attention,
          alignment_type=alignment_type)
      print_bytes(tf.compat.as_bytes(sentence), stream=stream)

  def transfer_weights(self, new_model, new_optimizer=None, optimizer=None, ignore_weights=None):
    updated_variables = []

    def _map_variables(inputter_fn, vars_fn):
      mapping, _ = vocab.get_mapping(
          inputter_fn(self).vocabulary_file,
          inputter_fn(new_model).vocabulary_file)
      vars_a, vocab_axes = vars_fn(self)
      vars_b, _ = vars_fn(new_model)
      for var_a, var_b, vocab_axis in zip(vars_a, vars_b, vocab_axes):
        if new_optimizer is not None and optimizer is not None:
          variables = vocab.update_variable_and_slots(
              var_a,
              var_b,
              optimizer,
              new_optimizer,
              mapping,
              vocab_axis=vocab_axis)
        else:
          variables = [vocab.update_variable(var_a, var_b, mapping, vocab_axis=vocab_axis)]
        updated_variables.extend(variables)
      return vars_b

class Priming_variational_SequenceToSequence(model.SequenceGenerator):
  """A sequence to sequence model."""

  def __init__(self,
               source_inputter,
               target_inputter,
               encoder,
               pre_encoder,
               decoder,
               version=1,
               share_embeddings=EmbeddingsSharingLevel.NONE):
    if not isinstance(target_inputter, inputters.WordEmbedder):
      raise TypeError("Target inputter must be a WordEmbedder")
    if EmbeddingsSharingLevel.share_input_embeddings(share_embeddings):
      if isinstance(source_inputter, inputters.ParallelInputter):
        source_inputters = source_inputter.inputters
      else:
        source_inputters = [source_inputter]
      for inputter in source_inputters:
        if not isinstance(inputter, inputters.WordEmbedder):
          raise TypeError("Sharing embeddings requires all inputters to be a "
                          "WordEmbedder")

    examples_inputter = Priming_SequenceToSequenceInputter(
        source_inputter,
        target_inputter,
        share_parameters=EmbeddingsSharingLevel.share_input_embeddings(share_embeddings))
    super(Priming_variational_SequenceToSequence, self).__init__(examples_inputter)
    self.encoder = encoder
    self.pre_encoder = pre_encoder
    self.decoder = decoder
    self.share_embeddings = share_embeddings
    self.version = version
    
    if self.version==5:
      print("share encoder")
      self.pre_encoder = self.encoder
    
    attention = MultiHeadAttention(
          num_heads,
          num_units,
          dropout=attention_dropout,
          return_attention=num_sources == 1)

    self.pre_selectioner = TransformerLayerWrapper(
          attention, dropout)

    self.ffn = FeedForwardNetwork(
        ffn_inner_dim,
        num_units,
        dropout=ffn_dropout,
        activation=ffn_activation)
    self.ffn = TransformerLayerWrapper(
        self.ffn, dropout)
  
  def auto_config(self, num_replicas=1):
    config = super(Priming_variational_SequenceToSequence, self).auto_config(num_replicas=num_replicas)
    return merge_dict(config, {
        "params": {
            "beam_width": 4
        },
        "train": {
            "sample_buffer_size": -1,
            "max_step": 500000
        },
        "infer": {
            "batch_size": 32,
            "length_bucket_width": 5
        }
    })

  def initialize(self, data_config, params=None):
    super(Priming_variational_SequenceToSequence, self).initialize(data_config, params=params)
    if self.params.get("contrastive_learning"):
      # Use the simplest and most effective CL_one from the paper.
      # https://www.aclweb.org/anthology/P19-1623
      noiser = noise.WordNoiser(
          noises=[noise.WordOmission(1)],
          subword_token=self.params.get("decoding_subword_token", "￭"),
          is_spacer=self.params.get("decoding_subword_token_is_spacer"))
      self.labels_inputter.set_noise(noiser, in_place=False)

  def build(self, input_shape):
    super(Priming_variational_SequenceToSequence, self).build(input_shape)
    output_layer = None
    if EmbeddingsSharingLevel.share_target_embeddings(self.share_embeddings):
      output_layer = layers.Dense(
          self.labels_inputter.vocabulary_size,
          weight=self.labels_inputter.embedding,
          transpose=True)
    self.decoder.initialize(
        vocab_size=self.labels_inputter.vocabulary_size,
        output_layer=output_layer)

  def create_variables(self, optimizer=None):
    """Creates the model variables by running it once.

    Args:
      optimizer: If set, also create the optimizer variables.
    """
    if self.built:
      return

    # Create input features from the input signatures. We remove the leading
    # batch dimension as sometimes assumed by make_features methods and set
    # unspecified dimensions to 1.
    features = tf.nest.map_structure(
        lambda spec: tf.fill(
            [dim or 1 for dim in spec.shape.as_list()[1:]],
            tf.constant("" if spec.dtype is tf.string else 1, dtype=spec.dtype)),
        self.examples_inputter.input_signature())
    #print("features: ",features)
    features = self.examples_inputter.make_features(features=features)
    #print("features: ",features)
    # Add the batch dimension back before calling the model.
    features, labels = tf.nest.map_structure(lambda x: tf.expand_dims(x, 0), features)
    _ = self(features, labels=labels, training=True, step=0)

    if optimizer is not None:
      _ = optimizer.iterations
      optimizer._create_hypers()  # pylint: disable=protected-access
      optimizer._create_slots(self.trainable_variables)  # pylint: disable=protected-access
  
  def call(self, features, labels=None, training=None, step=None):
    # Encode the source.
    source_length = self.features_inputter.get_length(features)
    source_inputs = self.features_inputter(features, training=training)

    source_length, pre_length = source_length
    source_inputs, pre_inputs = source_inputs

    encoder_outputs, encoder_state, encoder_sequence_length = self.encoder(
      source_inputs, sequence_length=source_length, training=training)
    pre_encoder_outputs, pre_encoder_state, pre_encoder_sequence_length = self.pre_encoder(
      pre_inputs, sequence_length=pre_length, training=training)
    
    outputs = None
    predictions = None

    # When a target is provided, compute the decoder outputs for it.
    if labels is not None:
      outputs = self._decode_target(
          labels,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length,
          pre_encoder_outputs,
          pre_encoder_state,
          pre_encoder_sequence_length,
          step=step,
          training=training)

    # When not in training, also compute the model predictions.
    if not training:
      predictions = self._dynamic_decode(
          features,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length,
          pre_encoder_outputs,
          pre_encoder_state,
          pre_encoder_sequence_length)

    return outputs, predictions

  def _decode_target(self,
                     labels,
                     encoder_outputs,
                     encoder_state,
                     encoder_sequence_length,
                     pre_encoder_outputs,
                     pre_encoder_state,
                     pre_encoder_sequence_length,
                     step=None,
                     training=None):
    params = self.params
    target_inputs = self.labels_inputter(labels, training=training)
    input_fn = lambda ids: self.labels_inputter({"ids": ids}, training=training)

    sampling_probability = None
    if training:
      sampling_probability = decoder_util.get_sampling_probability(
          step,
          read_probability=params.get("scheduled_sampling_read_probability"),
          schedule_type=params.get("scheduled_sampling_type"),
          k=params.get("scheduled_sampling_k"))

    

    if self.version in [1,5]:
      initial_state = self.decoder.initial_state(
        memory=tf.concat([encoder_outputs, pre_encoder_outputs], axis=1),
        memory_sequence_length= [encoder_sequence_length,pre_encoder_sequence_length],
        initial_state= None)

    logits, _, attention = self.decoder(
        target_inputs,
        self.labels_inputter.get_length(labels),
        state=initial_state,
        input_fn=input_fn,
        sampling_probability=sampling_probability,
        training=training)
    outputs = dict(logits=logits, attention=attention)

    return outputs

  def _dynamic_decode(self, features, encoder_outputs, encoder_state, encoder_sequence_length, pre_encoder_outputs,
          pre_encoder_state,
          pre_encoder_sequence_length):

    params = self.params
    batch_size = tf.shape(tf.nest.flatten(encoder_outputs)[0])[0]
    start_ids = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
    beam_size = params.get("beam_width", 1)

    if beam_size > 1:
      # Tile encoder outputs to prepare for beam search.
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      encoder_sequence_length = tfa.seq2seq.tile_batch(encoder_sequence_length, beam_size)
      if encoder_state is not None:
        encoder_state = tfa.seq2seq.tile_batch(encoder_state, beam_size)
      
      pre_encoder_outputs = tfa.seq2seq.tile_batch(pre_encoder_outputs, beam_size)
      pre_encoder_sequence_length = tfa.seq2seq.tile_batch(pre_encoder_sequence_length, beam_size)
      if encoder_state is not None:
        pre_encoder_state = tfa.seq2seq.tile_batch(pre_encoder_state, beam_size)
      
    # Dynamically decodes from the encoder outputs.
    if self.version in [1,5]:
      initial_state = self.decoder.initial_state(
        memory=tf.concat([encoder_outputs, pre_encoder_outputs], axis=1),
        memory_sequence_length= [encoder_sequence_length, pre_encoder_sequence_length],
        initial_state= None)

    sampled_ids, sampled_length, log_probs, alignment, _ = self.decoder.dynamic_decode(
        self.labels_inputter,
        start_ids,
        initial_state=initial_state,
        decoding_strategy=decoding.DecodingStrategy.from_params(params),
        sampler=decoding.Sampler.from_params(params),
        maximum_iterations=params.get("maximum_decoding_length", 250),
        minimum_iterations=params.get("minimum_decoding_length", 0))
    target_tokens = self.labels_inputter.ids_to_tokens.lookup(tf.cast(sampled_ids, tf.int64))

    predictions = {
        "tokens": target_tokens,
        "length": sampled_length,
        "log_probs": log_probs
    }
    if alignment is not None:
      predictions["alignment"] = alignment

    # Maybe restrict the number of returned hypotheses based on the user parameter.
    num_hypotheses = params.get("num_hypotheses", 1)
    if num_hypotheses > 0:
      if num_hypotheses > beam_size:
        raise ValueError("n_best cannot be greater than beam_width")
      for key, value in six.iteritems(predictions):
        predictions[key] = value[:, :num_hypotheses]
    return predictions

  def compute_loss(self, outputs, labels, training=True):
    params = self.params
    if not isinstance(outputs, dict):
      outputs = dict(logits=outputs)
    logits = outputs["logits"]
    noisy_logits = outputs.get("noisy_logits")
    attention = outputs.get("attention")
    if noisy_logits is not None and params.get("contrastive_learning"):
      return losses.max_margin_loss(
          logits,
          labels["ids_out"],
          labels["length"],
          noisy_logits,
          labels["noisy_ids_out"],
          labels["noisy_length"],
          eta=params.get("max_margin_eta", 0.1))
    labels_lengths = self.labels_inputter.get_length(labels)
    loss, loss_normalizer, loss_token_normalizer = losses.cross_entropy_sequence_loss(
        logits,
        labels["ids_out"],
        labels_lengths,
        label_smoothing=params.get("label_smoothing", 0.0),
        average_in_time=params.get("average_loss_in_time", False),
        training=training)
    if training:
      gold_alignments = labels.get("alignment")
      guided_alignment_type = params.get("guided_alignment_type")
      if gold_alignments is not None and guided_alignment_type is not None:
        if attention is None:
          tf.get_logger().warning("This model did not return attention vectors; "
                                  "guided alignment will not be applied")
        else:
          loss += losses.guided_alignment_cost(
              attention[:, :-1],  # Do not constrain last timestep.
              gold_alignments,
              sequence_length=labels_lengths - 1,
              cost_type=guided_alignment_type,
              weight=params.get("guided_alignment_weight", 1))
    return loss, loss_normalizer, loss_token_normalizer

  def print_prediction(self, prediction, params=None, stream=None):
    if params is None:
      params = {}
    num_hypotheses = len(prediction["tokens"])
    for i in range(num_hypotheses):
      target_length = prediction["length"][i]
      tokens = prediction["tokens"][i][:target_length]
      sentence = self.labels_inputter.tokenizer.detokenize(tokens)
      score = None
      attention = None
      alignment_type = None
      if params.get("with_scores"):
        score = prediction["log_probs"][i]
      if params.get("with_alignments"):
        attention = prediction["alignment"][i][:target_length]
        alignment_type = params["with_alignments"]
      sentence = format_translation_output(
          sentence,
          score=score,
          attention=attention,
          alignment_type=alignment_type)
      print_bytes(tf.compat.as_bytes(sentence), stream=stream)

  def transfer_weights(self, new_model, new_optimizer=None, optimizer=None, ignore_weights=None):
    updated_variables = []

    def _map_variables(inputter_fn, vars_fn):
      mapping, _ = vocab.get_mapping(
          inputter_fn(self).vocabulary_file,
          inputter_fn(new_model).vocabulary_file)
      vars_a, vocab_axes = vars_fn(self)
      vars_b, _ = vars_fn(new_model)
      for var_a, var_b, vocab_axis in zip(vars_a, vars_b, vocab_axes):
        if new_optimizer is not None and optimizer is not None:
          variables = vocab.update_variable_and_slots(
              var_a,
              var_b,
              optimizer,
              new_optimizer,
              mapping,
              vocab_axis=vocab_axis)
        else:
          variables = [vocab.update_variable(var_a, var_b, mapping, vocab_axis=vocab_axis)]
        updated_variables.extend(variables)
      return vars_b

    _map_variables(
        lambda model: model.features_inputter,
        lambda model: ([model.features_inputter.embedding], [0]))
    _map_variables(
        lambda model: model.labels_inputter,
        lambda model: ([
            model.labels_inputter.embedding,
            model.decoder.output_layer.kernel,
            model.decoder.output_layer.bias], [0, 1, 0]))

    return super(SequenceToSequence, self).transfer_weights(
        new_model,
        new_optimizer=new_optimizer,
        optimizer=optimizer,
        ignore_weights=updated_variables)

class Multi_domain_SequenceToSequence_TopK_sparse_multi_layer_v1(model.SequenceGenerator):

  """A sequence to sequence model."""

  def __init__(self,
               source_inputter,
               target_inputter,
               encoder,
               decoder,
               version=1,
               num_domains=6,
               dropout_rate=0.2,
               num_domain_unit_group=12,
               unit_group_size=16,
               num_shared_units=480,
               num_units=512,
               share_embeddings=EmbeddingsSharingLevel.NONE):

    if not isinstance(target_inputter, inputters.WordEmbedder) and not isinstance(target_inputter, My_inputter):
      raise TypeError("Target inputter must be a WordEmbedder or My_inputter")
    if EmbeddingsSharingLevel.share_input_embeddings(share_embeddings):
      if isinstance(source_inputter, inputters.ParallelInputter):
        source_inputters = source_inputter.inputters
      else:
        source_inputters = [source_inputter]
      for inputter in source_inputters:
        if not isinstance(inputter, inputters.WordEmbedder):
          raise TypeError("Sharing embeddings requires all inputters to be a "
                          "WordEmbedder")

    examples_inputter = Multi_domain_SequenceToSequenceInputter(
        source_inputter,
        target_inputter,
        share_parameters=EmbeddingsSharingLevel.share_input_embeddings(share_embeddings))
    super(Multi_domain_SequenceToSequence_TopK_sparse_multi_layer_v1, self).__init__(examples_inputter)
    self.encoder = encoder
    self.decoder = decoder
    self.share_embeddings = share_embeddings
    self.num_domains = num_domains
    self.num_domain_unit_group=num_domain_unit_group
    self.unit_group_size = unit_group_size
    self.dropout_rate = dropout_rate
    self.num_units = num_units
    self.num_shared_units = num_shared_units
    assert num_shared_units + unit_group_size * num_domain_unit_group == num_units
    self.version = version
    self.mask_num = encoder.num_layers + decoder.num_layers + 2
  
  def auto_config(self, num_replicas=1):
    config = super(Multi_domain_SequenceToSequence_TopK_sparse_multi_layer_v1, self).auto_config(num_replicas=num_replicas)
    return merge_dict(config, {
        "params": {
            "beam_width": 5
        },
        "train": {
            "sample_buffer_size": -1,
            "max_step": 200000
        },
        "infer": {
            "batch_size": 32,
            "length_bucket_width": 5
        }
    })

  def initialize(self, data_config, params=None):
    super(Multi_domain_SequenceToSequence_TopK_sparse_multi_layer_v1, self).initialize(data_config, params=params)
    if self.params.get("contrastive_learning"):
      noiser = noise.WordNoiser(
          noises=[noise.WordOmission(1)],
          subword_token=self.params.get("decoding_subword_token", "￭"),
          is_spacer=self.params.get("decoding_subword_token_is_spacer"))
      self.labels_inputter.set_noise(noiser, in_place=False)

  def create_variables(self, optimizer=None):
    """Creates the model variables by running it once.

    Args:
      optimizer: If set, also create the optimizer variables.
    """
    if self.built:
      return

    # Create input features from the input signatures. We remove the leading
    # batch dimension as sometimes assumed by make_features methods and set
    # unspecified dimensions to 1.
    features = tf.nest.map_structure(
        lambda spec: tf.fill(
            [dim or 1 for dim in spec.shape.as_list()[1:]],
            tf.constant("" if spec.dtype is tf.string else 1, dtype=spec.dtype)),
        self.examples_inputter.input_signature())
    features = self.examples_inputter.make_features(features=features)

    # Add the batch dimension back before calling the model.
    features, labels = tf.nest.map_structure(lambda x: tf.expand_dims(x, 0), features)
    _ = self(features, domain_dropout_mask=[tf.ones(self.num_units)]*(self.encoder.num_layers+self.decoder.num_layers+2), labels=labels, training=True, step=0)

    if optimizer is not None:
      _ = optimizer.iterations
      optimizer._create_hypers()  # pylint: disable=protected-access
      optimizer._create_slots(self.trainable_variables)  # pylint: disable=protected-access

  def build(self, input_shape):
    super(Multi_domain_SequenceToSequence_TopK_sparse_multi_layer_v1, self).build(input_shape)
    output_layer = None
    if EmbeddingsSharingLevel.share_target_embeddings(self.share_embeddings):
      output_layer = layers.Dense(
          self.labels_inputter.vocabulary_size,
          weight=self.labels_inputter.embedding,
          transpose=True)
    self.decoder.initialize(
        vocab_size=self.labels_inputter.vocabulary_size,
        output_layer=output_layer)

    self.latent_group_allocation_logit_per_layer = [self.add_weight("latent_group_allocation_logit_per_layer_%d"%i, initializer=tf.keras.initializers.Zeros(), shape=[self.num_domains, self.num_domain_unit_group]) for i in range(self.encoder.num_layers+self.decoder.num_layers+2)]

  def call(self, features, domain_dropout_mask=None, labels=None, training=None, step=None, internal_node_printing=False, return_domain_classification_logits=False, return_embedding=False, adapter_activate=True, inference=True):
    # Encode the source.
    assert isinstance(self.features_inputter, My_inputter)
    assert isinstance(self.labels_inputter, My_inputter)    
    assert domain_dropout_mask != None
    
    source_length = self.features_inputter.get_length(features)
    source_inputs = self.features_inputter(features, training=training)

    encoder_outputs, encoder_state, encoder_sequence_length = self.encoder(
        [source_inputs, features["domain"], domain_dropout_mask[:self.encoder.num_layers+1]], sequence_length=source_length, training=training)
    
    outputs = None
    predictions = None

    # When a target is provided, compute the decoder outputs for it.
    
    if labels is not None:
      outputs = self._decode_target(
          labels,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length,
          domain_dropout_mask[self.encoder.num_layers+1:],
          step=step,
          training=training)

    # When not in training, also compute the model predictions.
    if not training:
      predictions = self._dynamic_decode(
          features,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length,
          domain_dropout_mask[self.encoder.num_layers+1:])
    
    return outputs, predictions
  
  def _decode_target(self,
                     labels,
                     encoder_outputs,
                     encoder_state,
                     encoder_sequence_length,
                     domain_dropout_mask,
                     step=None,
                     training=None,
                     internal_node_printing=False):
    params = self.params
    target_inputs = self.labels_inputter(labels, training=training)
    input_fn = lambda ids: [self.labels_inputter({"ids": ids}, training=training), labels["domain"]]

    sampling_probability = None
    if training:
      sampling_probability = decoder_util.get_sampling_probability(
          step,
          read_probability=params.get("scheduled_sampling_read_probability"),
          schedule_type=params.get("scheduled_sampling_type"),
          k=params.get("scheduled_sampling_k"))

    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    logits, _, attention = self.decoder(
        [target_inputs, labels["domain"], domain_dropout_mask],
        self.labels_inputter.get_length(labels),
        state=initial_state,
        input_fn=input_fn,
        sampling_probability=sampling_probability,
        training=training)
    outputs = dict(logits=logits, attention=attention)

    return outputs
  
  def _dynamic_decode(self, features, encoder_outputs, encoder_state, encoder_sequence_length, domain_dropout_mask):
    params = self.params
    batch_size = tf.shape(tf.nest.flatten(encoder_outputs)[0])[0]
    start_ids = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
    beam_size = params.get("beam_width", 1)

    if beam_size > 1:
      # Tile encoder outputs to prepare for beam search.
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      encoder_sequence_length = tfa.seq2seq.tile_batch(encoder_sequence_length, beam_size)
      if encoder_state is not None:
        encoder_state = tfa.seq2seq.tile_batch(encoder_state, beam_size)

    # Dynamically decodes from the encoder outputs.
    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    sampled_ids, sampled_length, log_probs, alignment, _ = self.decoder.dynamic_decode(
        lambda ids: [self.labels_inputter({"ids": ids}), features["domain"], domain_dropout_mask],
        start_ids,
        initial_state=initial_state,
        decoding_strategy=decoding.DecodingStrategy.from_params(params),
        sampler=decoding.Sampler.from_params(params),
        maximum_iterations=params.get("maximum_decoding_length", 250),
        minimum_iterations=params.get("minimum_decoding_length", 0))
    target_tokens = self.labels_inputter.ids_to_tokens.lookup(tf.cast(sampled_ids, tf.int64))

    predictions = {
        "tokens": target_tokens,
        "length": sampled_length,
        "log_probs": log_probs
    }
    
    # Maybe restrict the number of returned hypotheses based on the user parameter.
    num_hypotheses = params.get("num_hypotheses", 1)
    if num_hypotheses > 0:
      if num_hypotheses > beam_size:
        raise ValueError("n_best cannot be greater than beam_width")
      for key, value in six.iteritems(predictions):
        predictions[key] = value[:, :num_hypotheses]
    return predictions

  def compute_loss(self, outputs, labels, training=True):
    params = self.params
    if not isinstance(outputs, dict):
      outputs = dict(logits=outputs)
    logits = outputs["logits"]
    labels_lengths = self.labels_inputter.get_length(labels)

    loss, loss_normalizer, loss_token_normalizer = losses.cross_entropy_sequence_loss(
        logits,
        labels["ids_out"],
        labels_lengths,
        label_smoothing=params.get("label_smoothing", 0.0),
        average_in_time=params.get("average_loss_in_time", False),
        training=training)
    
    return loss, loss_normalizer, loss_token_normalizer
  
  def print_prediction(self, prediction, params=None, stream=None):
    if params is None:
      params = {}
    num_hypotheses = len(prediction["tokens"])
    for i in range(num_hypotheses):
      target_length = prediction["length"][i]
      tokens = prediction["tokens"][i][:target_length]
      sentence = self.labels_inputter.tokenizer.detokenize(tokens)
      score = None
      attention = None
      alignment_type = None
      if params.get("with_scores"):
        score = prediction["log_probs"][i]
      if params.get("with_alignments"):
        attention = prediction["alignment"][i][:target_length]
        alignment_type = params["with_alignments"]
      sentence = format_translation_output(
          sentence,
          score=score,
          attention=attention,
          alignment_type=alignment_type)
      print_bytes(tf.compat.as_bytes(sentence), stream=stream)

  def transfer_weights(self, new_model, new_optimizer=None, optimizer=None, ignore_weights=None):
    updated_variables = []

    def _map_variables(inputter_fn, vars_fn):
      mapping, _ = vocab.get_mapping(
          inputter_fn(self).vocabulary_file,
          inputter_fn(new_model).vocabulary_file)
      vars_a, vocab_axes = vars_fn(self)
      vars_b, _ = vars_fn(new_model)
      for var_a, var_b, vocab_axis in zip(vars_a, vars_b, vocab_axes):
        if new_optimizer is not None and optimizer is not None:
          variables = vocab.update_variable_and_slots(
              var_a,
              var_b,
              optimizer,
              new_optimizer,
              mapping,
              vocab_axis=vocab_axis)
        else:
          variables = [vocab.update_variable(var_a, var_b, mapping, vocab_axis=vocab_axis)]
        updated_variables.extend(variables)
      return vars_b

class Multi_domain_SequenceToSequence_TopK_sparse_multi_layer_v0(model.SequenceGenerator):

  """A sequence to sequence model."""

  def __init__(self,
               source_inputter,
               target_inputter,
               encoder,
               decoder,
               version=1,
               num_domains=6,
               dropout_rate=0.2,
               num_domain_unit_group=12,
               unit_group_size=16,
               num_shared_units=480,
               num_units=512,
               share_embeddings=EmbeddingsSharingLevel.NONE):

    if not isinstance(target_inputter, inputters.WordEmbedder) and not isinstance(target_inputter, My_inputter):
      raise TypeError("Target inputter must be a WordEmbedder or My_inputter")
    if EmbeddingsSharingLevel.share_input_embeddings(share_embeddings):
      if isinstance(source_inputter, inputters.ParallelInputter):
        source_inputters = source_inputter.inputters
      else:
        source_inputters = [source_inputter]
      for inputter in source_inputters:
        if not isinstance(inputter, inputters.WordEmbedder):
          raise TypeError("Sharing embeddings requires all inputters to be a "
                          "WordEmbedder")

    examples_inputter = Multi_domain_SequenceToSequenceInputter(
        source_inputter,
        target_inputter,
        share_parameters=EmbeddingsSharingLevel.share_input_embeddings(share_embeddings))
    super(Multi_domain_SequenceToSequence_TopK_sparse_multi_layer_v0, self).__init__(examples_inputter)
    self.encoder = encoder
    self.decoder = decoder
    self.share_embeddings = share_embeddings
    self.num_domains = num_domains
    self.num_domain_unit_group=num_domain_unit_group
    self.unit_group_size = unit_group_size
    self.dropout_rate = dropout_rate
    self.num_units = num_units
    self.num_shared_units = num_shared_units
    assert num_shared_units + unit_group_size * num_domain_unit_group == num_units
    self.version = version
    self.mask_num = encoder.num_layers + decoder.num_layers 
  
  def auto_config(self, num_replicas=1):
    config = super(Multi_domain_SequenceToSequence_TopK_sparse_multi_layer_v0, self).auto_config(num_replicas=num_replicas)
    return merge_dict(config, {
        "params": {
            "beam_width": 5
        },
        "train": {
            "sample_buffer_size": -1,
            "max_step": 200000
        },
        "infer": {
            "batch_size": 32,
            "length_bucket_width": 5
        }
    })

  def initialize(self, data_config, params=None):
    super(Multi_domain_SequenceToSequence_TopK_sparse_multi_layer_v1, self).initialize(data_config, params=params)
    if self.params.get("contrastive_learning"):
      noiser = noise.WordNoiser(
          noises=[noise.WordOmission(1)],
          subword_token=self.params.get("decoding_subword_token", "￭"),
          is_spacer=self.params.get("decoding_subword_token_is_spacer"))
      self.labels_inputter.set_noise(noiser, in_place=False)

  def create_variables(self, optimizer=None):
    """Creates the model variables by running it once.

    Args:
      optimizer: If set, also create the optimizer variables.
    """
    if self.built:
      return

    # Create input features from the input signatures. We remove the leading
    # batch dimension as sometimes assumed by make_features methods and set
    # unspecified dimensions to 1.
    features = tf.nest.map_structure(
        lambda spec: tf.fill(
            [dim or 1 for dim in spec.shape.as_list()[1:]],
            tf.constant("" if spec.dtype is tf.string else 1, dtype=spec.dtype)),
        self.examples_inputter.input_signature())
    features = self.examples_inputter.make_features(features=features)

    # Add the batch dimension back before calling the model.
    features, labels = tf.nest.map_structure(lambda x: tf.expand_dims(x, 0), features)
    _ = self(features, domain_dropout_mask=[tf.ones(self.num_units)]*(self.encoder.num_layers+self.decoder.num_layers), labels=labels, training=True, step=0)

    if optimizer is not None:
      _ = optimizer.iterations
      optimizer._create_hypers()  # pylint: disable=protected-access
      optimizer._create_slots(self.trainable_variables)  # pylint: disable=protected-access

  def build(self, input_shape):
    super(Multi_domain_SequenceToSequence_TopK_sparse_multi_layer_v0, self).build(input_shape)
    output_layer = None
    if EmbeddingsSharingLevel.share_target_embeddings(self.share_embeddings):
      output_layer = layers.Dense(
          self.labels_inputter.vocabulary_size,
          weight=self.labels_inputter.embedding,
          transpose=True)
    self.decoder.initialize(
        vocab_size=self.labels_inputter.vocabulary_size,
        output_layer=output_layer)

    self.latent_group_allocation_logit_per_layer = [self.add_weight("latent_group_allocation_logit_per_layer_%d"%i, initializer=tf.keras.initializers.Zeros(), shape=[self.num_domains, self.num_domain_unit_group]) for i in range(self.encoder.num_layers+self.decoder.num_layers)]
    self.latent_topk_logit_per_layer = [self.add_weight("latent_topk_logit_per_layer_%d"%i, initializer=tf.keras.initializers.Zeros(), shape=[self.num_domains, self.num_domain_unit_group-1]) for i in range(self.encoder.num_layers+self.decoder.num_layers)]

  def call(self, features, domain_dropout_mask=None, labels=None, training=None, step=None, internal_node_printing=False, return_domain_classification_logits=False, return_embedding=False, adapter_activate=True, inference=True):
    # Encode the source.
    assert isinstance(self.features_inputter, My_inputter)
    assert isinstance(self.labels_inputter, My_inputter)    
    assert domain_dropout_mask != None
    
    source_length = self.features_inputter.get_length(features)
    source_inputs = self.features_inputter(features, training=training)

    encoder_outputs, encoder_state, encoder_sequence_length = self.encoder(
        [source_inputs, features["domain"], domain_dropout_mask[:self.encoder.num_layers]], sequence_length=source_length, training=training)
    
    outputs = None
    predictions = None

    # When a target is provided, compute the decoder outputs for it.
    
    if labels is not None:
      outputs = self._decode_target(
          labels,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length,
          domain_dropout_mask[self.encoder.num_layers:],
          step=step,
          training=training)

    # When not in training, also compute the model predictions.
    if not training:
      predictions = self._dynamic_decode(
          features,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length,
          domain_dropout_mask[self.encoder.num_layers:])
    
    return outputs, predictions
  
  def _decode_target(self,
                     labels,
                     encoder_outputs,
                     encoder_state,
                     encoder_sequence_length,
                     domain_dropout_mask,
                     step=None,
                     training=None,
                     internal_node_printing=False):
    params = self.params
    target_inputs = self.labels_inputter(labels, training=training)
    input_fn = lambda ids: [self.labels_inputter({"ids": ids}, training=training), labels["domain"]]

    sampling_probability = None
    if training:
      sampling_probability = decoder_util.get_sampling_probability(
          step,
          read_probability=params.get("scheduled_sampling_read_probability"),
          schedule_type=params.get("scheduled_sampling_type"),
          k=params.get("scheduled_sampling_k"))

    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    logits, _, attention = self.decoder(
        [target_inputs, labels["domain"], domain_dropout_mask],
        self.labels_inputter.get_length(labels),
        state=initial_state,
        input_fn=input_fn,
        sampling_probability=sampling_probability,
        training=training)
    outputs = dict(logits=logits, attention=attention)

    return outputs
  
  def _dynamic_decode(self, features, encoder_outputs, encoder_state, encoder_sequence_length, domain_dropout_mask):
    params = self.params
    batch_size = tf.shape(tf.nest.flatten(encoder_outputs)[0])[0]
    start_ids = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
    beam_size = params.get("beam_width", 1)

    if beam_size > 1:
      # Tile encoder outputs to prepare for beam search.
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      encoder_sequence_length = tfa.seq2seq.tile_batch(encoder_sequence_length, beam_size)
      if encoder_state is not None:
        encoder_state = tfa.seq2seq.tile_batch(encoder_state, beam_size)

    # Dynamically decodes from the encoder outputs.
    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    sampled_ids, sampled_length, log_probs, alignment, _ = self.decoder.dynamic_decode(
        lambda ids: [self.labels_inputter({"ids": ids}), features["domain"], domain_dropout_mask],
        start_ids,
        initial_state=initial_state,
        decoding_strategy=decoding.DecodingStrategy.from_params(params),
        sampler=decoding.Sampler.from_params(params),
        maximum_iterations=params.get("maximum_decoding_length", 250),
        minimum_iterations=params.get("minimum_decoding_length", 0))
    target_tokens = self.labels_inputter.ids_to_tokens.lookup(tf.cast(sampled_ids, tf.int64))

    predictions = {
        "tokens": target_tokens,
        "length": sampled_length,
        "log_probs": log_probs
    }
    
    # Maybe restrict the number of returned hypotheses based on the user parameter.
    num_hypotheses = params.get("num_hypotheses", 1)
    if num_hypotheses > 0:
      if num_hypotheses > beam_size:
        raise ValueError("n_best cannot be greater than beam_width")
      for key, value in six.iteritems(predictions):
        predictions[key] = value[:, :num_hypotheses]
    return predictions

  def compute_loss(self, outputs, labels, training=True):
    params = self.params
    if not isinstance(outputs, dict):
      outputs = dict(logits=outputs)
    logits = outputs["logits"]
    labels_lengths = self.labels_inputter.get_length(labels)

    loss, loss_normalizer, loss_token_normalizer = losses.cross_entropy_sequence_loss(
        logits,
        labels["ids_out"],
        labels_lengths,
        label_smoothing=params.get("label_smoothing", 0.0),
        average_in_time=params.get("average_loss_in_time", False),
        training=training)
    
    return loss, loss_normalizer, loss_token_normalizer
  
  def print_prediction(self, prediction, params=None, stream=None):
    if params is None:
      params = {}
    num_hypotheses = len(prediction["tokens"])
    for i in range(num_hypotheses):
      target_length = prediction["length"][i]
      tokens = prediction["tokens"][i][:target_length]
      sentence = self.labels_inputter.tokenizer.detokenize(tokens)
      score = None
      attention = None
      alignment_type = None
      if params.get("with_scores"):
        score = prediction["log_probs"][i]
      if params.get("with_alignments"):
        attention = prediction["alignment"][i][:target_length]
        alignment_type = params["with_alignments"]
      sentence = format_translation_output(
          sentence,
          score=score,
          attention=attention,
          alignment_type=alignment_type)
      print_bytes(tf.compat.as_bytes(sentence), stream=stream)

  def transfer_weights(self, new_model, new_optimizer=None, optimizer=None, ignore_weights=None):
    updated_variables = []

    def _map_variables(inputter_fn, vars_fn):
      mapping, _ = vocab.get_mapping(
          inputter_fn(self).vocabulary_file,
          inputter_fn(new_model).vocabulary_file)
      vars_a, vocab_axes = vars_fn(self)
      vars_b, _ = vars_fn(new_model)
      for var_a, var_b, vocab_axis in zip(vars_a, vars_b, vocab_axes):
        if new_optimizer is not None and optimizer is not None:
          variables = vocab.update_variable_and_slots(
              var_a,
              var_b,
              optimizer,
              new_optimizer,
              mapping,
              vocab_axis=vocab_axis)
        else:
          variables = [vocab.update_variable(var_a, var_b, mapping, vocab_axis=vocab_axis)]
        updated_variables.extend(variables)
      return vars_b

class Multi_domain_SequenceToSequence_Instace_Aware_TopK_sparse_multi_layer(model.SequenceGenerator):

  """A sequence to sequence model."""

  def __init__(self,
               source_inputter,
               target_inputter,
               encoder,
               decoder,
               meta_encoder=None,
               version=1,
               num_domains=6,
               dropout_rate=0.2,
               num_domain_unit_group=12,
               unit_group_size=16,
               num_shared_units=480,
               num_units=512,
               share_embeddings=EmbeddingsSharingLevel.NONE):

    if not isinstance(target_inputter, inputters.WordEmbedder) and not isinstance(target_inputter, My_inputter):
      raise TypeError("Target inputter must be a WordEmbedder or My_inputter")
    if EmbeddingsSharingLevel.share_input_embeddings(share_embeddings):
      if isinstance(source_inputter, inputters.ParallelInputter):
        source_inputters = source_inputter.inputters
      else:
        source_inputters = [source_inputter]
      for inputter in source_inputters:
        if not isinstance(inputter, inputters.WordEmbedder):
          raise TypeError("Sharing embeddings requires all inputters to be a "
                          "WordEmbedder")

    examples_inputter = Multi_domain_SequenceToSequenceInputter(
        source_inputter,
        target_inputter,
        share_parameters=EmbeddingsSharingLevel.share_input_embeddings(share_embeddings))
    super(Multi_domain_SequenceToSequence_Instace_Aware_TopK_sparse_multi_layer, self).__init__(examples_inputter)
    self.encoder = encoder
    self.decoder = decoder
    self.meta_encoder = meta_encoder
    self.share_embeddings = share_embeddings
    self.num_domains = num_domains
    self.num_domain_unit_group=num_domain_unit_group
    self.unit_group_size = unit_group_size
    self.dropout_rate = dropout_rate
    self.num_units = num_units
    self.num_shared_units = num_shared_units
    assert num_shared_units + unit_group_size * num_domain_unit_group == num_units
    self.version = version
    self.mask_num = encoder.num_layers + decoder.num_layers 
    self.mask_generators = [Dense(num_domain_unit_group, name="mask_generator_%d"%i) for i in range(self.mask_num)]
  
  def auto_config(self, num_replicas=1):
    config = super(Multi_domain_SequenceToSequence_Instace_Aware_TopK_sparse_multi_layer, self).auto_config(num_replicas=num_replicas)
    return merge_dict(config, {
        "params": {
            "beam_width": 5
        },
        "train": {
            "sample_buffer_size": -1,
            "max_step": 200000
        },
        "infer": {
            "batch_size": 32,
            "length_bucket_width": 5
        }
    })

  def initialize(self, data_config, params=None):
    super(Multi_domain_SequenceToSequence_Instace_Aware_TopK_sparse_multi_layer, self).initialize(data_config, params=params)
    if self.params.get("contrastive_learning"):
      noiser = noise.WordNoiser(
          noises=[noise.WordOmission(1)],
          subword_token=self.params.get("decoding_subword_token", "￭"),
          is_spacer=self.params.get("decoding_subword_token_is_spacer"))
      self.labels_inputter.set_noise(noiser, in_place=False)

  def create_variables(self, optimizer=None):
    """Creates the model variables by running it once.

    Args:
      optimizer: If set, also create the optimizer variables.
    """
    if self.built:
      return

    # Create input features from the input signatures. We remove the leading
    # batch dimension as sometimes assumed by make_features methods and set
    # unspecified dimensions to 1.
    features = tf.nest.map_structure(
        lambda spec: tf.fill(
            [dim or 1 for dim in spec.shape.as_list()[1:]],
            tf.constant("" if spec.dtype is tf.string else 1, dtype=spec.dtype)),
        self.examples_inputter.input_signature())
    features = self.examples_inputter.make_features(features=features)

    # Add the batch dimension back before calling the model.
    features, labels = tf.nest.map_structure(lambda x: tf.expand_dims(x, 0), features)
    batch_size = tf.shape(self.features_inputter.get_length(features))[0]
    _ = self(features, domain_dropout_mask=[tf.ones([batch_size, self.num_units])]*(self.encoder.num_layers+self.decoder.num_layers), labels=labels, training=True, step=0)

    if optimizer is not None:
      _ = optimizer.iterations
      optimizer._create_hypers()  # pylint: disable=protected-access
      optimizer._create_slots(self.trainable_variables)  # pylint: disable=protected-access

  def build(self, input_shape):
    super(Multi_domain_SequenceToSequence_Instace_Aware_TopK_sparse_multi_layer, self).build(input_shape)
    output_layer = None
    if EmbeddingsSharingLevel.share_target_embeddings(self.share_embeddings):
      output_layer = layers.Dense(
          self.labels_inputter.vocabulary_size,
          weight=self.labels_inputter.embedding,
          transpose=True)
    self.decoder.initialize(
        vocab_size=self.labels_inputter.vocabulary_size,
        output_layer=output_layer)

  def call(self, features, domain_dropout_mask=None, labels=None, training=None, step=None, internal_node_printing=False, return_domain_classification_logits=False, return_embedding=False, adapter_activate=True, inference=True):
    # Encode the source.
    assert isinstance(self.features_inputter, My_inputter)
    assert isinstance(self.labels_inputter, My_inputter)    
    assert domain_dropout_mask != None
    
    source_length = self.features_inputter.get_length(features)
    source_inputs = self.features_inputter(features, training=training)

    encoder_outputs, encoder_state, encoder_sequence_length = self.encoder(
        [source_inputs, features["domain"], domain_dropout_mask[:self.encoder.num_layers]], sequence_length=source_length, training=training)
    
    outputs = None
    predictions = None

    # When a target is provided, compute the decoder outputs for it.
    
    if labels is not None:
      outputs = self._decode_target(
          labels,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length,
          domain_dropout_mask[self.encoder.num_layers:],
          step=step,
          training=training)

    # When not in training, also compute the model predictions.
    if not training:
      predictions = self._dynamic_decode(
          features,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length,
          domain_dropout_mask[self.encoder.num_layers:])
    
    return outputs, predictions
  
  def _decode_target(self,
                     labels,
                     encoder_outputs,
                     encoder_state,
                     encoder_sequence_length,
                     domain_dropout_mask,
                     step=None,
                     training=None,
                     internal_node_printing=False):
    params = self.params
    target_inputs = self.labels_inputter(labels, training=training)
    input_fn = lambda ids: [self.labels_inputter({"ids": ids}, training=training), labels["domain"]]

    sampling_probability = None
    if training:
      sampling_probability = decoder_util.get_sampling_probability(
          step,
          read_probability=params.get("scheduled_sampling_read_probability"),
          schedule_type=params.get("scheduled_sampling_type"),
          k=params.get("scheduled_sampling_k"))

    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    logits, _, attention = self.decoder(
        [target_inputs, labels["domain"], domain_dropout_mask],
        self.labels_inputter.get_length(labels),
        state=initial_state,
        input_fn=input_fn,
        sampling_probability=sampling_probability,
        training=training)
    outputs = dict(logits=logits, attention=attention)

    return outputs
  
  def _dynamic_decode(self, features, encoder_outputs, encoder_state, encoder_sequence_length, domain_dropout_mask):
    params = self.params
    batch_size = tf.shape(tf.nest.flatten(encoder_outputs)[0])[0]
    start_ids = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
    beam_size = params.get("beam_width", 1)

    if beam_size > 1:
      # Tile encoder outputs to prepare for beam search.
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      encoder_sequence_length = tfa.seq2seq.tile_batch(encoder_sequence_length, beam_size)
      if encoder_state is not None:
        encoder_state = tfa.seq2seq.tile_batch(encoder_state, beam_size)

    # Dynamically decodes from the encoder outputs.
    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    sampled_ids, sampled_length, log_probs, alignment, _ = self.decoder.dynamic_decode(
        lambda ids: [self.labels_inputter({"ids": ids}), features["domain"], domain_dropout_mask],
        start_ids,
        initial_state=initial_state,
        decoding_strategy=decoding.DecodingStrategy.from_params(params),
        sampler=decoding.Sampler.from_params(params),
        maximum_iterations=params.get("maximum_decoding_length", 250),
        minimum_iterations=params.get("minimum_decoding_length", 0))
    target_tokens = self.labels_inputter.ids_to_tokens.lookup(tf.cast(sampled_ids, tf.int64))

    predictions = {
        "tokens": target_tokens,
        "length": sampled_length,
        "log_probs": log_probs
    }
    
    # Maybe restrict the number of returned hypotheses based on the user parameter.
    num_hypotheses = params.get("num_hypotheses", 1)
    if num_hypotheses > 0:
      if num_hypotheses > beam_size:
        raise ValueError("n_best cannot be greater than beam_width")
      for key, value in six.iteritems(predictions):
        predictions[key] = value[:, :num_hypotheses]
    return predictions

  def compute_loss(self, outputs, labels, training=True):
    params = self.params
    if not isinstance(outputs, dict):
      outputs = dict(logits=outputs)
    logits = outputs["logits"]
    labels_lengths = self.labels_inputter.get_length(labels)

    loss, loss_normalizer, loss_token_normalizer = losses.cross_entropy_sequence_loss(
        logits,
        labels["ids_out"],
        labels_lengths,
        label_smoothing=params.get("label_smoothing", 0.0),
        average_in_time=params.get("average_loss_in_time", False),
        training=training)
    
    return loss, loss_normalizer, loss_token_normalizer
  
  def print_prediction(self, prediction, params=None, stream=None):
    if params is None:
      params = {}
    num_hypotheses = len(prediction["tokens"])
    for i in range(num_hypotheses):
      target_length = prediction["length"][i]
      tokens = prediction["tokens"][i][:target_length]
      sentence = self.labels_inputter.tokenizer.detokenize(tokens)
      score = None
      attention = None
      alignment_type = None
      if params.get("with_scores"):
        score = prediction["log_probs"][i]
      if params.get("with_alignments"):
        attention = prediction["alignment"][i][:target_length]
        alignment_type = params["with_alignments"]
      sentence = format_translation_output(
          sentence,
          score=score,
          attention=attention,
          alignment_type=alignment_type)
      print_bytes(tf.compat.as_bytes(sentence), stream=stream)

  def transfer_weights(self, new_model, new_optimizer=None, optimizer=None, ignore_weights=None):
    updated_variables = []

    def _map_variables(inputter_fn, vars_fn):
      mapping, _ = vocab.get_mapping(
          inputter_fn(self).vocabulary_file,
          inputter_fn(new_model).vocabulary_file)
      vars_a, vocab_axes = vars_fn(self)
      vars_b, _ = vars_fn(new_model)
      for var_a, var_b, vocab_axis in zip(vars_a, vars_b, vocab_axes):
        if new_optimizer is not None and optimizer is not None:
          variables = vocab.update_variable_and_slots(
              var_a,
              var_b,
              optimizer,
              new_optimizer,
              mapping,
              vocab_axis=vocab_axis)
        else:
          variables = [vocab.update_variable(var_a, var_b, mapping, vocab_axis=vocab_axis)]
        updated_variables.extend(variables)
      return vars_b







































