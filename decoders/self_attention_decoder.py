"""Define self-attention decoder."""
import sys
sys.path.append("/gpfsdswork/projects/rech/sfz/utt84zy/anaconda3/envs/huggingface/lib/python3.7/site-packages")
import numpy as np
import tensorflow as tf
from opennmt.decoders.decoder import Decoder
from opennmt.decoders.self_attention_decoder import SelfAttentionDecoder
from layers import common, transformer
from opennmt.layers.position import SinusoidalPositionEncoder
from layers.layers import Multi_domain_classification_gate_v2, Regulation_Gate, Multi_domain_classification_gate, Multi_domain_FeedForwardNetwork, Multi_domain_FeedForwardNetwork_v8, Multi_domain_FeedForwardNetwork_v6, Multi_domain_Gate_v2, Multi_domain_FeedForwardNetwork_v2, DAFE, Multi_domain_Gate, Multi_domain_FeedForwardNetwork_v3, Multi_domain_FeedForwardNetwork_v9
from utils.utils_ import make_domain_mask
from layers.common import Multi_LayerNorm
from opennmt.utils import decoding
from opennmt import constants
from opennmt.inputters import text_inputter
from opennmt.utils.misc import shape_list

class Multi_domain_SelfAttentionDecoder(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
               num_domain_units=128,
               ADAP_layer_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               num_sources=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = common.LayerNorm()
    self.layers = [
        transformer.SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]
    self.mask = make_domain_mask(num_domains, num_units, num_domain_units=num_domain_units)
    self.multi_domain_layers = [
        Multi_domain_FeedForwardNetwork(num_domains*num_domain_units, num_units, name="ADAP_%d"%i)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient
  def initialize(self, vocab_size=None, output_layer=None):
    
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Dense(vocab_size)

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    domain = inputs[1]
    domain_mask = tf.nn.embedding_lookup(self.mask, domain)
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):

      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer(tf.stop_gradient(inputs), domain_mask) + inputs
      else:
        inputs = multi_domain_layer(inputs, domain_mask) + inputs

    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer(outputs)
    return logits, state, attention
    
  def forward_fn(self,
              inputs,
              args_dict,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run_forward_fn(
        inputs,
        args_dict,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer.forward_fn(outputs, args_dict)
    return logits, state, attention

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

  def _run_forward_fn(self,
           inputs,
           args_dict,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    domain = inputs[1]
    domain_mask = tf.nn.embedding_lookup(self.mask, domain)
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]
    
    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):

      inputs, layer_cache, attention = layer.forward_fn(
          inputs,
          args_dict,
          mask=mask,          
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), args_dict, domain_mask) + inputs
      else:
        inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain_mask) + inputs

    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, new_cache, attention

class Multi_domain_SelfAttentionDecoder_v1(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
               num_domain_units=128,
               ADAP_layer_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v2,
               num_sources=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_v1, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = common.LayerNorm()
    self.layers = [
        transformer.SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient

  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Multi_ADAP_Dense(vocab_size, self.num_units, Multi_domain_FeedForwardNetwork_v2)

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def call(self,
           inputs,
           length_or_step=None,
           state=None,
           input_fn=None,
           sampling_probability=None,
           training=None):
    
    self._assert_is_initialized()
    if isinstance(inputs, list):
      rank = inputs[0].shape.ndims
    else:
      rank = inputs.shape.ndims
    domain = inputs[1]
    domain = domain[0]
    if rank == 2:
      if length_or_step.shape.ndims != 0:
        raise ValueError("length_or_step should be a scalar with the current timestep")
      outputs, state, attention = self.step(
          inputs,
          length_or_step,
          state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          training=training)
      logits = self.output_layer(outputs, domain)
    elif rank == 3:
      if length_or_step.shape.ndims != 1:
        raise ValueError("length_or_step should contain the length of each sequence")
      logits, state, attention = self.forward(
          inputs,
          sequence_length=length_or_step,
          initial_state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
    else:
      raise ValueError("Unsupported input rank %d" % rank)
    return logits, state, attention

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):

      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer(tf.stop_gradient(inputs), domain, mask=mask, training=training) + inputs
      else:
        inputs = multi_domain_layer(inputs, domain, mask=mask, training=training) + inputs

    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    domain = inputs[1][0]
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer(outputs, domain)
    return logits, state, attention
    
  def forward_fn(self,
              inputs,
              args_dict,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    domain = inputs[1][0]
    outputs, state, attention = self._run_forward_fn(
        inputs,
        args_dict,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer.forward_fn(outputs, args_dict, domain)
    return logits, state, attention

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

  def _run_forward_fn(self,
           inputs,
           args_dict,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]
    
    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):

      inputs, layer_cache, attention = layer.forward_fn(
          inputs,
          args_dict,
          mask=mask,          
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training) + inputs
      else:
        inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain, mask=mask, training=training) + inputs

    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, new_cache, attention

class Multi_domain_SelfAttentionDecoder_v2(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
               num_domain_units=128,
               ADAP_layer_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               res_using_rate = 1.0,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v2,
               fake_domain_prob=0.1,
               noisy_prob=None,
               ADAP_contribution=None,
               num_sources=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_v2, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = common.LayerNorm()
    self.layers = [
        transformer.SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        if not(multi_domain_adapter_class == Multi_domain_FeedForwardNetwork_v6)
        else multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i, fake_domain_prob= fake_domain_prob, noisy_prob=noisy_prob)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient
    if ADAP_contribution==None:
      ADAP_contribution =[1.0] * num_layers

    self.ADAP_contribution = ADAP_contribution
    print("ADAP contribution", self.ADAP_contribution)
    self.res_using_rate = res_using_rate
  
  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Dense(vocab_size)

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def call(self,
           inputs,
           length_or_step=None,
           state=None,
           input_fn=None,
           sampling_probability=None,
           training=None):
    
    self._assert_is_initialized()
    if isinstance(inputs, list):
      rank = inputs[0].shape.ndims
    else:
      rank = inputs.shape.ndims
    domain = inputs[1]
    domain = domain[0]
    if rank == 2:
      if length_or_step.shape.ndims != 0:
        raise ValueError("length_or_step should be a scalar with the current timestep")
      outputs, state, attention = self.step(
          inputs,
          length_or_step,
          state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          training=training)
      logits = self.output_layer(outputs)
    elif rank == 3:
      if length_or_step.shape.ndims != 1:
        raise ValueError("length_or_step should contain the length of each sequence")
      logits, state, attention = self.forward(
          inputs,
          sequence_length=length_or_step,
          initial_state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
    else:
      raise ValueError("Unsupported input rank %d" % rank)
    return logits, state, attention

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    if training:
      keeping = tf.keras.backend.random_binomial([1], self.res_using_rate)
    else:
      keeping = 1.0
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):

      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer(tf.stop_gradient(inputs), domain, mask=mask, training=training) * self.ADAP_contribution[i] * keeping + inputs
      else:
        inputs = multi_domain_layer(inputs, domain, mask=mask, training=training) * self.ADAP_contribution[i] * keeping + inputs

    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer(outputs)
    return logits, state, attention
    
  def forward_fn(self,
              inputs,
              args_dict,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run_forward_fn(
        inputs,
        args_dict,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer.forward_fn(outputs, args_dict)
    return logits, state, attention

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

  def _run_forward_fn(self,
           inputs,
           args_dict,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]
    
    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):

      inputs, layer_cache, attention = layer.forward_fn(
          inputs,
          args_dict,
          mask=mask,          
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training) * self.ADAP_contribution[i] + inputs
      else:
        inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain, mask=mask, training=training) * self.ADAP_contribution[i] + inputs

    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, new_cache, attention

class Multi_domain_SelfAttentionDecoder_v3(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
               num_domain_units=128,
               ADAP_layer_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=DAFE,
               num_sources=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_v3, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = common.LayerNorm()
    self.layers = [
        transformer.SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="DAFE_%d"%i)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient

  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Dense(vocab_size)

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):

      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer(tf.stop_gradient(inputs), domain, mask=mask, training=training)
      else:
        inputs = multi_domain_layer(inputs, domain, mask=mask, training=training)

    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer(outputs)
    return logits, state, attention
    
  def forward_fn(self,
              inputs,
              args_dict,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run_forward_fn(
        inputs,
        args_dict,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer.forward_fn(outputs, args_dict)
    return logits, state, attention

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

  def _run_forward_fn(self,
           inputs,
           args_dict,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]
    
    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):

      inputs, layer_cache, attention = layer.forward_fn(
          inputs,
          args_dict,
          mask=mask,          
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training)
      else:
        inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain, mask=mask, training=training)

    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, new_cache, attention

class Multi_domain_SelfAttentionDecoder_v5(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
               num_domain_units=128,
               ADAP_layer_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v2,
               ADAP_contribution=None,
               num_sources=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_v5, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = common.LayerNorm()
    self.layers = [
        transformer.SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient
    if ADAP_contribution==None:
      ADAP_contribution=[1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution
  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Multi_ADAP_Dense_v1(vocab_size, self.num_units, Multi_domain_FeedForwardNetwork_v2)

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def call(self,
           inputs,
           length_or_step=None,
           state=None,
           input_fn=None,
           sampling_probability=None,
           training=None):
    
    self._assert_is_initialized()
    if isinstance(inputs, list):
      rank = inputs[0].shape.ndims
    else:
      rank = inputs.shape.ndims
    domain = inputs[1]
    domain = domain[0]
    if rank == 2:
      if length_or_step.shape.ndims != 0:
        raise ValueError("length_or_step should be a scalar with the current timestep")
      outputs, state, attention = self.step(
          inputs,
          length_or_step,
          state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          training=training)
      logits = self.output_layer(outputs, domain)
    elif rank == 3:
      if length_or_step.shape.ndims != 1:
        raise ValueError("length_or_step should contain the length of each sequence")
      logits, state, attention = self.forward(
          inputs,
          sequence_length=length_or_step,
          initial_state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
    else:
      raise ValueError("Unsupported input rank %d" % rank)
    return logits, state, attention

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):

      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer(tf.stop_gradient(inputs), domain, mask=mask, training=training) * self.ADAP_contribution[i] + inputs
      else:
        inputs = multi_domain_layer(inputs, domain, mask=mask, training=training) * self.ADAP_contribution[i] + inputs

    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    domain = inputs[1][0]
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer(outputs, domain)
    return logits, state, attention
    
  def forward_fn(self,
              inputs,
              args_dict,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    domain = inputs[1][0]
    outputs, state, attention = self._run_forward_fn(
        inputs,
        args_dict,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer.forward_fn(outputs, args_dict, domain)
    return logits, state, attention

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

  def _run_forward_fn(self,
           inputs,
           args_dict,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]
    
    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):

      inputs, layer_cache, attention = layer.forward_fn(
          inputs,
          args_dict,
          mask=mask,          
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training) * self.ADAP_contribution[i] + inputs
      else:
        inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain, mask=mask, training=training) * self.ADAP_contribution[i] + inputs

    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, new_cache, attention

class Multi_domain_SelfAttentionDecoder_v6(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
               num_domain_units=128,
               ADAP_layer_stopping_gradient=False,
               ADAP_gate_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
               multi_domain_adapter_gate_class=Multi_domain_Gate,
               ADAP_contribution=None,
               num_sources=1,
               fake_domain_prob=0.1,
               noisy_prob=None,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_v6, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = common.LayerNorm()
    self.layers = [
        transformer.SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        if not(multi_domain_adapter_class in [Multi_domain_FeedForwardNetwork_v6, Multi_domain_FeedForwardNetwork_v8])
        else multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i, fake_domain_prob= fake_domain_prob, noisy_prob=noisy_prob)
        for i in range(num_layers)]
    self.multi_domain_gates = [
        multi_domain_adapter_gate_class(num_units, num_units, num_units, domain_numb=num_domains, name="ADAP_gate_%d"%i)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient
    self.ADAP_gate_stopping_gradient = ADAP_gate_stopping_gradient
    if ADAP_contribution==None:
      ADAP_contribution =[1.0] * num_layers

    self.ADAP_contribution = ADAP_contribution
    print("ADAP contribution", self.ADAP_contribution)
  
  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Dense(vocab_size)

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def call(self,
           inputs,
           length_or_step=None,
           state=None,
           input_fn=None,
           sampling_probability=None,
           training=None,
           internal_node_printing=False):
    
    self._assert_is_initialized()
    if isinstance(inputs, list):
      rank = inputs[0].shape.ndims
    else:
      rank = inputs.shape.ndims
    domain = inputs[1]
    domain = domain[0]
    if rank == 2:
      if length_or_step.shape.ndims != 0:
        raise ValueError("length_or_step should be a scalar with the current timestep")
      outputs, state, attention = self.step(
          inputs,
          length_or_step,
          state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          training=training,
          internal_node_printing=internal_node_printing)
      logits = self.output_layer(outputs)
    elif rank == 3:
      if length_or_step.shape.ndims != 1:
        raise ValueError("length_or_step should contain the length of each sequence")
      logits, state, attention = self.forward(
          inputs,
          sequence_length=length_or_step,
          initial_state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training,
          internal_node_printing=internal_node_printing)
    else:
      raise ValueError("Unsupported input rank %d" % rank)
    return logits, state, attention

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None,
           internal_node_printing=False):
    # Process inputs.
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer, multi_domain_gate) in enumerate(zip(self.layers,self.multi_domain_layers,self.multi_domain_gates)):

      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      if self.ADAP_gate_stopping_gradient:
        g = multi_domain_gate(tf.stop_gradient(inputs), domain, mask=mask, training=training)
      else:
        g = multi_domain_gate(inputs, domain, mask=mask, training=training)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer(tf.stop_gradient(inputs), domain, mask=mask, training=training) * g + inputs * (1-g)
      else:
        inputs = multi_domain_layer(inputs, domain, mask=mask, training=training) * g + inputs * (1-g)
      if internal_node_printing:
        tf.print("###", self.name_scope(), "gate_mean_abs_pooling: ", tf.reduce_mean(tf.abs(g),-1)[0,:], "domain: ", domain, "###", sep="|", summarize=1000)
    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None,
              internal_node_printing=False):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training,
        internal_node_printing=internal_node_printing)
    logits = self.output_layer(outputs)
    return logits, state, attention
    
  def forward_fn(self,
              inputs,
              args_dict,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run_forward_fn(
        inputs,
        args_dict,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer.forward_fn(outputs, args_dict)
    return logits, state, attention

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None,
           internal_node_printing=False):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training,
        internal_node_printing=internal_node_printing)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

  def _run_forward_fn(self,
           inputs,
           args_dict,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]
    
    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer, multi_domain_gate) in enumerate(zip(self.layers,self.multi_domain_layers,self.multi_domain_gates)):

      inputs, layer_cache, attention = layer.forward_fn(
          inputs,
          args_dict,
          mask=mask,          
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      
      if self.ADAP_gate_stopping_gradient:
        g = multi_domain_gate.forward_fn(tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training)
      else:
        g = multi_domain_gate.forward_fn(inputs, args_dict, domain, mask=mask, training=training)
        
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training) * g + inputs * (1-g)
      else:
        inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain, mask=mask, training=training) * g + inputs * (1-g)

    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, new_cache, attention

class Multi_domain_SelfAttentionDecoder_v7(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
               num_domain_units=128,
               ADAP_layer_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_Gate,
               multi_domain_adapter_gate_class=Multi_domain_Gate,
               ADAP_contribution=None,
               num_sources=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_v7, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = common.LayerNorm()
    self.layers = [
        transformer.SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        for i in range(num_layers)]
    self.multi_domain_gates = [
        multi_domain_adapter_gate_class(num_units, num_units, num_units, domain_numb=num_domains, name="ADAP_gate_%d"%i)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient
    if ADAP_contribution==None:
      ADAP_contribution =[1.0] * num_layers

    self.ADAP_contribution = ADAP_contribution
    print("ADAP contribution", self.ADAP_contribution)
  
  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Dense(vocab_size)

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def call(self,
           inputs,
           length_or_step=None,
           state=None,
           input_fn=None,
           sampling_probability=None,
           training=None):
    
    self._assert_is_initialized()
    if isinstance(inputs, list):
      rank = inputs[0].shape.ndims
    else:
      rank = inputs.shape.ndims
    domain = inputs[1]
    domain = domain[0]
    if rank == 2:
      if length_or_step.shape.ndims != 0:
        raise ValueError("length_or_step should be a scalar with the current timestep")
      outputs, state, attention = self.step(
          inputs,
          length_or_step,
          state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          training=training)
      logits = self.output_layer(outputs)
    elif rank == 3:
      if length_or_step.shape.ndims != 1:
        raise ValueError("length_or_step should contain the length of each sequence")
      logits, state, attention = self.forward(
          inputs,
          sequence_length=length_or_step,
          initial_state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
    else:
      raise ValueError("Unsupported input rank %d" % rank)
    return logits, state, attention

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer, multi_domain_gate) in enumerate(zip(self.layers,self.multi_domain_layers,self.multi_domain_gates)):

      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      g = multi_domain_gate(inputs, domain, mask=mask, training=training)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer(tf.stop_gradient(inputs), domain, mask=mask, training=training) * g + inputs * (1-g)
      else:
        inputs = multi_domain_layer(inputs, domain, mask=mask, training=training) * g + inputs * (1-g)

    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer(outputs)
    return logits, state, attention
    
  def forward_fn(self,
              inputs,
              args_dict,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run_forward_fn(
        inputs,
        args_dict,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer.forward_fn(outputs, args_dict)
    return logits, state, attention

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

  def _run_forward_fn(self,
           inputs,
           args_dict,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]
    
    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer, multi_domain_gate) in enumerate(zip(self.layers,self.multi_domain_layers,self.multi_domain_gates)):

      inputs, layer_cache, attention = layer.forward_fn(
          inputs,
          args_dict,
          mask=mask,          
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      g = multi_domain_gate.forward_fn(inputs, args_dict, domain, mask=mask, training=training)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training) * g + inputs * (1-g)
      else:
        inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain, mask=mask, training=training) * g + inputs * (1-g)

    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, new_cache, attention

class Multi_domain_SelfAttentionDecoder_v8(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
               num_domain_units=128,
               ADAP_layer_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_Gate,
               multi_domain_adapter_gate_class=Multi_domain_Gate,
               ADAP_contribution=None,
               num_sources=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_v8, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = common.LayerNorm()
    self.layers = [
        transformer.SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        for i in range(num_layers)]
    self.multi_domain_gates = [
        multi_domain_adapter_gate_class(num_units, num_units, num_units, domain_numb=num_domains, name="ADAP_gate_%d"%i)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient
    if ADAP_contribution==None:
      ADAP_contribution =[1.0] * num_layers

    self.ADAP_contribution = ADAP_contribution
    print("ADAP contribution", self.ADAP_contribution)
  
  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Dense(vocab_size)

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def call(self,
           inputs,
           length_or_step=None,
           state=None,
           input_fn=None,
           sampling_probability=None,
           training=None):
    
    self._assert_is_initialized()
    if isinstance(inputs, list):
      rank = inputs[0].shape.ndims
    else:
      rank = inputs.shape.ndims
    domain = inputs[1]
    domain = domain[0]
    if rank == 2:
      if length_or_step.shape.ndims != 0:
        raise ValueError("length_or_step should be a scalar with the current timestep")
      outputs, state, attention = self.step(
          inputs,
          length_or_step,
          state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          training=training)
      logits = self.output_layer(outputs)
    elif rank == 3:
      if length_or_step.shape.ndims != 1:
        raise ValueError("length_or_step should contain the length of each sequence")
      logits, state, attention = self.forward(
          inputs,
          sequence_length=length_or_step,
          initial_state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
    else:
      raise ValueError("Unsupported input rank %d" % rank)
    return logits, state, attention

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer, multi_domain_gate) in enumerate(zip(self.layers,self.multi_domain_layers,self.multi_domain_gates)):

      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      g = multi_domain_gate(inputs, domain, mask=mask, training=training)
      inputs = multi_domain_layer(g*inputs+(1-g)*tf.stop_gradient(inputs), domain, mask=mask, training=training) + inputs
    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer(outputs)
    return logits, state, attention
    
  def forward_fn(self,
              inputs,
              args_dict,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run_forward_fn(
        inputs,
        args_dict,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer.forward_fn(outputs, args_dict)
    return logits, state, attention

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

  def _run_forward_fn(self,
           inputs,
           args_dict,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]
    
    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer, multi_domain_gate) in enumerate(zip(self.layers,self.multi_domain_layers,self.multi_domain_gates)):

      inputs, layer_cache, attention = layer.forward_fn(
          inputs,
          args_dict,
          mask=mask,          
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      g = multi_domain_gate.forward_fn(inputs, args_dict, domain, mask=mask, training=training)
      inputs = multi_domain_layer.forward_fn(g*inputs + (1-g)*tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training) + inputs
    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, new_cache, attention

class Multi_domain_SelfAttentionDecoder_v9(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
               num_domain_units=128,
               ADAP_layer_stopping_gradient=False,
               ADAP_gate_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
               multi_domain_adapter_gate_class=Multi_domain_Gate_v2,
               input_gate_regularization=False,
               ADAP_contribution=None,
               num_sources=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_v9, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.num_domains = num_domains
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = common.LayerNorm()
    self.layers = [
        transformer.SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        for i in range(num_layers)]
    self.multi_domain_forget_gate = multi_domain_adapter_gate_class(num_units, num_units, num_units, domain_numb=num_domains, name="ADAP_forget_gate")
    self.multi_domain_input_gate = multi_domain_adapter_gate_class(num_units, num_units, num_units, domain_numb=num_domains, name="ADAP_input_gate", output_regularization=input_gate_regularization)

    self.ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient
    self.ADAP_gate_stopping_gradient=ADAP_gate_stopping_gradient
    if ADAP_contribution==None:
      ADAP_contribution =[1.0] * num_layers

    self.ADAP_contribution = ADAP_contribution
    print("ADAP contribution", self.ADAP_contribution)
  
  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Dense(vocab_size)

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def call(self,
           inputs,
           length_or_step=None,
           state=None,
           input_fn=None,
           sampling_probability=None,
           training=None):
    
    self._assert_is_initialized()
    if isinstance(inputs, list):
      rank = inputs[0].shape.ndims
    else:
      rank = inputs.shape.ndims
    domain = inputs[1]
    domain = domain[0]
    if rank == 2:
      if length_or_step.shape.ndims != 0:
        raise ValueError("length_or_step should be a scalar with the current timestep")
      outputs, state, attention = self.step(
          inputs,
          length_or_step,
          state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          training=training)
      logits = self.output_layer(outputs)
    elif rank == 3:
      if length_or_step.shape.ndims != 1:
        raise ValueError("length_or_step should contain the length of each sequence")
      logits, state, attention = self.forward(
          inputs,
          sequence_length=length_or_step,
          initial_state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
    else:
      raise ValueError("Unsupported input rank %d" % rank)
    return logits, state, attention

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    multi_domain_forget_gate = self.multi_domain_forget_gate
    multi_domain_input_gate = self.multi_domain_input_gate
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):
      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      ADAP_input = multi_domain_layer(tf.stop_gradient(inputs), domain, mask=mask, training=training)        
      f = multi_domain_forget_gate(inputs, ADAP_input, mask=mask, training=training)
      i = multi_domain_input_gate(inputs, ADAP_input, mask=mask, training=training)
      inputs = inputs * f + ADAP_input * i
      #if not training:
      #  tf.print(self.name_scope(),"forget_gate:",tf.reduce_mean(tf.abs(f)),"input gate:",tf.reduce_mean(tf.abs(i)),sep="|",stream=sys.stderr)
    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer(outputs)
    return logits, state, attention
    
  def _adv_run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    multi_domain_forget_gate = self.multi_domain_forget_gate
    multi_domain_input_gate = self.multi_domain_input_gate
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):
      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      ADAP_input = tf.stop_gradient(multi_domain_layer(inputs, domain, mask=mask, training=training))
      f = multi_domain_forget_gate(inputs, ADAP_input, mask=mask, training=training)
      i = multi_domain_input_gate(inputs, ADAP_input, mask=mask, training=training)
      inputs = inputs * f + ADAP_input * i
      
    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def adv_forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._adv_run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer(outputs)
    return logits, state, attention

  def forward_fn(self,
              inputs,
              args_dict,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run_forward_fn(
        inputs,
        args_dict,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer.forward_fn(outputs, args_dict)
    return logits, state, attention

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

  def _run_forward_fn(self,
           inputs,
           args_dict,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]
    
    # Run each layer.
    new_cache = []
    multi_domain_forget_gate = self.multi_domain_forget_gate
    multi_domain_input_gate = self.multi_domain_input_gate
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):
      inputs, layer_cache, attention = layer.forward_fn(
          inputs,
          args_dict,
          mask=mask,          
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      if self.ADAP_layer_stopping_gradient: 
        ADAP_input = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), domain, mask=mask, training=training)
        if self.ADAP_gate_stopping_gradient:
          f = multi_domain_forget_gate.forward_fn(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
          i = multi_domain_input_gate.forward_fn(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
        else:
          f = multi_domain_forget_gate.forward_fn(inputs, ADAP_input, mask=mask, training=training)
          i = multi_domain_input_gate.forward_fn(inputs, ADAP_input, mask=mask, training=training)
        inputs = inputs * f + ADAP_input * i
      else:
        ADAP_input = multi_domain_layer.forward_fn(inputs, domain, mask=mask, training=training)
        if self.ADAP_gate_stopping_gradient:
          f = multi_domain_forget_gate.forward_fn(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
          i = multi_domain_input_gate.forward_fn(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
        else:
          f = multi_domain_forget_gate.forward_fn(inputs, ADAP_input, mask=mask, training=training)
          i = multi_domain_input_gate.forward_fn(inputs, ADAP_input, mask=mask, training=training)
        inputs = inputs * f + ADAP_input * i
    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, new_cache, attention

class Multi_domain_SelfAttentionDecoder_v10(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
               num_domain_units=128,
               ADAP_layer_stopping_gradient=False,
               ADAP_gate_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
               multi_domain_adapter_gate_class=Multi_domain_Gate_v2,
               input_gate_regularization=False,
               ADAP_contribution=None,
               num_sources=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_v10, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.num_domains = num_domains
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = common.LayerNorm()
    self.layers = [
        transformer.SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        for i in range(num_layers)]
    self.multi_domain_forget_gates = [multi_domain_adapter_gate_class(num_units, num_units, num_units, domain_numb=num_domains, name="ADAP_forget_gate_%d"%i)
        for i in range(num_layers)]
    self.multi_domain_input_gates = [multi_domain_adapter_gate_class(num_units, num_units, num_units, domain_numb=num_domains, name="ADAP_input_gate_%d"%i, output_regularization=input_gate_regularization)
        for i in range(num_layers)]

    self.ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient
    self.ADAP_gate_stopping_gradient=ADAP_gate_stopping_gradient
    if ADAP_contribution==None:
      ADAP_contribution =[1.0] * num_layers

    self.ADAP_contribution = ADAP_contribution
    print("ADAP contribution", self.ADAP_contribution)
  
  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Dense(vocab_size)

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def call(self,
           inputs,
           length_or_step=None,
           state=None,
           input_fn=None,
           sampling_probability=None,
           training=None):
    
    self._assert_is_initialized()
    if isinstance(inputs, list):
      rank = inputs[0].shape.ndims
    else:
      rank = inputs.shape.ndims
    domain = inputs[1]
    domain = domain[0]
    if rank == 2:
      if length_or_step.shape.ndims != 0:
        raise ValueError("length_or_step should be a scalar with the current timestep")
      outputs, state, attention = self.step(
          inputs,
          length_or_step,
          state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          training=training)
      logits = self.output_layer(outputs)
    elif rank == 3:
      if length_or_step.shape.ndims != 1:
        raise ValueError("length_or_step should contain the length of each sequence")
      logits, state, attention = self.forward(
          inputs,
          sequence_length=length_or_step,
          initial_state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
    else:
      raise ValueError("Unsupported input rank %d" % rank)
    return logits, state, attention

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer, multi_domain_input_gate, multi_domain_forget_gate) in enumerate(zip(self.layers, self.multi_domain_layers, self.multi_domain_input_gates, self.multi_domain_forget_gates)):
      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache) 
      ADAP_input = multi_domain_layer(tf.stop_gradient(inputs), domain, mask=mask, training=training)        
      f = multi_domain_forget_gate(tf.stop_gradient(inputs), tf.stop_gradient(ADAP_input), mask=mask, training=training)
      i_ = multi_domain_input_gate(tf.stop_gradient(inputs), tf.stop_gradient(ADAP_input), mask=mask, training=training)
      inputs = inputs * f + ADAP_input * i_
      if not training:
        tf.print(self.name_scope(),"forget_gate:",tf.reduce_mean(tf.abs(f)),"input gate:",tf.reduce_mean(tf.abs(i_)),sep="|",output_stream=sys.stdout)
    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer(outputs)
    return logits, state, attention
    
  def _adv_run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer, multi_domain_input_gate, multi_domain_forget_gate) in enumerate(zip(self.layers, self.multi_domain_layers, self.multi_domain_input_gates, self.multi_domain_forget_gates)):
      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      ADAP_input = tf.stop_gradient(multi_domain_layer(inputs, domain, mask=mask, training=training))
      f = multi_domain_forget_gate(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
      i_ = multi_domain_input_gate(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
      inputs = inputs * f + ADAP_input * i_
      
    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def adv_forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._adv_run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer(outputs)
    return logits, state, attention

  def forward_fn(self,
              inputs,
              args_dict,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run_forward_fn(
        inputs,
        args_dict,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer.forward_fn(outputs, args_dict)
    return logits, state, attention

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

  def _run_forward_fn(self,
           inputs,
           args_dict,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]
    
    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer, multi_domain_input_gate, multi_domain_forget_gate) in enumerate(zip(self.layers, self.multi_domain_layers, self.multi_domain_input_gates, self.multi_domain_forget_gates)):
      inputs, layer_cache, attention = layer.forward_fn(
          inputs,
          args_dict,
          mask=mask,          
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      ADAP_input = multi_domain_layer.forward_fn(inputs, domain, mask=mask, training=training)
      f = multi_domain_forget_gate.forward_fn(inputs, ADAP_input, mask=mask, training=training)
      i_ = multi_domain_input_gate.forward_fn(inputs, ADAP_input, mask=mask, training=training)
      inputs = inputs * f + ADAP_input * i_
    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, new_cache, attention

class Multi_domain_SelfAttentionDecoder_v11(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
               num_domain_units=128,
               ADAP_layer_stopping_gradient=False,
               ADAP_gate_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
               multi_domain_adapter_gate_class=Regulation_Gate,
               ADAP_contribution=None,
               num_sources=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_v11, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = common.LayerNorm()
    self.layers = [
        transformer.SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        for i in range(num_layers)]
    self.multi_domain_gates = [
        multi_domain_adapter_gate_class(num_units, num_units, num_units, domain_numb=num_domains, name="ADAP_gate_%d"%i)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient
    self.ADAP_gate_stopping_gradient = ADAP_gate_stopping_gradient
    if ADAP_contribution==None:
      ADAP_contribution =[1.0] * num_layers

    self.ADAP_contribution = ADAP_contribution
    print("ADAP contribution", self.ADAP_contribution)
  
  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Dense(vocab_size)

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def call(self,
           inputs,
           length_or_step=None,
           state=None,
           input_fn=None,
           sampling_probability=None,
           training=None):
    
    self._assert_is_initialized()
    if isinstance(inputs, list):
      rank = inputs[0].shape.ndims
    else:
      rank = inputs.shape.ndims
    domain = inputs[1]
    domain = domain[0]
    if rank == 2:
      if length_or_step.shape.ndims != 0:
        raise ValueError("length_or_step should be a scalar with the current timestep")
      outputs, state, attention = self.step(
          inputs,
          length_or_step,
          state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          training=training)
      logits = self.output_layer(outputs)
    elif rank == 3:
      if length_or_step.shape.ndims != 1:
        raise ValueError("length_or_step should contain the length of each sequence")
      logits, state, attention = self.forward(
          inputs,
          sequence_length=length_or_step,
          initial_state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
    else:
      raise ValueError("Unsupported input rank %d" % rank)
    return logits, state, attention

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer, multi_domain_gate) in enumerate(zip(self.layers,self.multi_domain_layers,self.multi_domain_gates)):

      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      if self.ADAP_gate_stopping_gradient:
        g = multi_domain_gate(tf.stop_gradient(inputs), domain, mask=mask, training=training)
      else:
        g = multi_domain_gate(inputs, domain, mask=mask, training=training)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer(tf.stop_gradient(inputs), domain, mask=mask, training=training) * g + inputs * (1-g)
      else:
        inputs = multi_domain_layer(inputs, domain, mask=mask, training=training) * g + inputs * (1-g)

    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer(outputs)
    return logits, state, attention
    
  def forward_fn(self,
              inputs,
              args_dict,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run_forward_fn(
        inputs,
        args_dict,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer.forward_fn(outputs, args_dict)
    return logits, state, attention

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

  def _run_forward_fn(self,
           inputs,
           args_dict,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]
    
    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer, multi_domain_gate) in enumerate(zip(self.layers,self.multi_domain_layers,self.multi_domain_gates)):

      inputs, layer_cache, attention = layer.forward_fn(
          inputs,
          args_dict,
          mask=mask,          
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      
      if self.ADAP_gate_stopping_gradient:
        g = multi_domain_gate.forward_fn(tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training)
      else:
        g = multi_domain_gate.forward_fn(inputs, args_dict, domain, mask=mask, training=training)
        
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training) * g + inputs * (1-g)
      else:
        inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain, mask=mask, training=training) * g + inputs * (1-g)

    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, new_cache, attention

class Multi_domain_SelfAttentionDecoder_v0(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
               num_domain_units=128,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v2,
               num_sources=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_v0, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = common.LayerNorm()
    self.layers = [
        transformer.SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        for i in range(num_layers)]

  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Dense(vocab_size)

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def call(self,
           inputs,
           length_or_step=None,
           state=None,
           input_fn=None,
           sampling_probability=None,
           training=None):
    
    self._assert_is_initialized()
    if isinstance(inputs, list):
      rank = inputs[0].shape.ndims
    else:
      rank = inputs.shape.ndims
    domain = inputs[1]
    domain = domain[0]
    if rank == 2:
      if length_or_step.shape.ndims != 0:
        raise ValueError("length_or_step should be a scalar with the current timestep")
      outputs, state, attention = self.step(
          inputs,
          length_or_step,
          state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          training=training)
      logits = self.output_layer(outputs)
    elif rank == 3:
      if length_or_step.shape.ndims != 1:
        raise ValueError("length_or_step should contain the length of each sequence")
      logits, state, attention = self.forward(
          inputs,
          sequence_length=length_or_step,
          initial_state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
    else:
      raise ValueError("Unsupported input rank %d" % rank)
    return logits, state, attention

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):

      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      inputs = multi_domain_layer(inputs, domain, mask=mask, training=training)

    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer(outputs)
    return logits, state, attention
    
  def forward_fn(self,
              inputs,
              args_dict,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run_forward_fn(
        inputs,
        args_dict,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer.forward_fn(outputs, args_dict)
    return logits, state, attention

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

  def _run_forward_fn(self,
           inputs,
           args_dict,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]
    
    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):

      inputs, layer_cache, attention = layer.forward_fn(
          inputs,
          args_dict,
          mask=mask,          
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain, mask=mask, training=training)

    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, new_cache, attention

class Multi_domain_SelfAttentionDecoder_WDC(Decoder):

  def __init__(self,
               num_layers,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               num_sources=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_WDC, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = common.LayerNorm()
    self.layers = [
        transformer.SelfAttentionDecoderLayer_v1(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]

    self.domain_share_gates = [
      transformer.MultiHeadAttention(
              self.num_heads,
              self.num_units,
              dropout=0.1,
              return_attention=True,)
        for i in range(num_layers)]

    self.domain_specific_gates = [
      transformer.MultiHeadAttention(
              self.num_heads,
              self.num_units,
              dropout=0.1,
              return_attention=False)
        for i in range(num_layers)]

    self.combine_gate = common.Dense(self.num_units, activation=tf.nn.sigmoid)

    """ self.feed_forwards = [
      transformer.FeedForwardNetwork(ffn_inner_dim,
               self.num_units,
               dropout=0.1,
               activation=tf.nn.relu)
        for i in range(num_layers)] """
    
    self.feed_forwards = [
      common.Dense(self.num_units)
      for i in range(num_layers)
    ]

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

  def call(self,
           inputs,
           length_or_step=None,
           state=None,
           input_fn=None,
           sampling_probability=None,
           training=None):
    
    self._assert_is_initialized()
    if isinstance(inputs, list):
      rank = inputs[0].shape.ndims
    else:
      rank = inputs.shape.ndims

    if rank == 2:
      if length_or_step.shape.ndims != 0:
        raise ValueError("length_or_step should be a scalar with the current timestep")
      outputs, state, attention = self.step(
          inputs,
          length_or_step,
          state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          training=training)
      logits = self.output_layer(outputs)
    elif rank == 3:
      if length_or_step.shape.ndims != 1:
        raise ValueError("length_or_step should contain the length of each sequence")
      logits, state, attention = self.forward(
          inputs,
          sequence_length=length_or_step,
          initial_state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
    else:
      raise ValueError("Unsupported input rank %d" % rank)
    return logits, state, attention

  def _run(self,
           inputs,
           h_r,
           h_s,
           encoder_mask,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    for i, (layer, domain_share_gate, domain_specific_gate, feed_forward) in enumerate(zip(self.layers, self.domain_share_gates, self.domain_specific_gates, self.feed_forwards)):
      inputs, self_kv = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      c_r, memory_kv, attention = domain_share_gate(inputs, memory = h_r, mask = encoder_mask, training=training)
      c_s, _ = domain_specific_gate(inputs, memory = h_s, mask = encoder_mask, training=training)
      g_c = self.combine_gate(tf.concat([inputs, c_r, c_s],-1))
      c_l = inputs + g_c * c_r + (1-g_c) * c_s
      inputs = c_l + feed_forward(c_l, training=training)
      layer_cache = dict(self_kv=self_kv, memory_kv=[memory_kv])
      new_cache.append(layer_cache)
      attention = attention[:, 0]
    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    inputs, h_r, h_s, encoder_mask = inputs
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        h_r,
        h_s,
        encoder_mask,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer(outputs)
    return logits, outputs, attention

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None):
    inputs, h_r, h_s, encoder_mask = inputs
    inputs = tf.expand_dims(inputs, 1)
    outputs, state, attention = self._run(
        inputs,
        h_r,
        h_s,
        encoder_mask,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention

  def _get_initial_state(self, batch_size, dtype, initial_state=None):
    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

class Multi_domain_SelfAttentionDecoder_v12(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
               num_domain_units=128,
               ADAP_layer_stopping_gradient=False,
               ADAP_gate_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
               multi_domain_adapter_gate_class=Multi_domain_Gate,
               ADAP_contribution=None,
               num_sources=1,
               fake_domain_prob=0.1,
               noisy_prob=None,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_v12, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = common.LayerNorm()
    self.layers = [
        transformer.SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        if not(multi_domain_adapter_class in [Multi_domain_FeedForwardNetwork_v6, Multi_domain_FeedForwardNetwork_v8])
        else multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i, fake_domain_prob= fake_domain_prob, noisy_prob=noisy_prob)
        for i in range(num_layers)]
    self.multi_domain_gate = transformer.SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
    self.ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient
    self.ADAP_gate_stopping_gradient = ADAP_gate_stopping_gradient
    if ADAP_contribution==None:
      ADAP_contribution =[1.0] * num_layers

    self.ADAP_contribution = ADAP_contribution
    print("ADAP contribution", self.ADAP_contribution)
  
  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Dense(vocab_size)

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def call(self,
           inputs,
           length_or_step=None,
           state=None,
           input_fn=None,
           sampling_probability=None,
           training=None,
           internal_node_printing=False):
    
    self._assert_is_initialized()
    if isinstance(inputs, list):
      rank = inputs[0].shape.ndims
    else:
      rank = inputs.shape.ndims
    domain = inputs[1]
    domain = domain[0]
    if rank == 2:
      if length_or_step.shape.ndims != 0:
        raise ValueError("length_or_step should be a scalar with the current timestep")
      outputs, state, attention = self.step(
          inputs,
          length_or_step,
          state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          training=training,
          internal_node_printing=internal_node_printing)
      logits = self.output_layer(outputs)
    elif rank == 3:
      if length_or_step.shape.ndims != 1:
        raise ValueError("length_or_step should contain the length of each sequence")
      logits, state, attention = self.forward(
          inputs,
          sequence_length=length_or_step,
          initial_state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training,
          internal_node_printing=internal_node_printing)
    else:
      raise ValueError("Unsupported input rank %d" % rank)
    return logits, state, attention

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None,
           internal_node_printing=False):
    # Process inputs.
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []

    if self.ADAP_gate_stopping_gradient:
      g, _, _ = self.multi_domain_gate(
          tf.stop_gradient(inputs),
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache= None,
          training=training)
    else:
      g, _, _ = self.multi_domain_gate(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache= None,
          training=training)

    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):

      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer(tf.stop_gradient(inputs), domain, mask=mask, training=training) * g + inputs * (1-g)
      else:
        inputs = multi_domain_layer(inputs, domain, mask=mask, training=training) * g + inputs * (1-g)

      if internal_node_printing:
        tf.print("###", self.name_scope(), "gate_mean_abs_pooling: ", tf.reduce_mean(tf.abs(g),-1)[0,:], "domain: ", domain, "###", sep="|", summarize=1000)

    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None,
              internal_node_printing=False):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training,
        internal_node_printing=internal_node_printing)
    logits = self.output_layer(outputs)
    return logits, state, attention

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None,
           internal_node_printing=False):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training,
        internal_node_printing=internal_node_printing)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

class Multi_domain_SelfAttentionDecoder_v15(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
               num_domain_units=128,
               ADAP_layer_stopping_gradient=False,
               ADAP_gate_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
               multi_domain_adapter_gate_class=Multi_domain_Gate_v2,
               input_gate_regularization=False,
               ADAP_contribution=None,
               num_sources=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_v15, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.num_domains = num_domains
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = common.LayerNorm()
    self.layers = [
        transformer.SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        for i in range(num_layers)]
    self.multi_domain_forget_gates = [multi_domain_adapter_gate_class(num_units, num_units, num_units, domain_numb=num_domains, name="ADAP_forget_gate")
                                    for i in range(num_layers)]
    self.multi_domain_input_gates = [multi_domain_adapter_gate_class(num_units, num_units, num_units, domain_numb=num_domains, name="ADAP_input_gate", output_regularization=input_gate_regularization)
                                    for i in range(num_layers)]

    self.ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient
    self.ADAP_gate_stopping_gradient=ADAP_gate_stopping_gradient
    if ADAP_contribution==None:
      ADAP_contribution =[1.0] * num_layers

    self.ADAP_contribution = ADAP_contribution
    print("ADAP contribution", self.ADAP_contribution)
  
  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Dense(vocab_size)

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def call(self,
           inputs,
           length_or_step=None,
           state=None,
           input_fn=None,
           sampling_probability=None,
           training=None):
    
    self._assert_is_initialized()
    if isinstance(inputs, list):
      rank = inputs[0].shape.ndims
    else:
      rank = inputs.shape.ndims
    domain = inputs[1]
    domain = domain[0]
    if rank == 2:
      if length_or_step.shape.ndims != 0:
        raise ValueError("length_or_step should be a scalar with the current timestep")
      outputs, state, attention = self.step(
          inputs,
          length_or_step,
          state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          training=training)
      logits = self.output_layer(outputs)
    elif rank == 3:
      if length_or_step.shape.ndims != 1:
        raise ValueError("length_or_step should contain the length of each sequence")
      logits, state, attention = self.forward(
          inputs,
          sequence_length=length_or_step,
          initial_state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
    else:
      raise ValueError("Unsupported input rank %d" % rank)
    return logits, state, attention

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    for i, (layer, multi_domain_layer, multi_domain_forget_gate, multi_domain_input_gate) in enumerate(zip(self.layers, self.multi_domain_layers, self.multi_domain_forget_gates, self.multi_domain_input_gates)):
      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)      
      ADAP_input = multi_domain_layer(inputs, domain, mask=mask, training=training)        
      f = multi_domain_forget_gate(inputs, ADAP_input, mask=mask, training=training)
      i = multi_domain_input_gate(inputs, ADAP_input, mask=mask, training=training)
      inputs = inputs * f + ADAP_input * i
      #if not training:
      #  tf.print(self.name_scope(),"forget_gate:",tf.reduce_mean(tf.abs(f)),"input gate:",tf.reduce_mean(tf.abs(i)),sep="|",stream=sys.stderr)
    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer(outputs)
    return logits, state, attention

  def forward_fn(self,
              inputs,
              args_dict,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run_forward_fn(
        inputs,
        args_dict,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer.forward_fn(outputs, args_dict)
    return logits, state, attention

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

  def _run_forward_fn(self,
           inputs,
           args_dict,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]
    
    # Run each layer.
    new_cache = []
    multi_domain_forget_gate = self.multi_domain_forget_gate
    multi_domain_input_gate = self.multi_domain_input_gate
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):
      inputs, layer_cache, attention = layer.forward_fn(
          inputs,
          args_dict,
          mask=mask,          
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      if self.ADAP_layer_stopping_gradient: 
        ADAP_input = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), domain, mask=mask, training=training)
        if self.ADAP_gate_stopping_gradient:
          f = multi_domain_forget_gate.forward_fn(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
          i = multi_domain_input_gate.forward_fn(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
        else:
          f = multi_domain_forget_gate.forward_fn(inputs, ADAP_input, mask=mask, training=training)
          i = multi_domain_input_gate.forward_fn(inputs, ADAP_input, mask=mask, training=training)
        inputs = inputs * f + ADAP_input * i
      else:
        ADAP_input = multi_domain_layer.forward_fn(inputs, domain, mask=mask, training=training)
        if self.ADAP_gate_stopping_gradient:
          f = multi_domain_forget_gate.forward_fn(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
          i = multi_domain_input_gate.forward_fn(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
        else:
          f = multi_domain_forget_gate.forward_fn(inputs, ADAP_input, mask=mask, training=training)
          i = multi_domain_input_gate.forward_fn(inputs, ADAP_input, mask=mask, training=training)
        inputs = inputs * f + ADAP_input * i
    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, new_cache, attention

class Multi_domain_SelfAttentionDecoder_v16(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
               num_domain_units=128,
               ADAP_layer_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
               inner_layer_norm=None,
               fake_domain_prob=0.1,
               noisy_prob=None,
               ADAP_contribution=None,
               num_sources=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_v16, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = common.LayerNorm()
    self.layers = [
        transformer.SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, inner_layer_norm=inner_layer_norm, name="ADAP_%d"%i)
        if not(multi_domain_adapter_class == Multi_domain_FeedForwardNetwork_v6)
        else multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i, fake_domain_prob= fake_domain_prob, noisy_prob=noisy_prob)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient
    if ADAP_contribution==None:
      ADAP_contribution =[1.0] * num_layers

    self.ADAP_contribution = ADAP_contribution
    print("ADAP contribution", self.ADAP_contribution)
  
  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Dense(vocab_size)

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def call(self,
           inputs,
           length_or_step=None,
           state=None,
           input_fn=None,
           sampling_probability=None,
           training=None):
    
    self._assert_is_initialized()
    if isinstance(inputs, list):
      rank = inputs[0].shape.ndims
    else:
      rank = inputs.shape.ndims
    domain = inputs[1]
    domain = domain[0]
    if rank == 2:
      if length_or_step.shape.ndims != 0:
        raise ValueError("length_or_step should be a scalar with the current timestep")
      outputs, state, attention = self.step(
          inputs,
          length_or_step,
          state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          training=training)
      logits = self.output_layer(outputs)
    elif rank == 3:
      if length_or_step.shape.ndims != 1:
        raise ValueError("length_or_step should contain the length of each sequence")
      logits, state, attention = self.forward(
          inputs,
          sequence_length=length_or_step,
          initial_state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
    else:
      raise ValueError("Unsupported input rank %d" % rank)
    return logits, state, attention

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    total_adapt = []
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):
      if self.ADAP_layer_stopping_gradient:
        adapt = multi_domain_layer(layer(
          tf.stop_gradient(inputs),
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)[0], domain, mask=mask, training=training)
        total_adapt.append(adapt)
        inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
        new_cache.append(layer_cache)
      else:
        inputs, layer_cache, attention = layer(
            inputs,
            mask=mask,
            memory=memory,
            memory_mask=memory_mask,
            cache=cache[i] if cache is not None else None,
            training=training)
        new_cache.append(layer_cache)
        adapt = multi_domain_layer(inputs, domain, mask=mask, training=training)
        total_adapt.append(adapt)
    outputs = self.layer_norm(inputs + tf.add_n(total_adapt))
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer(outputs)
    return logits, state, attention
    
  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

  def forward_fn(self,
              inputs,
              args_dict,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run_forward_fn(
        inputs,
        args_dict,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer.forward_fn(outputs, args_dict)
    return logits, state, attention

  def _run_forward_fn(self,
            inputs,
            args_dict,
            sequence_length=None,
            cache=None,
            memory=None,
            memory_sequence_length=None,
            step=None,
            training=None):
      domain = inputs[1]
      domain = domain[0]
      inputs = inputs[0]
      inputs *= self.num_units**0.5
      if self.position_encoder is not None:
        inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
      inputs = common.dropout(inputs, self.dropout, training=training)

      # Prepare query mask.
      mask = None
      if step is None:
        maximum_length = tf.shape(inputs)[1]
        if sequence_length is None:
          batch_size = tf.shape(inputs)[0]
          sequence_length = tf.fill([batch_size], maximum_length)
        mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

      # Prepare memory mask.
      memory_mask = None
      if memory is not None:
        if not isinstance(memory, (list, tuple)):
          memory = (memory,)
      if memory_sequence_length is not None:
        if not isinstance(memory_sequence_length, (list, tuple)):
          memory_sequence_length = (memory_sequence_length,)
        memory_mask = [
            tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
            for mem, mem_length in zip(memory, memory_sequence_length)]
      
      # Run each layer.
      new_cache = []
      total_adapt=[]
      for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):
        if self.ADAP_layer_stopping_gradient:
          adapt = multi_domain_layer.forward_fn(layer.forward_fn(
            tf.stop_gradient(inputs),
            args_dict,
            mask=mask,          
            memory=memory,
            memory_mask=memory_mask,
            cache=cache[i] if cache is not None else None,
            training=training)[0], args_dict, domain, mask=mask, training=training)
          total_adapt.append(adapt)
          inputs, layer_cache, attention = layer.forward_fn(
            inputs,
            args_dict,
            mask=mask,          
            memory=memory,
            memory_mask=memory_mask,
            cache=cache[i] if cache is not None else None,
            training=training)
          new_cache.append(layer_cache)
        else:
          inputs, layer_cache, attention = layer.forward_fn(
              inputs,
              args_dict,
              mask=mask,          
              memory=memory,
              memory_mask=memory_mask,
              cache=cache[i] if cache is not None else None,
              training=training)
          new_cache.append(layer_cache)
          adapt = multi_domain_layer.forward_fn(inputs, args_dict, domain, mask=mask, training=training)
          total_adapt.append(adapt)

      outputs = self.layer_norm.forward_fn(inputs+tf.add_n(total_adapt), args_dict)
      return outputs, new_cache, attention

class Multi_domain_SelfAttentionDecoder_v17(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
               num_domain_units=128,
               domain_region_sizes = None,
               ADAP_layer_stopping_gradient=False,
               ADAP_gate_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               training_res_using_rate=0.0,
               testing_res_using_rate=0.0,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
               multi_domain_adapter_gate_class=Multi_domain_classification_gate,
               ADAP_contribution=None,
               num_sources=1,
               fake_domain_prob=0.1,
               noisy_prob=None,
               version=1,
               inner_layer_norm=None,
               stop_gradient_version=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_v17, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.num_domains = num_domains
    self.num_layers = num_layers
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()

    if version==18:
      self.layer_norm = common.Multi_LayerNorm(num_domains)
      self.layers = [
          transformer.SelfAttentionDecoderLayer_v1(
              num_units,
              num_heads,
              ffn_inner_dim,
              domain_numb = num_domains,
              dropout=dropout,
              attention_dropout=attention_dropout,
              ffn_dropout=ffn_dropout,
              ffn_activation=ffn_activation)
          for i in range(num_layers)] 

    elif version==21:
      self.layer_norm = common.LayerNorm()
      self.layers = [
          transformer.SelfAttentionDecoderLayer(
              num_units,
              num_heads,
              ffn_inner_dim,
              dropout=dropout,
              attention_dropout=attention_dropout,
              ffn_dropout=ffn_dropout,
              ffn_activation=ffn_activation)
          for i in range(num_layers)] 
      
    elif version==20:
      self.layer_norm = common.Multi_LayerNorm(num_domains)
      self.layers = [
          transformer.SelfAttentionDecoderLayer_v3(
              num_units,
              num_heads,
              ffn_inner_dim,
              domain_numb = num_domains,
              dropout=dropout,
              attention_dropout=attention_dropout,
              ffn_dropout=ffn_dropout,
              ffn_activation=ffn_activation)
          for i in range(num_layers)] 
    elif version==19:
      self.layer_norm = common.Multi_LayerNorm(num_domains)
      self.layers = [
          transformer.SelfAttentionDecoderLayer_v2(
              num_units,
              num_heads,
              ffn_inner_dim,
              domain_numb = num_domains,
              num_domain_units = num_domain_units,
              dropout=dropout,
              attention_dropout=attention_dropout,
              ffn_dropout=ffn_dropout,
              ffn_activation=ffn_activation)
          for i in range(num_layers)] 
    else:
      self.layer_norm = common.LayerNorm()
      self.layers = [
          transformer.SelfAttentionDecoderLayer(
              num_units,
              num_heads,
              ffn_inner_dim,
              dropout=dropout,
              attention_dropout=attention_dropout,
              ffn_dropout=ffn_dropout,
              ffn_activation=ffn_activation)
          for i in range(num_layers)] 
    print("inner_layer_norm: ", inner_layer_norm)
    print("num_domains: ", num_domains)
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, inner_layer_norm=inner_layer_norm, name="ADAP_%d"%i)
        if not(multi_domain_adapter_class in [Multi_domain_FeedForwardNetwork_v6, Multi_domain_FeedForwardNetwork_v8])
        else multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i, fake_domain_prob= fake_domain_prob, noisy_prob=noisy_prob)
        for i in range(num_layers)]

    if version == 17:
      self.multi_domain_gate = Multi_domain_classification_gate_v2(num_units, num_units, domain_numb=num_domains, name="ADAP_gate")
    else:
      self.multi_domain_gate = multi_domain_adapter_gate_class(num_units, num_units, domain_numb=num_domains, name="ADAP_gate")
    
    if version in [18,19,20]:
      self.domain_mask = make_domain_mask(self.num_domains,  num_units=num_units, num_domain_units=num_domain_units, domain_region_sizes=domain_region_sizes)
      with np.printoptions(threshold=np.inf):
        print("domain_mask:",self.domain_mask)
    self.ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient
    self.ADAP_gate_stopping_gradient = ADAP_gate_stopping_gradient
    if ADAP_contribution==None:
      ADAP_contribution =[1.0] * num_layers
    self.version = version
    self.stop_gradient_version=stop_gradient_version
    if self.version==1:
      print("version 1: h' = h(1-z)+adap(h_[1,..6])*z")
    elif self.version==2:
      print("version 2: h' = h+adap(h_[1,..6])*z")
    elif self.version==3:
      print("version 3: h' = h")
    elif self.version==5:
      print("version 5: h' = h+adap(h_[1,..6])*activation(z)")
    elif self.version==6:
      print("version 6: h' = h(1-activation(z))+adap(h_[1,..6])*activation(z)")
    elif self.version==7:
      print("version 7: h' = h + adap(h_[1,..6])")
    elif self.version==8:
      print("version 8: h' = h+adap(h)*activation(z)")
    elif self.version==9:
      print("version 9: h' = h + adap(h)")
    elif self.version==10:
      print("version 10: h_3 = h_3 + adap(h_3)")
    elif self.version==11:
      print("version 11: h_3(1,5) = h_3(1,5) + adap(h_3(1,5))")
    elif self.version==12:
      print("version 12: h_1 = h_1 + adap(h_1)")
    elif self.version==15:
      print("version 15: h_{1..5} = h_{1..5} + adap(h_{1..5})")
    elif self.version==16:
      print("version 16: h_{1..5} = h_{1..5} * lhuc(h_{1..5})")  
    self.ADAP_contribution = ADAP_contribution
    self.training_res_using_rate = training_res_using_rate
    self.testing_res_using_rate = testing_res_using_rate
    print("ADAP contribution", self.ADAP_contribution)
    print("testing_res_using_rate: ", testing_res_using_rate)
    print("training_res_using_rate: ", training_res_using_rate)
  
  def build(self, input_shape):
    super(Multi_domain_SelfAttentionDecoder_v17, self).build(input_shape)
    scope_name = self.name_scope()
    self.lhuc_embedding = []
    if self.version==16:
      for i in range(self.num_layers):
        self.lhuc_embedding.append(self.add_weight("%s_lhuc_%d"%(scope_name,i), shape=[self.num_domains, self.num_units]))
  
  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Dense(vocab_size)
    
  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def call(self,
           inputs,
           length_or_step=None,
           state=None,
           input_fn=None,
           sampling_probability=None,
           training=None,
           adapter_activate=True,
           internal_node_printing=False):
    
    self._assert_is_initialized()
    if isinstance(inputs, list):
      rank = inputs[0].shape.ndims
    else:
      rank = inputs.shape.ndims
    domain = inputs[1]
    domain = domain[0]
    if rank == 2:
      if length_or_step.shape.ndims != 0:
        raise ValueError("length_or_step should be a scalar with the current timestep")
      outputs, state, attention = self.step(
          inputs,
          length_or_step,
          state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          training=training,
          adapter_activate=adapter_activate,
          internal_node_printing=internal_node_printing)
      logits = self.output_layer(outputs)
    elif rank == 3:
      if length_or_step.shape.ndims != 1:
        raise ValueError("length_or_step should contain the length of each sequence")
      logits, state, attention = self.forward(
          inputs,
          sequence_length=length_or_step,
          initial_state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training,
          adapter_activate=adapter_activate,
          internal_node_printing=internal_node_printing)
    else:
      raise ValueError("Unsupported input rank %d" % rank)
    return logits, state, attention

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None,
           adapter_activate=True,
           internal_node_printing=True):
    # Process inputs.
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)
    
    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    total_adapt = []
    if self.version==17:
      from collections import defaultdict
      total_adapt = defaultdict(list)
    if training:
      keeping = tf.keras.backend.random_binomial([1], self.training_res_using_rate)
    else:
      keeping = self.testing_res_using_rate
    
    if self.version in [18,19,20]:
        domain_mask = tf.nn.embedding_lookup(self.domain_mask, domain)
        inputs = tf.math.multiply(inputs, domain_mask)

    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):
      
      if self.version in [18,19,20]:        
        inputs, layer_cache, attention = layer(
          inputs,
          domain,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
        new_cache.append(layer_cache)
        domain_mask = tf.nn.embedding_lookup(self.domain_mask, domain)
        inputs = tf.math.multiply(inputs, domain_mask)
      elif self.version == 21:
        inputs, layer_cache, attention = layer(
          inputs,
          domain,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
        new_cache.append(layer_cache)
      else:
        inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
        new_cache.append(layer_cache)
        
      if self.version not in [3,8,9,10,11,12,15,16,17,18,19,20,21]:
        adapt = multi_domain_layer(inputs, domain, mask=mask, training=training)
        total_adapt.append(adapt)
      if self.version == 17:
        for d in range(self.num_domains-1):
          adapt = multi_domain_layer(inputs, d, mask=mask, training=training)
          total_adapt[d].append(adapt)
      if self.version == 10:
        if i==3:
          adapt = multi_domain_layer(inputs, domain, mask=mask, training=training)
          inputs = inputs + adapt
      if self.version == 11:
        if i in [1,3,5]:
          adapt = multi_domain_layer(inputs, domain, mask=mask, training=training)
          inputs = inputs + adapt
      if self.version == 12:
        if i==1:
          adapt = multi_domain_layer(inputs, domain, mask=mask, training=training)
          inputs = inputs + adapt
      if self.version == 15:
        adapt = multi_domain_layer(inputs, domain, mask=mask, training=training)
        inputs = inputs + tf.nn.dropout(adapt, 1-self.training_res_using_rate)
      if self.version == 16:
        lhuc_vector = tf.nn.embedding_lookup(self.lhuc_embedding[i], domain)
        lhuc_scale = 2 * tf.math.sigmoid(lhuc_vector)
        inputs = tf.math.multiply(inputs, lhuc_scale) + inputs

    if self.version not in [3,8,9,10,11,12,15,16,17,18,19,20,21]:
      total_adapt = tf.add_n(total_adapt)
    elif self.version in [8,9]:
      total_adapt = self.multi_domain_layers[-1](inputs, domain, mask=mask, training=training)
    if self.version not in [3,7,9,10,11,12,15,16,17,18,19,20,21]:
      g = self.multi_domain_gate(inputs, domain, mask=mask, training=training)
    if self.version == 17:
      g = self.multi_domain_gate(inputs, domain, mask=mask, training=training, tag="decoder prediction")
      #if internal_node_printing:
      #tf.print("###decoder", self.name_scope(), "gate_mean_abs_pooling: ", g[domain,:,0], "domain: ", domain, "###", sep="|", summarize=-1)
    if self.version==1:
      outputs = self.layer_norm(inputs * (1-g) + total_adapt * g)
    elif self.version==2:
      outputs = self.layer_norm(inputs + total_adapt * g)
    elif self.version==3:
      outputs = self.layer_norm(inputs)
    elif self.version==5:
      outputs = self.layer_norm(inputs + tf.exp((g-1)*2/g) * total_adapt)
    elif self.version==6:
      z = tf.exp((g-1)*2/g)
      outputs = self.layer_norm(inputs * (1-z) + z * total_adapt)
    elif self.version==17:
      for d in total_adapt:
        total_adapt[d] = tf.add_n(total_adapt[d])
        b = tf.shape(total_adapt[d])[0]
        total_adapt[d] = tf.expand_dims(tf.reshape(total_adapt[d],[-1,self.num_units]),0)
      
      inputs = tf.expand_dims(tf.reshape(inputs,[-1,self.num_units]),0)
      all_values = list(total_adapt.values()) + [inputs]
      #tf.print("g:",g.shape,tf.concat(all_values,1).shape)
      total_adapt = tf.math.reduce_sum(tf.concat(all_values,0) * tf.stop_gradient(g),0)
      total_adapt = tf.reshape(total_adapt,[b,-1,self.num_units])
      #tf.print("total_adapt",tf.shape(total_adapt))
      outputs = self.layer_norm(total_adapt)
    elif self.version==7:
      outputs = self.layer_norm(inputs + tf.nn.dropout(total_adapt,1-self.training_res_using_rate))
    elif self.version==9:
      outputs = self.layer_norm(inputs + total_adapt)
    elif self.version==8:
      outputs = self.layer_norm(inputs + tf.exp((g-1)*2/g) * total_adapt)
    elif self.version in [10,11,12]:
      outputs = self.layer_norm(inputs)
    elif self.version == 15:
      outputs = self.layer_norm(inputs)
    elif self.version ==16:
      outputs = self.layer_norm(inputs)
    elif self.version == 18:
      outputs = self.layer_norm(inputs, domain)
    elif self.version == 19:
      outputs = self.layer_norm(inputs, domain)
    elif self.version == 20:
      outputs = self.layer_norm(inputs, domain)
    elif self.version == 21:
      outputs = self.layer_norm(inputs, domain)

    return outputs, new_cache, attention
  
  def _adv_run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    total_adapt = []
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):

      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      if self.version!=3:
        adapt = multi_domain_layer(inputs, domain, mask=mask, training=training)
        total_adapt.append(adapt)
    
    if self.version!=3:
      total_adapt = tf.add_n(total_adapt)
      if self.stop_gradient_version==1:
        g = self.multi_domain_gate(tf.stop_gradient(inputs), domain, mask=mask, training=training)
      else:
        g = self.multi_domain_gate(inputs, domain, mask=mask, training=training)
      if self.stop_gradient_version==1:
        if self.ADAP_gate_stopping_gradient:
          if isinstance(self.ADAP_gate_stopping_gradient, float):
            print("stopping gradient at d_classifier in decoder: ", self.ADAP_gate_stopping_gradient)
            g = tf.stop_gradient(g * (1-self.ADAP_gate_stopping_gradient)) + g * self.ADAP_gate_stopping_gradient
          elif isinstance(self.ADAP_gate_stopping_gradient, bool):
            print("stopping gradient at d_classifier in decoder")
            g = tf.stop_gradient(g)
      
    if self.version==1:
      outputs = self.layer_norm(inputs * (1-g) + total_adapt * g)
    elif self.version==2:
      outputs = self.layer_norm(inputs + total_adapt * g)
    elif self.version==3:
      outputs = self.layer_norm(inputs)
    elif self.version==5:
      outputs = self.layer_norm(inputs + tf.exp((g-1)*2/g) * total_adapt)
    elif self.version==6:
      z = tf.exp((g-1)*2/g)
      outputs = self.layer_norm(inputs * (1-z) + z * total_adapt)
    elif self.version==7:
      outputs = self.layer_norm(inputs + total_adapt)
    elif self.version==8:
      z = tf.exp((g-1)*2/g)
      outputs = self.layer_norm(inputs * (1-z) + z * tf.linalg.normalize(total_adapt,axis=-1)[0])

    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None,
              adapter_activate=True,
              internal_node_printing=False):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training,
        adapter_activate=adapter_activate,
        internal_node_printing=internal_node_printing)
    logits = self.output_layer(outputs)
    return logits, state, attention

  def adv_forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._adv_run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer(outputs)
    return logits, state, attention
  
  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None,
           adapter_activate=True,
           internal_node_printing=False):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training,
        adapter_activate=adapter_activate,
        internal_node_printing=internal_node_printing)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

class Multi_domain_SelfAttentionDecoder_v18(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
               num_domain_units=128,
               ADAP_layer_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
               multi_domain_adapter_gate_class=Multi_domain_classification_gate,
               fake_domain_prob=0.1,
               noisy_prob=None,
               ADAP_contribution=None,
               inner_layer_norm=None,
               version=1,
               num_sources=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_v18, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = common.LayerNorm()
    self.layers = [
        transformer.SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]
    self.multi_domain_layers = [ multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains,inner_layer_norm=inner_layer_norm, name="ADAP_%d"%i) for i in range(num_layers)]
    self.multi_domain_gate = multi_domain_adapter_gate_class(num_units, num_units, domain_numb=num_domains, name="ADAP_gate")
    self.ADAP_layer_stopping_gradient=ADAP_layer_stopping_gradient
    if ADAP_contribution==None:
      ADAP_contribution =[1.0] * num_layers

    self.ADAP_contribution = ADAP_contribution
    self.version=version
    print("ADAP contribution", self.ADAP_contribution)
    print("dec version: ", self.version)
  
  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Dense(vocab_size)

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def call(self,
           inputs,
           length_or_step=None,
           state=None,
           input_fn=None,
           sampling_probability=None,
           training=None):
    
    self._assert_is_initialized()
    if isinstance(inputs, list):
      rank = inputs[0].shape.ndims
    else:
      rank = inputs.shape.ndims
    domain = inputs[1]
    domain = domain[0]
    if rank == 2:
      if length_or_step.shape.ndims != 0:
        raise ValueError("length_or_step should be a scalar with the current timestep")
      outputs, state, attention = self.step(
          inputs,
          length_or_step,
          state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          training=training)
      logits = self.output_layer(outputs)
    elif rank == 3:
      if length_or_step.shape.ndims != 1:
        raise ValueError("length_or_step should contain the length of each sequence")
      logits, state, attention = self.forward(
          inputs,
          sequence_length=length_or_step,
          initial_state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
    else:
      raise ValueError("Unsupported input rank %d" % rank)
    return logits, state, attention

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    total_adapt=[]
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):

      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      if self.version==1:
        inputs = multi_domain_layer(inputs, domain, mask=mask, training=training) * self.ADAP_contribution[i] + inputs
      elif self.version==2:
        adapt = multi_domain_layer(inputs, domain, mask=mask, training=training)
        total_adapt.append(adapt)
      elif self.version==7:
        adapt = multi_domain_layer(inputs, domain, mask=mask, training=training)
        total_adapt.append(adapt)

      if self.version == 10:
        if i==3:
          adapt = multi_domain_layer(inputs, domain, mask=mask, training=training)
          inputs = inputs + adapt
      if self.version == 11:
        if i in [1,3,5]:
          adapt = multi_domain_layer(inputs, domain, mask=mask, training=training)
          inputs = inputs + adapt
      if self.version == 12:
        if i==1:
          adapt = multi_domain_layer(inputs, domain, mask=mask, training=training)
          inputs = inputs + adapt
      if self.version == 9:
        if i==5:
          adapt = multi_domain_layer(inputs, domain, mask=mask, training=training)
          inputs = inputs + adapt

    if self.version in [6,7]:
      domain_g = self.multi_domain_gate(inputs,domain,mask=mask,training=training)

    if self.version==1:
      outputs = self.layer_norm(inputs)
    elif self.version==2:
      outputs = self.layer_norm(inputs + tf.add_n(total_adapt))
    elif self.version==7:
      outputs = self.layer_norm(inputs + tf.exp((domain_g-1)*2/domain_g) * tf.add_n(total_adapt))
    else:
      outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer(outputs)
    return logits, state, attention

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

class Multi_domain_SelfAttentionDecoder_v19(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
               num_domain_units=128,
               ADAP_layer_stopping_gradient=False,
               ADAP_gate_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v9,
               ADAP_contribution=None,
               num_sources=1,
               fake_domain_prob=0.1,
               noisy_prob=None,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_v19, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = common.LayerNorm()
    self.layers = [
        transformer.SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    self.ADAP_gate_stopping_gradient = ADAP_gate_stopping_gradient
    if ADAP_contribution==None:
      ADAP_contribution =[1.0] * num_layers

    self.ADAP_contribution = ADAP_contribution
    print("ADAP contribution", self.ADAP_contribution)
  
  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Dense(vocab_size)

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def call(self,
           inputs,
           length_or_step=None,
           state=None,
           input_fn=None,
           sampling_probability=None,
           training=None,
           internal_node_printing=False):
    
    self._assert_is_initialized()
    if isinstance(inputs, list):
      rank = inputs[0].shape.ndims
    else:
      rank = inputs.shape.ndims
    if rank == 2:
      if length_or_step.shape.ndims != 0:
        raise ValueError("length_or_step should be a scalar with the current timestep")
      outputs, state, attention = self.step(
          inputs,
          length_or_step,
          state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          training=training,
          internal_node_printing=internal_node_printing)
      logits = self.output_layer(outputs)
    elif rank == 3:
      if length_or_step.shape.ndims != 1:
        raise ValueError("length_or_step should contain the length of each sequence")
      logits, state, attention = self.forward(
          inputs,
          sequence_length=length_or_step,
          initial_state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training,
          internal_node_printing=internal_node_printing)
    else:
      raise ValueError("Unsupported input rank %d" % rank)
    return logits, state, attention

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None,
           internal_node_printing=False):
    # Process inputs.
    domain = inputs[1]
    #domain = domain[0] (domain is vector of prob now)
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    total_adapt = []
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):

      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
      if self.ADAP_contribution[i]>0:
        adapt = multi_domain_layer(inputs, domain, mask=mask, training=training)
        total_adapt.append(adapt)
    if len(total_adapt)>0:
      total_adapt = tf.add_n(total_adapt)
    else:
      total_adapt = 0
    outputs = self.layer_norm(inputs + total_adapt)
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None,
              internal_node_printing=False):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training,
        internal_node_printing=internal_node_printing)
    logits = self.output_layer(outputs)
    return logits, state, attention

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None,
           internal_node_printing=False):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training,
        internal_node_printing=internal_node_printing)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

  def dynamic_decode(self,
                     embeddings,
                     start_ids,
                     end_id=constants.END_OF_SENTENCE_ID,
                     initial_state=None,
                     decoding_strategy=None,
                     sampler=None,
                     maximum_iterations=None,
                     minimum_iterations=0):
    
    if isinstance(embeddings, text_inputter.WordEmbedder):
      print("2!")
      input_fn = lambda ids: embeddings({"ids": ids})
    elif callable(embeddings):
      print("1!")
      input_fn = embeddings
    else:
      print("3!")
      input_fn = lambda ids: tf.nn.embedding_lookup(embeddings, ids)

    return decoding.dynamic_decode(
        lambda ids, step, state: self(input_fn(ids), step, state),
        start_ids,
        end_id=end_id,
        initial_state=initial_state,
        decoding_strategy=decoding_strategy,
        sampler=sampler,
        maximum_iterations=maximum_iterations,
        minimum_iterations=minimum_iterations,
        attention_history=self.support_alignment_history,
        attention_size=tf.shape(self.memory)[1] if self.support_alignment_history else None)

class SelfAttentionDecoder_v1(Decoder):

  def __init__(self,
               num_layers,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               num_sources=1,
               **kwargs):
    
    super(SelfAttentionDecoder_v1, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = common.LayerNorm()
    self.layers = [
        transformer.SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    for i, layer in enumerate(self.layers):
      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)
    logits = self.output_layer(outputs)
    return (logits, outputs), state, attention

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None):
    inputs = tf.expand_dims(inputs, 1)
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention

  def _get_initial_state(self, batch_size, dtype, initial_state=None):
    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

class Multi_domain_SelfAttentionDecoder_sparse(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
               num_domain_units=128,
               domain_region_sizes = None,
               ADAP_layer_stopping_gradient=False,
               ADAP_gate_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               training_res_using_rate=0.0,
               testing_res_using_rate=0.0,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
               multi_domain_adapter_gate_class=Multi_domain_classification_gate,
               ADAP_contribution=None,
               num_sources=1,
               fake_domain_prob=0.1,
               noisy_prob=None,
               version=1,
               inner_layer_norm=None,
               stop_gradient_version=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_sparse, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.num_domains = num_domains
    self.num_layers = num_layers
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()

    self.version = version
    print("decoder_version",version)
    if version==1:
      self.layer_norm = common.Multi_LayerNorm(num_domains)
      self.layers = [
        transformer.SelfAttentionDecoderLayer_v1(
            num_units,
            num_heads,
            ffn_inner_dim,
            domain_numb = num_domains,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)] 
    else:
      self.layer_norm = common.LayerNorm()
      self.layers = [
        transformer.SelfAttentionDecoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)] 

           
  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Dense(vocab_size)
    
  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def call(self,
           inputs,
           length_or_step=None,
           state=None,
           input_fn=None,
           sampling_probability=None,
           training=None,
           adapter_activate=True,
           internal_node_printing=False):
    
    self._assert_is_initialized()
    if isinstance(inputs, list):
      rank = inputs[0].shape.ndims
    else:
      rank = inputs.shape.ndims
    
    if rank == 2:
      if length_or_step.shape.ndims != 0:
        raise ValueError("length_or_step should be a scalar with the current timestep")
      outputs, state, attention = self.step(
          inputs,
          length_or_step,
          state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          training=training,
          adapter_activate=adapter_activate,
          internal_node_printing=internal_node_printing)
      logits = self.output_layer(outputs)
    elif rank == 3:
      if length_or_step.shape.ndims != 1:
        raise ValueError("length_or_step should contain the length of each sequence")
      logits, state, attention = self.forward(
          inputs,
          sequence_length=length_or_step,
          initial_state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training,
          adapter_activate=adapter_activate,
          internal_node_printing=internal_node_printing)
    else:
      raise ValueError("Unsupported input rank %d" % rank)
    return logits, state, attention

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None,
           adapter_activate=True,
           internal_node_printing=True):
    # Process inputs.
    domain = inputs[1]
    domain_mask = inputs[2]
    inputs = inputs[0]
    domain = domain[0]
    
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)
    
    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
        
    for i, layer in enumerate(self.layers):
      if self.version==1:
        inputs, layer_cache, attention = layer(
          inputs,
          domain,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      else:
        inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)

      new_cache.append(layer_cache)
        
      inputs = tf.math.multiply(inputs, domain_mask)
    
    if self.version==1:
      outputs = self.layer_norm(inputs, domain)
    else:
      outputs = self.layer_norm(inputs)

    return outputs, new_cache, attention
  
  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None,
              adapter_activate=True,
              internal_node_printing=False):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training,
        adapter_activate=adapter_activate,
        internal_node_printing=internal_node_printing)
    logits = self.output_layer(outputs)
    return logits, state, attention
  
  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None,
           adapter_activate=True,
           internal_node_printing=False):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1], inputs[2]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training,
        adapter_activate=adapter_activate,
        internal_node_printing=internal_node_printing)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

class Multi_domain_SelfAttentionDecoder_sparse_multi_layer(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
               num_domain_units=128,
               domain_region_sizes = None,
               ADAP_layer_stopping_gradient=False,
               ADAP_gate_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               training_res_using_rate=0.0,
               testing_res_using_rate=0.0,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
               multi_domain_adapter_gate_class=Multi_domain_classification_gate,
               ADAP_contribution=None,
               num_sources=1,
               fake_domain_prob=0.1,
               noisy_prob=None,
               version=1,
               inner_layer_norm=None,
               stop_gradient_version=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_sparse_multi_layer, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.num_domains = num_domains
    self.num_layers = num_layers
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()

    self.version = version
    print("decoder_version",version)
    if version==1:
      self.layer_norm = common.Multi_LayerNorm(num_domains)
      self.layers = [
        transformer.SelfAttentionDecoderLayer_v1(
            num_units,
            num_heads,
            ffn_inner_dim,
            domain_numb = num_domains,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)] 
    elif version==3:
      self.layer_norm = common.LayerNorm_v2()
      self.layers = [
        transformer.SelfAttentionDecoderLayer_v4(
            num_units,
            num_heads,
            ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)] 
    else:
      self.layer_norm = common.LayerNorm()
      self.layers = [
        transformer.SelfAttentionDecoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)] 

           
  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Dense(vocab_size)
    
  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def call(self,
           inputs,
           length_or_step=None,
           state=None,
           input_fn=None,
           sampling_probability=None,
           training=None,
           adapter_activate=True,
           internal_node_printing=False):
    
    self._assert_is_initialized()
    if isinstance(inputs, list):
      rank = inputs[0].shape.ndims
    else:
      rank = inputs.shape.ndims
    
    if rank == 2:
      if length_or_step.shape.ndims != 0:
        raise ValueError("length_or_step should be a scalar with the current timestep")
      outputs, state, attention = self.step(
          inputs,
          length_or_step,
          state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          training=training,
          adapter_activate=adapter_activate,
          internal_node_printing=internal_node_printing)
      logits = self.output_layer(outputs)
    elif rank == 3:
      if length_or_step.shape.ndims != 1:
        raise ValueError("length_or_step should contain the length of each sequence")
      logits, state, attention = self.forward(
          inputs,
          sequence_length=length_or_step,
          initial_state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training,
          adapter_activate=adapter_activate,
          internal_node_printing=internal_node_printing)
    else:
      raise ValueError("Unsupported input rank %d" % rank)
    return logits, state, attention

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None,
           adapter_activate=True,
           internal_node_printing=True):
    # Process inputs.
    domain = inputs[1]
    domain_mask = inputs[2]
    inputs = inputs[0]
    domain = domain[0]
    
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)
    
    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
        
    for i, layer in enumerate(self.layers):
      if self.version==1:
        inputs, layer_cache, attention = layer(
          inputs,
          domain,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      else:
        inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)

      new_cache.append(layer_cache)
        
      inputs = tf.math.multiply(inputs, domain_mask[i])
    
    if self.version==1:
      outputs = self.layer_norm(inputs, domain)
    else:
      outputs = self.layer_norm(inputs)

    return outputs, new_cache, attention
  
  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None,
              adapter_activate=True,
              internal_node_printing=False):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training,
        adapter_activate=adapter_activate,
        internal_node_printing=internal_node_printing)
    logits = self.output_layer(outputs)
    return logits, state, attention
  
  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None,
           adapter_activate=True,
           internal_node_printing=False):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1], inputs[2]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training,
        adapter_activate=adapter_activate,
        internal_node_printing=internal_node_printing)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

class Multi_domain_SelfAttentionDecoder_sparse_multi_layer_v1(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
               num_domain_units=128,
               domain_region_sizes = None,
               ADAP_layer_stopping_gradient=False,
               ADAP_gate_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               training_res_using_rate=0.0,
               testing_res_using_rate=0.0,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
               multi_domain_adapter_gate_class=Multi_domain_classification_gate,
               ADAP_contribution=None,
               num_sources=1,
               fake_domain_prob=0.1,
               noisy_prob=None,
               version=1,
               inner_layer_norm=None,
               stop_gradient_version=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_sparse_multi_layer_v1, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.num_domains = num_domains
    self.num_layers = num_layers
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()

    self.version = version
    print("decoder_version",version)
    if version==1:
      self.layer_norm = common.Multi_LayerNorm(num_domains)
      self.layers = [
        transformer.SelfAttentionDecoderLayer_v1(
            num_units,
            num_heads,
            ffn_inner_dim,
            domain_numb = num_domains,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)] 
    else:
      self.layer_norm = common.LayerNorm()
      self.layers = [
        transformer.SelfAttentionDecoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)] 

           
  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Dense(vocab_size)
    
  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def call(self,
           inputs,
           length_or_step=None,
           state=None,
           input_fn=None,
           sampling_probability=None,
           training=None,
           adapter_activate=True,
           internal_node_printing=False):
    
    self._assert_is_initialized()
    if isinstance(inputs, list):
      rank = inputs[0].shape.ndims
    else:
      rank = inputs.shape.ndims
    
    if rank == 2:
      if length_or_step.shape.ndims != 0:
        raise ValueError("length_or_step should be a scalar with the current timestep")
      outputs, state, attention = self.step(
          inputs,
          length_or_step,
          state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          training=training,
          adapter_activate=adapter_activate,
          internal_node_printing=internal_node_printing)
      logits = self.output_layer(outputs)
    elif rank == 3:
      if length_or_step.shape.ndims != 1:
        raise ValueError("length_or_step should contain the length of each sequence")
      logits, state, attention = self.forward(
          inputs,
          sequence_length=length_or_step,
          initial_state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training,
          adapter_activate=adapter_activate,
          internal_node_printing=internal_node_printing)
    else:
      raise ValueError("Unsupported input rank %d" % rank)
    return logits, state, attention

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None,
           adapter_activate=True,
           internal_node_printing=True):
    # Process inputs.
    domain = inputs[1]
    domain_mask = inputs[2]
    inputs = inputs[0]
    domain = domain[0]
    
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)
    
    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    
    # 0th layer
    inputs = tf.math.multiply(inputs, domain_mask[0])

    for i, layer in enumerate(self.layers):
      if self.version==1:
        inputs, layer_cache, attention = layer(
          inputs,
          domain,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      else:
        inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)

      new_cache.append(layer_cache)
        
      inputs = tf.math.multiply(inputs, domain_mask[i+1])
    
    if self.version==1:
      outputs = self.layer_norm(inputs, domain)
    else:
      outputs = self.layer_norm(inputs)

    return outputs, new_cache, attention
  
  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None,
              adapter_activate=True,
              internal_node_printing=False):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training,
        adapter_activate=adapter_activate,
        internal_node_printing=internal_node_printing)
    logits = self.output_layer(outputs)
    return logits, state, attention
  
  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None,
           adapter_activate=True,
           internal_node_printing=False):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1], inputs[2]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training,
        adapter_activate=adapter_activate,
        internal_node_printing=internal_node_printing)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

class Multi_domain_SelfAttentionDecoder_sparse_multi_layer_v0(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
               num_domain_units=128,
               domain_region_sizes = None,
               ADAP_layer_stopping_gradient=False,
               ADAP_gate_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               training_res_using_rate=0.0,
               testing_res_using_rate=0.0,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
               multi_domain_adapter_gate_class=Multi_domain_classification_gate,
               ADAP_contribution=None,
               num_sources=1,
               fake_domain_prob=0.1,
               noisy_prob=None,
               version=1,
               inner_layer_norm=None,
               stop_gradient_version=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_sparse_multi_layer_v0, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.num_domains = num_domains
    self.num_layers = num_layers
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()

    self.version = version
    print("decoder_version",version)
    if version==1:
      self.layer_norm = common.Multi_LayerNorm(num_domains)
      self.layers = [
        transformer.SelfAttentionDecoderLayer_v1(
            num_units,
            num_heads,
            ffn_inner_dim,
            domain_numb = num_domains,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)] 
    else:
      self.layer_norm = common.LayerNorm()
      self.layers = [
        transformer.SelfAttentionDecoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)] 

           
  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Dense(vocab_size)
    
  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def call(self,
           inputs,
           length_or_step=None,
           state=None,
           input_fn=None,
           sampling_probability=None,
           training=None,
           adapter_activate=True,
           internal_node_printing=False):
    
    self._assert_is_initialized()
    if isinstance(inputs, list):
      rank = inputs[0].shape.ndims
    else:
      rank = inputs.shape.ndims
    
    if rank == 2:
      if length_or_step.shape.ndims != 0:
        raise ValueError("length_or_step should be a scalar with the current timestep")
      outputs, state, attention = self.step(
          inputs,
          length_or_step,
          state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          training=training,
          adapter_activate=adapter_activate,
          internal_node_printing=internal_node_printing)
      logits = self.output_layer(outputs)
    elif rank == 3:
      if length_or_step.shape.ndims != 1:
        raise ValueError("length_or_step should contain the length of each sequence")
      logits, state, attention = self.forward(
          inputs,
          sequence_length=length_or_step,
          initial_state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training,
          adapter_activate=adapter_activate,
          internal_node_printing=internal_node_printing)
    else:
      raise ValueError("Unsupported input rank %d" % rank)
    return logits, state, attention

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None,
           adapter_activate=True,
           internal_node_printing=True):
    # Process inputs.
    domain = inputs[1]
    domain_mask = inputs[2]
    inputs = inputs[0]
    domain = domain[0]
    
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)
    
    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
    
    # 0th layer
    #inputs = tf.math.multiply(inputs, domain_mask[0])

    for i, layer in enumerate(self.layers):
      if self.version==1:
        inputs, layer_cache, attention = layer(
          inputs,
          domain,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      else:
        inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)

      new_cache.append(layer_cache)
        
      inputs = tf.math.multiply(inputs, domain_mask[i])
    
    if self.version==1:
      outputs = self.layer_norm(inputs, domain)
    else:
      outputs = self.layer_norm(inputs)

    return outputs, new_cache, attention
  
  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None,
              adapter_activate=True,
              internal_node_printing=False):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training,
        adapter_activate=adapter_activate,
        internal_node_printing=internal_node_printing)
    logits = self.output_layer(outputs)
    return logits, state, attention
  
  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None,
           adapter_activate=True,
           internal_node_printing=False):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1], inputs[2]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training,
        adapter_activate=adapter_activate,
        internal_node_printing=internal_node_printing)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache

class Multi_domain_SelfAttentionDecoder_sparse_multi_layer_v2(Decoder):
  
  def __init__(self,
               num_layers,
               num_domains,
               num_domain_units=128,
               domain_region_sizes = None,
               ADAP_layer_stopping_gradient=False,
               ADAP_gate_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               training_res_using_rate=0.0,
               testing_res_using_rate=0.0,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
               multi_domain_adapter_gate_class=Multi_domain_classification_gate,
               ADAP_contribution=None,
               num_sources=1,
               fake_domain_prob=0.1,
               noisy_prob=None,
               version=1,
               inner_layer_norm=None,
               stop_gradient_version=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionDecoder_sparse_multi_layer_v2, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.num_domains = num_domains
    self.num_layers = num_layers
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()

    self.version = version
    print("decoder_version",version)
    
    self.layer_norm = common.LayerNorm()
    self.layers = [
      transformer.SelfAttentionDecoderLayer(
          num_units,
          num_heads,
          ffn_inner_dim,
          dropout=dropout,
          attention_dropout=attention_dropout,
          ffn_dropout=ffn_dropout,
          ffn_activation=ffn_activation)
      for i in range(num_layers)] 
           
  def initialize(self, vocab_size=None, output_layer=None):  
    if output_layer is not None:
      self.output_layer = output_layer
    else:
      if vocab_size is None:
        raise ValueError("One of vocab_size and output_layer must be set")
      self.output_layer = common.Dense(vocab_size)
    
  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  @property
  def support_alignment_history(self):
    return self.num_sources == 1

  def map_v1_weights(self, weights):
    m = []
    m += self.output_layer.map_v1_weights(weights["dense"])
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m
  
  def call(self,
           inputs,
           length_or_step=None,
           state=None,
           input_fn=None,
           sampling_probability=None,
           training=None,
           adapter_activate=True,
           internal_node_printing=False):
    
    self._assert_is_initialized()
    if isinstance(inputs, list):
      rank = inputs[0].shape.ndims
    else:
      rank = inputs.shape.ndims
    
    if rank == 2:
      if length_or_step.shape.ndims != 0:
        raise ValueError("length_or_step should be a scalar with the current timestep")
      outputs, state, attention = self.step(
          inputs,
          length_or_step,
          state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          training=training,
          adapter_activate=adapter_activate,
          internal_node_printing=internal_node_printing)
      logits = self.output_layer(outputs)
    elif rank == 3:
      if length_or_step.shape.ndims != 1:
        raise ValueError("length_or_step should contain the length of each sequence")
      logits, state, attention = self.forward(
          inputs,
          sequence_length=length_or_step,
          initial_state=state,
          memory=self.memory,
          memory_sequence_length=self.memory_sequence_length,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training,
          adapter_activate=adapter_activate,
          internal_node_printing=internal_node_printing)
    else:
      raise ValueError("Unsupported input rank %d" % rank)
    return logits, state, attention

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None,
           adapter_activate=True,
           internal_node_printing=True):
    # Process inputs.
    domain = inputs[1]
    domain_mask = inputs[2]
    inputs = inputs[0]
    domain = domain[0]
    
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)
    
    # Prepare query mask.
    mask = None
    if step is None:
      maximum_length = tf.shape(inputs)[1]
      if sequence_length is None:
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.fill([batch_size], maximum_length)
      mask = transformer.future_mask(sequence_length, maximum_length=maximum_length)

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
          for mem, mem_length in zip(memory, memory_sequence_length)]

    # Run each layer.
    new_cache = []
        
    for i, layer in enumerate(self.layers):
      
      inputs, layer_cache, attention = layer(
        inputs,
        mask=mask,
        memory=memory,
        memory_mask=memory_mask,
        cache=cache[i] if cache is not None else None,
        training=training)

      new_cache.append(layer_cache)
        
      inputs = tf.math.multiply(inputs, tf.expand_dims(domain_mask[i],1))
    
    outputs = self.layer_norm(inputs)

    return outputs, new_cache, attention
  
  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              input_fn=None,
              sampling_probability=None,
              training=None,
              adapter_activate=True,
              internal_node_printing=False):
    _ = initial_state
    _ = input_fn
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported by this decoder")
    outputs, state, attention = self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training,
        adapter_activate=adapter_activate,
        internal_node_printing=internal_node_printing)
    logits = self.output_layer(outputs)
    return logits, state, attention
  
  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None,
           adapter_activate=True,
           internal_node_printing=False):
    
    inputs = [tf.expand_dims(inputs[0], 1), inputs[1], inputs[2]]
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training,
        adapter_activate=adapter_activate,
        internal_node_printing=internal_node_printing)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention
    
  def _get_initial_state(self, batch_size, dtype, initial_state=None):

    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache




















































