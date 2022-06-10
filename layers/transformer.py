import sys
sys.path.append("/gpfsdswork/projects/rech/sfz/utt84zy/anaconda3/envs/huggingface/lib/python3.7/site-packages")

import tensorflow as tf
import numpy as np
from utils.utils_ import make_domain_mask

from layers import common
from opennmt.utils import misc
from opennmt.layers import Dense

def _lower_triangle_mask(sequence_length, maximum_length=None, dtype=tf.bool):
  batch_size = tf.shape(sequence_length)[0]
  if maximum_length is None:
    maximum_length = tf.reduce_max(sequence_length)
  mask = tf.ones([batch_size, maximum_length, maximum_length], dtype=dtype)
  mask = tf.linalg.band_part(mask, -1, 0)
  return mask

def future_mask(sequence_length, maximum_length=None, dtype=tf.bool):
  
  sequence_mask = tf.sequence_mask(sequence_length, maxlen=maximum_length, dtype=dtype)
  sequence_mask = tf.expand_dims(sequence_mask, axis=1)
  mask = _lower_triangle_mask(sequence_length, maximum_length=maximum_length, dtype=dtype)
  if dtype is tf.bool:
    return tf.math.logical_and(mask, sequence_mask)
  else:
    return mask * sequence_mask

def split_heads(inputs, num_heads):
  """Splits a tensor in depth.

  Args:
    inputs: A ``tf.Tensor`` of shape :math:`[B, T, D]`.
    num_heads: The number of heads :math:`H`.

  Returns:
    A ``tf.Tensor`` of shape :math:`[B, H, T, D / H]`.
  """
  shape = misc.shape_list(inputs)
  outputs = tf.reshape(inputs, [shape[0], shape[1], num_heads, shape[2] // num_heads])
  outputs = tf.transpose(outputs, perm=[0, 2, 1, 3])
  return outputs

def combine_heads(inputs):
  """Concatenates heads.

  Args:
    inputs: A ``tf.Tensor`` of shape :math:`[B, H, T, D]`.

  Returns:
    A ``tf.Tensor`` of shape :math:`[B, T, D * H]`.
  """
  shape = misc.shape_list(inputs)
  outputs = tf.transpose(inputs, perm=[0, 2, 1, 3])
  outputs = tf.reshape(outputs, [shape[0], shape[2], shape[1] * shape[3]])
  return outputs

class FeedForwardNetwork(tf.keras.layers.Layer):
  
  def __init__(self,
               inner_dim,
               output_dim,
               dropout=0.1,
               activation=tf.nn.relu,
               **kwargs):
    
    super(FeedForwardNetwork, self).__init__(**kwargs)
    self.inner = common.Dense(inner_dim, activation=activation)
    self.outer = common.Dense(output_dim)
    self.dropout = dropout

  def call(self, inputs, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    inner = self.inner(inputs)
    inner = common.dropout(inner, self.dropout, training=training)
    return self.outer(inner)

  def forward_fn(self, inputs, args_dict, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    inner = self.inner.forward_fn(inputs, args_dict)
    inner = common.dropout(inner, self.dropout, training=training)
    return self.outer.forward_fn(inner, args_dict)

  def map_v1_weights(self, weights):
    # V1 used conv1d layers that have a leading dimensions.
    weights = tf.nest.map_structure(np.squeeze, weights)
    m = []
    m += self.inner.map_v1_weights(weights["conv1d"])
    m += self.outer.map_v1_weights(weights["conv1d_1"])
    return m

class MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self,
               num_heads,
               num_units,
               dropout=0.1,
               return_attention=False,
               **kwargs):
    
    super(MultiHeadAttention, self).__init__(**kwargs)
    if num_units % num_heads != 0:
      raise ValueError("Multi head attention requires that num_units is a"
                       " multiple of %s" % num_heads)
    self.num_heads = num_heads
    self.num_units = num_units
    self.linear_queries = Dense(num_units)
    self.linear_keys = Dense(num_units)
    self.linear_values = Dense(num_units)
    self.linear_output = Dense(num_units)
    self.dropout = dropout
    self.return_attention = return_attention

  def map_v1_weights(self, weights):
    # V1 used conv1d layers that have a leading dimensions.
    weights = tf.nest.map_structure(np.squeeze, weights)

    # V1 used fused linear projections, so the weights should be split accordingly.
    def _partial_weights(key, num_splits, index):
      return tf.nest.map_structure(
          lambda w: np.split(w, num_splits, axis=0 if w.ndim == 1 else 1)[index],
          weights[key])

    m = []
    if "conv1d_2" not in weights:  # Case self-attention.
      m += self.linear_queries.map_v1_weights(_partial_weights("conv1d", 3, 0))
      m += self.linear_keys.map_v1_weights(_partial_weights("conv1d", 3, 1))
      m += self.linear_values.map_v1_weights(_partial_weights("conv1d", 3, 2))
      m += self.linear_output.map_v1_weights(weights["conv1d_1"])
    else:
      m += self.linear_queries.map_v1_weights(weights["conv1d"])
      m += self.linear_keys.map_v1_weights(_partial_weights("conv1d_1", 2, 0))
      m += self.linear_values.map_v1_weights(_partial_weights("conv1d_1", 2, 1))
      m += self.linear_output.map_v1_weights(weights["conv1d_2"])
    return m

  def call(self, inputs, memory=None, mask=None, cache=None, training=None):  # pylint: disable=arguments-differ
    def _compute_kv(x):
      keys = self.linear_keys(x)
      keys = split_heads(keys, self.num_heads)
      values = self.linear_values(x)
      values = split_heads(values, self.num_heads)
      return keys, values

    # Compute queries.
    queries = self.linear_queries(inputs)
    queries = split_heads(queries, self.num_heads)
    queries *= (self.num_units // self.num_heads)**-0.5

    # Compute keys and values.
    if memory is None:
      keys, values = _compute_kv(inputs)
      if cache:
        keys = tf.concat([cache[0], keys], axis=2)
        values = tf.concat([cache[1], values], axis=2)
    else:
      if cache:
        keys, values = tf.cond(
            tf.equal(tf.shape(cache[0])[2], 0),
            true_fn=lambda: _compute_kv(memory),
            false_fn=lambda: cache)
      else:
        keys, values = _compute_kv(memory)

    cache = (keys, values)

    # Dot product attention.
    dot = tf.matmul(queries, keys, transpose_b=True)
    if mask is not None:
      mask = tf.cast(mask, tf.float32)
      if mask.shape.rank == 2:
        mask = tf.expand_dims(mask, 1)  # Broadcast on time dimension.
      mask = tf.expand_dims(mask, 1)  # Broadcast on head dimension.
      dot = tf.cast(tf.cast(dot, tf.float32) * mask + ((1.0 - mask) * tf.float32.min), dot.dtype)
    attn = tf.cast(tf.nn.softmax(tf.cast(dot, tf.float32)), dot.dtype)
    drop_attn = common.dropout(attn, self.dropout, training=training)
    heads = tf.matmul(drop_attn, values)

    # Concatenate all heads output.
    combined = combine_heads(heads)
    outputs = self.linear_output(combined)
    if self.return_attention:
      return outputs, cache, attn
    return outputs, cache

  def forward_fn(self, inputs, args_dict, memory=None, mask=None, cache=None, training=None):  # pylint: disable=arguments-differ
    def _compute_kv(x):
      keys = self.linear_keys.forward_fn(x, args_dict)
      keys = split_heads(keys, self.num_heads)
      values = self.linear_values.forward_fn(x, args_dict)
      values = split_heads(values, self.num_heads)
      return keys, values

    # Compute queries.
    queries = self.linear_queries.forward_fn(inputs, args_dict)
    queries = split_heads(queries, self.num_heads)
    queries *= (self.num_units // self.num_heads)**-0.5

    # Compute keys and values.
    if memory is None:
      keys, values = _compute_kv(inputs)
      if cache:
        keys = tf.concat([cache[0], keys], axis=2)
        values = tf.concat([cache[1], values], axis=2)
    else:
      if cache:
        keys, values = tf.cond(
            tf.equal(tf.shape(cache[0])[2], 0),
            true_fn=lambda: _compute_kv(memory),
            false_fn=lambda: cache)
      else:
        keys, values = _compute_kv(memory)

    cache = (keys, values)

    # Dot product attention.
    dot = tf.matmul(queries, keys, transpose_b=True)
    if mask is not None:
      mask = tf.cast(mask, tf.float32)
      if mask.shape.rank == 2:
        mask = tf.expand_dims(mask, 1)  # Broadcast on time dimension.
      mask = tf.expand_dims(mask, 1)  # Broadcast on head dimension.
      dot = tf.cast(tf.cast(dot, tf.float32) * mask + ((1.0 - mask) * tf.float32.min), dot.dtype)
    attn = tf.cast(tf.nn.softmax(tf.cast(dot, tf.float32)), dot.dtype)
    drop_attn = common.dropout(attn, self.dropout, training=training)
    heads = tf.matmul(drop_attn, values)

    # Concatenate all heads output.
    combined = combine_heads(heads)
    outputs = self.linear_output.forward_fn(combined, args_dict)
    if self.return_attention:
      return outputs, cache, attn
    return outputs, cache

class TransformerLayerWrapper(common.LayerWrapper):
  
  def __init__(self, layer, output_dropout, **kwargs):
    
    super(TransformerLayerWrapper, self).__init__(
        layer,
        normalize_input=True,
        output_dropout=output_dropout,
        residual_connection=True,
        **kwargs)

  def map_v1_weights(self, weights):
    m = []
    m += self.input_layer_norm.map_v1_weights(weights["LayerNorm"])
    m += self.layer.map_v1_weights(weights)
    return m

class TransformerLayerWrapper_v2(common.LayerWrapper_v2):
  
  def __init__(self, layer, output_dropout, **kwargs):
    
    super(TransformerLayerWrapper_v2, self).__init__(
        layer,
        normalize_input=True,
        output_dropout=output_dropout,
        residual_connection=True,
        **kwargs)

  def map_v1_weights(self, weights):
    m = []
    m += self.input_layer_norm.map_v1_weights(weights["LayerNorm"])
    m += self.layer.map_v1_weights(weights)
    return m

class TransformerLayerWrapper_v1(common.LayerWrapper_v1):
  
  def __init__(self, layer, output_dropout, domain_numb=5, **kwargs):
    
    super(TransformerLayerWrapper_v1, self).__init__(
        layer,
        domain_numb = domain_numb,
        normalize_input=True,
        output_dropout=output_dropout,
        residual_connection=True,
        **kwargs)

  def map_v1_weights(self, weights):
    m = []
    m += self.input_layer_norm.map_v1_weights(weights["LayerNorm"])
    m += self.layer.map_v1_weights(weights)
    return m

class SelfAttentionEncoderLayer(tf.keras.layers.Layer):
  """Implements one self-attention encoding layer."""

  def __init__(self,
               num_units,
               num_heads,
               ffn_inner_dim,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               **kwargs):
    
    super(SelfAttentionEncoderLayer, self).__init__(**kwargs)
    self.self_attention = MultiHeadAttention(
        num_heads, num_units, dropout=attention_dropout)
    self.self_attention = TransformerLayerWrapper(
        self.self_attention, dropout)
    self.ffn = FeedForwardNetwork(
        ffn_inner_dim,
        num_units,
        dropout=ffn_dropout,
        activation=ffn_activation)
    self.ffn = TransformerLayerWrapper(
        self.ffn, dropout)

  def call(self, x, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the encoder layer."""
    y, _ = self.self_attention(x, mask=mask, training=training)
    y = self.ffn(y, training=training)
    return y

  def forward_fn(self, x, args_dict, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the encoder layer."""
    y, _ = self.self_attention.forward_fn(x, args_dict, mask=mask, training=training)
    y = self.ffn.forward_fn(y, args_dict, training=training)
    return y

  def map_v1_weights(self, weights):
    m = []
    m += self.self_attention.map_v1_weights(weights["multi_head"])
    m += self.ffn.map_v1_weights(weights["ffn"])
    return m

class SelfAttentionDecoderLayer(tf.keras.layers.Layer):
  """Implements one self-attention decoding layer."""

  def __init__(self,
               num_units,
               num_heads,
               ffn_inner_dim,
               num_sources=1,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               **kwargs):
    
    super(SelfAttentionDecoderLayer, self).__init__(**kwargs)
    self.self_attention = MultiHeadAttention(
        num_heads,
        num_units,
        dropout=attention_dropout)
    self.self_attention = TransformerLayerWrapper(
        self.self_attention, dropout)
    self.attention = []
    for _ in range(num_sources):
      attention = MultiHeadAttention(
          num_heads,
          num_units,
          dropout=attention_dropout,
          return_attention=num_sources == 1)
      attention = TransformerLayerWrapper(
          attention, dropout)
      self.attention.append(attention)
    self.ffn = FeedForwardNetwork(
        ffn_inner_dim,
        num_units,
        dropout=ffn_dropout,
        activation=ffn_activation)
    self.ffn = TransformerLayerWrapper(
        self.ffn, dropout)

  def map_v1_weights(self, weights):
    m = []
    m += self.self_attention.map_v1_weights(weights["masked_multi_head"])
    m += self.attention[0].map_v1_weights(weights["multi_head"])
    m += self.ffn.map_v1_weights(weights["ffn"])
    return m

  # pylint: disable=arguments-differ
  def call(self,
           inputs,
           mask=None,
           memory=None,
           memory_mask=None,
           cache=None,
           training=None):
    """Runs the decoder layer."""
    if cache is None:
      cache = {}

    outputs, self_kv = self.self_attention(
        inputs,
        mask=mask,
        cache=cache.get("self_kv"),
        training=training)

    attention = None
    memory_kv = []
    if memory is not None:
      memory_cache = cache.get("memory_kv")
      if memory_cache is None:
        memory_cache = [None] * len(self.attention)
      for layer, mem, mem_mask, mem_cache in zip(
          self.attention, memory, memory_mask, memory_cache):
        result = layer(
            outputs,
            memory=mem,
            mask=mem_mask,
            cache=mem_cache,
            training=training)
        if len(result) == 3:
          outputs, memory_kv_i, attention = result
          attention = attention[:, 0]  # Use the first head for the attention vector.
        else:
          outputs, memory_kv_i = result
        memory_kv.append(memory_kv_i)

    outputs = self.ffn(outputs, training=training)
    cache = dict(self_kv=self_kv, memory_kv=memory_kv)
    return outputs, cache, attention

  def forward_fn(self,
           inputs,
           args_dict,
           mask=None,
           memory=None,
           memory_mask=None,
           cache=None,
           training=None):
    """Runs the decoder layer."""
    if cache is None:
      cache = {}

    outputs, self_kv = self.self_attention.forward_fn(
        inputs,
        args_dict,
        mask=mask,
        cache=cache.get("self_kv"),
        training=training)

    attention = None
    memory_kv = []
    if memory is not None:
      memory_cache = cache.get("memory_kv")
      if memory_cache is None:
        memory_cache = [None] * len(self.attention)
      for layer, mem, mem_mask, mem_cache in zip(
          self.attention, memory, memory_mask, memory_cache):
        result = layer.forward_fn(
            outputs,
            args_dict,
            memory=mem,
            mask=mem_mask,
            cache=mem_cache,
            training=training)
        if len(result) == 3:
          outputs, memory_kv_i, attention = result
          attention = attention[:, 0]  # Use the first head for the attention vector.
        else:
          outputs, memory_kv_i = result
        memory_kv.append(memory_kv_i)

    outputs = self.ffn.forward_fn(outputs, args_dict, training=training)
    cache = dict(self_kv=self_kv, memory_kv=memory_kv)
    return outputs, cache, attention
  
class Priming_SelfAttentionDecoderLayer(tf.keras.layers.Layer):
  """Implements one self-attention decoding layer."""

  def __init__(self,
               num_units,
               num_heads,
               ffn_inner_dim,
               num_sources=1,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               **kwargs):
    
    super(Priming_SelfAttentionDecoderLayer, self).__init__(**kwargs)
    self.self_attention = MultiHeadAttention(
        num_heads,
        num_units,
        dropout=attention_dropout)
    self.self_attention = TransformerLayerWrapper(
        self.self_attention, dropout)
    self.attention = []
    #############
    attention = MultiHeadAttention(
        num_heads,
        num_units,
        dropout=attention_dropout,
        return_attention=num_sources == 1)
    attention = TransformerLayerWrapper(
        attention, dropout)
    self.attention.append(attention)
    #############

    attention = Priming_MultiHeadAttention(
        num_heads,
        num_units,
        dropout=attention_dropout,
        return_attention=num_sources == 1)
    attention = TransformerLayerWrapper(
        attention, dropout)
    self.attention.append(attention)

    #############
    self.ffn = FeedForwardNetwork(
        ffn_inner_dim,
        num_units,
        dropout=ffn_dropout,
        activation=ffn_activation)
    self.ffn = TransformerLayerWrapper(
        self.ffn, dropout)

  def map_v1_weights(self, weights):
    m = []
    m += self.self_attention.map_v1_weights(weights["masked_multi_head"])
    m += self.attention[0].map_v1_weights(weights["multi_head"])
    m += self.ffn.map_v1_weights(weights["ffn"])
    return m

  # pylint: disable=arguments-differ
  def call(self,
           inputs,
           mask=None,
           memory=None,
           memory_mask=None,
           cache=None,
           training=None):
    """Runs the decoder layer."""
    if cache is None:
      cache = {}

    outputs, self_kv = self.self_attention(
        inputs,
        mask=mask,
        cache=cache.get("self_kv"),
        training=training)

    attention = None
    memory_kv = []
    if memory is not None:
      memory_cache = cache.get("memory_kv")
      if memory_cache is None:
        memory_cache = [None] * len(self.attention)
      for layer, mem, mem_mask, mem_cache in zip(
          self.attention, memory, memory_mask, memory_cache):
        result = layer(
            outputs,
            memory=mem,
            mask=mem_mask,
            cache=mem_cache,
            training=training)
        if len(result) == 3:
          outputs, memory_kv_i, attention = result
          attention = attention[:, 0]  # Use the first head for the attention vector.
        else:
          outputs, memory_kv_i = result
        memory_kv.append(memory_kv_i)

    outputs = self.ffn(outputs, training=training)
    cache = dict(self_kv=self_kv, memory_kv=memory_kv)
    return outputs, cache, attention

class Priming_MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self,
               num_heads,
               num_units,
               dropout=0.1,
               return_attention=False,
               **kwargs):
    
    super(MultiHeadAttention, self).__init__(**kwargs)
    if num_units % num_heads != 0:
      raise ValueError("Multi head attention requires that num_units is a"
                       " multiple of %s" % num_heads)
    self.num_heads = num_heads
    self.num_units = num_units
    self.linear_queries = Dense(num_units)
    self.linear_keys = Dense(num_units)
    self.linear_values = Dense(num_units)
    self.linear_output = Dense(num_units)
    self.dropout = dropout
    self.return_attention = return_attention

  def map_v1_weights(self, weights):
    # V1 used conv1d layers that have a leading dimensions.
    weights = tf.nest.map_structure(np.squeeze, weights)

    # V1 used fused linear projections, so the weights should be split accordingly.
    def _partial_weights(key, num_splits, index):
      return tf.nest.map_structure(
          lambda w: np.split(w, num_splits, axis=0 if w.ndim == 1 else 1)[index],
          weights[key])

    m = []
    if "conv1d_2" not in weights:  # Case self-attention.
      m += self.linear_queries.map_v1_weights(_partial_weights("conv1d", 3, 0))
      m += self.linear_keys.map_v1_weights(_partial_weights("conv1d", 3, 1))
      m += self.linear_values.map_v1_weights(_partial_weights("conv1d", 3, 2))
      m += self.linear_output.map_v1_weights(weights["conv1d_1"])
    else:
      m += self.linear_queries.map_v1_weights(weights["conv1d"])
      m += self.linear_keys.map_v1_weights(_partial_weights("conv1d_1", 2, 0))
      m += self.linear_values.map_v1_weights(_partial_weights("conv1d_1", 2, 1))
      m += self.linear_output.map_v1_weights(weights["conv1d_2"])
    return m

  def call(self, inputs, memory=None, mask=None, cache=None, training=None):  # pylint: disable=arguments-differ
    def _compute_kv(x):
      keys = self.linear_keys(x)
      keys = split_heads(keys, self.num_heads)
      values = self.linear_values(x)
      values = split_heads(values, self.num_heads)
      return keys, values

    # Compute queries.
    queries = self.linear_queries(inputs)
    queries = split_heads(queries, self.num_heads)
    queries *= (self.num_units // self.num_heads)**-0.5

    # Compute keys and values.
    if memory is None:
      keys, values = _compute_kv(inputs)
      if cache:
        keys = tf.concat([cache[0], keys], axis=2)
        values = tf.concat([cache[1], values], axis=2)
    else:
      if cache:
        keys, values = tf.cond(
            tf.equal(tf.shape(cache[0])[2], 0),
            true_fn=lambda: _compute_kv(memory),
            false_fn=lambda: cache)
      else:
        keys, values = _compute_kv(memory)

    cache = (keys, values)

    # Dot product attention.
    dot = tf.matmul(queries, keys, transpose_b=True)
    if mask is not None:
      mask = tf.cast(mask, tf.float32)
      if mask.shape.rank == 2:
        mask = tf.expand_dims(mask, 1)  # Broadcast on time dimension.
      mask = tf.expand_dims(mask, 1)  # Broadcast on head dimension.
      dot = tf.cast(tf.cast(dot, tf.float32) * mask + ((1.0 - mask) * tf.float32.min), dot.dtype)
    attn = tf.nn.relu(tf.math.multiply(tf.cast(tf.nn.softmax(tf.cast(dot, tf.float32)), dot.dtype) * tf.expand_dims(tf.reduce_max(tf.cast(dot, tf.float32),-1),1)))
    drop_attn = common.dropout(attn, self.dropout, training=training)
    heads = tf.matmul(drop_attn, values)

    # Concatenate all heads output.
    combined = combine_heads(heads)
    outputs = self.linear_output(combined)
    if self.return_attention:
      return outputs, cache, attn
    return outputs, cache

class Cross_SelfAttentionEncoderLayer(tf.keras.layers.Layer):
  """Implements one self-attention encoding layer."""

  def __init__(self,
               num_units,
               num_heads,
               ffn_inner_dim,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               **kwargs):
    
    super(Cross_SelfAttentionEncoderLayer, self).__init__(**kwargs)

    self.self_attention = MultiHeadAttention(
        num_heads, num_units, dropout=attention_dropout)
    self.self_attention = TransformerLayerWrapper(
        self.self_attention, dropout)

    self.self_xattention = MultiHeadAttention(
        num_heads, num_units, dropout=attention_dropout)
    self.self_xattention = TransformerLayerWrapper(
        self.self_xattention, dropout)

    self.ffn = FeedForwardNetwork(
        ffn_inner_dim,
        num_units,
        dropout=ffn_dropout,
        activation=ffn_activation)
    self.ffn = TransformerLayerWrapper(
        self.ffn, dropout)
    
    self.xffn = FeedForwardNetwork(
        ffn_inner_dim,
        num_units,
        dropout=ffn_dropout,
        activation=ffn_activation)
    self.xffn = TransformerLayerWrapper(
        self.xffn, dropout)


  def call(self, x, pre, pre_mask, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the encoder layer."""
    y, _ = self.self_attention(x, mask=mask, training=training)
    y = self.ffn(y, training=training)
    y, _ = self.self_xattention(y, memory=pre, mask=pre_mask, training=training)
    y = self.xffn(y, training=training)
    return y

  def map_v1_weights(self, weights):
    m = []
    m += self.self_attention.map_v1_weights(weights["multi_head"])
    m += self.ffn.map_v1_weights(weights["ffn"])
    return m

class SelfAttentionEncoderLayer_v1(tf.keras.layers.Layer):
  """Implements one self-attention encoding layer."""

  def __init__(self,
               num_units,
               num_heads,
               ffn_inner_dim,
               domain_numb = 5,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               **kwargs):
    
    super(SelfAttentionEncoderLayer_v1, self).__init__(**kwargs)
    self.self_attention = MultiHeadAttention(
        num_heads, num_units, dropout=attention_dropout)
    self.self_attention = TransformerLayerWrapper_v1(
        self.self_attention, dropout, domain_numb=domain_numb)
    self.ffn = FeedForwardNetwork(
        ffn_inner_dim,
        num_units,
        dropout=ffn_dropout,
        activation=ffn_activation)
    self.ffn = TransformerLayerWrapper_v1(
        self.ffn, dropout, domain_numb=domain_numb)

  def call(self, x, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the encoder layer."""
    y, _ = self.self_attention(x, domain, mask=mask, training=training)
    y = self.ffn(y, domain, training=training)
    return y

  def map_v1_weights(self, weights):
    m = []
    m += self.self_attention.map_v1_weights(weights["multi_head"])
    m += self.ffn.map_v1_weights(weights["ffn"])
    return m

class SelfAttentionDecoderLayer_v1(tf.keras.layers.Layer):
  """Implements one self-attention decoding layer."""

  def __init__(self,
               num_units,
               num_heads,
               ffn_inner_dim,
               domain_numb = 5,
               num_sources=1,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               **kwargs):
    
    super(SelfAttentionDecoderLayer_v1, self).__init__(**kwargs)
    self.self_attention = MultiHeadAttention(
        num_heads,
        num_units,
        dropout=attention_dropout)
    self.self_attention = TransformerLayerWrapper_v1(
        self.self_attention, dropout, domain_numb = domain_numb)
    self.attention = []
    for _ in range(num_sources):
      attention = MultiHeadAttention(
          num_heads,
          num_units,
          dropout=attention_dropout,
          return_attention=num_sources == 1)
      attention = TransformerLayerWrapper_v1(
          attention, dropout, domain_numb = domain_numb)
      self.attention.append(attention)
    self.ffn = FeedForwardNetwork(
        ffn_inner_dim,
        num_units,
        dropout=ffn_dropout,
        activation=ffn_activation)
    self.ffn = TransformerLayerWrapper_v1(
        self.ffn, dropout, domain_numb = domain_numb)

  def map_v1_weights(self, weights):
    m = []
    m += self.self_attention.map_v1_weights(weights["masked_multi_head"])
    m += self.attention[0].map_v1_weights(weights["multi_head"])
    m += self.ffn.map_v1_weights(weights["ffn"])
    return m

  # pylint: disable=arguments-differ
  def call(self,
           inputs,
           domain,
           mask=None,
           memory=None,
           memory_mask=None,
           cache=None,
           training=None):
    """Runs the decoder layer."""
    if cache is None:
      cache = {}

    outputs, self_kv = self.self_attention(
        inputs,
        domain,
        mask=mask,
        cache=cache.get("self_kv"),
        training=training)

    attention = None
    memory_kv = []
    if memory is not None:
      memory_cache = cache.get("memory_kv")
      if memory_cache is None:
        memory_cache = [None] * len(self.attention)
      for layer, mem, mem_mask, mem_cache in zip(
          self.attention, memory, memory_mask, memory_cache):
        result = layer(
            outputs,
            domain,
            memory=mem,
            mask=mem_mask,
            cache=mem_cache,
            training=training)
        if len(result) == 3:
          outputs, memory_kv_i, attention = result
          attention = attention[:, 0]  # Use the first head for the attention vector.
        else:
          outputs, memory_kv_i = result
        memory_kv.append(memory_kv_i)

    outputs = self.ffn(outputs, domain, training=training)
    cache = dict(self_kv=self_kv, memory_kv=memory_kv)
    return outputs, cache, attention

class SelfAttentionDecoderLayer_v2(tf.keras.layers.Layer):
  """Implements one self-attention decoding layer."""

  def __init__(self,
               num_units,
               num_heads,
               ffn_inner_dim,
               domain_numb = 5,
               num_domain_units = 32,
               num_sources=1,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               **kwargs):
    
    super(SelfAttentionDecoderLayer_v2, self).__init__(**kwargs)
    self.self_attention = MultiHeadAttention(
        num_heads,
        num_units,
        dropout=attention_dropout)
    self.self_attention = TransformerLayerWrapper_v1(
        self.self_attention, dropout, domain_numb = domain_numb)
    self.attention = []
    self.domain_mask = make_domain_mask(domain_numb,  num_units=num_units, num_domain_units=num_domain_units)
    for _ in range(num_sources):
      attention = MultiHeadAttention(
          num_heads,
          num_units,
          dropout=attention_dropout,
          return_attention=num_sources == 1)
      attention = TransformerLayerWrapper_v1(
          attention, dropout, domain_numb = domain_numb)
      self.attention.append(attention)
    self.ffn = FeedForwardNetwork(
        ffn_inner_dim,
        num_units,
        dropout=ffn_dropout,
        activation=ffn_activation)
    self.ffn = TransformerLayerWrapper_v1(
        self.ffn, dropout, domain_numb = domain_numb)

  def map_v1_weights(self, weights):
    m = []
    m += self.self_attention.map_v1_weights(weights["masked_multi_head"])
    m += self.attention[0].map_v1_weights(weights["multi_head"])
    m += self.ffn.map_v1_weights(weights["ffn"])
    return m

  # pylint: disable=arguments-differ
  def call(self,
           inputs,
           domain,
           mask=None,
           memory=None,
           memory_mask=None,
           cache=None,
           training=None):
    """Runs the decoder layer."""
    if cache is None:
      cache = {}

    outputs, self_kv = self.self_attention(
        inputs,
        domain,
        mask=mask,
        cache=cache.get("self_kv"),
        training=training)

    domain_mask = tf.nn.embedding_lookup(self.domain_mask, domain)
    outputs = tf.math.multiply(outputs, domain_mask)

    attention = None
    memory_kv = []
    if memory is not None:
      memory_cache = cache.get("memory_kv")
      if memory_cache is None:
        memory_cache = [None] * len(self.attention)
      for layer, mem, mem_mask, mem_cache in zip(
          self.attention, memory, memory_mask, memory_cache):
        result = layer(
            outputs,
            domain,
            memory=mem,
            mask=mem_mask,
            cache=mem_cache,
            training=training)
        if len(result) == 3:
          outputs, memory_kv_i, attention = result
          attention = attention[:, 0]  # Use the first head for the attention vector.
        else:
          outputs, memory_kv_i = result
        memory_kv.append(memory_kv_i)
    outputs = tf.math.multiply(outputs, domain_mask)
    outputs = self.ffn(outputs, domain, training=training)
    outputs = tf.math.multiply(outputs, domain_mask)
    cache = dict(self_kv=self_kv, memory_kv=memory_kv)
    return outputs, cache, attention

class SelfAttentionEncoderLayer_v2(tf.keras.layers.Layer):
  """Implements one self-attention encoding layer."""

  def __init__(self,
               num_units,
               num_heads,
               ffn_inner_dim,
               domain_numb = 5,
               num_domain_units = 32,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               **kwargs):
    
    super(SelfAttentionEncoderLayer_v2, self).__init__(**kwargs)
    self.self_attention = MultiHeadAttention(
        num_heads, num_units, dropout=attention_dropout)
    
    self.domain_mask = make_domain_mask(domain_numb,  num_units=num_units, num_domain_units=num_domain_units)
    self.self_attention = TransformerLayerWrapper_v1(
        self.self_attention, dropout, domain_numb=domain_numb)
    self.ffn = FeedForwardNetwork(
        ffn_inner_dim,
        num_units,
        dropout=ffn_dropout,
        activation=ffn_activation)
    self.ffn = TransformerLayerWrapper_v1(
        self.ffn, dropout, domain_numb=domain_numb)

  def call(self, x, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the encoder layer."""
    y, _ = self.self_attention(x, domain, mask=mask, training=training)
    domain_mask = tf.nn.embedding_lookup(self.domain_mask, domain)
    y = tf.math.multiply(y, domain_mask)
    y = self.ffn(y, domain, training=training)
    y = tf.math.multiply(y, domain_mask)
    return y

  def map_v1_weights(self, weights):
    m = []
    m += self.self_attention.map_v1_weights(weights["multi_head"])
    m += self.ffn.map_v1_weights(weights["ffn"])
    return m

class SelfAttentionEncoderLayer_v3(tf.keras.layers.Layer):
  """Implements one self-attention encoding layer."""

  def __init__(self,
               num_units,
               num_heads,
               ffn_inner_dim,
               domain_numb = 5,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               **kwargs):
    
    super(SelfAttentionEncoderLayer_v3, self).__init__(**kwargs)
    self.self_attention = MultiHeadAttention(
        num_heads, 512, dropout=attention_dropout)
    self.self_attention = TransformerLayerWrapper_v1(
        self.self_attention, dropout, domain_numb=domain_numb)
    self.ffn = FeedForwardNetwork(
        ffn_inner_dim,
        num_units,
        dropout=ffn_dropout,
        activation=ffn_activation)
    self.ffn = TransformerLayerWrapper_v1(
        self.ffn, dropout, domain_numb=domain_numb)

  def call(self, x, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the encoder layer."""
    y, _ = self.self_attention(x, domain, mask=mask, training=training)
    y = self.ffn(y, domain, training=training)
    return y

  def map_v1_weights(self, weights):
    m = []
    m += self.self_attention.map_v1_weights(weights["multi_head"])
    m += self.ffn.map_v1_weights(weights["ffn"])
    return m

class SelfAttentionDecoderLayer_v3(tf.keras.layers.Layer):
  """Implements one self-attention decoding layer."""

  def __init__(self,
               num_units,
               num_heads,
               ffn_inner_dim,
               domain_numb = 5,
               num_sources=1,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               **kwargs):
    
    super(SelfAttentionDecoderLayer_v3, self).__init__(**kwargs)
    self.self_attention = MultiHeadAttention(
        num_heads,
        512,
        dropout=attention_dropout)
    self.self_attention = TransformerLayerWrapper_v1(
        self.self_attention, dropout, domain_numb = domain_numb)
    self.attention = []
    for _ in range(num_sources):
      attention = MultiHeadAttention(
          num_heads,
          512,
          dropout=attention_dropout,
          return_attention=num_sources == 1)
      attention = TransformerLayerWrapper_v1(
          attention, dropout, domain_numb = domain_numb)
      self.attention.append(attention)
    self.ffn = FeedForwardNetwork(
        ffn_inner_dim,
        num_units,
        dropout=ffn_dropout,
        activation=ffn_activation)
    self.ffn = TransformerLayerWrapper_v1(
        self.ffn, dropout, domain_numb = domain_numb)

  def map_v1_weights(self, weights):
    m = []
    m += self.self_attention.map_v1_weights(weights["masked_multi_head"])
    m += self.attention[0].map_v1_weights(weights["multi_head"])
    m += self.ffn.map_v1_weights(weights["ffn"])
    return m

  # pylint: disable=arguments-differ
  def call(self,
           inputs,
           domain,
           mask=None,
           memory=None,
           memory_mask=None,
           cache=None,
           training=None):
    """Runs the decoder layer."""
    if cache is None:
      cache = {}

    outputs, self_kv = self.self_attention(
        inputs,
        domain,
        mask=mask,
        cache=cache.get("self_kv"),
        training=training)

    attention = None
    memory_kv = []
    if memory is not None:
      memory_cache = cache.get("memory_kv")
      if memory_cache is None:
        memory_cache = [None] * len(self.attention)
      for layer, mem, mem_mask, mem_cache in zip(
          self.attention, memory, memory_mask, memory_cache):
        result = layer(
            outputs,
            domain,
            memory=mem,
            mask=mem_mask,
            cache=mem_cache,
            training=training)
        if len(result) == 3:
          outputs, memory_kv_i, attention = result
          attention = attention[:, 0]  # Use the first head for the attention vector.
        else:
          outputs, memory_kv_i = result
        memory_kv.append(memory_kv_i)

    outputs = self.ffn(outputs, domain, training=training)
    cache = dict(self_kv=self_kv, memory_kv=memory_kv)
    return outputs, cache, attention

class SelfAttentionEncoderLayer_v4(tf.keras.layers.Layer):
  """Implements one self-attention encoding layer."""

  def __init__(self,
               num_units,
               num_heads,
               ffn_inner_dim,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               **kwargs):
    
    super(SelfAttentionEncoderLayer_v4, self).__init__(**kwargs)
    self.self_attention = MultiHeadAttention(
        num_heads, num_units, dropout=attention_dropout)
    self.self_attention = TransformerLayerWrapper_v2(
        self.self_attention, dropout)
    self.ffn = FeedForwardNetwork(
        ffn_inner_dim,
        num_units,
        dropout=ffn_dropout,
        activation=ffn_activation)
    self.ffn = TransformerLayerWrapper_v2(
        self.ffn, dropout)

  def call(self, x, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the encoder layer."""
    y, _ = self.self_attention(x, mask=mask, training=training)
    y = self.ffn(y, training=training)
    return y

  def map_v1_weights(self, weights):
    m = []
    m += self.self_attention.map_v1_weights(weights["multi_head"])
    m += self.ffn.map_v1_weights(weights["ffn"])
    return m

class SelfAttentionDecoderLayer_v4(tf.keras.layers.Layer):
  """Implements one self-attention decoding layer."""

  def __init__(self,
               num_units,
               num_heads,
               ffn_inner_dim,
               num_sources=1,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               **kwargs):
    
    super(SelfAttentionDecoderLayer_v4, self).__init__(**kwargs)
    self.self_attention = MultiHeadAttention(
        num_heads,
        num_units,
        dropout=attention_dropout)
    self.self_attention = TransformerLayerWrapper_v2(
        self.self_attention, dropout)
    self.attention = []
    for _ in range(num_sources):
      attention = MultiHeadAttention(
          num_heads,
          num_units,
          dropout=attention_dropout,
          return_attention=num_sources == 1)
      attention = TransformerLayerWrapper_v2(
          attention, dropout)
      self.attention.append(attention)
    self.ffn = FeedForwardNetwork(
        ffn_inner_dim,
        num_units,
        dropout=ffn_dropout,
        activation=ffn_activation)
    self.ffn = TransformerLayerWrapper_v2(
        self.ffn, dropout)

  def map_v1_weights(self, weights):
    m = []
    m += self.self_attention.map_v1_weights(weights["masked_multi_head"])
    m += self.attention[0].map_v1_weights(weights["multi_head"])
    m += self.ffn.map_v1_weights(weights["ffn"])
    return m

  # pylint: disable=arguments-differ
  def call(self,
           inputs,
           mask=None,
           memory=None,
           memory_mask=None,
           cache=None,
           training=None):
    """Runs the decoder layer."""
    if cache is None:
      cache = {}

    outputs, self_kv = self.self_attention(
        inputs,
        mask=mask,
        cache=cache.get("self_kv"),
        training=training)

    attention = None
    memory_kv = []
    if memory is not None:
      memory_cache = cache.get("memory_kv")
      if memory_cache is None:
        memory_cache = [None] * len(self.attention)
      for layer, mem, mem_mask, mem_cache in zip(
          self.attention, memory, memory_mask, memory_cache):
        result = layer(
            outputs,
            memory=mem,
            mask=mem_mask,
            cache=mem_cache,
            training=training)
        if len(result) == 3:
          outputs, memory_kv_i, attention = result
          attention = attention[:, 0]  # Use the first head for the attention vector.
        else:
          outputs, memory_kv_i = result
        memory_kv.append(memory_kv_i)

    outputs = self.ffn(outputs, training=training)
    cache = dict(self_kv=self_kv, memory_kv=memory_kv)
    return outputs, cache, attention






















































































































































