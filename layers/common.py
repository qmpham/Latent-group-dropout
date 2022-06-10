"""Defines common layers."""
import sys
from matplotlib.pyplot import axis
sys.path.append("/gpfsdswork/projects/rech/sfz/utt84zy/anaconda3/envs/huggingface/lib/python3.7/site-packages")

import tensorflow as tf
import numpy as np

from opennmt.utils.misc import shape_list


def dropout(x, rate, training=None):
  """Simple dropout layer."""
  if not training or rate == 0:
    return x
  return tf.nn.dropout(x, rate)

def gelu(x):
  """Gaussian Error Linear Unit activation function described in
  https://arxiv.org/abs/1606.08415.
  """
  return 0.5 * x * (1 + tf.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

class Dense(tf.keras.layers.Dense):
  
  def __init__(self, units, weight=None, transpose=False, **kwargs):
    
    super(Dense, self).__init__(units, **kwargs)
    self.weight = weight
    self.transpose = transpose

  def add_weight(self, name, *args, **kwargs):  # pylint: disable=arguments-differ
    if self.weight is not None and name == "kernel":
      return self.weight
    return super(Dense, self).add_weight(name, *args, **kwargs)

  def call(self, inputs):
    #print("where we are? ______________", self.name_scope(), self.kernel.name, self.bias.name)
    shape = shape_list(inputs)
    rank = len(shape)
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    outputs = tf.matmul(inputs, self.kernel, transpose_b=self.transpose)
    if self.use_bias:
      outputs = tf.nn.bias_add(outputs, self.bias)
    if self.activation is not None:
      outputs = self.activation(outputs)  # pylint: disable=not-callable
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.units])
    return outputs

  def forward_fn(self, inputs, args_dict):
    #print("where we are? ______________", self.name_scope())
    shape = shape_list(inputs)
    rank = len(shape)
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    kernel = args_dict[self.kernel.name]
    outputs = tf.matmul(inputs, kernel, transpose_b=self.transpose)
    if self.use_bias:
      bias = args_dict[self.bias.name]
      outputs = tf.nn.bias_add(outputs, bias)
    if self.activation is not None:
      outputs = self.activation(outputs)  # pylint: disable=not-callable
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.units])
    return outputs

  def map_v1_weights(self, weights):
    m = [(self.kernel, weights["kernel"])]
    if self.use_bias:
      m.append((self.bias, weights["bias"]))
    return m

class LayerNorm(tf.keras.layers.Layer):
  
  def __init__(self, epsilon=1e-6, **kwargs):
    
    super(LayerNorm, self).__init__(**kwargs)
    self.epsilon = epsilon

  def build(self, input_shape):
    """Creates the variables."""
    depth = input_shape[-1]
    self.beta = self.add_weight(
        "beta", [depth], initializer=tf.keras.initializers.Constant(0))
    self.gamma = self.add_weight(
        "gamma", [depth], initializer=tf.keras.initializers.Constant(1))
    super(LayerNorm, self).build(input_shape)

  def call(self, x):  # pylint: disable=arguments-differ
    """Normalizes :obj:`x`."""
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.math.rsqrt(variance + self.epsilon)
    return norm_x * self.gamma + self.beta

  def forward_fn(self, x, args_dict):  # pylint: disable=arguments-differ
    """Normalizes :obj:`x`."""
    gamma = args_dict[self.gamma.name]
    beta = args_dict[self.beta.name]
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.math.rsqrt(variance + self.epsilon)
    return norm_x * gamma + beta

  def map_v1_weights(self, weights):
    return [
        (self.beta, weights["beta"]),
        (self.gamma, weights["gamma"])
    ]

class LayerNorm_v2(tf.keras.layers.Layer):
  
  def __init__(self, epsilon=1e-6, **kwargs):
    
    super(LayerNorm_v2, self).__init__(**kwargs)
    self.epsilon = epsilon

  def build(self, input_shape):
    """Creates the variables."""
    depth = input_shape[-1]
    # self.beta = self.add_weight(
    #     "beta", [depth], initializer=tf.keras.initializers.Constant(0))
    self.gamma = self.add_weight(
        "gamma", [depth], initializer=tf.keras.initializers.Constant(1))
    super(LayerNorm_v2, self).build(input_shape)

  def call(self, x):  # pylint: disable=arguments-differ
    """Normalizes :obj:`x`."""
    #mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x), axis=[-1], keepdims=True)

    norm_x = x * tf.math.rsqrt(variance + self.epsilon)
    return norm_x * self.gamma 

  def map_v1_weights(self, weights):
    return [
        (self.gamma, weights["gamma"])
    ]
    # return [
    #     (self.beta, weights["beta"]),
    #     (self.gamma, weights["gamma"])
    # ]
  
class LayerNorm_v1(tf.keras.layers.Layer):
  
  def __init__(self, epsilon=1e-6, domain_numb=6, num_domain_units=[128, 2048, 2048, 1024, 1024, 128], **kwargs):
    
    super(LayerNorm_v1, self).__init__(**kwargs)
    self.epsilon = epsilon
    self.domain_numb = domain_numb
    self.num_domain_units = tf.constant(num_domain_units)

  def build(self, input_shape):
    """Creates the variables."""
    self.beta = self.add_weight(
        "beta", [sum(self.num_domain_units)], initializer=tf.keras.initializers.Constant(0))
    self.gamma = self.add_weight(
        "gamma", [sum(self.num_domain_units)], initializer=tf.keras.initializers.Constant(1))
    super(LayerNorm_v1, self).build(input_shape)

  def call(self, x, domain):  # pylint: disable=arguments-differ
    """Normalizes :obj:`x`."""
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.math.rsqrt(variance + self.epsilon)
    return norm_x * self.gamma[tf.reduce_sum(self.num_domain_units[:domain]) : tf.reduce_sum(self.num_domain_units[:domain+1])] + self.beta[tf.reduce_sum(self.num_domain_units[:domain]) : tf.reduce_sum(self.num_domain_units[:domain+1])]

  def map_v1_weights(self, weights):
    return [
        (self.beta, weights["beta"]),
        (self.gamma, weights["gamma"])
    ]

class LayerWrapper(tf.keras.layers.Layer):
  
  def __init__(self,
               layer,
               normalize_input=False,
               normalize_output=False,
               input_dropout=0,
               output_dropout=0,
               residual_connection=False,
               **kwargs):
    
    super(LayerWrapper, self).__init__(**kwargs)
    self.layer = layer
    self.input_layer_norm = LayerNorm() if normalize_input else None
    self.output_layer_norm = LayerNorm() if normalize_output else None
    self.input_dropout = input_dropout
    self.output_dropout = output_dropout
    self.residual_connection = residual_connection

  def call(self, inputs, *args, **kwargs):  # pylint: disable=arguments-differ
    
    training = kwargs.get("training")
    x = inputs
    if self.input_layer_norm is not None:
      x = self.input_layer_norm(x)  # pylint: disable=not-callable
    x = dropout(x, self.input_dropout, training=training)
    
    all_outputs = self.layer(x, *args, **kwargs)
    if isinstance(all_outputs, tuple):
      outputs = all_outputs[0]
      extra_outputs = list(all_outputs)[1:]
    else:
      outputs = all_outputs
      extra_outputs = None

    outputs = dropout(outputs, self.output_dropout, training=training)
    if self.residual_connection and outputs.shape[-1] == inputs.shape[-1]:
      outputs += inputs
    if self.output_layer_norm is not None:
      outputs = self.output_layer_norm(outputs)  # pylint: disable=not-callable

    if extra_outputs:
      return tuple([outputs] + extra_outputs)
    return outputs

  def forward_fn(self, inputs, args_dict, *args, **kwargs):  # pylint: disable=arguments-differ
    
    training = kwargs.get("training")
    x = inputs
    if self.input_layer_norm is not None:
      x = self.input_layer_norm.forward_fn(x, args_dict)  # pylint: disable=not-callable
    x = dropout(x, self.input_dropout, training=training)

    all_outputs = self.layer.forward_fn(x, args_dict, *args, **kwargs)
    if isinstance(all_outputs, tuple):
      outputs = all_outputs[0]
      extra_outputs = list(all_outputs)[1:]
    else:
      outputs = all_outputs
      extra_outputs = None

    outputs = dropout(outputs, self.output_dropout, training=training)
    if self.residual_connection and outputs.shape[-1] == inputs.shape[-1]:
      outputs += inputs
    if self.output_layer_norm is not None:
      outputs = self.output_layer_norm.forward_fn(outputs, args_dict)  # pylint: disable=not-callable

    if extra_outputs:
      return tuple([outputs] + extra_outputs)
    return outputs

  def get_config(self):
    """Returns the layer wrapper configuration."""
    config = {
        "layer": tf.keras.layers.serialize(self.layer),
        "normalize_input": self.input_layer_norm is not None,
        "normalize_output": self.output_layer_norm is not None,
        "input_dropout": self.input_dropout,
        "output_dropout": self.output_dropout,
        "residual_connection": self.residual_connection
    }
    base_config = super(LayerWrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    """Creates a layer wrapper from its configuration."""
    layer = tf.keras.layers.deserialize(config.pop("layer"))
    return cls(layer, **config)

class LayerWrapper_v2(tf.keras.layers.Layer):
  
  def __init__(self,
               layer,
               normalize_input=False,
               normalize_output=False,
               input_dropout=0,
               output_dropout=0,
               residual_connection=False,
               **kwargs):
    
    super(LayerWrapper_v2, self).__init__(**kwargs)
    self.layer = layer
    self.input_layer_norm = LayerNorm_v2() if normalize_input else None
    self.output_layer_norm = LayerNorm_v2() if normalize_output else None
    self.input_dropout = input_dropout
    self.output_dropout = output_dropout
    self.residual_connection = residual_connection

  def call(self, inputs, *args, **kwargs):  # pylint: disable=arguments-differ
    
    training = kwargs.get("training")
    x = inputs
    if self.input_layer_norm is not None:
      x = self.input_layer_norm(x)  # pylint: disable=not-callable
    x = dropout(x, self.input_dropout, training=training)
    #tf.print("inner self attention after norm: ", x[0,0,:], summarize=-1)
    all_outputs = self.layer(x, *args, **kwargs)
    if isinstance(all_outputs, tuple):
      outputs = all_outputs[0]
      extra_outputs = list(all_outputs)[1:]
    else:
      outputs = all_outputs
      extra_outputs = None

    outputs = dropout(outputs, self.output_dropout, training=training)
    if self.residual_connection and outputs.shape[-1] == inputs.shape[-1]:
      outputs += inputs
    if self.output_layer_norm is not None:
      outputs = self.output_layer_norm(outputs)  # pylint: disable=not-callable

    if extra_outputs:
      return tuple([outputs] + extra_outputs)
    return outputs

  def get_config(self):
    """Returns the layer wrapper configuration."""
    config = {
        "layer": tf.keras.layers.serialize(self.layer),
        "normalize_input": self.input_layer_norm is not None,
        "normalize_output": self.output_layer_norm is not None,
        "input_dropout": self.input_dropout,
        "output_dropout": self.output_dropout,
        "residual_connection": self.residual_connection
    }
    base_config = super(LayerWrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    """Creates a layer wrapper from its configuration."""
    layer = tf.keras.layers.deserialize(config.pop("layer"))
    return cls(layer, **config)

class Multi_ADAP_Dense(tf.keras.layers.Dense):
  
  def __init__(self, units, input_units, multi_domain_adapter_class, weight=None, transpose=False, num_domain_units=128, num_domains=6, **kwargs):
    
    super(Multi_ADAP_Dense, self).__init__(units, **kwargs)
    self.weight = weight
    self.transpose = transpose
    self.adapter = multi_domain_adapter_class(input_units, num_domain_units, input_units, domain_numb=num_domains, name="ADAP_output_layer")

  def add_weight(self, name, *args, **kwargs):  # pylint: disable=arguments-differ
    if self.weight is not None and name == "kernel":
      return self.weight
    return super(Multi_ADAP_Dense, self).add_weight(name, *args, **kwargs)

  def call(self, inputs, domain):

    shape = shape_list(inputs)
    rank = len(shape)
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    kernel = self.kernel
    kernel = tf.transpose(self.adapter(tf.transpose(tf.stop_gradient(kernel)), domain)) + kernel
    outputs = tf.matmul(inputs, kernel, transpose_b=self.transpose)
    if self.use_bias:
      outputs = tf.nn.bias_add(outputs, self.bias)
    if self.activation is not None:
      outputs = self.activation(outputs)  # pylint: disable=not-callable
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.units])
    return outputs

  def forward_fn(self, inputs, args_dict, domain):

    shape = shape_list(inputs)
    rank = len(shape)
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    kernel = args_dict[self.kernel.name]
    kernel = tf.transpose(self.adapter(tf.transpose(tf.stop_gradient(kernel)), domain)) + kernel
    outputs = tf.matmul(inputs, kernel, transpose_b=self.transpose)
    if self.use_bias:
      bias = args_dict[self.bias.name]
      outputs = tf.nn.bias_add(outputs, bias)
    if self.activation is not None:
      outputs = self.activation(outputs)  # pylint: disable=not-callable
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.units])
    return outputs

class Multi_ADAP_Dense_v1(tf.keras.layers.Dense):
  
  def __init__(self, units, input_units, multi_domain_adapter_class, weight=None, transpose=False, num_domain_units=128, num_domains=6, **kwargs):
    
    super(Multi_ADAP_Dense_v1, self).__init__(units, **kwargs)
    self.transpose = transpose
    self.use_bias = False
    self.use_multi_bias = True
    self.domain_numb = num_domains
    self.adapter = multi_domain_adapter_class(input_units, num_domain_units, input_units, domain_numb=num_domains, name="ADAP_output_layer")

  def build(self, input_shape):
    super(Multi_ADAP_Dense_v1, self).build(input_shape)
    scope_name = self.name_scope()
    self.multi_bias = self.add_weight("%s_multi_bias"%scope_name, shape=[self.domain_numb, self.units])

  def call(self, inputs, domain):
    shape = shape_list(inputs)
    rank = len(shape)
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    kernel = self.kernel
    kernel = tf.transpose(self.adapter(tf.transpose(tf.stop_gradient(kernel)), domain)) + kernel
    outputs = tf.matmul(inputs, kernel, transpose_b=self.transpose)
    if self.use_multi_bias:
      print("Using Multi_bias")
      bias = tf.nn.embedding_lookup(self.multi_bias, domain)
      outputs = tf.nn.bias_add(outputs, bias)
    if self.activation is not None:
      outputs = self.activation(outputs)  # pylint: disable=not-callable
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.units])
    return outputs

  def forward_fn(self, inputs, args_dict, domain):
    shape = shape_list(inputs)
    rank = len(shape)
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    kernel = args_dict[self.kernel.name]
    kernel = tf.transpose(self.adapter(tf.transpose(tf.stop_gradient(kernel)), domain)) + kernel
    outputs = tf.matmul(inputs, kernel, transpose_b=self.transpose)
    if self.use_multi_bias:
      print("Using Multi_bias")
      multi_bias = args_dict[self.multi_bias.name]
      bias = tf.nn.embedding_lookup(multi_bias, domain)
      outputs = tf.nn.bias_add(outputs, bias)
    if self.activation is not None:
      outputs = self.activation(outputs)  # pylint: disable=not-callable
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.units])
    return outputs

class Multi_LayerNorm(tf.keras.layers.Layer):
  
  def __init__(self, domain_numb, epsilon=1e-6, **kwargs):
    
    super(Multi_LayerNorm, self).__init__(**kwargs)
    self.epsilon = epsilon
    self.domain_numb = domain_numb

  def build(self, input_shape):
    """Creates the variables."""
    depth = input_shape[-1]
    #depth = self.input_dims_max
    self.beta = self.add_weight(
        "beta", [self.domain_numb, depth], initializer=tf.keras.initializers.Constant(0))
    self.gamma = self.add_weight(
        "gamma", [self.domain_numb, depth], initializer=tf.keras.initializers.Constant(1))
    super(Multi_LayerNorm, self).build(input_shape)

  def call(self, x, domain):  # pylint: disable=arguments-differ
    """Normalizes :obj:`x`."""
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.math.rsqrt(variance + self.epsilon)
    gamma = tf.nn.embedding_lookup(self.gamma, domain)
    beta = tf.nn.embedding_lookup(self.beta, domain)
    return norm_x * gamma + beta

  def forward_fn(self, x, args_dict, domain):  # pylint: disable=arguments-differ
    """Normalizes :obj:`x`."""
    dims = self.input_dims[domain]
    gamma = args_dict[self.gamma.name]
    gamma = tf.nn.embedding_lookup(gamma, domain)[:dims]
    beta = args_dict[self.beta.name]
    beta = tf.nn.embedding_lookup(beta, domain)[:dims]
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.math.rsqrt(variance + self.epsilon)
    return norm_x * gamma + beta

  def map_v1_weights(self, weights):
    return [
        (self.beta, weights["beta"]),
        (self.gamma, weights["gamma"])
    ]

class LayerWrapper_v1(tf.keras.layers.Layer):
  
  def __init__(self,
               layer,
               domain_numb = 5,
               normalize_input=False,
               normalize_output=False,
               input_dropout=0,
               output_dropout=0,
               residual_connection=False,
               **kwargs):
    
    super(LayerWrapper_v1, self).__init__(**kwargs)
    self.layer = layer
    self.input_layer_norm = Multi_LayerNorm(domain_numb=domain_numb) if normalize_input else None
    self.output_layer_norm = Multi_LayerNorm(domain_numb=domain_numb) if normalize_output else None
    self.input_dropout = input_dropout
    self.output_dropout = output_dropout
    self.residual_connection = residual_connection

  def call(self, inputs, domain, *args, **kwargs):  # pylint: disable=arguments-differ
    
    training = kwargs.get("training")
    x = inputs
    if self.input_layer_norm is not None:
      x = self.input_layer_norm(x, domain)  # pylint: disable=not-callable
    x = dropout(x, self.input_dropout, training=training)

    all_outputs = self.layer(x, *args, **kwargs)
    if isinstance(all_outputs, tuple):
      outputs = all_outputs[0]
      extra_outputs = list(all_outputs)[1:]
    else:
      outputs = all_outputs
      extra_outputs = None

    outputs = dropout(outputs, self.output_dropout, training=training)
    if self.residual_connection and outputs.shape[-1] == inputs.shape[-1]:
      outputs += inputs
    if self.output_layer_norm is not None:
      outputs = self.output_layer_norm(outputs, domain)  # pylint: disable=not-callable

    if extra_outputs:
      return tuple([outputs] + extra_outputs)
    return outputs

  def get_config(self):
    """Returns the layer wrapper configuration."""
    config = {
        "layer": tf.keras.layers.serialize(self.layer),
        "normalize_input": self.input_layer_norm is not None,
        "normalize_output": self.output_layer_norm is not None,
        "input_dropout": self.input_dropout,
        "output_dropout": self.output_dropout,
        "residual_connection": self.residual_connection
    }
    base_config = super(LayerWrapper_v1, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    """Creates a layer wrapper from its configuration."""
    layer = tf.keras.layers.deserialize(config.pop("layer"))
    return cls(layer, **config)
