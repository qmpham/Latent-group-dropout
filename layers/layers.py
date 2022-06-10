import sys
sys.path.append("/gpfsdswork/projects/rech/sfz/utt84zy/anaconda3/envs/huggingface/lib/python3.7/site-packages")

import tensorflow as tf
import numpy as np
from layers import common
from opennmt.utils import misc
from opennmt.utils.misc import shape_list
import sys
from opennmt.layers.rnn import _RNNWrapper
from opennmt.layers import reducer as reducer_lib
from opennmt.utils.losses import _smooth_one_hot_labels


class Classification_layer(tf.keras.layers.Layer):
  def __init__(self,   
               input_dim,            
               domain_numb=6,
               kernel_size=512,
               dropout=0.1,
               **kwargs):
    
    super(Classification_layer, self).__init__(**kwargs)
    self.domain_numb = domain_numb
    self.input_dim = input_dim
    self.layer_norm = common.LayerNorm()
    self.kernel_size = kernel_size
    self.ff_layer_1 = common.Dense(2048, use_bias=True, activation=tf.nn.relu)
    self.ff_layer_2 = common.Dense(2048, use_bias=True, activation=tf.nn.relu)
    self.ff_layer_end = common.Dense(domain_numb, use_bias=True, activation=tf.nn.tanh)

  def build(self, input_shape):
    super(Classification_layer, self).build(input_shape)
    scope_name = self.name_scope()
    self.v = self.add_weight("%s_v_a"%scope_name, shape=[self.kernel_size])
    self.W = self.add_weight("%s_W_a"%scope_name, shape=[self.input_dim, self.kernel_size])

  def call(self, inputs, src_length, training=True):
    #print("inputs:", inputs)    
    v = self.v
    W = self.W
    v_a = tf.expand_dims(tf.expand_dims(v, 0),2)
    v_a = tf.tile(v_a, [tf.shape(inputs)[0], 1, 1])
    W_a = tf.expand_dims(W, 0)
    W_a = tf.tile(W_a, [tf.shape(inputs)[0],1,1])
    attention_weight = tf.matmul(tf.tanh(tf.matmul(inputs, W_a)), v_a)
    adv_mask = tf.sequence_mask(src_length, maxlen=tf.shape(attention_weight)[1], dtype=tf.float32)
    adv_mask = tf.expand_dims(adv_mask, -1)
    attention_weight = tf.cast(tf.cast(attention_weight, tf.float32) * adv_mask + ((1.0 - adv_mask) * tf.float32.min), attention_weight.dtype)
    attention_weight = tf.cast(tf.nn.softmax(tf.cast(attention_weight, tf.float32)), attention_weight.dtype)
    attention_weight = tf.squeeze(attention_weight,-1)
    attention_weight = tf.expand_dims(attention_weight, 1)
    e = tf.matmul(attention_weight, inputs)
    e = tf.squeeze(e,1)
    e = common.dropout(e, rate=0.3, training=training)
    #logits = self.ff_layer_1(tf.nn.relu(e))          
    #logits = common.dropout(logits, rate=0.3, training=training)
    #logits = self.ff_layer_2(logits)
    #logits = common.dropout(logits, rate=0.3, training=training)
    logits = self.ff_layer_end(tf.nn.relu(e))
    return e, logits
  
class Multi_domain_FeedForwardNetwork(tf.keras.layers.Layer):

  def __init__(self,
               inner_dim,
               output_dim,
               dropout=0.1,
               activation=tf.nn.relu,
               **kwargs):
    
    super(Multi_domain_FeedForwardNetwork, self).__init__(**kwargs)
    self.inner = common.Dense(inner_dim, activation=activation)
    self.outer = common.Dense(output_dim)
    self.dropout = dropout
    self.layer_norm = common.LayerNorm()

  def call(self, inputs, mask, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    inputs = self.layer_norm(inputs)
    inner = self.inner(inputs)
    inner = inner * tf.broadcast_to(tf.expand_dims(mask,1), tf.shape(inner))
    inner = common.dropout(inner, self.dropout, training=training)
    outputs = self.outer(inner)
    self.add_loss(tf.reduce_mean(tf.reduce_sum(tf.abs(tf.reshape(outputs,[-1,tf.shape(outputs)[-1]])),axis=-1)))
    if not training:
      tf.print("#######")
      tf.print(self.name_scope(), "Inputs_max_abs_pooling: ", tf.reduce_max(tf.abs(inputs)), "ADAP_max_abs_pooling: ", 
                tf.reduce_max(tf.abs(outputs)), "ADAP_min_abs_pooling: ", tf.reduce_min(tf.abs(outputs)), sep="|")
      tf.print("#######")
    return outputs

  def forward_fn(self, inputs, args_dict, mask, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    inputs = self.layer_norm(inputs)
    inner = self.inner.forward_fn(inputs, args_dict)
    inner = inner * tf.broadcast_to(tf.expand_dims(mask,1), tf.shape(inner))
    inner = common.dropout(inner, self.dropout, training=training)
    return self.outer.forward_fn(inner, args_dict)

class Multi_domain_FeedForwardNetwork_v2(tf.keras.layers.Layer):

  def __init__(self,
               input_dim, 
               inner_dim,
               output_dim,
               domain_numb=6,
               dropout=0.1,
               activation=tf.nn.relu,
               outer_activation=None,
               **kwargs):
    
    super(Multi_domain_FeedForwardNetwork_v2, self).__init__(**kwargs)
    self.dropout = dropout
    self.domain_numb = domain_numb
    self.input_dim = input_dim
    self.inner_dim = inner_dim
    self.output_dim = output_dim
    self.layer_norm = common.LayerNorm()
    self.inner_transpose = False
    self.outer_transpose = False
    self.inner_use_bias = True
    self.outer_use_bias = True
    self.inner_activation = activation
    self.outer_activation = outer_activation
  
  def build(self, input_shape):
    super(Multi_domain_FeedForwardNetwork_v2, self).build(input_shape)
    scope_name = self.name_scope()
    self.inner_kernel = self.add_weight("%s_inner_weight"%scope_name, shape=[self.domain_numb, self.input_dim*self.inner_dim])
    self.inner_bias = self.add_weight("%s_inner_bias"%scope_name, shape=[self.domain_numb, self.inner_dim])
    self.outer_kernel = self.add_weight("%s_outer_weight"%scope_name, shape=[self.domain_numb, self.inner_dim*self.output_dim])
    self.outer_bias = self.add_weight("%s_outer_bias"%scope_name, shape=[self.domain_numb, self.output_dim])
    
  def call(self, inputs, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    if not(mask is None):
      mask=tf.cast(mask,tf.float32)
    mask=None
    inputs = self.layer_norm(inputs)
    ##### inner layer
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    dom_inner_kernel = tf.nn.embedding_lookup(self.inner_kernel, domain)
    dom_inner_bias = tf.nn.embedding_lookup(self.inner_bias, domain)
    dom_inner_kernel = tf.reshape(dom_inner_kernel, [-1, self.inner_dim])
    inner = tf.matmul(inputs, dom_inner_kernel, transpose_b=self.inner_transpose)
    if self.inner_use_bias:
      inner = tf.nn.bias_add(inner, dom_inner_bias)
    if self.inner_activation is not None:
      inner = self.inner_activation(inner)  # pylint: disable=not-callable
    if rank > 2:
      inner = tf.reshape(inner, shape[:-1] + [self.inner_dim])
    ##### output layer
    inner = common.dropout(inner, self.dropout, training=training)
    shape = shape_list(inner)
    rank = len(shape)      
    if rank > 2:
      inner = tf.reshape(inner, [-1, shape[-1]])
    dom_outer_kernel = tf.nn.embedding_lookup(self.outer_kernel, domain)
    dom_outer_bias = tf.nn.embedding_lookup(self.outer_bias, domain)
    dom_outer_kernel = tf.reshape(dom_outer_kernel, [-1, self.output_dim])
    outputs = tf.matmul(inner, dom_outer_kernel, transpose_b=self.outer_transpose)
    if self.outer_use_bias:
      outputs = tf.nn.bias_add(outputs, dom_outer_bias)
    if self.outer_activation is not None:
      outputs = self.outer_activation(outputs)  # pylint: disable=not-callable
    if mask is not None:
      self.add_loss(tf.divide(tf.reduce_sum(mask * tf.reduce_sum(tf.abs(outputs),axis=-1)), tf.reduce_sum(mask)))
    else:
      self.add_loss(tf.reduce_mean(tf.reduce_sum(tf.abs(outputs),axis=-1)))
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])   
    
    if not training:
      tf.print("###", self.name_scope(), "Inputs_max_abs_pooling: ", tf.reduce_max(tf.abs(inputs)), "ADAP_max_abs_pooling: ", 
                tf.reduce_max(tf.abs(outputs)), "ADAP_min_abs_pooling: ", tf.reduce_min(tf.abs(outputs)), "domain: ", domain, "###", sep="|")    
    return outputs

  def forward_fn(self, inputs, args_dict, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    if not(mask is None):
      mask=tf.cast(mask,tf.float32)
    mask=None
    inner_kernel = args_dict[self.inner_kernel.name]
    outer_kernel = args_dict[self.outer_kernel.name]
    inner_bias = args_dict[self.inner_bias.name]
    outer_bias = args_dict[self.outer_bias.name]

    inputs = self.layer_norm(inputs)
    ##### inner layer
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    dom_inner_kernel = tf.nn.embedding_lookup(inner_kernel, domain)
    dom_inner_bias = tf.nn.embedding_lookup(inner_bias, domain)
    dom_inner_kernel = tf.reshape(dom_inner_kernel, [-1, self.inner_dim])
    inner = tf.matmul(inputs, dom_inner_kernel, transpose_b=self.inner_transpose)
    if self.inner_use_bias:
      inner = tf.nn.bias_add(inner, dom_inner_bias)
    if self.inner_activation is not None:
      inner = self.inner_activation(inner)  # pylint: disable=not-callable
    if rank > 2:
      inner = tf.reshape(inner, shape[:-1] + [self.inner_dim])
    ##### output layer
    inner = common.dropout(inner, self.dropout, training=training)
    shape = shape_list(inner)
    rank = len(shape)      
    if rank > 2:
      inner = tf.reshape(inner, [-1, shape[-1]])
    dom_outer_kernel = tf.nn.embedding_lookup(outer_kernel, domain)
    dom_outer_bias = tf.nn.embedding_lookup(outer_bias, domain)
    dom_outer_kernel = tf.reshape(dom_outer_kernel, [-1, self.output_dim])
    outputs = tf.matmul(inner, dom_outer_kernel, transpose_b=self.outer_transpose)
    if self.outer_use_bias:
      outputs = tf.nn.bias_add(outputs, dom_outer_bias)
    if self.outer_activation is not None:
      outputs = self.outer_activation(outputs)  # pylint: disable=not-callable
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])
    return outputs

class Multi_domain_FeedForwardNetwork_v3(tf.keras.layers.Layer):

  def __init__(self,
               input_dim, 
               inner_dim,
               output_dim,
               domain_numb=6,
               dropout=0.1,
               activation=tf.nn.relu,
               inner_layer_norm=None,
               outer_activation=None,
               **kwargs):
    
    super(Multi_domain_FeedForwardNetwork_v3, self).__init__(**kwargs)
    self.dropout = dropout
    self.domain_numb = domain_numb
    self.input_dim = input_dim
    self.inner_dim = inner_dim
    self.output_dim = output_dim
    
    if inner_layer_norm:
      self.layer_norm = inner_layer_norm(domain_numb)
      self.inner_layer_norm = inner_layer_norm(domain_numb)
    else:
      self.layer_norm = common.LayerNorm()
      self.inner_layer_norm = common.LayerNorm()
    self.inner_transpose = False
    self.outer_transpose = False
    self.inner_use_bias = True
    self.outer_use_bias = True
    self.inner_activation = activation
    self.outer_activation = outer_activation
  
  def build(self, input_shape):
    super(Multi_domain_FeedForwardNetwork_v3, self).build(input_shape)
    scope_name = self.name_scope()
    #print("self.domain_numb, self.input_dim, self.inner_dim: ", self.domain_numb, self.input_dim, self.inner_dim)
    self.inner_kernel = self.add_weight("%s_inner_weight"%scope_name, shape=[self.domain_numb, self.input_dim*self.inner_dim])
    self.inner_bias = self.add_weight("%s_inner_bias"%scope_name, shape=[self.domain_numb, self.inner_dim])
    self.outer_kernel = self.add_weight("%s_outer_weight"%scope_name, shape=[self.domain_numb, self.inner_dim*self.output_dim])
    self.outer_bias = self.add_weight("%s_outer_bias"%scope_name, shape=[self.domain_numb, self.output_dim])
    
  def call(self, inputs, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    if not(mask is None):
      mask=tf.cast(mask,tf.float32)
    mask=None

    if isinstance(self.layer_norm, common.LayerNorm):
      inputs = self.layer_norm(inputs)
    else:
      inputs = self.layer_norm(inputs, domain)
    ##### inner layer
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    dom_inner_kernel = tf.nn.embedding_lookup(self.inner_kernel, domain)
    dom_inner_bias = tf.nn.embedding_lookup(self.inner_bias, domain)
    dom_inner_kernel = tf.reshape(dom_inner_kernel, [-1, self.inner_dim])
    inner = tf.matmul(inputs, dom_inner_kernel, transpose_b=self.inner_transpose)
    if self.inner_use_bias:
      inner = tf.nn.bias_add(inner, dom_inner_bias)
    if self.inner_activation is not None:
      if isinstance(self.inner_layer_norm, common.LayerNorm):
        inner = self.inner_layer_norm(inner)
      else:
        inner = self.inner_layer_norm(inner,domain)
      inner = self.inner_activation(inner)  # pylint: disable=not-callable
    if rank > 2:
      inner = tf.reshape(inner, shape[:-1] + [self.inner_dim])
    ##### output layer
    inner = common.dropout(inner, self.dropout, training=training)
    shape = shape_list(inner)
    rank = len(shape)      
    if rank > 2:
      inner = tf.reshape(inner, [-1, shape[-1]])
    dom_outer_kernel = tf.nn.embedding_lookup(self.outer_kernel, domain)
    dom_outer_bias = tf.nn.embedding_lookup(self.outer_bias, domain)
    dom_outer_kernel = tf.reshape(dom_outer_kernel, [-1, self.output_dim])
    outputs = tf.matmul(inner, dom_outer_kernel, transpose_b=self.outer_transpose)
    if self.outer_use_bias:
      outputs = tf.nn.bias_add(outputs, dom_outer_bias)
    if self.outer_activation is not None:
      outputs = self.outer_activation(outputs)  # pylint: disable=not-callable
    self.add_loss(tf.reduce_mean(tf.reduce_sum(outputs*outputs,axis=-1)))
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])   
    
    # if not training:
    #   tf.print("###", self.name_scope(), "Inputs_max_abs_pooling: ", tf.reduce_max(tf.abs(inputs)), "ADAP_max_abs_pooling: ", 
    #             tf.reduce_max(tf.abs(outputs)), "ADAP_min_abs_pooling: ", tf.reduce_min(tf.abs(outputs)), "domain: ", domain, "###", sep="|")    
    return outputs

  def forward_fn(self, inputs, args_dict, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    if not(mask is None):
      mask=tf.cast(mask,tf.float32)
    mask=None
    inner_kernel = args_dict[self.inner_kernel.name]
    outer_kernel = args_dict[self.outer_kernel.name]
    inner_bias = args_dict[self.inner_bias.name]
    outer_bias = args_dict[self.outer_bias.name]

    inputs = self.layer_norm(inputs)
    ##### inner layer
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    dom_inner_kernel = tf.nn.embedding_lookup(inner_kernel, domain)
    dom_inner_bias = tf.nn.embedding_lookup(inner_bias, domain)
    dom_inner_kernel = tf.reshape(dom_inner_kernel, [-1, self.inner_dim])
    inner = tf.matmul(inputs, dom_inner_kernel, transpose_b=self.inner_transpose)
    if self.inner_use_bias:
      inner = tf.nn.bias_add(inner, dom_inner_bias)
    if self.inner_activation is not None:
      inner = self.inner_layer_norm(inner)
      inner = self.inner_activation(inner)  # pylint: disable=not-callable
    if rank > 2:
      inner = tf.reshape(inner, shape[:-1] + [self.inner_dim])
    ##### output layer
    inner = common.dropout(inner, self.dropout, training=training)
    shape = shape_list(inner)
    rank = len(shape)      
    if rank > 2:
      inner = tf.reshape(inner, [-1, shape[-1]])
    dom_outer_kernel = tf.nn.embedding_lookup(outer_kernel, domain)
    dom_outer_bias = tf.nn.embedding_lookup(outer_bias, domain)
    dom_outer_kernel = tf.reshape(dom_outer_kernel, [-1, self.output_dim])
    outputs = tf.matmul(inner, dom_outer_kernel, transpose_b=self.outer_transpose)
    if self.outer_use_bias:
      outputs = tf.nn.bias_add(outputs, dom_outer_bias)
    if self.outer_activation is not None:
      outputs = self.outer_activation(outputs)  # pylint: disable=not-callable
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])
    return outputs

class Multi_domain_FeedForwardNetwork_v7(tf.keras.layers.Layer):

  def __init__(self,
               input_dim, 
               inner_dim,
               output_dim,
               domain_numb=6,
               dropout=0.1,
               activation=tf.nn.relu,
               outer_activation=None,
               **kwargs):
    
    super(Multi_domain_FeedForwardNetwork_v7, self).__init__(**kwargs)
    self.dropout = dropout
    self.domain_numb = domain_numb
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.layer_norm = common.LayerNorm()
    self.inner_layer_norm = common.LayerNorm_v1(domain_numb=domain_numb, num_domain_units=inner_dim)
    self.inner_transpose = False
    self.outer_transpose = False
    self.inner_use_bias = True
    self.outer_use_bias = True
    self.inner_activation = activation
    self.outer_activation = outer_activation
    self.inner_dim = tf.constant(inner_dim)
  
  def build(self, input_shape):
    super(Multi_domain_FeedForwardNetwork_v7, self).build(input_shape)
    scope_name = self.name_scope()
    print("self.domain_numb, self.input_dim, self.inner_dim: ", self.domain_numb, self.input_dim, self.inner_dim)
    self.inner_kernel = self.add_weight("%s_inner_weight"%scope_name, shape=[self.input_dim * sum(self.inner_dim)])
    self.inner_bias = self.add_weight("%s_inner_bias"%scope_name, shape=[sum(self.inner_dim)])
    self.outer_kernel = self.add_weight("%s_outer_weight"%scope_name, shape=[sum(self.inner_dim) * self.output_dim])
    self.outer_bias = self.add_weight("%s_outer_bias"%scope_name, shape=[self.domain_numb, self.output_dim])
    
  def call(self, inputs, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""

    if not(mask is None):
      mask=tf.cast(mask,tf.float32)
    mask=None
    inputs = self.layer_norm(inputs)
    ##### inner layer
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    dom_inner_kernel = self.inner_kernel[tf.reduce_sum(self.inner_dim[:domain]) * self.input_dim : tf.reduce_sum(self.inner_dim[:domain+1]) * self.input_dim] #tf.nn.embedding_lookup(self.inner_kernel, domain)
    dom_inner_bias = self.inner_bias[tf.reduce_sum(self.inner_dim[:domain]) : tf.reduce_sum(self.inner_dim[:domain+1])] #tf.nn.embedding_lookup(self.inner_bias, domain)
    dom_inner_kernel = tf.reshape(dom_inner_kernel, [-1, self.inner_dim[domain]])
    inner = tf.matmul(inputs, dom_inner_kernel, transpose_b=self.inner_transpose)
    if self.inner_use_bias:
      inner = tf.nn.bias_add(inner, dom_inner_bias)
    if self.inner_activation is not None:
      inner = self.inner_layer_norm(inner, domain)
      inner = self.inner_activation(inner)  # pylint: disable=not-callable
    if rank > 2:
      inner = tf.reshape(inner, shape[:-1] + [self.inner_dim[domain]])
    ##### output layer
    inner = common.dropout(inner, self.dropout, training=training)
    shape = shape_list(inner)
    rank = len(shape)      
    if rank > 2:
      inner = tf.reshape(inner, [-1, shape[-1]])
    dom_outer_kernel = self.outer_kernel[tf.reduce_sum(self.inner_dim[:domain]) * self.output_dim : tf.reduce_sum(self.inner_dim[:domain+1]) * self.output_dim] #tf.nn.embedding_lookup(self.outer_kernel, domain)
    dom_outer_bias = tf.nn.embedding_lookup(self.outer_bias, domain)
    dom_outer_kernel = tf.reshape(dom_outer_kernel, [-1, self.output_dim])
    outputs = tf.matmul(inner, dom_outer_kernel, transpose_b=self.outer_transpose)
    if self.outer_use_bias:
      outputs = tf.nn.bias_add(outputs, dom_outer_bias)
    if self.outer_activation is not None:
      outputs = self.outer_activation(outputs)  # pylint: disable=not-callable
    if mask is not None:
      self.add_loss(tf.divide(tf.reduce_sum(mask * tf.reduce_sum(tf.abs(outputs),axis=-1)), tf.reduce_sum(mask)))
    else:
      self.add_loss(tf.reduce_mean(tf.reduce_sum(tf.abs(outputs),axis=-1)))
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])   
    
    return outputs

  def forward_fn(self, inputs, args_dict, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    if not(mask is None):
      mask=tf.cast(mask,tf.float32)
    mask=None
    inner_kernel = args_dict[self.inner_kernel.name]
    outer_kernel = args_dict[self.outer_kernel.name]
    inner_bias = args_dict[self.inner_bias.name]
    outer_bias = args_dict[self.outer_bias.name]

    inputs = self.layer_norm(inputs)
    ##### inner layer
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    dom_inner_kernel = tf.nn.embedding_lookup(inner_kernel, domain)
    dom_inner_bias = tf.nn.embedding_lookup(inner_bias, domain)
    dom_inner_kernel = tf.reshape(dom_inner_kernel, [-1, self.inner_dim])
    inner = tf.matmul(inputs, dom_inner_kernel, transpose_b=self.inner_transpose)
    if self.inner_use_bias:
      inner = tf.nn.bias_add(inner, dom_inner_bias)
    if self.inner_activation is not None:
      inner = self.inner_layer_norm(inner)
      inner = self.inner_activation(inner)  # pylint: disable=not-callable
    if rank > 2:
      inner = tf.reshape(inner, shape[:-1] + [self.inner_dim])
    ##### output layer
    inner = common.dropout(inner, self.dropout, training=training)
    shape = shape_list(inner)
    rank = len(shape)      
    if rank > 2:
      inner = tf.reshape(inner, [-1, shape[-1]])
    dom_outer_kernel = tf.nn.embedding_lookup(outer_kernel, domain)
    dom_outer_bias = tf.nn.embedding_lookup(outer_bias, domain)
    dom_outer_kernel = tf.reshape(dom_outer_kernel, [-1, self.output_dim])
    outputs = tf.matmul(inner, dom_outer_kernel, transpose_b=self.outer_transpose)
    if self.outer_use_bias:
      outputs = tf.nn.bias_add(outputs, dom_outer_bias)
    if self.outer_activation is not None:
      outputs = self.outer_activation(outputs)  # pylint: disable=not-callable
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])
    return outputs

class Multi_domain_FeedForwardNetwork_v1(tf.keras.layers.Layer):

  def __init__(self,
               input_dim, 
               inner_dim,
               output_dim,
               domain_numb=6,
               dropout=0.1,
               activation=tf.nn.tanh,
               outer_activation=None,
               **kwargs):
    
    super(Multi_domain_FeedForwardNetwork_v1, self).__init__(**kwargs)
    self.dropout = dropout
    self.domain_numb = domain_numb
    self.input_dim = input_dim
    self.inner_dim = inner_dim
    self.output_dim = output_dim
    self.layer_norm = common.LayerNorm()
    self.inner_layer_norm = common.LayerNorm()
    self.inner_transpose = False
    self.outer_transpose = False
    self.inner_use_bias = True
    self.outer_use_bias = True
    self.inner_activation = activation
    self.outer_activation = outer_activation
  
  def build(self, input_shape):
    super(Multi_domain_FeedForwardNetwork_v1, self).build(input_shape)
    scope_name = self.name_scope()
    self.inner_kernel = self.add_weight("%s_inner_weight"%scope_name, shape=[self.domain_numb, self.input_dim*self.inner_dim])
    self.inner_bias = self.add_weight("%s_inner_bias"%scope_name, shape=[self.domain_numb, self.inner_dim])
    self.outer_kernel = self.add_weight("%s_outer_weight"%scope_name, shape=[self.domain_numb, self.inner_dim*self.output_dim])
    self.outer_bias = self.add_weight("%s_outer_bias"%scope_name, shape=[self.domain_numb, self.output_dim])
    
  def call(self, inputs, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    if not(mask is None):
      mask=tf.cast(mask,tf.float32)
    mask=None
    inputs = self.layer_norm(inputs)
    ##### inner layer
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    dom_inner_kernel = tf.nn.embedding_lookup(self.inner_kernel, domain)
    dom_inner_bias = tf.nn.embedding_lookup(self.inner_bias, domain)
    dom_inner_kernel = tf.reshape(dom_inner_kernel, [-1, self.inner_dim])
    inner = tf.matmul(inputs, dom_inner_kernel, transpose_b=self.inner_transpose)
    if self.inner_use_bias:
      inner = tf.nn.bias_add(inner, dom_inner_bias)
    if self.inner_activation is not None:
      inner = self.inner_layer_norm(inner)
      inner = self.inner_activation(inner)  # pylint: disable=not-callable
    if rank > 2:
      inner = tf.reshape(inner, shape[:-1] + [self.inner_dim])
    ##### output layer
    inner = common.dropout(inner, self.dropout, training=training)
    shape = shape_list(inner)
    rank = len(shape)      
    if rank > 2:
      inner = tf.reshape(inner, [-1, shape[-1]])
    dom_outer_kernel = tf.nn.embedding_lookup(self.outer_kernel, domain)
    dom_outer_bias = tf.nn.embedding_lookup(self.outer_bias, domain)
    dom_outer_kernel = tf.reshape(dom_outer_kernel, [-1, self.output_dim])
    outputs = tf.matmul(inner, dom_outer_kernel, transpose_b=self.outer_transpose)
    if self.outer_use_bias:
      outputs = tf.nn.bias_add(outputs, dom_outer_bias)
    if self.outer_activation is not None:
      outputs = self.outer_activation(outputs)  # pylint: disable=not-callable
    if mask is not None:
      self.add_loss(tf.divide(tf.reduce_sum(mask * tf.reduce_sum(tf.abs(inner),axis=-1)), tf.reduce_sum(mask)))
    else:
      self.add_loss(tf.reduce_mean(tf.reduce_sum(tf.abs(inner),axis=-1)))

    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])   
    if not training:
      tf.print("###", self.name_scope(), "Inputs_max_abs_pooling: ", tf.reduce_max(tf.abs(inputs)), "ADAP_max_abs_pooling: ", 
                tf.reduce_max(tf.abs(inner)), "ADAP_min_abs_pooling: ", tf.reduce_min(tf.abs(inner)), "domain: ", domain, "###", sep="|")    
    return outputs

  def forward_fn(self, inputs, args_dict, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    if not(mask is None):
      mask=tf.cast(mask,tf.float32)
    mask=None
    inner_kernel = args_dict[self.inner_kernel.name]
    outer_kernel = args_dict[self.outer_kernel.name]
    inner_bias = args_dict[self.inner_bias.name]
    outer_bias = args_dict[self.outer_bias.name]

    inputs = self.layer_norm(inputs)
    ##### inner layer
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    dom_inner_kernel = tf.nn.embedding_lookup(inner_kernel, domain)
    dom_inner_bias = tf.nn.embedding_lookup(inner_bias, domain)
    dom_inner_kernel = tf.reshape(dom_inner_kernel, [-1, self.inner_dim])
    inner = tf.matmul(inputs, dom_inner_kernel, transpose_b=self.inner_transpose)
    if self.inner_use_bias:
      inner = tf.nn.bias_add(inner, dom_inner_bias)
    if self.inner_activation is not None:
      inner = self.inner_layer_norm(inner)
      inner = self.inner_activation(inner)  # pylint: disable=not-callable
    if rank > 2:
      inner = tf.reshape(inner, shape[:-1] + [self.inner_dim])
    ##### output layer
    inner = common.dropout(inner, self.dropout, training=training)
    shape = shape_list(inner)
    rank = len(shape)      
    if rank > 2:
      inner = tf.reshape(inner, [-1, shape[-1]])
    dom_outer_kernel = tf.nn.embedding_lookup(outer_kernel, domain)
    dom_outer_bias = tf.nn.embedding_lookup(outer_bias, domain)
    dom_outer_kernel = tf.reshape(dom_outer_kernel, [-1, self.output_dim])
    outputs = tf.matmul(inner, dom_outer_kernel, transpose_b=self.outer_transpose)
    if self.outer_use_bias:
      outputs = tf.nn.bias_add(outputs, dom_outer_bias)
    if self.outer_activation is not None:
      outputs = self.outer_activation(outputs)  # pylint: disable=not-callable
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])
    return outputs

class Multi_domain_FeedForwardNetwork_v0(tf.keras.layers.Layer):

  def __init__(self,
               input_dim, 
               inner_dim,
               output_dim,
               domain_numb=6,
               dropout=0.1,
               activation=tf.nn.relu,
               outer_activation=None,
               **kwargs):
    
    super(Multi_domain_FeedForwardNetwork_v0, self).__init__(**kwargs)
    self.dropout = dropout
    self.domain_numb = domain_numb
    self.input_dim = input_dim
    self.inner_dim = inner_dim
    self.output_dim = output_dim
    self.layer_norm = common.LayerNorm()
    self.inner_layer_norm = common.LayerNorm()
    self.inner_transpose = False
    self.outer_transpose = False
    self.inner_use_bias = True
    self.outer_use_bias = True
    self.inner_activation = activation
    self.outer_activation = outer_activation
  
  def build(self, input_shape):
    super(Multi_domain_FeedForwardNetwork_v0, self).build(input_shape)
    scope_name = self.name_scope()
    self.inner_kernel = self.add_weight("%s_inner_weight"%scope_name, shape=[self.domain_numb, self.input_dim*self.inner_dim])
    self.inner_bias = self.add_weight("%s_inner_bias"%scope_name, shape=[self.domain_numb, self.inner_dim])
    self.outer_kernel = self.add_weight("%s_outer_weight"%scope_name, shape=[self.domain_numb, self.inner_dim*self.output_dim])
    self.outer_bias = self.add_weight("%s_outer_bias"%scope_name, shape=[self.domain_numb, self.output_dim])
    
  def call(self, inputs, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    if not(mask is None):
      mask=tf.cast(mask,tf.float32)
    mask=None
    #inputs = self.layer_norm(inputs)
    ##### inner layer
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    dom_inner_kernel = tf.nn.embedding_lookup(self.inner_kernel, domain)
    dom_inner_bias = tf.nn.embedding_lookup(self.inner_bias, domain)
    dom_inner_kernel = tf.reshape(dom_inner_kernel, [-1, self.inner_dim])
    inner = tf.matmul(inputs, dom_inner_kernel, transpose_b=self.inner_transpose)
    if self.inner_use_bias:
      inner = tf.nn.bias_add(inner, dom_inner_bias)
    if self.inner_activation is not None:
      inner = self.inner_layer_norm(inner)
      inner = self.inner_activation(inner)  # pylint: disable=not-callable
    if rank > 2:
      inner = tf.reshape(inner, shape[:-1] + [self.inner_dim])
    ##### output layer
    inner = common.dropout(inner, self.dropout, training=training)
    shape = shape_list(inner)
    rank = len(shape)      
    if rank > 2:
      inner = tf.reshape(inner, [-1, shape[-1]])
    dom_outer_kernel = tf.nn.embedding_lookup(self.outer_kernel, domain)
    dom_outer_bias = tf.nn.embedding_lookup(self.outer_bias, domain)
    dom_outer_kernel = tf.reshape(dom_outer_kernel, [-1, self.output_dim])
    outputs = tf.matmul(inner, dom_outer_kernel, transpose_b=self.outer_transpose)
    if self.outer_use_bias:
      outputs = tf.nn.bias_add(outputs, dom_outer_bias)
    if self.outer_activation is not None:
      outputs = self.outer_activation(outputs)  # pylint: disable=not-callable
    if mask is not None:
      self.add_loss(tf.divide(tf.reduce_sum(mask * tf.reduce_sum(tf.abs(outputs),axis=-1)), tf.reduce_sum(mask)))
    else:
      self.add_loss(tf.reduce_mean(tf.reduce_sum(tf.abs(outputs),axis=-1)))
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])   
    
    if not training:
      tf.print("###", self.name_scope(), "Inputs_max_abs_pooling: ", tf.reduce_max(tf.abs(inputs)), "ADAP_max_abs_pooling: ", 
                tf.reduce_max(tf.abs(outputs)), "ADAP_min_abs_pooling: ", tf.reduce_min(tf.abs(outputs)), "domain: ", domain, "###", sep="|")    
    return outputs

  def forward_fn(self, inputs, args_dict, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    if not(mask is None):
      mask=tf.cast(mask,tf.float32)
    mask=None
    inner_kernel = args_dict[self.inner_kernel.name]
    outer_kernel = args_dict[self.outer_kernel.name]
    inner_bias = args_dict[self.inner_bias.name]
    outer_bias = args_dict[self.outer_bias.name]

    ##### inner layer
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    dom_inner_kernel = tf.nn.embedding_lookup(inner_kernel, domain)
    dom_inner_bias = tf.nn.embedding_lookup(inner_bias, domain)
    dom_inner_kernel = tf.reshape(dom_inner_kernel, [-1, self.inner_dim])
    inner = tf.matmul(inputs, dom_inner_kernel, transpose_b=self.inner_transpose)
    if self.inner_use_bias:
      inner = tf.nn.bias_add(inner, dom_inner_bias)
    if self.inner_activation is not None:
      inner = self.inner_layer_norm(inner)
      inner = self.inner_activation(inner)  # pylint: disable=not-callable
    if rank > 2:
      inner = tf.reshape(inner, shape[:-1] + [self.inner_dim])
    ##### output layer
    inner = common.dropout(inner, self.dropout, training=training)
    shape = shape_list(inner)
    rank = len(shape)      
    if rank > 2:
      inner = tf.reshape(inner, [-1, shape[-1]])
    dom_outer_kernel = tf.nn.embedding_lookup(outer_kernel, domain)
    dom_outer_bias = tf.nn.embedding_lookup(outer_bias, domain)
    dom_outer_kernel = tf.reshape(dom_outer_kernel, [-1, self.output_dim])
    outputs = tf.matmul(inner, dom_outer_kernel, transpose_b=self.outer_transpose)
    if self.outer_use_bias:
      outputs = tf.nn.bias_add(outputs, dom_outer_bias)
    if self.outer_activation is not None:
      outputs = self.outer_activation(outputs)  # pylint: disable=not-callable
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])
    return outputs

class Multi_domain_Gate(tf.keras.layers.Layer):

  def __init__(self,
               input_dim, 
               inner_dim,
               output_dim,
               domain_numb=6,
               dropout=0.1,
               activation=tf.nn.sigmoid,
               outer_activation=None,
               **kwargs):
    
    super(Multi_domain_Gate, self).__init__(**kwargs)
    self.dropout = dropout
    self.domain_numb = domain_numb
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.layer_norm = common.LayerNorm()
    self.inner_layer_norm = common.LayerNorm()
    self.outer_transpose = False
    self.outer_use_bias = True
    self.outer_activation = activation
  
  def build(self, input_shape):
    super(Multi_domain_Gate, self).build(input_shape)
    scope_name = self.name_scope()
    self.outer_kernel = self.add_weight("%s_outer_weight"%scope_name, shape=[self.domain_numb, self.input_dim*self.output_dim], initializer=tf.zeros_initializer)
    self.outer_bias = self.add_weight("%s_outer_bias"%scope_name, shape=[self.domain_numb, self.output_dim], initializer=tf.zeros_initializer)
    
  def call(self, inputs, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    dom_outer_kernel = tf.nn.embedding_lookup(self.outer_kernel, domain)
    dom_outer_bias = tf.nn.embedding_lookup(self.outer_bias, domain)
    dom_outer_kernel = tf.reshape(dom_outer_kernel, [-1, self.output_dim])
    outputs = tf.matmul(inputs, dom_outer_kernel, transpose_b=self.outer_transpose)
    
    if self.outer_use_bias:
      outputs = tf.nn.bias_add(outputs, dom_outer_bias)
    outputs = self.layer_norm(outputs)

    if self.outer_activation is not None:
      outputs = self.outer_activation(outputs)  # pylint: disable=not-callable
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])   
    
    #if not training:
    #  tf.print("###", self.name_scope(), "ADAP_gate_norm_L1: ", tf.reduce_mean(tf.reduce_sum(tf.abs(outputs),0),0), "domain: ", domain, "###", sep="|", output_stream=sys.stdout)
      
      #tf.print("###", self.name_scope(), "ADAP_gate: ", outputs[0:2,tf.math.floordiv(tf.shape(outputs)[1],2),:], summarize=2048)

    return outputs

  def forward_fn(self, inputs, args_dict, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    
    outer_kernel = args_dict[self.outer_kernel.name]
    outer_bias = args_dict[self.outer_bias.name]
    inputs = self.layer_norm(inputs)
    ##### inner layer
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    dom_outer_kernel = tf.nn.embedding_lookup(outer_kernel, domain)
    dom_outer_bias = tf.nn.embedding_lookup(outer_bias, domain)
    dom_outer_kernel = tf.reshape(dom_outer_kernel, [-1, self.output_dim])
    outputs = tf.matmul(inputs, dom_outer_kernel, transpose_b=self.outer_transpose)
    if self.outer_use_bias:
      outputs = tf.nn.bias_add(outputs, dom_outer_bias)
    outputs = self.layer_norm.forward_fn(outputs, args_dict)
    if self.outer_activation is not None:
      outputs = self.outer_activation.forward_fn(outputs, args_dict)  # pylint: disable=not-callable
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])
    return outputs

class Multi_domain_Gate_v1(tf.keras.layers.Layer):

  def __init__(self,
               input_dim, 
               inner_dim,
               output_dim,
               domain_numb=6,
               dropout=0.1,
               activation=tf.nn.sigmoid,
               outer_activation=None,
               **kwargs):
    
    super(Multi_domain_Gate_v1, self).__init__(**kwargs)
    self.dropout = dropout
    self.domain_numb = domain_numb
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.layer_norm = common.LayerNorm()
    self.input_norm = common.LayerNorm()
    self.outer_transpose = False
    self.outer_use_bias = True
    self.outer_activation = activation
  
  def build(self, input_shape):
    super(Multi_domain_Gate_v1, self).build(input_shape)
    scope_name = self.name_scope()
    self.outer_kernel = self.add_weight("%s_outer_weight"%scope_name, shape=[self.domain_numb, self.input_dim*self.output_dim])
    self.outer_bias = self.add_weight("%s_outer_bias"%scope_name, shape=[self.domain_numb, self.output_dim])
    
  def call(self, inputs, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    inputs = self.input_norm(inputs)
    dom_outer_kernel = tf.nn.embedding_lookup(self.outer_kernel, domain)
    dom_outer_bias = tf.nn.embedding_lookup(self.outer_bias, domain)
    dom_outer_kernel = tf.reshape(dom_outer_kernel, [-1, self.output_dim])
    outputs = tf.matmul(inputs, dom_outer_kernel, transpose_b=self.outer_transpose)
    
    if self.outer_use_bias:
      outputs = tf.nn.bias_add(outputs, dom_outer_bias)
    outputs = self.layer_norm(outputs)

    if self.outer_activation is not None:
      outputs = self.outer_activation(outputs)  # pylint: disable=not-callable
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])   
    
    #if not training:
      #tf.print("###", self.name_scope(), "Inputs_max_abs_pooling: ", tf.reduce_max(tf.abs(inputs)), "ADAP_gate_max_abs_pooling: ", 
      #          tf.reduce_max(tf.abs(outputs)), "ADAP_gate_min_abs_pooling: ", tf.reduce_min(tf.abs(outputs)), "ADAP_gate_avg_abs_pooling: ", tf.reduce_mean(tf.abs(outputs)), "domain: ", domain, "###", sep="|")
      
    #  tf.print("###", self.name_scope(), "ADAP_gate: ", outputs[0:2,tf.math.floordiv(tf.shape(outputs)[1],2),:], summarize=2048)

    return outputs

  def forward_fn(self, inputs, args_dict, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    
    outer_kernel = args_dict[self.outer_kernel.name]
    outer_bias = args_dict[self.outer_bias.name]
    inputs = self.input_norm(inputs)
    ##### inner layer
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    dom_outer_kernel = tf.nn.embedding_lookup(outer_kernel, domain)
    dom_outer_bias = tf.nn.embedding_lookup(outer_bias, domain)
    dom_outer_kernel = tf.reshape(dom_outer_kernel, [-1, self.output_dim])
    outputs = tf.matmul(inputs, dom_outer_kernel, transpose_b=self.outer_transpose)
    if self.outer_use_bias:
      outputs = tf.nn.bias_add(outputs, dom_outer_bias)
    outputs = self.layer_norm(outputs)
    if self.outer_activation is not None:
      outputs = self.outer_activation(outputs)  # pylint: disable=not-callable
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])
    return outputs

class Multi_domain_FeedForwardNetwork_v5(tf.keras.layers.Layer):

  def __init__(self,
               input_dim, 
               inner_dim,
               output_dim,
               domain_numb=6,
               dropout=0.1,
               activation=tf.nn.relu,
               outer_activation=None,
               **kwargs):
    
    super(Multi_domain_FeedForwardNetwork_v5, self).__init__(**kwargs)
    self.dropout = dropout
    self.domain_numb = domain_numb
    self.input_dim = input_dim
    self.inner_dim = tf.constant(inner_dim)
    self.inner_dim_max = 1024
    self.output_dim = output_dim
    self.layer_norm = common.LayerNorm()
    self.inner_layer_norm = common.Multi_LayerNorm(domain_numb, inner_dim)
    self.inner_transpose = False
    self.outer_transpose = False
    self.inner_use_bias = True
    self.outer_use_bias = True
    self.inner_activation = activation
    self.outer_activation = outer_activation
  
  def build(self, input_shape):
    super(Multi_domain_FeedForwardNetwork_v5, self).build(input_shape)
    scope_name = self.name_scope()
    self.inner_kernel = self.add_weight("%s_inner_weight"%scope_name, shape=[self.domain_numb * self.input_dim * self.inner_dim_max])
    self.inner_bias = self.add_weight("%s_inner_bias"%scope_name, shape=[self.domain_numb * self.inner_dim_max])
    self.outer_kernel = self.add_weight("%s_outer_weight"%scope_name, shape=[self.domain_numb * self.inner_dim_max * self.output_dim])
    self.outer_bias = self.add_weight("%s_outer_bias"%scope_name, shape=[self.domain_numb, self.output_dim])
  
  def call(self, inputs, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    if not(mask is None):
      mask=tf.cast(mask,tf.float32)
    mask=None
    inputs = self.layer_norm(inputs)
    ##### inner layer
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    
    inner_dim = self.inner_dim[domain]

    dom_inner_kernel = self.inner_kernel[domain * self.input_dim * self.inner_dim_max: domain * self.input_dim * self.inner_dim_max + self.input_dim * inner_dim] #tf.nn.embedding_lookup(self.inner_kernel, domain)
    dom_inner_bias = self.inner_bias[domain * self.inner_dim_max : domain * self.inner_dim_max + inner_dim] #tf.nn.embedding_lookup(self.inner_bias, domain)
    dom_inner_kernel = tf.reshape(dom_inner_kernel, [self.input_dim, -1])
    inner = tf.matmul(inputs, dom_inner_kernel, transpose_b=self.inner_transpose)
    if self.inner_use_bias:
      inner = tf.nn.bias_add(inner, dom_inner_bias)
    if self.inner_activation is not None:
      inner = self.inner_layer_norm(inner, domain)
      inner = self.inner_activation(inner)  # pylint: disable=not-callable
    if rank > 2:
      inner = tf.reshape(inner, shape[:-1] + [inner_dim])
    ##### output layer
    inner = common.dropout(inner, self.dropout, training=training)
    shape = shape_list(inner)
    rank = len(shape)      
    if rank > 2:
      inner = tf.reshape(inner, [-1, shape[-1]])
    dom_outer_kernel = self.outer_kernel[domain * self.output_dim * self.inner_dim_max : domain * self.output_dim * self.inner_dim_max + self.output_dim * inner_dim] #tf.nn.embedding_lookup(self.outer_kernel, domain)
    dom_outer_bias = tf.nn.embedding_lookup(self.outer_bias, domain)
    dom_outer_kernel = tf.reshape(dom_outer_kernel, [-1, self.output_dim])
    outputs = tf.matmul(inner, dom_outer_kernel, transpose_b=self.outer_transpose)
    if self.outer_use_bias:
      outputs = tf.nn.bias_add(outputs, dom_outer_bias)
    if self.outer_activation is not None:
      outputs = self.outer_activation(outputs)  # pylint: disable=not-callable
    if mask is not None:
      self.add_loss(tf.divide(tf.reduce_sum(mask * tf.reduce_sum(tf.abs(outputs),axis=-1)), tf.reduce_sum(mask)))
    else:
      self.add_loss(tf.reduce_mean(tf.reduce_sum(tf.abs(outputs),axis=-1)))
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])   
    
    if not training:
      tf.print("###", self.name_scope(), "Inputs_max_abs_pooling: ", tf.reduce_max(tf.abs(inputs)), "ADAP_max_abs_pooling: ", 
                tf.reduce_max(tf.abs(outputs)), "ADAP_min_abs_pooling: ", tf.reduce_min(tf.abs(outputs)), "domain: ", domain, "###", sep="|")    
    return outputs

  def forward_fn(self, inputs, args_dict, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    if not(mask is None):
      mask=tf.cast(mask,tf.float32)
    mask=None
    inner_kernel = args_dict[self.inner_kernel.name]
    outer_kernel = args_dict[self.outer_kernel.name]
    inner_bias = args_dict[self.inner_bias.name]
    outer_bias = args_dict[self.outer_bias.name]
    inputs = self.layer_norm(inputs)
    ##### inner layer
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    inner_dim = self.inner_dim[domain]
    dom_inner_kernel = inner_kernel[domain * self.input_dim * self.inner_dim_max: domain * self.input_dim * self.inner_dim_max + self.input_dim * inner_dim] #tf.nn.embedding_lookup(inner_kernel, domain)
    dom_inner_bias = inner_bias[domain * self.inner_dim_max : domain * self.inner_dim_max + inner_dim] #tf.nn.embedding_lookup(inner_bias, domain)
    dom_inner_kernel = tf.reshape(dom_inner_kernel, [self.input_dim, -1])
    inner = tf.matmul(inputs, dom_inner_kernel, transpose_b=self.inner_transpose)
    if self.inner_use_bias:
      inner = tf.nn.bias_add(inner, dom_inner_bias)
    if self.inner_activation is not None:
      inner = self.inner_layer_norm(inner, domain)
      inner = self.inner_activation(inner)  # pylint: disable=not-callable
    if rank > 2:
      inner = tf.reshape(inner, shape[:-1] + [self.inner_dim])
    ##### output layer
    inner = common.dropout(inner, self.dropout, training=training)
    shape = shape_list(inner)
    rank = len(shape)      
    if rank > 2:
      inner = tf.reshape(inner, [-1, shape[-1]])
    dom_outer_kernel = outer_kernel[domain * self.output_dim * self.inner_dim_max : domain * self.output_dim * self.inner_dim_max + self.output_dim * inner_dim] #tf.nn.embedding_lookup(outer_kernel, domain)
    dom_outer_bias = tf.nn.embedding_lookup(outer_bias, domain)
    dom_outer_kernel = tf.reshape(dom_outer_kernel, [-1, self.output_dim])
    outputs = tf.matmul(inner, dom_outer_kernel, transpose_b=self.outer_transpose)
    if self.outer_use_bias:
      outputs = tf.nn.bias_add(outputs, dom_outer_bias)
    if self.outer_activation is not None:
      outputs = self.outer_activation(outputs)  # pylint: disable=not-callable
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])
    return outputs

class Multi_domain_Gate_v2(tf.keras.layers.Layer):

  def __init__(self,
               input_dim, 
               inner_dim,
               output_dim,
               domain_numb=6,
               dropout=0.1,
               activation=tf.nn.sigmoid,
               outer_activation=None,
               output_regularization=False,
               **kwargs):
    
    super(Multi_domain_Gate_v2, self).__init__(**kwargs)
    self.dropout = dropout
    self.domain_numb = domain_numb
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.layer_norm = common.LayerNorm()
    self.inner_layer_norm = common.LayerNorm()
    self.outer_transpose = False
    self.outer_use_bias = True
    self.outer_activation = activation
    self.output_regularization = output_regularization
  
  def build(self, input_shape):
    super(Multi_domain_Gate_v2, self).build(input_shape)
    scope_name = self.name_scope()
    self.input_outer_kernel = self.add_weight("%s_input_outer_weight"%scope_name, shape=[self.input_dim, self.output_dim])
    self.update_outer_kernel = self.add_weight("%s_update_outer_weight"%scope_name, shape=[self.input_dim, self.output_dim])
    self.outer_bias = self.add_weight("%s_outer_bias"%scope_name, shape=[self.output_dim])
    
  def call(self, inputs, updates, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
      updates = tf.reshape(updates, [-1, shape[-1]])

    outputs = tf.matmul(inputs, self.input_outer_kernel, transpose_b=self.outer_transpose) + tf.matmul(updates, self.update_outer_kernel, transpose_b=self.outer_transpose)
    
    if self.outer_use_bias:
      outputs = tf.nn.bias_add(outputs, self.outer_bias)
    outputs = self.layer_norm(outputs)

    if self.outer_activation is not None:
      outputs = self.outer_activation(outputs)  # pylint: disable=not-callable
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])   
    if self.output_regularization:
      self.add_loss(tf.reduce_mean(tf.reduce_sum(tf.abs(outputs),axis=-1)))
    return outputs

  def forward_fn(self, inputs, args_dict, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    
    input_outer_kernel = args_dict[self.input_outer_kernel.name]
    update_outer_kernel = args_dict[self.update_outer_kernel.name]
    outer_bias = args_dict[self.outer_bias.name]
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
      updates = tf.reshape(updates, [-1, shape[-1]])

    outputs = tf.matmul(inputs, input_outer_kernel, transpose_b=self.outer_transpose) + tf.matmul(updates, update_outer_kernel, transpose_b=self.outer_transpose)
    
    if self.outer_use_bias:
      outputs = tf.nn.bias_add(outputs, outer_bias)
    outputs = self.layer_norm.forward_fn(outputs, args_dict)

    if self.outer_activation is not None:
      outputs = self.outer_activation(outputs, args_dict)  # pylint: disable=not-callable
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim]) 

    return outputs

class DAFE(tf.keras.layers.Layer):
  
  def __init__(self,
               input_dim, 
               domain_numb=6,
               dropout=0.1,
               **kwargs):
    
    super(DAFE, self).__init__(**kwargs)
    self.domain_numb = domain_numb
    self.input_dim = input_dim
    self.layer_norm = common.LayerNorm()

  def build(self, input_shape):
    super(DAFE, self).build(input_shape)
    scope_name = self.name_scope()
    self.inner_bias = self.add_weight("%s_inner_bias"%scope_name, shape=[self.domain_numb, self.inner_dim])
    
  def call(self, inputs, domain, mask=None,  training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    inputs = self.layer_norm(inputs)
    ##### inner layer
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    dom_inner_bias = tf.nn.embedding_lookup(self.inner_bias, domain)
    outputs = tf.nn.bias_add(inputs, dom_inner_bias)    
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])

    return outputs

  def forward_fn(self, inputs, args_dict, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    inputs = self.layer_norm(inputs)
    ##### inner layer
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    dom_inner_bias = tf.nn.embedding_lookup(self.inner_bias, domain)
    outputs = tf.nn.bias_add(inputs, dom_inner_bias)    
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])
    return outputs

class GRU(tf.keras.layers.Layer):

  def __init__(self,
               num_layers,
               num_units,
               bidirectional=False,
               reducer=reducer_lib.ConcatReducer(),
               dropout=0,
               residual_connections=False,
               **kwargs):
    
    super(GRU, self).__init__(**kwargs)
    rnn_layers = [
        _RNNWrapper(
            tf.keras.layers.GRU(num_units, return_sequences=True, return_state=True),
            bidirectional=bidirectional,
            reducer=reducer)
        for _ in range(num_layers)]
    self.layers = [
        common.LayerWrapper(
            layer,
            output_dropout=dropout,
            residual_connection=residual_connections)
        for layer in rnn_layers]

  def call(self, inputs, sequence_length=None, mask=None, training=None, initial_state=None):  # pylint: disable=arguments-differ
    all_states = []
    for i, layer in enumerate(self.layers):
      outputs, states = layer(
          inputs,
          mask=mask,
          training=training,
          initial_state=initial_state[i] if initial_state is not None else None)
      all_states.append(states)
      inputs = outputs
    return outputs, tuple(all_states)

class CondGRUCell(tf.keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        self.cell1 = tf.keras.layers.GRUCell(units)
        self.cell2 = tf.keras.layers.GRUCell(units)
        self.gate = tf.keras.layers.Dense(units, activation=tf.nn.sigmoid)
        
        super(CondGRUCell, self).__init__(**kwargs)       

    def call(self, inputs, state):

      state_prime, _ = self.cell1(inputs, state)
      context = attention_mechanism_1(state_prime)
      d_context = attention_mechanism_2(state_prime)
      gate_ = self.gate(tf.concat([state_prime, context, d_context],-1))
      context = gate_ * context + (1 - gate_) * d_context
      next_state, _ = self.cell2(context, state_prime)
        
      return

class Regulation_Gate(tf.keras.layers.Layer):

  def __init__(self,
               input_dim, 
               inner_dim,
               output_dim,
               domain_numb=6,
               dropout=0.1,
               activation=tf.nn.relu,
               outer_activation=tf.nn.sigmoid,
               **kwargs):
    
    super(Regulation_Gate, self).__init__(**kwargs)
    self.dropout = dropout
    self.domain_numb = domain_numb
    self.input_dim = input_dim
    self.inner_dim = inner_dim
    self.output_dim = output_dim
    self.layer_norm = common.LayerNorm()
    self.inner_layer_norm = common.LayerNorm()
    self.inner_transpose = False
    self.outer_transpose = False
    self.inner_use_bias = True
    self.outer_use_bias = True
    self.inner_activation = activation
    self.outer_activation = outer_activation
  
  def build(self, input_shape):
    super(Regulation_Gate, self).build(input_shape)
    scope_name = self.name_scope()
    self.inner_kernel = self.add_weight("%s_inner_weight"%scope_name, shape=[self.input_dim, self.inner_dim])
    self.inner_bias = self.add_weight("%s_inner_bias"%scope_name, shape=[self.inner_dim])
    self.outer_kernel = self.add_weight("%s_outer_weight"%scope_name, shape=[self.inner_dim, self.output_dim])
    self.outer_bias = self.add_weight("%s_outer_bias"%scope_name, shape=[self.output_dim])
    
  def call(self, inputs, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    if not(mask is None):
      mask=tf.cast(mask,tf.float32)
    mask=None
    inputs = self.layer_norm(inputs)
    ##### inner layer
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    dom_inner_kernel = self.inner_kernel
    dom_inner_bias = self.inner_bias    
    inner = tf.matmul(inputs, dom_inner_kernel, transpose_b=self.inner_transpose)
    if self.inner_use_bias:
      inner = tf.nn.bias_add(inner, dom_inner_bias)
    if self.inner_activation is not None:
      inner = self.inner_layer_norm(inner)
      inner = self.inner_activation(inner)  # pylint: disable=not-callable
    if rank > 2:
      inner = tf.reshape(inner, shape[:-1] + [self.inner_dim])
    ##### output layer
    inner = common.dropout(inner, self.dropout, training=training)
    shape = shape_list(inner)
    rank = len(shape)      
    if rank > 2:
      inner = tf.reshape(inner, [-1, shape[-1]])
    dom_outer_kernel = self.outer_kernel
    dom_outer_bias = self.outer_bias
    outputs = tf.matmul(inner, dom_outer_kernel, transpose_b=self.outer_transpose)
    if self.outer_use_bias:
      outputs = tf.nn.bias_add(outputs, dom_outer_bias)
    if self.outer_activation is not None:
      outputs = self.outer_activation(outputs)  # pylint: disable=not-callable    
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])   
    
    #if not training:
    #   tf.print("###", self.name_scope(), "gate_mean_abs_pooling: ", tf.reduce_mean(tf.abs(outputs),-1)[0,:], "domain: ", domain, "###", sep="|")    
    return outputs

  def forward_fn(self, inputs, args_dict, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    if not(mask is None):
      mask=tf.cast(mask,tf.float32)
    mask=None
    inner_kernel = args_dict[self.inner_kernel.name]
    outer_kernel = args_dict[self.outer_kernel.name]
    inner_bias = args_dict[self.inner_bias.name]
    outer_bias = args_dict[self.outer_bias.name]

    inputs = self.layer_norm(inputs)
    ##### inner layer
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    dom_inner_kernel = inner_kernel
    dom_inner_bias = inner_bias
    inner = tf.matmul(inputs, dom_inner_kernel, transpose_b=self.inner_transpose)
    if self.inner_use_bias:
      inner = tf.nn.bias_add(inner, dom_inner_bias)
    if self.inner_activation is not None:
      inner = self.inner_layer_norm(inner)
      inner = self.inner_activation(inner)  # pylint: disable=not-callable
    if rank > 2:
      inner = tf.reshape(inner, shape[:-1] + [self.inner_dim])
    ##### output layer
    inner = common.dropout(inner, self.dropout, training=training)
    shape = shape_list(inner)
    rank = len(shape)      
    if rank > 2:
      inner = tf.reshape(inner, [-1, shape[-1]])
    dom_outer_kernel = outer_kernel
    dom_outer_bias = outer_bias
    outputs = tf.matmul(inner, dom_outer_kernel, transpose_b=self.outer_transpose)
    if self.outer_use_bias:
      outputs = tf.nn.bias_add(outputs, dom_outer_bias)
    if self.outer_activation is not None:
      outputs = self.outer_activation(outputs)  # pylint: disable=not-callable
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])
    return outputs

class Multi_domain_FeedForwardNetwork_v6(tf.keras.layers.Layer):

  def __init__(self,
               input_dim, 
               inner_dim,
               output_dim,
               domain_numb=6,
               dropout=0.1,
               fake_domain_prob=0.3,
               noisy_prob= None, #[0.014967015314468146, 0.20679040170628904, 0.14494109872507957, 0.07797983876280723, 0.3068415724589217, 0.24848007303243427],
               activation=tf.nn.relu,
               outer_activation=None,
               **kwargs):
    
    super(Multi_domain_FeedForwardNetwork_v6, self).__init__(**kwargs)
    self.dropout = dropout
    self.domain_numb = domain_numb
    self.input_dim = input_dim
    self.inner_dim = inner_dim
    self.output_dim = output_dim
    self.layer_norm = common.LayerNorm()
    self.inner_layer_norm = common.LayerNorm()
    self.inner_transpose = False
    self.outer_transpose = False
    self.inner_use_bias = True
    self.outer_use_bias = True
    self.inner_activation = activation
    self.outer_activation = outer_activation
    self.fake_domain_prob = fake_domain_prob
    if noisy_prob == None:
      self.noisy_prob = [1.0/(domain_numb-1)]*(domain_numb-1)
    else:
      self.noisy_prob = noisy_prob
    print("noisy prob:", self.noisy_prob)

  def build(self, input_shape):
    super(Multi_domain_FeedForwardNetwork_v6, self).build(input_shape)
    scope_name = self.name_scope()
    print("self.domain_numb, self.input_dim, self.inner_dim: ", self.domain_numb, self.input_dim, self.inner_dim)
    self.inner_kernel = self.add_weight("%s_inner_weight"%scope_name, shape=[self.domain_numb, self.input_dim*self.inner_dim])
    self.inner_bias = self.add_weight("%s_inner_bias"%scope_name, shape=[self.domain_numb, self.inner_dim])
    self.outer_kernel = self.add_weight("%s_outer_weight"%scope_name, shape=[self.domain_numb, self.inner_dim*self.output_dim])
    self.outer_bias = self.add_weight("%s_outer_bias"%scope_name, shape=[self.domain_numb, self.output_dim])
    
  def call(self, inputs, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    if not(mask is None):
      mask=tf.cast(mask,tf.float32)
    mask=None
    inputs = self.layer_norm(inputs)
    
    if training:
      print("noisy prob:", self.noisy_prob)
      print("fake_domain_prob:", self.fake_domain_prob)
      fake_domain_prob = self.fake_domain_prob
    else:
      fake_domain_prob = 0.0
    domain_ =  tf.random.categorical(tf.math.log([self.noisy_prob]), 1, dtype=tf.int32)[0,0] 
    #tf.math.mod(tf.cast(domain,tf.int32) + tf.constant(1, dtype=tf.int32) + tf.random.categorical(tf.math.log([self.noisy_prob]), 1, dtype=tf.int32)[0,0], self.domain_numb)
    #tf.print("noisy domain: ", domain_, "domain: ", domain, sep="|")
    ##### inner layer
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    dom_inner_kernel = tf.nn.embedding_lookup(self.inner_kernel, domain)
    dom_inner_bias = tf.nn.embedding_lookup(self.inner_bias, domain)
    dom_inner_kernel = tf.reshape(dom_inner_kernel, [-1, self.inner_dim])
    inner = tf.matmul(inputs, dom_inner_kernel, transpose_b=self.inner_transpose)
    if self.inner_use_bias:
      inner = tf.nn.bias_add(inner, dom_inner_bias)
    if self.inner_activation is not None:
      inner = self.inner_layer_norm(inner)
      inner = self.inner_activation(inner)  # pylint: disable=not-callable
    if rank > 2:
      inner = tf.reshape(inner, shape[:-1] + [self.inner_dim])
    ##### output layer
    inner = common.dropout(inner, self.dropout, training=training)
    shape = shape_list(inner)
    rank = len(shape)      
    if rank > 2:
      inner = tf.reshape(inner, [-1, shape[-1]])
    dom_outer_kernel = tf.nn.embedding_lookup(self.outer_kernel, domain)
    dom_outer_bias = tf.nn.embedding_lookup(self.outer_bias, domain)
    dom_outer_kernel = tf.reshape(dom_outer_kernel, [-1, self.output_dim])
    outputs = tf.matmul(inner, dom_outer_kernel, transpose_b=self.outer_transpose)
    if self.outer_use_bias:
      outputs = tf.nn.bias_add(outputs, dom_outer_bias)
    if self.outer_activation is not None:
      outputs = self.outer_activation(outputs)  # pylint: disable=not-callable
    if mask is not None:
      self.add_loss(tf.divide(tf.reduce_sum(mask * tf.reduce_sum(tf.abs(outputs),axis=-1)), tf.reduce_sum(mask)))
    else:
      self.add_loss(tf.reduce_mean(tf.reduce_sum(tf.abs(outputs),axis=-1)))

    # Noisy inputs
          ##### inner layer    
    noisy_dom_inner_kernel = tf.nn.embedding_lookup(self.inner_kernel, domain_)
    noisy_dom_inner_bias = tf.nn.embedding_lookup(self.inner_bias, domain_)
    noisy_dom_inner_kernel = tf.reshape(noisy_dom_inner_kernel, [-1, self.inner_dim])
    noisy_inner = tf.matmul(inputs, noisy_dom_inner_kernel, transpose_b=self.inner_transpose)
    if self.inner_use_bias:
      noisy_inner = tf.nn.bias_add(noisy_inner, noisy_dom_inner_bias)

    if self.inner_activation is not None:
      noisy_inner = self.inner_layer_norm(noisy_inner)
      noisy_inner = self.inner_activation(noisy_inner)  # pylint: disable=not-callable
    if rank > 2:
      noisy_inner = tf.reshape(noisy_inner, shape[:-1] + [self.inner_dim])
         ##### output layer
    noisy_inner = common.dropout(noisy_inner, self.dropout, training=training)
    shape = shape_list(noisy_inner)
    rank = len(shape)      
    if rank > 2:
      noisy_inner = tf.reshape(noisy_inner, [-1, shape[-1]])
    noisy_dom_outer_kernel = tf.nn.embedding_lookup(self.outer_kernel, domain_)
    noisy_dom_outer_bias = tf.nn.embedding_lookup(self.outer_bias, domain_)
    noisy_dom_outer_kernel = tf.reshape(noisy_dom_outer_kernel, [-1, self.output_dim])
    noisy_outputs = tf.matmul(noisy_inner, noisy_dom_outer_kernel, transpose_b=self.outer_transpose)
    if self.outer_use_bias:
      noisy_outputs = tf.nn.bias_add(noisy_outputs, noisy_dom_outer_bias)
    if self.outer_activation is not None:
      noisy_outputs = self.outer_activation(noisy_outputs)

    ####
    keeping = tf.keras.backend.random_binomial(tf.expand_dims(tf.shape(inputs)[0],0),1-fake_domain_prob)
    keeping = tf.tile(tf.reshape(keeping,[-1,1]),[1,512])
    if training:
      outputs = outputs * keeping + (1-keeping) * noisy_outputs
    
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])   
    
    return outputs

class Multi_domain_FeedForwardNetwork_v8(tf.keras.layers.Layer):

  def __init__(self,
               input_dim, 
               inner_dim,
               output_dim,
               domain_numb=6,
               dropout=0.1,
               fake_domain_prob=0.3,
               noisy_prob= None,
               activation=tf.nn.relu,
               outer_activation=None,
               **kwargs):
    
    super(Multi_domain_FeedForwardNetwork_v8, self).__init__(**kwargs)
    self.dropout = dropout
    self.domain_numb = domain_numb
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.layer_norm = common.LayerNorm()
    self.inner_layer_norm = common.LayerNorm_v1(domain_numb=domain_numb, num_domain_units=inner_dim)
    self.inner_transpose = False
    self.outer_transpose = False
    self.inner_use_bias = True
    self.outer_use_bias = True
    self.inner_activation = activation
    self.outer_activation = outer_activation
    self.inner_dim = tf.constant(inner_dim)
    self.fake_domain_prob = fake_domain_prob
    if noisy_prob == None:
      self.noisy_prob = [1.0/(domain_numb-1)]*(domain_numb-1)
    else:
      self.noisy_prob = noisy_prob
    print("noisy prob:", self.noisy_prob)

  def build(self, input_shape):
    super(Multi_domain_FeedForwardNetwork_v8, self).build(input_shape)
    scope_name = self.name_scope()
    print("self.domain_numb, self.input_dim, self.inner_dim: ", self.domain_numb, self.input_dim, self.inner_dim)
    self.inner_kernel = self.add_weight("%s_inner_weight"%scope_name, shape=[self.input_dim * sum(self.inner_dim)])
    self.inner_bias = self.add_weight("%s_inner_bias"%scope_name, shape=[sum(self.inner_dim)])
    self.outer_kernel = self.add_weight("%s_outer_weight"%scope_name, shape=[sum(self.inner_dim) * self.output_dim])
    self.outer_bias = self.add_weight("%s_outer_bias"%scope_name, shape=[self.domain_numb, self.output_dim])
    
  def call(self, inputs, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""

    if not(mask is None):
      mask=tf.cast(mask,tf.float32)
    mask=None

    if training:
      print("noisy prob:", self.noisy_prob)
      print("fake_domain_prob:", self.fake_domain_prob)
      fake_domain_prob = self.fake_domain_prob
    else:
      fake_domain_prob = 0.0
    domain_ =  tf.random.categorical(tf.math.log([self.noisy_prob]), 1, dtype=tf.int32)[0,0]

    ### true outputs
    inputs = self.layer_norm(inputs)
        ##### inner layer
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    dom_inner_kernel = self.inner_kernel[tf.reduce_sum(self.inner_dim[:domain]) * self.input_dim : tf.reduce_sum(self.inner_dim[:domain+1]) * self.input_dim] #tf.nn.embedding_lookup(self.inner_kernel, domain)
    dom_inner_bias = self.inner_bias[tf.reduce_sum(self.inner_dim[:domain]) : tf.reduce_sum(self.inner_dim[:domain+1])] #tf.nn.embedding_lookup(self.inner_bias, domain)
    dom_inner_kernel = tf.reshape(dom_inner_kernel, [-1, self.inner_dim[domain]])
    inner = tf.matmul(inputs, dom_inner_kernel, transpose_b=self.inner_transpose)
    if self.inner_use_bias:
      inner = tf.nn.bias_add(inner, dom_inner_bias)
    if self.inner_activation is not None:
      inner = self.inner_layer_norm(inner, domain)
      inner = self.inner_activation(inner)  # pylint: disable=not-callable
    if rank > 2:
      inner = tf.reshape(inner, shape[:-1] + [self.inner_dim[domain]])
        ##### output layer
    inner = common.dropout(inner, self.dropout, training=training)
    shape = shape_list(inner)
    rank = len(shape)      
    if rank > 2:
      inner = tf.reshape(inner, [-1, shape[-1]])
    dom_outer_kernel = self.outer_kernel[tf.reduce_sum(self.inner_dim[:domain]) * self.output_dim : tf.reduce_sum(self.inner_dim[:domain+1]) * self.output_dim] #tf.nn.embedding_lookup(self.outer_kernel, domain)
    dom_outer_bias = tf.nn.embedding_lookup(self.outer_bias, domain)
    dom_outer_kernel = tf.reshape(dom_outer_kernel, [-1, self.output_dim])
    outputs = tf.matmul(inner, dom_outer_kernel, transpose_b=self.outer_transpose)
    if self.outer_use_bias:
      outputs = tf.nn.bias_add(outputs, dom_outer_bias)
    if self.outer_activation is not None:
      outputs = self.outer_activation(outputs)  # pylint: disable=not-callable
    if mask is not None:
      self.add_loss(tf.divide(tf.reduce_sum(mask * tf.reduce_sum(tf.abs(outputs),axis=-1)), tf.reduce_sum(mask)))
    else:
      self.add_loss(tf.reduce_mean(tf.reduce_sum(tf.abs(outputs),axis=-1)))
    

    ### Noisy outputs
        ##### inner layer
    noisy_dom_inner_kernel = self.inner_kernel[tf.reduce_sum(self.inner_dim[:domain_]) * self.input_dim : tf.reduce_sum(self.inner_dim[:domain_+1]) * self.input_dim] #tf.nn.embedding_lookup(self.inner_kernel, domain)
    noisy_dom_inner_bias = self.inner_bias[tf.reduce_sum(self.inner_dim[:domain_]) : tf.reduce_sum(self.inner_dim[:domain_+1])] #tf.nn.embedding_lookup(self.inner_bias, domain)
    noisy_dom_inner_kernel = tf.reshape(noisy_dom_inner_kernel, [-1, self.inner_dim[domain_]])
    noisy_inner = tf.matmul(inputs, noisy_dom_inner_kernel, transpose_b=self.inner_transpose)
    if self.inner_use_bias:
      noisy_inner = tf.nn.bias_add(noisy_inner, noisy_dom_inner_bias)
    if self.inner_activation is not None:
      noisy_inner = self.inner_layer_norm(noisy_inner, domain_)
      noisy_inner = self.inner_activation(noisy_inner)  # pylint: disable=not-callable
    if rank > 2:
      noisy_inner = tf.reshape(noisy_inner, shape[:-1] + [self.inner_dim[domain_]])
        ##### output layer
    noisy_inner = common.dropout(noisy_inner, self.dropout, training=training)
    shape = shape_list(noisy_inner)
    rank = len(shape)      
    if rank > 2:
      noisy_inner = tf.reshape(noisy_inner, [-1, shape[-1]])
    noisy_dom_outer_kernel = self.outer_kernel[tf.reduce_sum(self.inner_dim[:domain_]) * self.output_dim : tf.reduce_sum(self.inner_dim[:domain_+1]) * self.output_dim] #tf.nn.embedding_lookup(self.outer_kernel, domain)
    noisy_dom_outer_bias = tf.nn.embedding_lookup(self.outer_bias, domain_)
    noisy_dom_outer_kernel = tf.reshape(noisy_dom_outer_kernel, [-1, self.output_dim])
    noisy_outputs = tf.matmul(noisy_inner, noisy_dom_outer_kernel, transpose_b=self.outer_transpose)
    if self.outer_use_bias:
      noisy_outputs = tf.nn.bias_add(noisy_outputs, noisy_dom_outer_bias)
    if self.outer_activation is not None:
      noisy_outputs = self.outer_activation(noisy_outputs)

    ####
    keeping = tf.keras.backend.random_binomial(tf.expand_dims(tf.shape(inputs)[0],0),1-fake_domain_prob)
    keeping = tf.tile(tf.reshape(keeping,[-1,1]),[1,512])
    if training:
      outputs = outputs * keeping + (1-keeping) * noisy_outputs
    
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim]) 

    return outputs

class Multi_domain_FeedForwardNetwork_v9(tf.keras.layers.Layer):

  def __init__(self,
               input_dim, 
               inner_dim,
               output_dim,
               domain_numb=6,
               dropout=0.1,
               activation=tf.nn.relu,
               inner_layer_norm=None,
               outer_activation=None,
               **kwargs):
    
    super(Multi_domain_FeedForwardNetwork_v9, self).__init__(**kwargs)
    self.dropout = dropout
    self.domain_numb = domain_numb
    self.input_dim = input_dim
    self.inner_dim = inner_dim
    self.output_dim = output_dim
    self.layer_norm = common.LayerNorm()
    if inner_layer_norm:
      self.inner_layer_norm = inner_layer_norm(domain_numb)
    else:
      self.inner_layer_norm = common.LayerNorm()
    self.inner_transpose = False
    self.outer_transpose = False
    self.inner_use_bias = True
    self.outer_use_bias = True
    self.inner_activation = activation
    self.outer_activation = outer_activation
  
  def build(self, input_shape):
    super(Multi_domain_FeedForwardNetwork_v9, self).build(input_shape)
    scope_name = self.name_scope()
    #print("self.domain_numb, self.input_dim, self.inner_dim: ", self.domain_numb, self.input_dim, self.inner_dim)
    self.inner_kernel = self.add_weight("%s_inner_weight"%scope_name, shape=[self.input_dim, self.domain_numb * self.inner_dim])
    self.inner_bias = self.add_weight("%s_inner_bias"%scope_name, shape=[self.domain_numb * self.inner_dim])
    self.outer_kernel = self.add_weight("%s_outer_weight"%scope_name, shape=[self.domain_numb, self.inner_dim , self.output_dim])
    self.outer_bias = self.add_weight("%s_outer_bias"%scope_name, shape=[self.domain_numb, self.output_dim])
    
  def call(self, inputs, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the layer."""
    if not(mask is None):
      mask=tf.cast(mask,tf.float32)
    mask=None
    inputs = self.layer_norm(inputs)
    ##### inner layer
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    inner = tf.matmul(inputs, self.inner_kernel, transpose_b=self.inner_transpose)
    if self.inner_use_bias:
      inner = tf.nn.bias_add(inner, self.inner_bias)
    if self.inner_activation is not None:
      inner = tf.reshape(inner, [-1, self.inner_dim])
      inner = self.inner_layer_norm(inner)
      inner = tf.reshape(inner, [-1, self.inner_dim * self.domain_numb])
      inner = self.inner_activation(inner)  # pylint: disable=not-callable
    if rank > 2:
      inner = tf.reshape(inner, shape[:-1] + [self.inner_dim * self.domain_numb])
    ##### output layer
    inner = common.dropout(inner, self.dropout, training=training)
    shape = shape_list(inner)
    rank = len(shape)      
    if rank > 2:
      inner = tf.reshape(inner, [-1, shape[-1]])
    inner = tf.transpose(inner)
    inner = tf.reshape(inner,[self.domain_numb, self.inner_dim, -1])
    @tf.function
    def my_map(*args, **kwargs):
      return tf.map_fn(*args, **kwargs)
    #outputs = my_map(lambda x: tf.transpose(tf.nn.bias_add(tf.matmul(tf.transpose(x[0]), x[1] , transpose_b=self.outer_transpose), x[2])), (inner, self.outer_kernel, self.outer_bias), dtype=tf.float32, parallel_iterations=self.domain_numb)
    #######
    #tf.print("before_domain_mixing: ", tf.shape(outputs))
    domain = tf.transpose(domain)
    outputs = my_map(lambda x: tf.transpose(tf.nn.bias_add(tf.matmul(tf.transpose(x[0]), x[1] , transpose_b=self.outer_transpose), x[2])) * tf.tile(tf.reshape(tf.tile(tf.expand_dims(x[3],0),[tf.shape(x[0])[1]//tf.shape(domain)[1],1]),[1,-1]),[self.output_dim,1]), (inner, self.outer_kernel, self.outer_bias, domain), dtype=tf.float32, parallel_iterations=self.domain_numb)
    #outputs = my_map(lambda x: x[0] * tf.tile(tf.reshape(tf.tile(tf.expand_dims(x[1],0),[tf.shape(x[0])[1]//tf.shape(domain)[1],1]),[1,-1]),[self.output_dim,1]), (outputs, domain), dtype=tf.float32, parallel_iterations=self.domain_numb)
    #tf.print("after_domain_mixing: ", tf.shape(outputs))
    outputs = tf.transpose(tf.reduce_sum(outputs,0))
    #######
    """
    outputs = tf.reshape(outputs, [self.domain_numb * self.output_dim, -1])
    outputs = tf.transpose(outputs)
    outputs = tf.reshape(outputs,[tf.shape(domain)[0], -1, self.domain_numb * self.output_dim])
    outputs = my_map(lambda x: tf.reduce_sum(tf.reshape(x[0] * tf.tile(tf.reshape(tf.transpose(tf.tile(tf.expand_dims(x[1],0),[self.output_dim,1])),[1,-1]),[tf.shape(x[0])[0],1]), [-1, self.domain_numb, self.output_dim]),1), (outputs, domain), dtype=tf.float32, parallel_iterations=0)
    """
    if mask is not None:
      self.add_loss(tf.divide(tf.reduce_sum(mask * tf.reduce_sum(tf.abs(outputs),axis=-1)), tf.reduce_sum(mask)))
    else:
      self.add_loss(tf.reduce_mean(tf.reduce_sum(tf.abs(outputs),axis=-1)))

    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])   
    # if not training:
    #   tf.print("###", self.name_scope(), "Inputs_max_abs_pooling: ", tf.reduce_max(tf.abs(inputs)), "ADAP_max_abs_pooling: ", 
    #             tf.reduce_max(tf.abs(outputs)), "ADAP_min_abs_pooling: ", tf.reduce_min(tf.abs(outputs)), "domain: ", domain, "###", sep="|")    
    return outputs

class Multi_domain_classification_gate_v2(tf.keras.layers.Layer):

  def __init__(self,
               input_dim, 
               num_units,
               domain_numb=6,
               dropout=0.1,
               activation=tf.nn.sigmoid,
               outer_activation=None,
               **kwargs):
    
    super(Multi_domain_classification_gate_v2, self).__init__(**kwargs)
    self.dropout = dropout
    self.domain_numb = domain_numb
    self.layer_norm = common.LayerNorm()
    self.inner_layer_norm = common.LayerNorm()
    self.output_dim = num_units
    self.outer_transpose = False
    self.outer_use_bias = True
    self.outer_activation = activation
    self.ff_layer_1 = common.Dense(2048, use_bias=True, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.001), bias_regularizer=tf.keras.regularizers.l2(0.001))
    self.ff_layer_2 = common.Dense(2048, use_bias=True, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.001), bias_regularizer=tf.keras.regularizers.l2(0.001))
    self.ff_layer_end = common.Dense(domain_numb, use_bias=True, kernel_initializer='zeros', bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(0.001), bias_regularizer=tf.keras.regularizers.l2(0.001))
  
  def build(self, input_shape):
    super(Multi_domain_classification_gate_v2, self).build(input_shape)
  
  def call(self, inputs, domain, mask=None, training=None, tag=""):  # pylint: disable=arguments-differ
    """Runs the layer."""
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    inputs = self.layer_norm(inputs)
    inputs = common.dropout(inputs, rate=0.3, training=training)
    logits = self.ff_layer_1(inputs)
    #tf.print("logits 1", logits)
    logits = common.dropout(logits, rate=0.3, training=training)
    logits = self.ff_layer_2(logits)
    #tf.print("logits 2", logits)
    logits = common.dropout(logits, rate=0.3, training=training)
    logits = self.ff_layer_end(logits)
    #tf.print("logits 3: ", logits, summarize=1000)
    #tf.print("%s outputs: "%(self.name_scope()), tf.math.softmax(logits),summarize=1000)
    outputs = tf.math.softmax(logits)
    #tf.print("%s##probs: "%tag,tf.shape(outputs), outputs[:,domain], summarize=-1)
    #tf.print("prediction loss", tf.nn.softmax_cross_entropy_with_logits(smoothed_labels, logits))
    if training:
      label_smoothing = 0.1
      labels = tf.fill([tf.shape(logits)[0]], domain)
      smoothed_labels = _smooth_one_hot_labels(logits, labels, label_smoothing)
      self.add_loss(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(smoothed_labels, logits)))
    """
    if rank > 2:
      outputs = tf.tile(tf.expand_dims(tf.transpose(outputs),-1),[1,1,self.output_dim])
      outputs = tf.reshape(outputs, [shape[0], self.domain_numb, -1])   
    else:
      outputs = tf.tile(tf.expand_dims(tf.transpose(outputs),-1),[1,1,self.output_dim])
      outputs = tf.reshape(outputs, [shape[0], self.domain_numb,-1])
    """
    outputs = tf.reshape(tf.transpose(tf.tile(tf.expand_dims(tf.reshape(tf.transpose(outputs),[-1]),0),[self.output_dim,1])),[self.domain_numb,-1,self.output_dim])
    #tf.print("domain_logits:",tf.shape(outputs), outputs[0,:,:], summarize=-1)

    return outputs
  
class Multi_domain_classification_gate(tf.keras.layers.Layer):

  def __init__(self,
               input_dim, 
               num_units,
               domain_numb=6,
               dropout=0.1,
               activation=tf.nn.sigmoid,
               outer_activation=None,
               **kwargs):
    
    super(Multi_domain_classification_gate, self).__init__(**kwargs)
    self.dropout = dropout
    self.domain_numb = domain_numb
    self.layer_norm = common.LayerNorm()
    self.inner_layer_norm = common.LayerNorm()
    self.output_dim = num_units
    self.outer_transpose = False
    self.outer_use_bias = True
    self.outer_activation = activation
    self.ff_layer_1 = common.Dense(2048, use_bias=True, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.001), bias_regularizer=tf.keras.regularizers.l2(0.001))
    self.ff_layer_2 = common.Dense(2048, use_bias=True, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.001), bias_regularizer=tf.keras.regularizers.l2(0.001))
    self.ff_layer_end = common.Dense(domain_numb, use_bias=True, kernel_initializer='zeros', bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.l2(0.001), bias_regularizer=tf.keras.regularizers.l2(0.001))
  
  def build(self, input_shape):
    super(Multi_domain_classification_gate, self).build(input_shape)
    
  def call(self, inputs, domain, mask=None, training=None):  # pylint: disable=arguments-differ
    shape = shape_list(inputs)
    rank = len(shape)      
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    inputs = self.layer_norm(inputs)
    inputs = common.dropout(inputs, rate=0.3, training=training)
    logits = self.ff_layer_1(inputs)
    #tf.print("logits 1", logits)
    logits = common.dropout(logits, rate=0.3, training=training)
    logits = self.ff_layer_2(logits)
    #tf.print("logits 2", logits)
    logits = common.dropout(logits, rate=0.3, training=training)
    logits = self.ff_layer_end(logits)
    #tf.print("logits 3: ", logits, summarize=1000)
    #tf.print("%s outputs: "%(self.name_scope()), tf.math.softmax(logits),summarize=1000)
    outputs = tf.math.softmax(logits)[:,domain]
    #tf.print("prediction loss", tf.nn.softmax_cross_entropy_with_logits(smoothed_labels, logits))
    if training:
      label_smoothing = 0.1
      labels = tf.fill([tf.shape(logits)[0]], domain)
      smoothed_labels = _smooth_one_hot_labels(logits, labels, label_smoothing)
      self.add_loss(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(smoothed_labels, logits)))

    outputs = tf.tile(tf.expand_dims(outputs,1),[1,self.output_dim])
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.output_dim])   

    return outputs
  

class CondGRU(tf.keras.layers.Layer):
  def __init__(self,
                num_layers,
                num_units,
                bidirectional=False,
                reducer=reducer_lib.ConcatReducer(),
                dropout=0,
                residual_connections=False,
                **kwargs):
      
      super(CondGRU, self).__init__(**kwargs)
      rnn_layers = [
          _RNNWrapper(
              tf.keras.layers.RNN(CondGRUCell(num_units, return_sequences=True, return_state=True)))
          for _ in range(num_layers)]
      self.layers = [
          common.LayerWrapper(
              layer,
              output_dropout=dropout,
              residual_connection=residual_connections)
          for layer in rnn_layers]

  def call(self, inputs, mask=None, training=None, initial_state=None):
    all_states = []
    for i, layer in enumerate(self.layers):
      outputs, states = layer(
          inputs,
          mask=mask,
          training=training,
          initial_state=initial_state[i] if initial_state is not None else None)
      all_states.append(states)
      inputs = outputs
    return outputs, tuple(all_states)