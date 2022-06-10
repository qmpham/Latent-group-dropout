"""Define the self-attention encoder."""
import sys
sys.path.append("/gpfsdswork/projects/rech/sfz/utt84zy/anaconda3/envs/huggingface/lib/python3.7/site-packages")

import tensorflow as tf

from layers import transformer
from opennmt.utils.misc import shape_list
from opennmt.encoders.encoder import Encoder
from opennmt.encoders.self_attention_encoder import SelfAttentionEncoder
from opennmt.layers.position import SinusoidalPositionEncoder
from opennmt.layers import common
from layers.common import LayerNorm, LayerNorm_v1, Multi_LayerNorm, LayerNorm_v2
from utils.utils_ import make_domain_mask
from layers.layers import Regulation_Gate, Multi_domain_classification_gate, Multi_domain_classification_gate_v2, Multi_domain_FeedForwardNetwork_v9, Multi_domain_FeedForwardNetwork_v6, Multi_domain_FeedForwardNetwork_v8, Multi_domain_FeedForwardNetwork_v7, Multi_domain_FeedForwardNetwork, Multi_domain_FeedForwardNetwork_v2, Multi_domain_FeedForwardNetwork_v3, DAFE, Multi_domain_Gate, Multi_domain_Gate_v2
from opennmt.utils.misc import shape_list
class Multi_domain_SelfAttentionEncoder(Encoder):

  def __init__(self,
               num_layers,
               num_domains=6,
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
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder, self).__init__(**kwargs)
    self.num_units = num_units
    self.num_domains = num_domains
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = LayerNorm()
    self.layers = [
        transformer.SelfAttentionEncoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]
    self.multi_domain_layers = [
        Multi_domain_FeedForwardNetwork(num_domains*num_domain_units, num_units, name="ADAP_%d"%i)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient

  def call(self, inputs, sequence_length=None, training=None):
    domain = inputs[1]
    domain_mask = tf.nn.embedding_lookup(self.mask, domain)
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    #for layer in self.layers:
    for layer, multi_domain_layer in zip(self.layers,self.multi_domain_layers):
      inputs = layer(inputs, mask=mask, training=training)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer(tf.stop_gradient(inputs), domain_mask, training=training) + inputs
      else:
        inputs = multi_domain_layer(inputs, domain_mask, training=training) + inputs
    outputs = self.layer_norm(inputs)
    return outputs, None, sequence_length

  def forward_fn(self, inputs, args_dict, sequence_length=None, training=None):
    domain = inputs[1]
    domain_mask = tf.nn.embedding_lookup(self.mask, domain)
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    #for layer in self.layers:
    for layer, multi_domain_layer in zip(self.layers,self.multi_domain_layers):
      inputs = layer.forward_fn(inputs, args_dict, mask=mask, training=training)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), args_dict, domain_mask, training=training) + inputs
      else:
        inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain_mask, training=training) + inputs
    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_v2(Encoder):

  def __init__(self,
               num_layers,
               num_domains=6,
               num_domain_units=128,
               ADAP_layer_stopping_gradient=False,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               res_using_rate=1.0,
               position_encoder_class=SinusoidalPositionEncoder,
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v3,
               fake_domain_prob=0.1,
               noisy_prob=None,
               ADAP_contribution=None,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v2, self).__init__(**kwargs)
    self.num_units = num_units
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = LayerNorm()
    self.layers = [
        transformer.SelfAttentionEncoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
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
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    if ADAP_contribution == None:
      ADAP_contribution = [1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution
    self.res_using_rate = res_using_rate
  
  def call(self, inputs, sequence_length=None, training=None, internal_node_printing=False, adapter_activate=True):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]    
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    if training:
      keeping = tf.keras.backend.random_binomial([1], self.res_using_rate)
    else:
      keeping = 1.0
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers, self.multi_domain_layers)):
      inputs = layer(inputs, mask=mask, training=training)
      
      if self.ADAP_layer_stopping_gradient:
        adapt = multi_domain_layer(tf.stop_gradient(inputs), domain, mask=mask, training=training)
        inputs = adapt * self.ADAP_contribution[i] * keeping + inputs
      else:
        adapt = multi_domain_layer(inputs, domain, mask=mask, training=training)
        inputs = adapt * self.ADAP_contribution[i] * keeping + inputs
      """
      if internal_node_printing:
        tf.print("layers: ", i , "ADAP mean pooling: ", tf.reduce_mean(tf.abs(adapt),-1)[0,:], "domain: ", domain, "###", sep="|", summarize=1000)
      """
    outputs = self.layer_norm(inputs)
    
    return outputs, None, sequence_length

  def forward_fn(self, inputs, args_dict, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    #for layer in self.layers:
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):
      inputs = layer.forward_fn(inputs, args_dict, mask=mask, training=training)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training) * self.ADAP_contribution[i] + inputs
      else:
        inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain, mask=mask, training=training) * self.ADAP_contribution[i] + inputs
    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_v3(Encoder):

  def __init__(self,
               num_layers,
               num_domains=6,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder_class=SinusoidalPositionEncoder,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v3, self).__init__(**kwargs)
    self.num_units = num_units
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = LayerNorm()
    self.layers = [
        transformer.SelfAttentionEncoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]    
    self.multi_domain_layers = [
        DAFE(num_units, domain_numb=num_domains, name="DAFE_%d"%i)
        for i in range(num_layers)]

  def call(self, inputs, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    #for layer in self.layers:
    for layer, multi_domain_layer in zip(self.layers,self.multi_domain_layers):
      inputs = layer(inputs, mask=mask, training=training)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer(tf.stop_gradient(inputs), domain)
      else:
        inputs = multi_domain_layer(inputs, domain)
    outputs = self.layer_norm(inputs)
    return outputs, None, sequence_length

  def forward_fn(self, inputs, args_dict, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    #for layer in self.layers:
    for layer, multi_domain_layer in zip(self.layers,self.multi_domain_layers):
      inputs = layer.forward_fn(inputs, args_dict, mask=mask, training=training)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), args_dict, domain)
      else:
        inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain)
    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_v0(Encoder):

  def __init__(self,
               num_layers,
               num_domains=6,
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
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v0, self).__init__(**kwargs)
    self.num_units = num_units
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = LayerNorm()
    self.layers = [
        transformer.SelfAttentionEncoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]    
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        for i in range(num_layers)]

  def call(self, inputs, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    for layer, multi_domain_layer in zip(self.layers, self.multi_domain_layers):
      inputs = layer(inputs, mask=mask, training=training)
      inputs = multi_domain_layer(inputs, domain, mask=mask, training=training)
    outputs = self.layer_norm(inputs)
    
    return outputs, None, sequence_length

  def forward_fn(self, inputs, args_dict, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    #for layer in self.layers:
    for layer, multi_domain_layer in zip(self.layers,self.multi_domain_layers):
      inputs = layer.forward_fn(inputs, args_dict, mask=mask, training=training)
      inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain, mask=mask, training=training)
    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_v1(Encoder):
  
  def __init__(self,
               num_layers,
               num_domains=6,
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
               fake_domain_prob=0.1,
               noisy_prob=None,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v1, self).__init__(**kwargs)
    self.num_units = num_units
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = LayerNorm()
    self.layers = [
        transformer.SelfAttentionEncoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)] 
    print("multi_domain_adapter_class == Multi_domain_FeedForwardNetwork_v6", multi_domain_adapter_class == Multi_domain_FeedForwardNetwork_v6)
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        if not multi_domain_adapter_class in [Multi_domain_FeedForwardNetwork_v6, Multi_domain_FeedForwardNetwork_v8]
        else multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i, 
        fake_domain_prob=fake_domain_prob, noisy_prob=noisy_prob)
        for i in range(num_layers)]
    self.multi_domain_gates = [
        multi_domain_adapter_gate_class(num_units, num_units, num_units, domain_numb=num_domains, name="ADAP_gate_%d"%i)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    self.ADAP_gate_stopping_gradient = ADAP_gate_stopping_gradient
    if ADAP_contribution == None:
      ADAP_contribution = [1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution
  
  def call(self, inputs, sequence_length=None, training=None, internal_node_printing=False):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    for layer, multi_domain_layer, multi_domain_gate in zip(self.layers, self.multi_domain_layers, self.multi_domain_gates):
      inputs = layer(inputs, mask=mask, training=training)
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
    
    return outputs, None, sequence_length

  def forward_fn(self, inputs, args_dict, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    for layer, multi_domain_layer, multi_domain_gate in zip(self.layers,self.multi_domain_layers,self.multi_domain_gates):
      inputs = layer.forward_fn(inputs, args_dict, mask=mask, training=training)
      
      if self.ADAP_gate_stopping_gradient:
        g = multi_domain_gate.forward_fn(tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training)
      else:
        g = multi_domain_gate.forward_fn(inputs, args_dict, domain, mask=mask, training=training)
        
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training) * g + inputs * (1-g)
      else:
        inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain, mask=mask, training=training) * g + inputs * (1-g)
    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_v5(Encoder):
  
  def __init__(self,
               num_layers,
               num_domains=6,
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
               multi_domain_adapter_gate_class=Multi_domain_Gate,
               ADAP_contribution=None,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v5, self).__init__(**kwargs)
    self.num_units = num_units
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = LayerNorm()
    self.layers = [
        transformer.SelfAttentionEncoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
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
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    if ADAP_contribution == None:
      ADAP_contribution = [1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution
  def call(self, inputs, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    for layer, multi_domain_layer, multi_domain_gate in zip(self.layers, self.multi_domain_layers, self.multi_domain_gates):
      inputs = layer(inputs, mask=mask, training=training)
      g = multi_domain_gate(inputs, domain, mask=mask, training=training)
      inputs = multi_domain_layer(g * inputs + (1-g) * tf.stop_gradient(inputs), domain, mask=mask, training=training) + inputs
    outputs = self.layer_norm(inputs)
    
    return outputs, None, sequence_length

  def forward_fn(self, inputs, args_dict, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    for layer, multi_domain_layer, multi_domain_gate in zip(self.layers,self.multi_domain_layers,self.multi_domain_gates):
      inputs = layer.forward_fn(inputs, args_dict, mask=mask, training=training)
      g = multi_domain_gate.forward_fn(inputs, domain, mask=mask, training=training)
      inputs = multi_domain_layer.forward_fn(g*inputs+(1-g)*tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training) + inputs
    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m    

class Multi_domain_SelfAttentionEncoder_v4(Encoder):

  def __init__(self,
               num_layers,
               num_domains=6,
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
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v4, self).__init__(**kwargs)
    self.num_units = num_units
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = LayerNorm()
    self.layers = [
        transformer.SelfAttentionEncoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
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
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    if ADAP_contribution == None:
      ADAP_contribution = [1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution
  def call(self, inputs, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    for layer, multi_domain_layer, multi_domain_gate in zip(self.layers, self.multi_domain_layers, self.multi_domain_gates):
      inputs = layer(inputs, mask=mask, training=training)
      g = multi_domain_gate(inputs, domain, mask=mask, training=training)
      if self.ADAP_layer_stopping_gradient:        
        inputs = multi_domain_layer(tf.stop_gradient(inputs), domain, mask=mask, training=training) * g + inputs * (1-g)
      else:
        inputs = multi_domain_layer(inputs, domain, mask=mask, training=training) * g + inputs * (1-g)
    outputs = self.layer_norm(inputs)
    
    return outputs, None, sequence_length

  def forward_fn(self, inputs, args_dict, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    for layer, multi_domain_layer, multi_domain_gate in zip(self.layers,self.multi_domain_layers,self.multi_domain_gates):
      inputs = layer.forward_fn(inputs, args_dict, mask=mask, training=training)
      g = multi_domain_gate.forward_fn(inputs, domain, mask=mask, training=training)
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training) * g + inputs * (1-g)
      else:
        inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain, mask=mask, training=training) * g + inputs * (1-g)
    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_v6(Encoder):
  
  def __init__(self,
               num_layers,
               num_domains=6,
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
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v6, self).__init__(**kwargs)
    self.num_units = num_units
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = LayerNorm()
    self.layers = [
        transformer.SelfAttentionEncoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
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
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    self.ADAP_gate_stopping_gradient = ADAP_gate_stopping_gradient
    if ADAP_contribution == None:
      ADAP_contribution = [1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution

  def call(self, inputs, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    multi_domain_forget_gate = self.multi_domain_forget_gate
    multi_domain_input_gate = self.multi_domain_input_gate
    for layer, multi_domain_layer in zip(self.layers, self.multi_domain_layers):
      inputs = layer(inputs, mask=mask, training=training)
      ADAP_input = multi_domain_layer(inputs, domain, mask=mask, training=training)
      f = multi_domain_forget_gate(inputs, ADAP_input, mask=mask, training=training)
      i = multi_domain_input_gate(inputs, ADAP_input, mask=mask, training=training)
      inputs = inputs * f + ADAP_input * i
      #if not training:
      #  tf.print(self.name_scope(),"forget_gate:",tf.reduce_mean(tf.abs(f)),"input gate:",tf.reduce_mean(tf.abs(i)),sep="|")
    outputs = self.layer_norm(inputs)
    
    return outputs, None, sequence_length

  def adv_call(self, inputs, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    multi_domain_forget_gate = self.multi_domain_forget_gate
    multi_domain_input_gate = self.multi_domain_input_gate
    for layer, multi_domain_layer in zip(self.layers, self.multi_domain_layers):
      inputs = layer(inputs, mask=mask, training=training)      
      ADAP_input = tf.stop_gradient(multi_domain_layer(inputs, domain, mask=mask, training=training))
      f = multi_domain_forget_gate(inputs, ADAP_input, mask=mask, training=training)
      i = multi_domain_input_gate(inputs, ADAP_input, mask=mask, training=training)
      inputs = inputs * f + ADAP_input * i
      
    outputs = self.layer_norm(inputs)
    
    return outputs, None, sequence_length

  def forward_fn(self, inputs, args_dict, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    multi_domain_forget_gate = self.multi_domain_forget_gate
    multi_domain_input_gate = self.multi_domain_input_gate
    for layer, multi_domain_layer in zip(self.layers,self.multi_domain_layers):
      inputs = layer(inputs, mask=mask, training=training)
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
        ADAP_input = multi_domain_layer(inputs, domain, mask=mask, training=training)
        if self.ADAP_gate_stopping_gradient:
          f = multi_domain_forget_gate.forward_fn(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
          i = multi_domain_input_gate.forward_fn(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
        else:
          f = multi_domain_forget_gate.forward_fn(inputs, mask=mask, training=training)
          i = multi_domain_input_gate.forward_fn(inputs, mask=mask, training=training)
        inputs = inputs * f + ADAP_input * i

    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_v8(Encoder):
  
  def __init__(self,
               num_layers,
               num_domains=6,
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
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v8, self).__init__(**kwargs)
    self.num_units = num_units
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = LayerNorm()
    self.layers = [
        transformer.SelfAttentionEncoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
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
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    self.ADAP_gate_stopping_gradient = ADAP_gate_stopping_gradient
    if ADAP_contribution == None:
      ADAP_contribution = [1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution

  def call(self, inputs, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    for i, (layer, multi_domain_layer, multi_domain_input_gate, multi_domain_forget_gate) in enumerate(zip(self.layers, self.multi_domain_layers, self.multi_domain_input_gates, self.multi_domain_forget_gates)):
      inputs = layer(inputs, mask=mask, training=training)
      ADAP_input = multi_domain_layer(tf.stop_gradient(inputs), domain, mask=mask, training=training)
      f = multi_domain_forget_gate(tf.stop_gradient(inputs), tf.stop_gradient(ADAP_input), mask=mask, training=training)
      i_ = multi_domain_input_gate(tf.stop_gradient(inputs), tf.stop_gradient(ADAP_input), mask=mask, training=training)
      inputs = inputs * f + ADAP_input * i_
      if not training:
        tf.print(self.name_scope(),"forget_gate:",tf.reduce_mean(tf.abs(f)),"input gate:",tf.reduce_mean(tf.abs(i_)),sep="|", output_stream=sys.stdout)
    outputs = self.layer_norm(inputs)
    
    return outputs, None, sequence_length

  def adv_call(self, inputs, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    for i, (layer, multi_domain_layer, multi_domain_input_gate, multi_domain_forget_gate) in enumerate(zip(self.layers, self.multi_domain_layers, self.multi_domain_input_gates, self.multi_domain_forget_gates)):
      inputs = layer(inputs, mask=mask, training=training)      
      ADAP_input = tf.stop_gradient(multi_domain_layer(inputs, domain, mask=mask, training=training))
      f = multi_domain_forget_gate(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
      i_ = multi_domain_input_gate(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
      inputs = inputs * f + ADAP_input * i_
    outputs = self.layer_norm(inputs)
    
    return outputs, None, sequence_length

  def forward_fn(self, inputs, args_dict, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    for i, (layer, multi_domain_layer, multi_domain_input_gate, multi_domain_forget_gate) in enumerate(zip(self.layers, self.multi_domain_layers, self.multi_domain_input_gates, self.multi_domain_forget_gates)):
      inputs = layer(inputs, mask=mask, training=training)
      ADAP_input = multi_domain_layer(inputs, domain, mask=mask, training=training)
      f = multi_domain_forget_gate.forward_fn(inputs, mask=mask, training=training)
      i_ = multi_domain_input_gate.forward_fn(inputs, mask=mask, training=training)
      inputs = inputs * f + ADAP_input * i_
    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_v9(Encoder):
  
  def __init__(self,
               num_layers,
               num_domains=6,
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
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v9, self).__init__(**kwargs)
    self.num_units = num_units
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = LayerNorm()
    self.layers = [
        transformer.SelfAttentionEncoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
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
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    self.ADAP_gate_stopping_gradient = ADAP_gate_stopping_gradient
    if ADAP_contribution == None:
      ADAP_contribution = [1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution
  def call(self, inputs, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    for layer, multi_domain_layer, multi_domain_gate in zip(self.layers, self.multi_domain_layers, self.multi_domain_gates):
      inputs = layer(inputs, mask=mask, training=training)
      if self.ADAP_gate_stopping_gradient:
        g = multi_domain_gate(tf.stop_gradient(inputs), domain, mask=mask, training=training)
      else:
        g = multi_domain_gate(inputs, domain, mask=mask, training=training)
      if self.ADAP_layer_stopping_gradient:        
        inputs = multi_domain_layer(tf.stop_gradient(inputs), domain, mask=mask, training=training) * g + inputs * (1-g)
      else:
        inputs = multi_domain_layer(inputs, domain, mask=mask, training=training) * g + inputs * (1-g)
    outputs = self.layer_norm(inputs)
    
    return outputs, None, sequence_length

  def forward_fn(self, inputs, args_dict, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    for layer, multi_domain_layer, multi_domain_gate in zip(self.layers,self.multi_domain_layers,self.multi_domain_gates):
      inputs = layer.forward_fn(inputs, args_dict, mask=mask, training=training)
      
      if self.ADAP_gate_stopping_gradient:
        g = multi_domain_gate.forward_fn(tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training)
      else:
        g = multi_domain_gate.forward_fn(inputs, args_dict, domain, mask=mask, training=training)
        
      if self.ADAP_layer_stopping_gradient:
        inputs = multi_domain_layer.forward_fn(tf.stop_gradient(inputs), args_dict, domain, mask=mask, training=training) * g + inputs * (1-g)
      else:
        inputs = multi_domain_layer.forward_fn(inputs, args_dict, domain, mask=mask, training=training) * g + inputs * (1-g)
    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_v7(Encoder):
  
  def __init__(self,
               num_layers,
               num_domains=6,
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
               ADAP_contribution=None,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v7, self).__init__(**kwargs)
    self.num_units = num_units
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = LayerNorm()
    self.layers = [
        transformer.SelfAttentionEncoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]    
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        for i in range(num_layers)]
    self.multi_domain_forget_gate = multi_domain_adapter_gate_class(num_units, num_units, num_units, domain_numb=num_domains, name="ADAP_forget_gate")
    self.multi_domain_input_gate = multi_domain_adapter_gate_class(num_units, num_units, num_units, domain_numb=num_domains, name="ADAP_input_gate")
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    self.ADAP_gate_stopping_gradient = ADAP_gate_stopping_gradient
    if ADAP_contribution == None:
      ADAP_contribution = [1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution

  def call(self, inputs, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    for layer in self.layers:
      inputs = layer(inputs, mask=mask, training=training)
    outputs = self.layer_norm(inputs)
    
    return outputs, None, sequence_length

  def forward_fn(self, inputs, args_dict, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    for layer in self.layers:
      inputs = layer.forward_fn(inputs, mask=mask, training=training)
    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_v10(Encoder):
  
  def __init__(self,
               num_layers,
               num_domains=6,
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
               fake_domain_prob=0.1,
               noisy_prob=None,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v10, self).__init__(**kwargs)
    self.num_units = num_units
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = LayerNorm()
    self.layers = [
        transformer.SelfAttentionEncoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)] 
    print("multi_domain_adapter_class == Multi_domain_FeedForwardNetwork_v6", multi_domain_adapter_class == Multi_domain_FeedForwardNetwork_v6)
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        if not multi_domain_adapter_class in [Multi_domain_FeedForwardNetwork_v6, Multi_domain_FeedForwardNetwork_v8]
        else multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i, 
        fake_domain_prob=fake_domain_prob, noisy_prob=noisy_prob)
        for i in range(num_layers)]
    self.multi_domain_gate = transformer.SelfAttentionEncoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    self.ADAP_gate_stopping_gradient = ADAP_gate_stopping_gradient
    if ADAP_contribution == None:
      ADAP_contribution = [1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution
  
  def call(self, inputs, sequence_length=None, training=None, internal_node_printing=False):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)

    if self.ADAP_gate_stopping_gradient:
      g = self.multi_domain_gate(tf.stop_gradient(inputs), mask=mask, training=training)
    else:
      g = self.multi_domain_gate(inputs, mask=mask, training=training)

    for layer, multi_domain_layer in zip(self.layers, self.multi_domain_layers):
      inputs = layer(inputs, mask=mask, training=training)
      
      if self.ADAP_layer_stopping_gradient:        
        inputs = multi_domain_layer(tf.stop_gradient(inputs), domain, mask=mask, training=training) * g + inputs * (1-g)
      else:
        inputs = multi_domain_layer(inputs, domain, mask=mask, training=training) * g + inputs * (1-g)
      if internal_node_printing:
        tf.print("###", self.name_scope(), "gate_mean_abs_pooling: ", tf.reduce_mean(tf.abs(g),-1)[0,:], "domain: ", domain, "###", sep="|", summarize=1000)

    outputs = self.layer_norm(inputs)
    
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_v11(Encoder):
  
  def __init__(self,
               num_layers,
               num_domains=6,
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
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v11, self).__init__(**kwargs)
    self.num_units = num_units
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = LayerNorm()
    self.layers = [
        transformer.SelfAttentionEncoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
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
    self.multi_domain_input_gates = [multi_domain_adapter_gate_class(num_units, num_units, num_units, 
                                    domain_numb=num_domains, name="ADAP_input_gate", output_regularization=input_gate_regularization)
                                    for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    self.ADAP_gate_stopping_gradient = ADAP_gate_stopping_gradient
    if ADAP_contribution == None:
      ADAP_contribution = [1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution

  def call(self, inputs, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    for layer, multi_domain_layer, multi_domain_forget_gate, multi_domain_input_gate in zip(self.layers, self.multi_domain_layers, self.multi_domain_forget_gates, self.multi_domain_input_gates):
      inputs = layer(inputs, mask=mask, training=training)
      ADAP_input = multi_domain_layer(inputs, domain, mask=mask, training=training)
      f = multi_domain_forget_gate(inputs, ADAP_input, mask=mask, training=training)
      i = multi_domain_input_gate(inputs, ADAP_input, mask=mask, training=training)
      inputs = inputs * f + ADAP_input * i
      #if not training:
      #  tf.print(self.name_scope(),"forget_gate:",tf.reduce_mean(tf.abs(f)),"input gate:",tf.reduce_mean(tf.abs(i)),sep="|")
    outputs = self.layer_norm(inputs)
    
    return outputs, None, sequence_length

  def forward_fn(self, inputs, args_dict, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    multi_domain_forget_gate = self.multi_domain_forget_gate
    multi_domain_input_gate = self.multi_domain_input_gate
    for layer, multi_domain_layer in zip(self.layers,self.multi_domain_layers):
      inputs = layer(inputs, mask=mask, training=training)
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
        ADAP_input = multi_domain_layer(inputs, domain, mask=mask, training=training)
        if self.ADAP_gate_stopping_gradient:
          f = multi_domain_forget_gate.forward_fn(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
          i = multi_domain_input_gate.forward_fn(tf.stop_gradient(inputs), ADAP_input, mask=mask, training=training)
        else:
          f = multi_domain_forget_gate.forward_fn(inputs, mask=mask, training=training)
          i = multi_domain_input_gate.forward_fn(inputs, mask=mask, training=training)
        inputs = inputs * f + ADAP_input * i

    outputs = self.layer_norm.forward_fn(inputs, args_dict)
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_v12(Encoder):

  def __init__(self,
               num_layers,
               num_domains=6,
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
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v12, self).__init__(**kwargs)
    self.num_units = num_units
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = LayerNorm()
    self.layers = [
        transformer.SelfAttentionEncoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
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
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    if ADAP_contribution == None:
      ADAP_contribution = [1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution
  
  def call(self, inputs, sequence_length=None, training=None, internal_node_printing=False):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]    
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    total_adapt = []
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers, self.multi_domain_layers)):
      if self.ADAP_layer_stopping_gradient:
        adapt = multi_domain_layer(layer(tf.stop_gradient(inputs), mask=mask, training=training), domain, mask=mask, training=training)
        total_adapt.append(adapt)
        inputs = layer(inputs, mask=mask, training=training)                
      else:
        inputs = layer(inputs, mask=mask, training=training)
        adapt = multi_domain_layer(inputs, domain, mask=mask, training=training)
        total_adapt.append(adapt)
      
    total_adapt = tf.add_n(total_adapt)
    if internal_node_printing:
        tf.print("Encoder ADAP mean pooling: ", tf.reduce_mean(tf.abs(total_adapt),-1)[0,:], "domain: ", domain, "###", sep="|", summarize=1000)
    outputs = self.layer_norm(inputs+total_adapt)
    
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

  def forward_fn(self, inputs, args_dict, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    total_adapt=[]
    #for layer in self.layers:
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers,self.multi_domain_layers)):
      if self.ADAP_layer_stopping_gradient:
        adapt = multi_domain_layer.forward_fn(layer(tf.stop_gradient(inputs), args_dict, mask=mask, training=training), args_dict, domain, mask=mask, training=training)
        total_adapt.append(adapt)
        inputs = layer.forward_fn(inputs, args_dict, mask=mask, training=training)                
      else:
        inputs = layer.forward_fn(inputs, args_dict, mask=mask, training=training)
        adapt = multi_domain_layer.forward_fn(inputs, args_dict, domain, mask=mask, training=training)
        total_adapt.append(adapt)
    outputs = self.layer_norm.forward_fn(inputs + tf.add_n(total_adapt), args_dict)
    return outputs, None, sequence_length

class Multi_domain_SelfAttentionEncoder_v15(Encoder):
  
  def __init__(self,
               num_layers,
               num_domains=6,
               num_domain_units=128,
               domain_region_sizes=None,
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
               fake_domain_prob=0.1,
               noisy_prob=None,
               version=1,
               inner_layer_norm=None,
               stop_gradient_version=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v15, self).__init__(**kwargs)
    self.num_units = num_units
    self.dropout = dropout
    self.position_encoder = None
    self.num_layers = num_layers
    self.num_domains = num_domains
    if version in [18,19,20]:
      self.domain_mask = make_domain_mask(self.num_domains,  num_units=num_units, num_domain_units=num_domain_units, domain_region_sizes=domain_region_sizes)
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    
    if version==18:
      self.layer_norm = Multi_LayerNorm(num_domains)

      self.layers = [
          transformer.SelfAttentionEncoderLayer_v1(
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
      self.layer_norm = LayerNorm()
      self.layers = [
          transformer.SelfAttentionEncoderLayer(
              num_units,
              num_heads,
              ffn_inner_dim,
              dropout=dropout,
              attention_dropout=attention_dropout,
              ffn_dropout=ffn_dropout,
              ffn_activation=ffn_activation)
          for i in range(num_layers)] 
    elif version == 20:
      self.layer_norm = Multi_LayerNorm(num_domains)

      self.layers = [
          transformer.SelfAttentionEncoderLayer_v3(
              num_units,
              num_heads,
              ffn_inner_dim,
              domain_numb = num_domains,
              dropout=dropout,
              attention_dropout=attention_dropout,
              ffn_dropout=ffn_dropout,
              ffn_activation=ffn_activation)
          for i in range(num_layers)] 
    elif version == 19:
      self.layer_norm = Multi_LayerNorm(num_domains)
      self.layers = [
          transformer.SelfAttentionEncoderLayer_v2(
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
      self.layer_norm = LayerNorm()
      self.layers = [
          transformer.SelfAttentionEncoderLayer(
              num_units,
              num_heads,
              ffn_inner_dim,
              dropout=dropout,
              attention_dropout=attention_dropout,
              ffn_dropout=ffn_dropout,
              ffn_activation=ffn_activation)
          for i in range(num_layers)] 

    print("inner_layer_norm: ", inner_layer_norm)
    print("multi_domain_adapter_class == Multi_domain_FeedForwardNetwork_v6", multi_domain_adapter_class == Multi_domain_FeedForwardNetwork_v6)
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, inner_layer_norm=inner_layer_norm, name="ADAP_%d"%i)
        if not multi_domain_adapter_class in [Multi_domain_FeedForwardNetwork_v6, Multi_domain_FeedForwardNetwork_v8]
        else multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i, 
        fake_domain_prob=fake_domain_prob, noisy_prob=noisy_prob)
        for i in range(num_layers)]
    if version==17:
      self.multi_domain_gate = Multi_domain_classification_gate_v2(num_units, num_units, domain_numb=num_domains, name="ADAP_gate")
    else:
      self.multi_domain_gate = multi_domain_adapter_gate_class(num_units, num_units, domain_numb=num_domains, name="ADAP_gate")

    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    self.ADAP_gate_stopping_gradient = ADAP_gate_stopping_gradient
    if ADAP_contribution == None:
      ADAP_contribution = [1.0] * num_layers
    
    self.ADAP_contribution = ADAP_contribution
    self.version = version
    self.stop_gradient_version = stop_gradient_version
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

    self.training_res_using_rate = training_res_using_rate
    self.testing_res_using_rate = testing_res_using_rate
    print("testing_res_using_rate: ", testing_res_using_rate)
    print("training_res_using_rate: ", training_res_using_rate)
  
  def build(self, input_shape):
    super(Multi_domain_SelfAttentionEncoder_v15, self).build(input_shape)
    scope_name = self.name_scope()
    self.lhuc_embedding = []
    if self.version==16:
      for i in range(self.num_layers):
        self.lhuc_embedding.append(self.add_weight("%s_lhuc_%d"%(scope_name,i), shape=[self.num_domains, self.num_units]))

  def call(self, inputs, sequence_length=None, training=None, internal_node_printing=False, adapter_activate=True):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5
    
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    total_adapt=[]
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
        inputs = layer(inputs, domain, mask=mask, training=training)
        domain_mask = tf.nn.embedding_lookup(self.domain_mask, domain)
        inputs = tf.math.multiply(inputs, domain_mask)
      elif self.version ==21:
        inputs = layer(inputs, mask=mask, training=training)
        domain_mask = tf.nn.embedding_lookup(self.domain_mask, domain)
        inputs = tf.math.multiply(inputs, domain_mask)     
      else:
        inputs = layer(inputs, mask=mask, training=training)

      if self.version not in [3,8,10,11,9,12,15,16,17,18,19,20,21]:
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
        if adapter_activate:
          adapt = multi_domain_layer(inputs, domain, mask=mask, training=training)
          inputs = inputs + common.dropout(adapt, 1-self.training_res_using_rate, training=training)
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
      if internal_node_printing:
        tf.print("###", self.name_scope(), "gate_mean_abs_pooling: ", tf.reduce_mean(g,-1)[0,:], "domain: ", domain, "###", sep="|", summarize=1000)
    if self.version == 17:
      g = self.multi_domain_gate(inputs, domain, mask=mask, training=training, tag="encoder prediction")
      #if internal_node_printing:
        #tf.print("###", self.name_scope(), "gate_mean_abs_pooling: ", g[domain,:,0], "domain: ", domain, "###", sep="|", summarize=-1)
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
      #tf.print("g:",g.shape,tf.concat(all_values,1).shape, g[:,0,:])
      total_adapt = tf.math.reduce_sum(tf.concat(all_values,0) * tf.stop_gradient(g),0)
      total_adapt = tf.reshape(total_adapt,[b,-1,self.num_units])
      #tf.print("total_adapt",tf.shape(total_adapt))
      outputs = self.layer_norm(total_adapt)
    elif self.version==7:
      outputs = self.layer_norm(inputs + common.dropout(total_adapt, 1-self.training_res_using_rate, training=training))
    elif self.version==9:
      outputs = self.layer_norm(inputs + total_adapt)
    elif self.version==8:
      outputs = self.layer_norm(inputs + tf.exp((g-1)*2/g) * total_adapt)
    elif self.version in [10,11,12]:
      outputs = self.layer_norm(inputs)
    elif self.version ==15:
      outputs = self.layer_norm(inputs)
    elif self.version ==16:
      outputs = self.layer_norm(inputs)
    elif self.version ==18:
      outputs = self.layer_norm(inputs, domain)
    elif self.version ==19:
      outputs = self.layer_norm(inputs, domain)
    elif self.version ==20:
      outputs = self.layer_norm(inputs, domain)
    elif self.version ==21:
      outputs = self.layer_norm(inputs)
    return outputs, None, sequence_length

  def adv_call(self, inputs, sequence_length=None, training=None):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    total_adapt=[]
    for layer, multi_domain_layer in zip(self.layers, self.multi_domain_layers):
      inputs = layer(inputs, mask=mask, training=training)
      if self.version!=3:
        adapt = multi_domain_layer(inputs, domain, mask=mask, training=training)
        total_adapt.append(adapt)

    if self.version!=3:
      if self.stop_gradient_version==1:
        g = self.multi_domain_gate(tf.stop_gradient(inputs), domain, mask=mask, training=training)
      else:
        g = self.multi_domain_gate(inputs, domain, mask=mask, training=training)
      total_adapt = tf.add_n(total_adapt)
      if self.stop_gradient_version==1:
        if self.ADAP_gate_stopping_gradient:
          if isinstance(self.ADAP_gate_stopping_gradient, float):
            print("stopping gradient at d_classifier in encoder: ", self.ADAP_gate_stopping_gradient)
            g = tf.stop_gradient(g * (1-self.ADAP_gate_stopping_gradient)) + g * self.ADAP_gate_stopping_gradient
          elif isinstance(self.ADAP_gate_stopping_gradient, bool):
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

    return outputs, None, sequence_length

  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_v16(Encoder):

  def __init__(self,
               num_layers,
               num_domains=6,
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
               inner_layer_norm=None,
               fake_domain_prob=0.1,
               noisy_prob=None,
               ADAP_contribution=None,
               version=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v16, self).__init__(**kwargs)
    self.num_units = num_units
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = LayerNorm()
    self.layer_norm_2 = LayerNorm(name="enc_layernorm_2")
    self.layers = [
        transformer.SelfAttentionEncoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]    
    self.multi_domain_layers = [ multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains,inner_layer_norm=inner_layer_norm, name="ADAP_%d"%i) for i in range(num_layers)]
    self.noisy_layers = [multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=2, inner_layer_norm=None, name="noisy_ADAP_%d"%i) for i in range(num_layers)]
    self.multi_domain_gate = multi_domain_adapter_gate_class(num_units, num_units, domain_numb=num_domains, name="ADAP_gate")
    self.noisy_gate = multi_domain_adapter_gate_class(num_units, num_units, domain_numb=2, name="noisy_gate")
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    if ADAP_contribution == None:
      ADAP_contribution = [1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution
    self.version = version
    print("enc version: ", version)
  
  def call(self, inputs, sequence_length=None, training=None, internal_node_printing=False):
    is_noisy = inputs[2]
    is_noisy = is_noisy[0]
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]    
    inputs *= self.num_units**0.5
    print("internal_node_printing: ", internal_node_printing)
    total_adapt=[]
    total_noisy_adapt=[]
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    for i, (layer, multi_domain_layer, noisy_layer) in enumerate(zip(self.layers, self.multi_domain_layers, self.noisy_layers)):
      inputs = layer(inputs, mask=mask, training=training)
      if self.version==1:
        adapt = multi_domain_layer(inputs, domain, mask=mask, training=training)
        inputs = adapt * self.ADAP_contribution[i] + inputs
      elif self.version==2:
        adapt = multi_domain_layer(inputs, domain, mask=mask, training=training)
        total_adapt.append(adapt)
      elif self.version in [5,6]:
        noisy_adapt= noisy_layer(inputs,is_noisy,mask=mask,training=training)
        total_noisy_adapt.append(noisy_adapt)
      elif self.version==7:
        noisy_adapt= noisy_layer(inputs,is_noisy,mask=mask,training=training)
        total_noisy_adapt.append(noisy_adapt)
        adapt = multi_domain_layer(inputs, domain, mask=mask, training=training)
        total_adapt.append(adapt)
      elif self.version==8:
        noisy_adapt= noisy_layer(inputs,is_noisy,mask=mask,training=training)
        inputs = inputs + noisy_adapt
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



    if self.version in [3,5,6,7]:
      g = self.noisy_gate(inputs, is_noisy, mask=mask, training=training)
      if internal_node_printing:
        tf.print("###", self.name_scope(), "noisy_gate_mean_abs_pooling: ", tf.reduce_mean(g,-1)[0,:], "is_noisy: ", is_noisy, "###", sep="|", summarize=1000)
    if self.version in [6,7]:
      domain_g = self.multi_domain_gate(inputs, domain, mask=mask, training=training)
      if internal_node_printing:
        tf.print("###", self.name_scope(), "domain_gate_mean_abs_pooling: ", tf.reduce_mean(domain_g,-1)[0,:], "domain: ", domain, "###", sep="|", summarize=1000)
    if self.version==1:
      outputs = self.layer_norm(inputs)
    elif self.version==2:
      outputs = self.layer_norm(inputs + tf.add_n(total_adapt))
    elif self.version in [5,6]:
      outputs = self.layer_norm(inputs + tf.exp((g-1)*2/g) * tf.add_n(total_noisy_adapt))
    elif self.version==7:
      outputs = self.layer_norm(inputs + tf.exp((g-1)*2/g) * tf.add_n(total_noisy_adapt))
      outputs = self.layer_norm_2(outputs + tf.exp((domain_g-1)*2/domain_g) * tf.add_n(total_adapt))
    else:
      outputs = self.layer_norm(inputs)
    
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_v17(Encoder):
  
  def __init__(self,
               num_layers,
               num_domains=6,
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
               fake_domain_prob=0.1,
               noisy_prob=None,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v17, self).__init__(**kwargs)
    self.num_units = num_units
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = LayerNorm()
    self.layers = [
        transformer.SelfAttentionEncoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)] 
    print("multi_domain_adapter_class == Multi_domain_FeedForwardNetwork_v6", multi_domain_adapter_class == Multi_domain_FeedForwardNetwork_v6)
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i)
        if not multi_domain_adapter_class in [Multi_domain_FeedForwardNetwork_v6, Multi_domain_FeedForwardNetwork_v8]
        else multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, name="ADAP_%d"%i, 
        fake_domain_prob=fake_domain_prob, noisy_prob=noisy_prob)
        for i in range(num_layers)]
    self.multi_domain_gates = [
        multi_domain_adapter_gate_class(num_units, num_units, num_units, domain_numb=num_domains, name="ADAP_gate_%d"%i)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    self.ADAP_gate_stopping_gradient = ADAP_gate_stopping_gradient
    if ADAP_contribution == None:
      ADAP_contribution = [1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution
  
  def call(self, inputs, sequence_length=None, training=None, internal_node_printing=False):
    domain = inputs[1]
    domain = domain[0]
    inputs = inputs[0]
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    total_adapt=[]
    for layer, multi_domain_layer, multi_domain_gate in zip(self.layers, self.multi_domain_layers, self.multi_domain_gates):
      inputs = layer(inputs, mask=mask, training=training)
      adapt = multi_domain_layer(inputs, domain, mask=mask, training=training)
      total_adapt.append(adapt)

    g = multi_domain_gate(inputs, domain, mask=mask, training=training)
    total_adapt = tf.add_n(total_adapt)
    if internal_node_printing:
      tf.print("###", self.name_scope(), "gate_mean_abs_pooling: ", tf.reduce_mean(g,-1)[0,:], "adapt_mean_abs_pooling: ", tf.reduce_mean(tf.abs(total_adapt),-1)[0,:], "domain: ", domain, "###", sep="|", summarize=1000)

    outputs = self.layer_norm(inputs * (1-g) + total_adapt * g)
    
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_v18(Encoder):

  def __init__(self,
               num_layers,
               num_domains=6,
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
               multi_domain_adapter_class=Multi_domain_FeedForwardNetwork_v9,
               inner_layer_norm=None,
               fake_domain_prob=0.1,
               noisy_prob=None,
               ADAP_contribution=None,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_v18, self).__init__(**kwargs)
    self.num_units = num_units
    self.dropout = dropout
    self.position_encoder = None
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    self.layer_norm = LayerNorm()
    self.layers = [
        transformer.SelfAttentionEncoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]    
    self.multi_domain_layers = [
        multi_domain_adapter_class(num_units, num_domain_units, num_units, domain_numb=num_domains, inner_layer_norm=inner_layer_norm, name="ADAP_%d"%i)
        for i in range(num_layers)]
    self.ADAP_layer_stopping_gradient = ADAP_layer_stopping_gradient
    if ADAP_contribution == None:
      ADAP_contribution = [1.0] * num_layers
    self.ADAP_contribution = ADAP_contribution
  
  def call(self, inputs, sequence_length=None, training=None, internal_node_printing=False):
    domain = inputs[1]
    inputs = inputs[0]    
    inputs *= self.num_units**0.5

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    total_adapt = []
    for i, (layer, multi_domain_layer) in enumerate(zip(self.layers, self.multi_domain_layers)):
      inputs = layer(inputs, mask=mask, training=training)
      if self.ADAP_contribution[i]>0:
        adapt = multi_domain_layer(inputs, domain, mask=mask, training=training)
        total_adapt.append(adapt)
    if len(total_adapt)>0:
      total_adapt = tf.add_n(total_adapt)
    else:
      total_adapt = 0
    if internal_node_printing:
        tf.print("Encoder ADAP mean pooling: ", tf.reduce_mean(tf.abs(total_adapt),-1)[0,:], "domain: ", domain, "###", sep="|", summarize=1000)
    outputs = self.layer_norm(inputs+total_adapt)
    
    return outputs, None, sequence_length
    
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_sparse(Encoder):
  
  def __init__(self,
               num_layers,
               num_domains=6,
               num_domain_units=128,
               domain_region_sizes=None,
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
               fake_domain_prob=0.1,
               noisy_prob=None,
               version=1,
               inner_layer_norm=None,
               stop_gradient_version=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_sparse, self).__init__(**kwargs)
    self.num_units = num_units
    self.dropout = dropout
    self.position_encoder = None
    self.num_layers = num_layers
    self.num_domains = num_domains
    self.version = version
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    print("encoder_version",version)
    if version ==1:
      self.layer_norm = Multi_LayerNorm(num_domains)
      self.layers = [
        transformer.SelfAttentionEncoderLayer_v1(
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
      self.layer_norm = LayerNorm()
      self.layers = [
        transformer.SelfAttentionEncoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]  
    
  def call(self, inputs, sequence_length=None, training=None, internal_node_printing=False, adapter_activate=True):
    domain = inputs[1]    
    domain_mask = inputs[2]
    inputs = inputs[0]
    domain = domain[0]

    inputs *= self.num_units**0.5
    
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    
    inputs = tf.math.multiply(inputs, domain_mask)
        
    for i, layer in enumerate(self.layers):
      if self.version ==1:
        inputs = layer(inputs, domain, mask=mask, training=training)
      else:
        inputs = layer(inputs, mask=mask, training=training)
      inputs = tf.math.multiply(inputs, domain_mask)
           
    if self.version==1:
      outputs = self.layer_norm(inputs, domain)
    else:
      outputs = self.layer_norm(inputs)
    return outputs, None, sequence_length
  
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_sparse_multi_layer(Encoder):
  
  def __init__(self,
               num_layers,
               num_domains=6,
               num_domain_units=128,
               domain_region_sizes=None,
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
               fake_domain_prob=0.1,
               noisy_prob=None,
               version=1,
               inner_layer_norm=None,
               stop_gradient_version=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_sparse_multi_layer, self).__init__(**kwargs)
    self.num_units = num_units
    self.dropout = dropout
    self.position_encoder = None
    self.num_layers = num_layers
    self.num_domains = num_domains
    self.version = version
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    print("encoder_version",version)
    if version == 1:
      self.layer_norm = Multi_LayerNorm(num_domains)
      self.layers = [
        transformer.SelfAttentionEncoderLayer_v1(
            num_units,
            num_heads,
            ffn_inner_dim,
            domain_numb = num_domains,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]  
    elif version == 3:
      self.layer_norm = LayerNorm_v2()
      self.layers = [
        transformer.SelfAttentionEncoderLayer_v4(
            num_units,
            num_heads,
            ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)] 
    else:
      self.layer_norm = LayerNorm()
      self.layers = [
        transformer.SelfAttentionEncoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]  
    
  def call(self, inputs, sequence_length=None, training=None, internal_node_printing=False, adapter_activate=True):
    domain = inputs[1]    
    domain_mask = inputs[2]
    inputs = inputs[0]
    domain = domain[0]

    inputs *= self.num_units**0.5
    
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    
    inputs = tf.math.multiply(inputs, domain_mask[0])
        
    for i, layer in enumerate(self.layers):
      if self.version ==1:
        inputs = layer(inputs, domain, mask=mask, training=training)
      else:
        inputs = layer(inputs, mask=mask, training=training)
      inputs = tf.math.multiply(inputs, domain_mask[i+1])
           
    if self.version==1:
      outputs = self.layer_norm(inputs, domain)
    else:
      outputs = self.layer_norm(inputs)
    return outputs, None, sequence_length
  
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_sparse_multi_layer_v0(Encoder):
  
  def __init__(self,
               num_layers,
               num_domains=6,
               num_domain_units=128,
               domain_region_sizes=None,
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
               fake_domain_prob=0.1,
               noisy_prob=None,
               version=1,
               inner_layer_norm=None,
               stop_gradient_version=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_sparse_multi_layer_v0, self).__init__(**kwargs)
    self.num_units = num_units
    self.dropout = dropout
    self.position_encoder = None
    self.num_layers = num_layers
    self.num_domains = num_domains
    self.version = version
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    print("encoder_version",version)
    if version ==1:
      self.layer_norm = Multi_LayerNorm(num_domains)
      self.layers = [
        transformer.SelfAttentionEncoderLayer_v1(
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
      self.layer_norm = LayerNorm()
      self.layers = [
        transformer.SelfAttentionEncoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation)
        for i in range(num_layers)]  
    
  def call(self, inputs, sequence_length=None, training=None, internal_node_printing=False, adapter_activate=True):
    domain = inputs[1]    
    domain_mask = inputs[2]
    inputs = inputs[0]
    domain = domain[0]

    inputs *= self.num_units**0.5
    
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    
    #inputs = tf.math.multiply(inputs, domain_mask[0])
        
    for i, layer in enumerate(self.layers):
      if self.version ==1:
        inputs = layer(inputs, domain, mask=mask, training=training)
      else:
        inputs = layer(inputs, mask=mask, training=training)
      inputs = tf.math.multiply(inputs, domain_mask[i])
           
    if self.version==1:
      outputs = self.layer_norm(inputs, domain)
    else:
      outputs = self.layer_norm(inputs)
    return outputs, None, sequence_length
  
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m

class Multi_domain_SelfAttentionEncoder_sparse_multi_layer_v1(Encoder):
  
  def __init__(self,
               num_layers,
               num_domains=6,
               num_domain_units=128,
               domain_region_sizes=None,
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
               fake_domain_prob=0.1,
               noisy_prob=None,
               version=1,
               inner_layer_norm=None,
               stop_gradient_version=1,
               **kwargs):
    
    super(Multi_domain_SelfAttentionEncoder_sparse_multi_layer_v1, self).__init__(**kwargs)
    self.num_units = num_units
    self.dropout = dropout
    self.position_encoder = None
    self.num_layers = num_layers
    self.num_domains = num_domains
    self.version = version
    if position_encoder_class is not None:
      self.position_encoder = position_encoder_class()
    print("encoder_version",version)
    
    self.layer_norm = LayerNorm()
    self.layers = [
      transformer.SelfAttentionEncoderLayer(
          num_units,
          num_heads,
          ffn_inner_dim,
          dropout=dropout,
          attention_dropout=attention_dropout,
          ffn_dropout=ffn_dropout,
          ffn_activation=ffn_activation)
      for i in range(num_layers)]  
    
  def call(self, inputs, sequence_length=None, training=None, internal_node_printing=False, adapter_activate=True):
    domain = inputs[1]    
    domain_mask = inputs[2]
    inputs = inputs[0]
    domain = domain[0]

    inputs *= self.num_units**0.5
    
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = self.build_mask(inputs, sequence_length=sequence_length)
            
    for i, layer in enumerate(self.layers):
      if self.version ==1:
        inputs = layer(inputs, domain, mask=mask, training=training)
      else:
        inputs = layer(inputs, mask=mask, training=training)
      inputs = tf.math.multiply(inputs, tf.expand_dims(domain_mask[i],1))
           
    if self.version==1:
      outputs = self.layer_norm(inputs, domain)
    else:
      outputs = self.layer_norm(inputs)
    return outputs, None, sequence_length
  
  def map_v1_weights(self, weights):
    m = []
    m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
    for i, layer in enumerate(self.layers):
      m += layer.map_v1_weights(weights["layer_%d" % i])
    return m






























