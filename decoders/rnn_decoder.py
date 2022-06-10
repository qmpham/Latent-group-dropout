import sys
sys.path.append("/gpfsdswork/projects/rech/sfz/utt84zy/anaconda3/envs/huggingface/lib/python3.7/site-packages")

from opennmt.decoders.rnn_decoder import RNNDecoder
import tensorflow_addons as tfa
import tensorflow as tf
from layers import common

class AttentionalRNNDecoder(RNNDecoder):
  """A RNN decoder with attention."""

  def __init__(self,
               num_layers,
               num_units,
               bridge_class=None,
               attention_mechanism_class=None,
               cell_class=None,
               dropout=0.3,
               residual_connections=False,
               first_layer_attention=False,
               **kwargs):
    
    super(AttentionalRNNDecoder, self).__init__(
        num_layers,
        num_units,
        bridge_class=bridge_class,
        cell_class=cell_class,
        dropout=dropout,
        residual_connections=residual_connections,
        **kwargs)
    if attention_mechanism_class is None:
      attention_mechanism_class = tfa.seq2seq.BahdanauAttention
    self.attention_mechanism_1 = attention_mechanism_class(self.cell.output_size)
    self.attention_mechanism_2 = attention_mechanism_class(self.cell.output_size)    
    self.dropout = dropout
    
  @property
  def support_alignment_history(self):
    return True

  def _get_initial_state(self, batch_size, dtype, initial_state=None):
    ####
    assert isinstance(self.memory,list)
    self.attention_mechanism_1.setup_memory(
        self.memory[0], memory_sequence_length=self.memory_sequence_length)
    self.attention_mechanism_2.setup_memory(
        self.memory[1], memory_sequence_length=self.memory_sequence_length)
    ####
    decoder_state = self.cell.get_initial_state(batch_size=batch_size, dtype=dtype)
    if initial_state is not None:
      if self.first_layer_attention:
        cell_state = list(decoder_state)
        cell_state[0] = decoder_state[0].cell_state
        cell_state = self.bridge(initial_state, cell_state)
        cell_state[0] = decoder_state[0].clone(cell_state=cell_state[0])
        decoder_state = tuple(cell_state)
      else:
        cell_state = self.bridge(initial_state, decoder_state.cell_state)
        decoder_state = decoder_state.clone(cell_state=cell_state)
    return decoder_state

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None):
    outputs, state = self.cell(inputs, state, self.attention_mechanism_1, self.attention_mechanism_2,  training=training)
    outputs = common.dropout(outputs, self.dropout, training=training)
    if self.first_layer_attention:
      attention = state[0].alignments
    else:
      attention = state.alignments
    return outputs, state, attention