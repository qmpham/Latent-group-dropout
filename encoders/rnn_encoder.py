import sys
sys.path.append("/gpfsdswork/projects/rech/sfz/utt84zy/anaconda3/envs/huggingface/lib/python3.7/site-packages")

from opennmt.encoders.rnn_encoder import _RNNEncoderBase
import tensorflow as tf
import tensorflow_addons as tfa
from opennmt.encoders.encoder import Encoder, SequentialEncoder
from opennmt.layers.reducer import ConcatReducer, JoinReducer, pad_in_time
from opennmt.layers import common
from opennmt.layers import rnn
from layers.layers import GRU

class GRUEncoder(_RNNEncoderBase):

  def __init__(self,
               num_layers,
               num_units,
               bidirectional=False,
               residual_connections=False,
               dropout=0.3,
               reducer=ConcatReducer(),
               **kwargs):
    
    lstm_layer = GRU(
        num_layers,
        num_units,
        bidirectional=bidirectional,
        reducer=reducer,
        dropout=dropout,
        residual_connections=residual_connections)
    super(GRUEncoder, self).__init__(lstm_layer, **kwargs)