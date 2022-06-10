import sys
from numpy import dtype
from opennmt.inputters import inputter
sys.path.append("/gpfsdswork/projects/rech/sfz/utt84zy/anaconda3/envs/huggingface/lib/python3.7/site-packages")

from opennmt.inputters.text_inputter import WordEmbedder, _get_field, TextInputter
from opennmt.inputters.inputter import ParallelInputter, Inputter
import tensorflow as tf
from opennmt import inputters
from opennmt.models.sequence_to_sequence import _shift_target_sequence
from opennmt.data import text
from opennmt.data import dataset as dataset_util
from opennmt.utils import misc
from utils.utils_ import make_domain_mask
from opennmt.inputters.text_inputter import load_pretrained_embeddings
from opennmt.layers import common
from opennmt import constants, tokenizers
from utils.utils_ import make_domain_mask

class My_inputter(TextInputter):
    def __init__(self, embedding_size=None, dropout=0.0, **kwargs):        
        super(My_inputter, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.embedding_file = None
        self.dropout = dropout

    def forward_fn(self, features, args_dict, training=None):
        embedding = args_dict[self.embedding.name]
        #print("where are we? ________________",embedding)
        outputs = tf.nn.embedding_lookup(embedding, features["ids"])
        outputs = common.dropout(outputs, self.dropout, training=training)
        return outputs

    def initialize(self, data_config, asset_prefix=""):
        super(My_inputter, self).initialize(data_config, asset_prefix=asset_prefix)
        embedding = _get_field(data_config, "embedding", prefix=asset_prefix)
        if embedding is None and self.embedding_size is None:
            raise ValueError("embedding_size must be set")
        if embedding is not None:
            self.embedding_file = embedding["path"]
            self.trainable = embedding.get("trainable", True)
            self.embedding_file_with_header = embedding.get("with_header", True)
            self.case_insensitive_embeddings = embedding.get("case_insensitive", True)

    def build(self, input_shape):
        if self.embedding_file:
            pretrained = load_pretrained_embeddings(
                self.embedding_file,
                self.vocabulary_file,
                num_oov_buckets=self.num_oov_buckets,
                with_header=self.embedding_file_with_header,
                case_insensitive_embeddings=self.case_insensitive_embeddings)
            self.embedding_size = pretrained.shape[-1]
            initializer = tf.constant_initializer(value=pretrained.astype(self.dtype))
        else:
            initializer = None
            scope_name = self.name_scope()
            self.embedding = self.add_weight(
                "%s_embedding"%scope_name,
                [self.vocabulary_size, self.embedding_size],
                initializer=initializer,
                trainable=self.trainable)
        super(My_inputter, self).build(input_shape)

    def call(self, features, training=None):
      outputs = tf.nn.embedding_lookup(self.embedding, features["ids"])
      outputs = common.dropout(outputs, self.dropout, training=training)
      return outputs

    def make_features(self, element=None, features=None, domain=1, is_noisy=1, training=None):
        features = super(My_inputter, self).make_features(
            element=element, features=features, training=training)

        if "ids" in features and "domain" in features:
          return features

        features["ids"] = self.tokens_to_ids.lookup(features["tokens"])
        features["domain"] = tf.constant(domain)
        features["is_noisy"] = tf.constant(is_noisy)
        return features
    
    def make_inference_dataset(self,
                             feature_file,
                             batch_size,
                             domain=1,
                             is_noisy=1,
                             length_bucket_width=None,
                             num_threads=1,
                             prefetch_buffer_size=None):
    
        map_func = lambda *arg: self.make_features(misc.item_or_tuple(arg), domain=domain, is_noisy=is_noisy, training=False)
        dataset = self.make_dataset(feature_file, training=False)
        dataset = dataset.apply(dataset_util.inference_pipeline(
            batch_size,
            process_fn=map_func,
            length_bucket_width=length_bucket_width,
            length_fn=self.get_length,
            num_threads=num_threads,
            prefetch_buffer_size=prefetch_buffer_size))
        return dataset
    
    def make_training_dataset(self,
                            features_file,
                            labels_file,
                            batch_size,
                            domain=1,
                            is_noisy=1,
                            batch_type="tokens",
                            batch_multiplier=1,
                            batch_size_multiple=1,
                            shuffle_buffer_size=None,
                            length_bucket_width=None,
                            maximum_features_length=None,
                            maximum_labels_length=None,
                            single_pass=False,
                            num_shards=1,
                            shard_index=0,
                            num_threads=4,
                            prefetch_buffer_size=None):
        """See :meth:`opennmt.inputters.ExampleInputter.make_training_dataset`."""
        _ = labels_file
        dataset = self.make_dataset(features_file, training=True)
        map_func = lambda *arg: self.make_features(misc.item_or_tuple(arg), domain=domain, is_noisy=is_noisy, training=True)
        print("batch_type", batch_type)
        dataset = dataset.apply(dataset_util.training_pipeline(
            batch_size,
            batch_type=batch_type,
            batch_multiplier=batch_multiplier,
            length_bucket_width=length_bucket_width,
            single_pass=single_pass,
            process_fn=map_func,
            num_threads=num_threads,
            shuffle_buffer_size=shuffle_buffer_size,
            prefetch_buffer_size=prefetch_buffer_size,
            maximum_features_length=maximum_features_length,
            maximum_labels_length=maximum_labels_length,
            features_length_fn=self.get_length,
            batch_size_multiple=batch_size_multiple,
            num_shards=num_shards,
            shard_index=shard_index))
        return dataset
    
    def make_evaluation_dataset(self,
                              features_file,
                              labels_file,
                              batch_size,
                              domain=1,
                              is_noisy=1,
                              num_threads=1,
                              prefetch_buffer_size=None):
        """See :meth:`opennmt.inputters.ExampleInputter.make_evaluation_dataset`."""
        _ = labels_file
        dataset = self.make_dataset(features_file, training=False)
        map_func = lambda *arg: self.make_features(misc.item_or_tuple(arg), domain=domain, is_noisy=is_noisy, training=False)
        dataset = dataset.apply(dataset_util.inference_pipeline(
            batch_size,
            process_fn=map_func,
            num_threads=num_threads,
            prefetch_buffer_size=prefetch_buffer_size))
        return dataset

class LDR_inputter(WordEmbedder):
    def __init__(self, embedding_size=None, num_units=512 , num_domains=6, num_domain_units=8, dropout=0.0, **kwargs):        
        super(LDR_inputter, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.embedding_file = None
        self.dropout = dropout
        self.fusion_layer = tf.keras.layers.Dense(num_units, use_bias=False)
        self.num_domain_units = num_domain_units
        self.num_domains = num_domains
        self.mask = make_domain_mask(self.num_domains,  num_units=num_units, num_domain_units=num_domain_units)

    def initialize(self, data_config, asset_prefix=""):
        super(LDR_inputter, self).initialize(data_config, asset_prefix=asset_prefix)
        embedding = _get_field(data_config, "embedding", prefix=asset_prefix)
        if embedding is None and self.embedding_size is None:
            raise ValueError("embedding_size must be set")
        if embedding is not None:
            self.embedding_file = embedding["path"]
            self.trainable = embedding.get("trainable", True)
            self.embedding_file_with_header = embedding.get("with_header", True)
            self.case_insensitive_embeddings = embedding.get("case_insensitive", True)
    
    def make_features(self, element=None, features=None, domain=1, training=None):
        features = super(LDR_inputter, self).make_features(
            element=element, features=features, training=training)
        if "domain" in features:
            return features
        features["domain"] = tf.constant(domain)

        return features

    def call(self, features, domain=None, training=None):
        outputs = tf.nn.embedding_lookup(self.embedding, features["ids"])
        outputs = common.dropout(outputs, self.dropout, training=training)
                
        if domain==None:
            domain = features["domain"][0]
            #ldr_inputs = tf.nn.embedding_lookup(self.ldr_embed, features["ids"])
            #ldr_inputs = ldr_inputs #ldr_inputs[:,:,self.num_domain_units * domain : self.num_domain_units * (domain+1)]
            #outputs = tf.concat([outputs, ldr_inputs],-1)
            #outputs = tf.reshape(outputs, [tf.shape(outputs)[0], tf.shape(outputs)[1], 512])
            mask = tf.nn.embedding_lookup(self.mask, domain)
        else:
            #ldr_inputs = tf.nn.embedding_lookup(self.ldr_embed, features["ids"])
            #ldr_inputs = ldr_inputs[:,self.num_domain_units * domain : self.num_domain_units * (domain+1)]
            #outputs = tf.concat([outputs, ldr_inputs],-1)
            mask = tf.nn.embedding_lookup(self.mask, domain)
            #outputs = tf.reshape(outputs, [-1, 512])
        outputs = tf.math.multiply(outputs, mask)
        #tf.print("output shape: ", tf.shape(outputs))
        return outputs
    
    def build(self, input_shape):
        self.ldr_embed = self.add_weight(
                                "domain_embedding",
                                [self.vocabulary_size, self.num_domain_units * self.num_domains],
                                initializer=None,
                                trainable=True)
        super(LDR_inputter, self).build(input_shape)

    def make_inference_dataset(self,
                             feature_file,
                             batch_size,
                             domain=1,
                             is_noisy=False,
                             length_bucket_width=None,
                             num_threads=1,
                             prefetch_buffer_size=None):
    
        map_func = lambda *arg: self.make_features(misc.item_or_tuple(arg), domain=domain, training=False)
        dataset = self.make_dataset(feature_file, training=False)
        dataset = dataset.apply(dataset_util.inference_pipeline(
            batch_size,
            process_fn=map_func,
            length_bucket_width=length_bucket_width,
            length_fn=self.get_length,
            num_threads=num_threads,
            prefetch_buffer_size=prefetch_buffer_size))
        return dataset

class LLR_inputter(WordEmbedder):
    def __init__(self, embedding_size=None, num_units=512 , num_src_langues=6, num_lang_units=8, dropout=0.0, **kwargs):        
        super(LLR_inputter, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.embedding_file = None
        self.dropout = dropout
        self.fusion_layer = tf.keras.layers.Dense(num_units, use_bias=False)
        self.num_lang_units = num_lang_units
        self.num_src_langues = num_src_langues

    def initialize(self, data_config, asset_prefix=""):
        super(LLR_inputter, self).initialize(data_config, asset_prefix=asset_prefix)
        embedding = _get_field(data_config, "embedding", prefix=asset_prefix)
        if embedding is None and self.embedding_size is None:
            raise ValueError("embedding_size must be set")
        if embedding is not None:
            self.embedding_file = embedding["path"]
            self.trainable = embedding.get("trainable", True)
            self.embedding_file_with_header = embedding.get("with_header", True)
            self.case_insensitive_embeddings = embedding.get("case_insensitive", True)
    
    def make_features(self, element=None, features=None, lang=1, training=None):
        features = super(LLR_inputter, self).make_features(
            element=element, features=features, training=training)
        if "lang" in features:
            return features
        features["lang"] = tf.constant(lang)

        return features

    def call(self, features, lang=None, training=None):
        outputs = tf.nn.embedding_lookup(self.embedding, features["ids"])
        outputs = common.dropout(outputs, self.dropout, training=training)
                
        if lang==None:
            lang = features["lang"][0]
            llr_inputs = tf.nn.embedding_lookup(self.llr_embed, features["ids"])
            llr_inputs = llr_inputs[:,:,self.num_lang_units * lang : self.num_lang_units * (lang+1)]
            outputs = tf.concat([outputs, llr_inputs],-1)
            outputs = tf.reshape(outputs, [tf.shape(outputs)[0], tf.shape(outputs)[1], 512])
        else:
            llr_inputs = tf.nn.embedding_lookup(self.llr_embed, features["ids"])
            llr_inputs = llr_inputs[:,self.num_lang_units * lang : self.num_lang_units * (lang+1)]
            outputs = tf.concat([outputs, llr_inputs],-1)
            outputs = tf.reshape(outputs, [-1, 512])
        
        return outputs
    
    def build(self, input_shape):
        self.llr_embed = self.add_weight(
                                "language_embedding",
                                [self.vocabulary_size, self.num_lang_units * self.num_src_langues],
                                initializer=None,
                                trainable=True)
        super(LDR_inputter, self).build(input_shape)

    def make_inference_dataset(self,
                             feature_file,
                             batch_size,
                             domain=1,
                             is_noisy=False,
                             length_bucket_width=None,
                             num_threads=1,
                             prefetch_buffer_size=None):
    
        map_func = lambda *arg: self.make_features(misc.item_or_tuple(arg), domain=domain, training=False)
        dataset = self.make_dataset(feature_file, training=False)
        dataset = dataset.apply(dataset_util.inference_pipeline(
            batch_size,
            process_fn=map_func,
            length_bucket_width=length_bucket_width,
            length_fn=self.get_length,
            num_threads=num_threads,
            prefetch_buffer_size=prefetch_buffer_size))
        return dataset

class DC_inputter(WordEmbedder):
    def __init__(self, embedding_size=None, num_units=512 , num_domains=6, num_domain_units=8, dropout=0.0, **kwargs):        
        super(DC_inputter, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.embedding_file = None
        self.dropout = dropout
        self.num_domain_units = num_domain_units
        self.num_domains = num_domains

    def initialize(self, data_config, asset_prefix=""):
        super(DC_inputter, self).initialize(data_config, asset_prefix=asset_prefix)
        embedding = _get_field(data_config, "embedding", prefix=asset_prefix)
        if embedding is None and self.embedding_size is None:
            raise ValueError("embedding_size must be set")
        if embedding is not None:
            self.embedding_file = embedding["path"]
            self.trainable = embedding.get("trainable", True)
            self.embedding_file_with_header = embedding.get("with_header", True)
            self.case_insensitive_embeddings = embedding.get("case_insensitive", True)
    
    def make_features(self, element=None, features=None, domain=1, training=None):
        features = super(DC_inputter, self).make_features(
            element=element, features=features, training=training)
        if "domain" in features:
            return features
        features["domain"] = tf.constant(domain)

        return features
    
    def call(self, features, domain=None, training=None):
        outputs = tf.nn.embedding_lookup(self.embedding, features["ids"])
        outputs = common.dropout(outputs, self.dropout, training=training)
        if domain==None:
            ldr_inputs = tf.nn.embedding_lookup(self.ldr_embed, features["domain"])
            ldr_inputs = tf.tile(tf.expand_dims(ldr_inputs,1), (1,tf.shape(outputs)[1],1))
        else:
            ldr_inputs = tf.nn.embedding_lookup(self.ldr_embed, domain)
            ldr_inputs = tf.tile(tf.expand_dims(ldr_inputs,0), (tf.shape(outputs)[0],1))
        outputs = tf.concat([outputs, ldr_inputs],-1)
        return outputs
    
    def build(self, input_shape):
        self.ldr_embed = self.add_weight(
                                "domain_embedding",
                                [self.num_domains, self.num_domain_units],
                                initializer=None,
                                trainable=True)
        super(DC_inputter, self).build(input_shape)
    
    def make_inference_dataset(self,
                             feature_file,
                             batch_size,
                             domain=1,
                             is_noisy=False,
                             length_bucket_width=None,
                             num_threads=1,
                             prefetch_buffer_size=None):
    
        map_func = lambda *arg: self.make_features(misc.item_or_tuple(arg), domain=domain, training=False)
        dataset = self.make_dataset(feature_file, training=False)
        dataset = dataset.apply(dataset_util.inference_pipeline(
            batch_size,
            process_fn=map_func,
            length_bucket_width=length_bucket_width,
            length_fn=self.get_length,
            num_threads=num_threads,
            prefetch_buffer_size=prefetch_buffer_size))
        return dataset

class Multi_domain_SequenceToSequenceInputter(inputters.ExampleInputter):
    def __init__(self,
               features_inputter,
               labels_inputter,
               share_parameters=False):
        super(Multi_domain_SequenceToSequenceInputter, self).__init__(
            features_inputter, labels_inputter, share_parameters=share_parameters)
        self.alignment_file = None

    def initialize(self, data_config, asset_prefix=""):
        super(Multi_domain_SequenceToSequenceInputter, self).initialize(data_config, asset_prefix=asset_prefix)
        self.alignment_file = data_config.get("train_alignments")

    def make_dataset(self, data_file, training=None):
        dataset = super(Multi_domain_SequenceToSequenceInputter, self).make_dataset(
        data_file, training=training)
        if self.alignment_file is None or not training:
            return dataset
        return tf.data.Dataset.zip((dataset, tf.data.TextLineDataset(self.alignment_file)))

    def make_features(self, element=None, features=None, domain=1, is_noisy=1, training=None):
        if training and self.alignment_file is not None:
            element, alignment = element
        else:
            alignment = None
        features, labels = super(Multi_domain_SequenceToSequenceInputter, self).make_features(
            element=element, features=features, training=training)
        if alignment is not None:
            labels["alignment"] = text.alignment_matrix_from_pharaoh(
                alignment,
                self.features_inputter.get_length(features),
                self.labels_inputter.get_length(labels))
        _shift_target_sequence(labels)
        if "noisy_ids" in labels:
            _shift_target_sequence(labels, prefix="noisy_")
        features["domain"] = tf.constant(domain)
        labels["domain"] = tf.constant(domain)
        print("make features noisy: ", is_noisy)
        features["is_noisy"] = tf.constant(is_noisy)
        labels["is_noisy"] = tf.constant(is_noisy)
        return features, labels

    def make_inference_dataset(self,
                             features_file,
                             batch_size,
                             domain,
                             is_noisy=1,
                             length_bucket_width=None,
                             num_threads=1,
                             prefetch_buffer_size=None):
        return self.features_inputter.make_inference_dataset(
            features_file,
            batch_size,
            domain=domain,
            is_noisy=is_noisy,
            length_bucket_width=length_bucket_width,
            num_threads=num_threads,
            prefetch_buffer_size=prefetch_buffer_size)

    def make_evaluation_dataset(self,
                                features_file,
                                labels_file,
                                batch_size,
                                domain,
                                is_noisy=1,
                                num_threads=1,
                                prefetch_buffer_size=None):
        
        map_func = lambda *arg: self.make_features(arg, domain=domain, is_noisy=is_noisy, training=False)
        dataset = self.make_dataset([features_file, labels_file], training=False)
        dataset = dataset.apply(dataset_util.inference_pipeline(
            batch_size,
            process_fn=map_func,
            num_threads=num_threads,
            prefetch_buffer_size=prefetch_buffer_size))
        return dataset

    def make_training_dataset(self,
                                features_file,
                                labels_file,
                                batch_size,
                                domain=0,
                                is_noisy=1,
                                batch_type="examples",
                                batch_multiplier=1,
                                batch_size_multiple=1,
                                shuffle_buffer_size=None,
                                length_bucket_width=None,
                                maximum_features_length=None,
                                maximum_labels_length=None,
                                single_pass=False,
                                num_shards=1,
                                shard_index=0,
                                num_threads=4,
                                prefetch_buffer_size=None):
        
        map_func = lambda *arg: self.make_features(arg, domain=domain, is_noisy=is_noisy, training=True)
        dataset = self.make_dataset([features_file, labels_file], training=True)
        dataset = dataset.apply(dataset_util.training_pipeline(
            batch_size,
            batch_type=batch_type,
            batch_multiplier=batch_multiplier,
            batch_size_multiple=batch_size_multiple,
            process_fn=map_func,
            length_bucket_width=length_bucket_width,
            features_length_fn=self.features_inputter.get_length,
            labels_length_fn=self.labels_inputter.get_length,
            maximum_features_length=maximum_features_length,
            maximum_labels_length=maximum_labels_length,
            single_pass=single_pass,
            num_shards=num_shards,
            shard_index=shard_index,
            num_threads=num_threads,
            shuffle_buffer_size=shuffle_buffer_size,
            prefetch_buffer_size=prefetch_buffer_size))
        return dataset

class Multi_domain_SequenceToSequenceInputter_withprob(ParallelInputter):
    
    def __init__(self,
               features_inputter,
               labels_inputter,
               probs_inputter,
               share_parameters=False):
        self.features_inputter = features_inputter
        self.labels_inputter = labels_inputter
        self.probs_inputter = probs_inputter
        self.features_inputter.asset_prefix = "source"
        self.labels_inputter.asset_prefix = "target"
        self.probs_inputter.asset_prefix = "source"
        super(Multi_domain_SequenceToSequenceInputter_withprob, self).__init__(
            [features_inputter, labels_inputter, probs_inputter], share_parameters=share_parameters, combine_features=False)
    
    def initialize(self, data_config, asset_prefix=""):
        super(Multi_domain_SequenceToSequenceInputter_withprob, self).initialize(data_config, asset_prefix=asset_prefix)

    def make_dataset(self, data_file, training=None):
        dataset = super(Multi_domain_SequenceToSequenceInputter_withprob, self).make_dataset(
        data_file, training=training)
        return dataset

    def make_features(self, element=None, features=None, training=None):
        features, labels, probs = super(Multi_domain_SequenceToSequenceInputter_withprob, self).make_features(
            element=element, features=features, training=training)
        _shift_target_sequence(labels)
        features["domain"] = tf.math.softmax(probs["probs"])
        labels["domain"] = tf.math.softmax(probs["probs"])
        
        return features, labels   

    def make_inference_dataset(self,
                             features_file,
                             probs_file,
                             batch_size,
                             length_bucket_width=None,
                             num_threads=1,
                             prefetch_buffer_size=None):
        def add_prob(f,p):
            feats = self.features_inputter.make_features(f)
            probs = self.probs_inputter.make_features(p)
            feats["domain"] = tf.math.softmax(probs["probs"])
            return feats
        
        #dataset = self.make_dataset([features_file, probs_file], training=True)
        datasets = [self.features_inputter.make_dataset(features_file), self.probs_inputter.make_dataset(probs_file)]
        dataset = tf.data.Dataset.zip(tuple(datasets))
        dataset = dataset.apply(dataset_util.inference_pipeline(
                       batch_size,
                       process_fn=add_prob,
                       num_threads=num_threads))
        return dataset        

    def make_training_dataset(self,
                                features_file,
                                labels_file,
                                probs_file,
                                batch_size,
                                batch_type="examples",
                                batch_multiplier=1,
                                batch_size_multiple=1,
                                shuffle_buffer_size=None,
                                length_bucket_width=None,
                                maximum_features_length=None,
                                maximum_labels_length=None,
                                single_pass=False,
                                num_shards=1,
                                shard_index=0,
                                num_threads=4,
                                prefetch_buffer_size=None):
        
        map_func = lambda *arg: self.make_features(arg, training=True)
        dataset = self.make_dataset([features_file, labels_file, probs_file], training=True)
        dataset = dataset.apply(dataset_util.training_pipeline(
            batch_size,
            batch_type=batch_type,
            batch_multiplier=batch_multiplier,
            batch_size_multiple=batch_size_multiple,
            process_fn=map_func,
            length_bucket_width=length_bucket_width,
            features_length_fn=self.features_inputter.get_length,
            labels_length_fn=self.labels_inputter.get_length,
            maximum_features_length=maximum_features_length,
            maximum_labels_length=maximum_labels_length,
            single_pass=single_pass,
            num_shards=num_shards,
            shard_index=shard_index,
            num_threads=num_threads,
            shuffle_buffer_size=shuffle_buffer_size,
            prefetch_buffer_size=prefetch_buffer_size))
        return dataset

class ProbInputter(Inputter):
  """An abstract inputter that processes text."""

  def __init__(self, domain_numb, **kwargs):
    super(ProbInputter, self).__init__(**kwargs)
    self.domain_numb = domain_numb
  def initialize(self, data_config, asset_prefix=""):
    self.tokenizer = tokenizers.make_tokenizer(None)

  def make_dataset(self, data_file, training=None):
    return tf.data.TextLineDataset(
        data_file, compression_type="GZIP" if misc.is_gzip_file(data_file) else None)

  def make_features(self, element=None, features=None, training=None):
    """Tokenizes raw text."""
    if features is None:
      features = {}
    if "probs" in features:
      return features
    if "text" in features:
      element = features.pop("text")
    tokens = self.tokenizer.tokenize(element)
    features["probs"] = tf.strings.to_number(tokens)
    return features

  def input_signature(self):
    return {
          "probs": tf.TensorSpec([None, self.domain_numb], tf.float32),
      }

class ProbInputter_v1(Inputter):
  """An abstract inputter that processes text."""

  def __init__(self, **kwargs):
    super(ProbInputter_v1, self).__init__(**kwargs)
  def initialize(self, data_config, asset_prefix=""):
    self.tokenizer = tokenizers.make_tokenizer(None)

  def make_dataset(self, data_file, training=None):
    return tf.data.TextLineDataset(
        data_file, compression_type="GZIP" if misc.is_gzip_file(data_file) else None)

  def make_features(self, element=None, features=None, training=None):
    """Tokenizes raw text."""
    if features is None:
      features = {}
    if "probs" in features:
      return features
    if "text" in features:
      element = features.pop("text")
    features["probs"] = tf.strings.to_number(element)
    return features

  def input_signature(self):
    return {
          "probs": tf.TensorSpec([None], tf.float32),
      }

class Multi_domain_SequenceToSequenceInputter_DRO(ParallelInputter):
    
    def __init__(self,
               features_inputter,
               labels_inputter,
               probs_inputter,
               share_parameters=False):
        self.features_inputter = features_inputter
        self.labels_inputter = labels_inputter
        self.probs_inputter = probs_inputter
        self.features_inputter.asset_prefix = "source"
        self.labels_inputter.asset_prefix = "target"
        self.probs_inputter.asset_prefix = "probs"
        super(Multi_domain_SequenceToSequenceInputter_DRO, self).__init__(
            [features_inputter, labels_inputter, probs_inputter], share_parameters=share_parameters, combine_features=False)
    
    def initialize(self, data_config, asset_prefix=""):
        super(Multi_domain_SequenceToSequenceInputter_DRO, self).initialize(data_config, asset_prefix=asset_prefix)

    def make_dataset(self, data_file, training=None):
        dataset = super(Multi_domain_SequenceToSequenceInputter_DRO, self).make_dataset(
        data_file, training=training)
        return dataset

    def make_features(self, element=None, features=None, domain=1, training=None):
        
        features, labels, logprob = super(Multi_domain_SequenceToSequenceInputter_DRO, self).make_features(
            element=element, features=features, training=training)
        
        _shift_target_sequence(labels)
        
        features["domain"] = tf.constant(domain)
        labels["domain"] = tf.constant(domain)
        features["logprobs"] = logprob["probs"]
        return features, labels
    def make_inference_dataset(self,
                             features_file,
                             batch_size,
                             domain,
                             length_bucket_width=None,
                             num_threads=1,
                             prefetch_buffer_size=None):
        return self.features_inputter.make_inference_dataset(
            features_file,
            batch_size,
            domain=domain,
            length_bucket_width=length_bucket_width,
            num_threads=num_threads,
            prefetch_buffer_size=prefetch_buffer_size)
    def make_training_dataset(self,
                                features_file,
                                labels_file,
                                logprobs_file,
                                batch_size,
                                domain,
                                batch_type="examples",
                                batch_multiplier=1,
                                batch_size_multiple=1,
                                shuffle_buffer_size=None,
                                length_bucket_width=None,
                                maximum_features_length=None,
                                maximum_labels_length=None,
                                single_pass=False,
                                num_shards=1,
                                shard_index=0,
                                num_threads=4,
                                prefetch_buffer_size=None):
        
        map_func = lambda *arg: self.make_features(arg, domain=domain, training=True)
        dataset = self.make_dataset([features_file, labels_file, logprobs_file], training=True)
        dataset = dataset.apply(dataset_util.training_pipeline(
            batch_size,
            batch_type=batch_type,
            batch_multiplier=batch_multiplier,
            batch_size_multiple=batch_size_multiple,
            process_fn=map_func,
            length_bucket_width=length_bucket_width,
            features_length_fn=self.features_inputter.get_length,
            labels_length_fn=self.labels_inputter.get_length,
            maximum_features_length=maximum_features_length,
            maximum_labels_length=maximum_labels_length,
            single_pass=single_pass,
            num_shards=num_shards,
            shard_index=shard_index,
            num_threads=num_threads,
            shuffle_buffer_size=shuffle_buffer_size,
            prefetch_buffer_size=prefetch_buffer_size))
        return dataset

class Priming_SequenceToSequenceInputter(inputters.ExampleInputter):
  """A custom :class:`opennmt.inputters.ExampleInputter` for sequence to
  sequence models.
  """

  def __init__(self,
               features_inputter,
               labels_inputter,
               share_parameters=False):
    super(Priming_SequenceToSequenceInputter, self).__init__(
        features_inputter, labels_inputter, share_parameters=share_parameters)

  def initialize(self, data_config, asset_prefix=""):
    super(Priming_SequenceToSequenceInputter, self).initialize(data_config, asset_prefix=asset_prefix)

  def make_dataset(self, data_file, training=None):
    dataset = super(Priming_SequenceToSequenceInputter, self).make_dataset(
        data_file, training=training)
    label_dataset = self.labels_inputter.make_dataset(data_file[1])
    feature_dataset = self.features_inputter.make_dataset(data_file[0])
    return tf.data.Dataset.zip(tuple([feature_dataset,label_dataset]))

  def make_features(self, element=None, features=None, training=None):
    #src = self.features_inputter.make_features(element=element[0],training=training)
    #labels = self.labels_inputter.make_features(element=element[1],training=training)
    temp = [None]*len(self.inputters)
    
    for i, inputter in enumerate(self.inputters):        
        temp[i] = inputter.make_features(element=element[i] if element is not None else None, 
                                        features=features[i] if features is not None else None, training=training)
    src, labels = temp
    _shift_target_sequence(labels)
    if "noisy_ids" in labels:
      _shift_target_sequence(labels, prefix="noisy_")
    return src, labels

class Priming_SequenceToSequenceInputter_adv(inputters.ExampleInputter):
  """A custom :class:`opennmt.inputters.ExampleInputter` for sequence to
  sequence models.
  """

  def __init__(self,
               features_inputter,
               labels_inputter,
               share_parameters=False):
    super(Priming_SequenceToSequenceInputter, self).__init__(
        features_inputter, labels_inputter, share_parameters=share_parameters)

  def initialize(self, data_config, asset_prefix=""):
    super(Priming_SequenceToSequenceInputter, self).initialize(data_config, asset_prefix=asset_prefix)

  def make_dataset(self, data_file, training=None):
    dataset = super(Priming_SequenceToSequenceInputter, self).make_dataset(
        data_file, training=training)
    label_dataset = self.labels_inputter.make_dataset(data_file[1])
    feature_dataset = self.features_inputter.make_dataset(data_file[0])
    return tf.data.Dataset.zip(tuple([feature_dataset,label_dataset]))

  def make_features(self, element=None, features=None, training=None):
    #src = self.features_inputter.make_features(element=element[0],training=training)
    #labels = self.labels_inputter.make_features(element=element[1],training=training)
    temp = [None]*len(self.inputters)
    
    for i, inputter in enumerate(self.inputters):        
        temp[i] = inputter.make_features(element=element[i] if element is not None else None, 
                                        features=features[i] if features is not None else None, training=training)
    src, labels = temp
    for sub_label in labels:
        _shift_target_sequence(sub_label)
    if "noisy_ids" in labels:
      _shift_target_sequence(labels, prefix="noisy_")
    return src, labels

class My_multilingual_inputter(TextInputter):
    def __init__(self, embedding_size=None, dropout=0.0, **kwargs):        
        super(My_multilingual_inputter, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.embedding_file = None
        self.dropout = dropout

    def initialize(self, data_config, asset_prefix=""):
        super(My_multilingual_inputter, self).initialize(data_config, asset_prefix=asset_prefix)
        embedding = _get_field(data_config, "embedding", prefix=asset_prefix)
        if embedding is None and self.embedding_size is None:
            raise ValueError("embedding_size must be set")
        if embedding is not None:
            self.embedding_file = embedding["path"]
            self.trainable = embedding.get("trainable", True)
            self.embedding_file_with_header = embedding.get("with_header", True)
            self.case_insensitive_embeddings = embedding.get("case_insensitive", True)

    def build(self, input_shape):
        if self.embedding_file:
            pretrained = load_pretrained_embeddings(
                self.embedding_file,
                self.vocabulary_file,
                num_oov_buckets=self.num_oov_buckets,
                with_header=self.embedding_file_with_header,
                case_insensitive_embeddings=self.case_insensitive_embeddings)
            self.embedding_size = pretrained.shape[-1]
            initializer = tf.constant_initializer(value=pretrained.astype(self.dtype))
        else:
            initializer = None
            scope_name = self.name_scope()
            self.embedding = self.add_weight(
                "%s_embedding"%scope_name,
                [self.vocabulary_size, self.embedding_size],
                initializer=initializer,
                trainable=self.trainable)
        super(My_multilingual_inputter, self).build(input_shape)

    def call(self, features, training=None):
      outputs = tf.nn.embedding_lookup(self.embedding, features["ids"])
      outputs = common.dropout(outputs, self.dropout, training=training)
      return outputs

    def make_features(self, element=None, features=None, lang_src=1, lang_tgt=1,  is_noisy=1, training=None):
        features = super(My_inputter, self).make_features(
            element=element, features=features, training=training)

        if "ids" in features and "domain" in features:
          return features

        features["ids"] = self.tokens_to_ids.lookup(features["tokens"])
        features["lang_src"] = tf.constant(lang_src)
        features["lang_tgt"] = tf.constant(lang_tgt)
        features["is_noisy"] = tf.constant(is_noisy)
        return features
    
    def make_inference_dataset(self,
                             feature_file,
                             batch_size,
                             lang_src=1, 
                             lang_tgt=1,
                             is_noisy=1,
                             length_bucket_width=None,
                             num_threads=1,
                             prefetch_buffer_size=None):
    
        map_func = lambda *arg: self.make_features(misc.item_or_tuple(arg), lang_src=1, lang_tgt=1, is_noisy=is_noisy, training=False)
        dataset = self.make_dataset(feature_file, training=False)
        dataset = dataset.apply(dataset_util.inference_pipeline(
            batch_size,
            process_fn=map_func,
            length_bucket_width=length_bucket_width,
            length_fn=self.get_length,
            num_threads=num_threads,
            prefetch_buffer_size=prefetch_buffer_size))
        return dataset
    
    def make_training_dataset(self,
                            features_file,
                            labels_file,
                            batch_size,
                            lang_src=1, lang_tgt=1,
                            is_noisy=1,
                            batch_type="tokens",
                            batch_multiplier=1,
                            batch_size_multiple=1,
                            shuffle_buffer_size=None,
                            length_bucket_width=None,
                            maximum_features_length=None,
                            maximum_labels_length=None,
                            single_pass=False,
                            num_shards=1,
                            shard_index=0,
                            num_threads=4,
                            prefetch_buffer_size=None):
        """See :meth:`opennmt.inputters.ExampleInputter.make_training_dataset`."""
        _ = labels_file
        dataset = self.make_dataset(features_file, training=True)
        map_func = lambda *arg: self.make_features(misc.item_or_tuple(arg), lang_src=1, lang_tgt=1, is_noisy=is_noisy, training=True)
        print("batch_type", batch_type)
        dataset = dataset.apply(dataset_util.training_pipeline(
            batch_size,
            batch_type=batch_type,
            batch_multiplier=batch_multiplier,
            length_bucket_width=length_bucket_width,
            single_pass=single_pass,
            process_fn=map_func,
            num_threads=num_threads,
            shuffle_buffer_size=shuffle_buffer_size,
            prefetch_buffer_size=prefetch_buffer_size,
            maximum_features_length=maximum_features_length,
            maximum_labels_length=maximum_labels_length,
            features_length_fn=self.get_length,
            batch_size_multiple=batch_size_multiple,
            num_shards=num_shards,
            shard_index=shard_index))
        return dataset
    
    def make_evaluation_dataset(self,
                              features_file,
                              labels_file,
                              batch_size,
                              lang_src=1, lang_tgt=1,
                              is_noisy=1,
                              num_threads=1,
                              prefetch_buffer_size=None):
        """See :meth:`opennmt.inputters.ExampleInputter.make_evaluation_dataset`."""
        _ = labels_file
        dataset = self.make_dataset(features_file, training=False)
        map_func = lambda *arg: self.make_features(misc.item_or_tuple(arg), lang_src=1, lang_tgt=1, is_noisy=is_noisy, training=False)
        dataset = dataset.apply(dataset_util.inference_pipeline(
            batch_size,
            process_fn=map_func,
            num_threads=num_threads,
            prefetch_buffer_size=prefetch_buffer_size))
        return dataset























        












