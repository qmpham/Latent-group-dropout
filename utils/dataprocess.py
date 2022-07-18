"""Dataset creation and transformations."""
import sys
from typing import List

import numpy as np
import tensorflow as tf


def make_cardinality_multiple_of(divisor):
  """Transformation that ensures that the dataset cardinality is a multiple of
  :obj:`divisor`.
  Example:
    >>> dataset = tf.data.Dataset.range(7)
    >>> dataset = dataset.apply(opennmt.data.make_cardinality_multiple_of(10))
    >>> len(list(iter(dataset)))
    10
  Args:
    divisor: The value that should divide the dataset size.
  Returns:
    A ``tf.data.Dataset`` transformation.
  Tip:
    This transformation is useful when training multiple replicas on a finite
    dataset. It ensures that each replica receives a non empty batch in the last
    training iteration.
  """
  if divisor == 1:
    return lambda dataset: dataset

  def _continue_iter(num_consumed, element):
    # Continue iterating if the current element is from the original dataset or
    # if the number of consumed batches is not a multiple of divisor.
    is_original = element[0]
    return tf.math.logical_or(is_original, tf.math.not_equal(num_consumed % divisor, 0))

  def _retrieve_element(num_consumed, element):
    _ = num_consumed
    return element[1]

  def _transform(dataset):
    # Nothing to do for infinite datasets.
    if tf.data.experimental.cardinality(dataset) == tf.data.experimental.INFINITE_CARDINALITY:
      return dataset

    # Concatenate extra batches with a flag.
    extra_batches = dataset.repeat()
    dataset = dataset.map(lambda *x: (tf.constant(True), x))
    extra_batches = extra_batches.map(lambda *x: (tf.constant(False), x))
    dataset = dataset.concatenate(extra_batches)

    # Take all original batches and the number of extra batches required.
    dataset = dataset.enumerate()
    dataset = dataset.apply(tf.data.experimental.take_while(_continue_iter))
    return dataset.map(_retrieve_element)  # Retrieve the element only.

  return _transform

def random_shard(shard_size, dataset_size):
  """Transformation that shards the dataset in a random order.
  Example:
    >>> dataset = tf.data.Dataset.range(6)
    >>> dataset = dataset.apply(opennmt.data.random_shard(2, 6)
    >>> list(dataset.as_numpy_iterator())
    [0, 1, 4, 5, 2, 3]
  Args:
    shard_size: The number of examples in each shard.
    dataset_size: The total number of examples in the dataset.
  Returns:
    A ``tf.data.Dataset`` transformation.
  """
  num_shards = -(-dataset_size // shard_size)  # Ceil division.
  offsets = np.linspace(0, dataset_size, num=num_shards, endpoint=False, dtype=np.int64)

  def _random_shard(dataset):
    sharded_dataset = tf.data.Dataset.from_tensor_slices(offsets)
    sharded_dataset = sharded_dataset.shuffle(num_shards)
    sharded_dataset = sharded_dataset.flat_map(
        lambda offset: dataset.skip(offset).take(shard_size))
    return sharded_dataset

  return _random_shard

def count_lines(filename):
  """Returns the number of lines of the file :obj:`filename`."""
  with open(filename, mode="rb") as f:
    i = 0
    for i, _ in enumerate(f):
      pass
    return i + 1

def make_batch_per_replica_1_(num_replicas_in_sync):
  def fixing_shape(*args):
    src, tgt = args
    new_src = {}
    new_tgt = {}
    for feature in list(src.keys()):
      batch = src[feature]
      dim = batch.shape.ndims
      if dim==2:
        batch = tf.expand_dims(batch,0)
        new_batch = tf.reshape(batch,[-1,tf.shape(batch)[1]//num_replicas_in_sync,tf.shape(batch)[-1]])
        new_src.update({feature:new_batch})
    for feature in list(tgt.keys()):
      batch = tgt[feature]
      dim = batch.shape.ndims
      if dim==2:
        batch = tf.expand_dims(batch,0)
        new_batch = tf.reshape(batch,[-1,tf.shape(batch)[1]//num_replicas_in_sync,tf.shape(batch)[-1]])
        new_tgt.update({feature:new_batch})
    return new_src, new_tgt
  return fixing_shape

def make_batch_per_replica_(num_replicas_in_sync):
  def fixing_shape(*args):
    src, tgt = args
    new_src = {}
    new_tgt = {}
    for feature in list(src.keys()):
      batch = src[feature]
      dim = batch.shape.ndims
      if dim==1:
        batch = tf.expand_dims(batch,0)
        new_batch = tf.reshape(batch,[num_replicas_in_sync,-1])
      elif dim==2:
        batch = tf.expand_dims(batch,0)
        new_batch = tf.reshape(batch,[num_replicas_in_sync,-1,tf.shape(batch)[-1]])
      new_src.update({feature:new_batch})
    for feature in list(tgt.keys()):
      batch = tgt[feature]
      dim = batch.shape.ndims
      if dim==1:
        batch = tf.expand_dims(batch,0)
        new_batch = tf.reshape(batch,[num_replicas_in_sync,-1])
      elif dim==2:
        batch = tf.expand_dims(batch,0)
        new_batch = tf.reshape(batch,[num_replicas_in_sync,-1,tf.shape(batch)[-1]])
      new_tgt.update({feature:new_batch})
    return new_src, new_tgt
  return fixing_shape

def merge_map_fn(*args):
  
  src_batches = []
  tgt_batches = []
  for (src,tgt) in args:
    src_batches.append(src)
    tgt_batches.append(tgt)
  print("element numb: ",len(src_batches))
  src_batch = {}
  tgt_batch = {}
  #print(src_batches[0].keys())
  for feature in list(src_batches[0].keys()):
    if feature!="ids" and feature!="tokens":
      #print(feature, src_batches[0][feature])
      src_batch.update({feature: tf.concat([b[feature] for b in src_batches],0)})
    else:
      #print(feature, src_batches[0][feature])
      len_max = tf.reduce_max([tf.shape(batch[feature])[1] for batch in src_batches])
      if src_batches[0][feature].dtype == tf.string:
        src_batch.update({feature: tf.concat([tf.concat([batch[feature], tf.fill([tf.shape(batch[feature])[0], 
                                              len_max-tf.shape(batch[feature])[1]],"")],1) for batch in src_batches],0)})
      else:
        src_batch.update({feature: tf.concat([tf.concat([batch[feature], tf.cast(tf.fill([tf.shape(batch[feature])[0], 
                                              len_max-tf.shape(batch[feature])[1]],0),tf.int64)],1) for batch in src_batches],0)})
    
  for feature in list(tgt_batches[0].keys()):
    if feature!="ids" and feature!="tokens" and feature!="ids_out":
      #print(feature, tgt_batches[0][feature])
      tgt_batch.update({feature: tf.concat([b[feature] for b in tgt_batches],0)})    
    else:
      #print(feature, tgt_batches[0][feature])
      len_max = tf.reduce_max([tf.shape(batch[feature])[1] for batch in tgt_batches])
      if tgt_batches[0][feature].dtype == tf.string:
        tgt_batch.update({feature: tf.concat([tf.concat([batch[feature], tf.fill([tf.shape(batch[feature])[0], 
                                              len_max-tf.shape(batch[feature])[1]],"")],1) for batch in tgt_batches],0)})
      else:
        tgt_batch.update({feature: tf.concat([tf.concat([batch[feature], tf.cast(tf.fill([tf.shape(batch[feature])[0], 
                                              len_max-tf.shape(batch[feature])[1]],0),tf.int64)],1) for batch in tgt_batches],0)})
  #print(src_batch,tgt_batch)
  return src_batch, tgt_batch

def ragged_map(*args):
  src, tgt = args  
  src_batch = {}
  tgt_batch = {}
  for feature in list(src.keys()):
    src_batch.update({feature: tf.RaggedTensor.from_tensor(tf.expand_dims(src[feature],0))})
    
  for feature in list(tgt.keys()):
    tgt_batch.update({feature: tf.RaggedTensor.from_tensor(tf.expand_dims(tgt[feature],0))})

  return src_batch, tgt_batch

def create_multi_domain_meta_training_dataset(strategy, model, domain, source_file, target_file, batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length, meta_train_picking_prob=None, meta_test_picking_prob=None):
  meta_train_datasets = [] 
  meta_test_datasets = [] 
  print("batch_type: ", batch_type)
  for i, src,tgt in zip(domain,source_file,target_file):
    meta_train_datasets.append(model.examples_inputter.make_training_dataset(src, tgt,
              batch_size=batch_meta_train_size,
              batch_type=batch_type,
              batch_multiplier=1,
              domain=i,
              shuffle_buffer_size=shuffle_buffer_size,
              length_bucket_width=1,  # Bucketize sequences by the same length for efficiency.
              maximum_features_length=maximum_length,
              maximum_labels_length=maximum_length))

    meta_test_datasets.append(model.examples_inputter.make_training_dataset(src, tgt,
              batch_size= batch_meta_test_size,
              batch_type=batch_type,
              batch_multiplier=1,
              domain=i,
              shuffle_buffer_size=shuffle_buffer_size,
              length_bucket_width=1,  # Bucketize sequences by the same length for efficiency.
              maximum_features_length=maximum_length,
              maximum_labels_length=maximum_length))
  if meta_train_picking_prob=="Natural":
    datasets_size = [count_lines(src) for src in source_file]
    meta_train_picking_prob = [data_size/sum(datasets_size) for data_size in datasets_size]
    #picking_prob = [1.0,0.01,0.01,0.01,0.01,0.01]
    print("picking probability: ", meta_train_picking_prob)
  elif meta_train_picking_prob=="Anneal":
    import itertools
    datasets_size = [count_lines(src) for src in source_file]
    picking_prob_ = [data_size/sum(datasets_size) for data_size in datasets_size]
    def anneal(i, end=200000 * strategy.num_replicas_in_sync):
      i = (end-i)/end
      prob_ = [p**i  for p in picking_prob_]
      return [p/sum(prob_) for p in prob_]
    tensor = tf.Variable(np.array([anneal(i) for i in range(200000)]))
    meta_train_picking_prob = tf.data.Dataset.from_tensor_slices(tensor)
    print("picking probability: ", meta_train_picking_prob)
  else:
    print("picking probability: ", meta_train_picking_prob)

  if meta_test_picking_prob=="Natural":
    datasets_size = [count_lines(src) for src in source_file]
    meta_test_picking_prob = [data_size/sum(datasets_size) for data_size in datasets_size]
    #picking_prob = [1.0,0.01,0.01,0.01,0.01,0.01]
    print("picking probability: ", meta_test_picking_prob)
  elif meta_test_picking_prob=="Anneal":
    import itertools
    datasets_size = [count_lines(src) for src in source_file]
    picking_prob_ = [data_size/sum(datasets_size) for data_size in datasets_size]
    tensor = tf.Variable(np.array([anneal(i) for i in range(200000)]))
    meta_test_picking_prob = tf.data.Dataset.from_tensor_slices(tensor)
    print("picking probability: ", meta_test_picking_prob)
  else:
    print("picking probability: ", meta_test_picking_prob)

  meta_train_dataset = tf.data.experimental.sample_from_datasets(meta_train_datasets, weights=meta_train_picking_prob) #tf.data.Dataset.zip(tuple(meta_train_datasets)).map(merge_map_fn) #tf.data.experimental.sample_from_datasets(meta_train_datasets)
  meta_test_dataset = tf.data.experimental.sample_from_datasets(meta_test_datasets, weights=meta_test_picking_prob) #tf.data.Dataset.zip(tuple(meta_test_datasets)).map(merge_map_fn)
  
  with strategy.scope():    
    meta_train_base_dataset = meta_train_dataset      
    meta_train_dataset = strategy.experimental_distribute_datasets_from_function(
          lambda _: meta_train_base_dataset)
  with strategy.scope():
    meta_test_base_dataset = meta_test_dataset      
    meta_test_dataset = strategy.experimental_distribute_datasets_from_function(
          lambda _: meta_test_base_dataset)
  
  return meta_train_dataset, meta_test_dataset

def create_multi_domain_meta_training_dataset_v1(strategy, model, domain, source_file, target_file, batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length):
  meta_train_datasets = [] 
  meta_test_datasets = [] 
  print("batch_type: ", batch_type)
  datasets_size = [count_lines(src) for src in source_file]
  datasets_numb = len(source_file)
  batch_size_ratios = [data_size/sum(datasets_size) for data_size in datasets_size]
  meta_train_batches_size = [round(batch_meta_train_size * datasets_numb * ratio) for ratio in batch_size_ratios]
  meta_test_batches_size = [round(batch_meta_test_size * datasets_numb * ratio) for ratio in batch_size_ratios]
  print("meta_train_batches_size per domain: ", meta_train_batches_size)
  print("meta_test_batches_size per domain: ", meta_test_batches_size)
  for i, src,tgt in zip(domain,source_file,target_file):
    meta_train_datasets.append(model.examples_inputter.make_training_dataset(src, tgt,
              batch_size=meta_train_batches_size[i],
              batch_type=batch_type,
              batch_multiplier=1,
              domain=i,
              shuffle_buffer_size=shuffle_buffer_size,
              length_bucket_width=1,  # Bucketize sequences by the same length for efficiency.
              maximum_features_length=maximum_length,
              maximum_labels_length=maximum_length))

    meta_test_datasets.append(model.examples_inputter.make_training_dataset(src, tgt,
              batch_size= meta_test_batches_size[i],
              batch_type=batch_type,
              batch_multiplier=1,
              domain=i,
              shuffle_buffer_size=shuffle_buffer_size,
              length_bucket_width=1,  # Bucketize sequences by the same length for efficiency.
              maximum_features_length=maximum_length,
              maximum_labels_length=maximum_length))
  
  meta_train_dataset = tf.data.experimental.sample_from_datasets(meta_train_datasets) #tf.data.Dataset.zip(tuple(meta_train_datasets)).map(merge_map_fn) #tf.data.experimental.sample_from_datasets(meta_train_datasets)
  meta_test_dataset = tf.data.experimental.sample_from_datasets(meta_test_datasets) #tf.data.Dataset.zip(tuple(meta_test_datasets)).map(merge_map_fn)
  
  with strategy.scope():    
    meta_train_base_dataset = meta_train_dataset      
    meta_train_dataset = strategy.experimental_distribute_datasets_from_function(
          lambda _: meta_train_base_dataset)
  with strategy.scope():
    meta_test_base_dataset = meta_test_dataset      
    meta_test_dataset = strategy.experimental_distribute_datasets_from_function(
          lambda _: meta_test_base_dataset)
  
  return meta_train_dataset, meta_test_dataset

def create_meta_training_dataset(strategy, model, domain, source_file, target_file, batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length):
  meta_train_datasets = [] 
  meta_test_datasets = [] 
  for i, src,tgt in zip(domain,source_file,target_file):
    meta_train_datasets.append(model.examples_inputter.make_training_dataset(src, tgt,
              batch_size=batch_meta_train_size,
              batch_type=batch_type,              
              shuffle_buffer_size=shuffle_buffer_size,
              length_bucket_width=1,  # Bucketize sequences by the same length for efficiency.
              maximum_features_length=maximum_length,
              maximum_labels_length=maximum_length))

    meta_test_datasets.append(model.examples_inputter.make_training_dataset(src, tgt,
              batch_size= batch_meta_test_size,
              batch_type=batch_type,
              shuffle_buffer_size=shuffle_buffer_size,
              length_bucket_width=1,  # Bucketize sequences by the same length for efficiency.
              maximum_features_length=maximum_length,
              maximum_labels_length=maximum_length))
  
  meta_train_dataset = tf.data.Dataset.zip(tuple(meta_train_datasets)).map(merge_map_fn) #tf.data.experimental.sample_from_datasets(meta_train_datasets)
  meta_test_dataset = tf.data.Dataset.zip(tuple(meta_test_datasets)).map(merge_map_fn)
  with strategy.scope():
    base_dataset = meta_train_dataset      
    meta_train_dataset = strategy.experimental_distribute_datasets_from_function(
          lambda _: base_dataset)
    base_dataset = meta_test_dataset      
    meta_test_dataset = strategy.experimental_distribute_datasets_from_function(
          lambda _: base_dataset)

  return meta_train_dataset, meta_test_dataset

def create_training_dataset(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, maximum_length, src_langue=None, tgt_langue=None, single_pass=False, length_bucket_width=None, multi_domain=True, multi_lingual=False, picking_prob=None, temperature=1.0, pick_in_order=False, window_size=None):

  print("maximum_length", maximum_length)
  train_datasets = [] 
  if multi_domain:
    print(batch_type)
    for i,src,tgt in zip(domain,source_file,target_file):
      train_datasets.append(model.examples_inputter.make_training_dataset(src, tgt,
              batch_size=batch_train_size,
              batch_type=batch_type,
              domain=i,
              single_pass=single_pass,
              shuffle_buffer_size=shuffle_buffer_size,
              length_bucket_width=length_bucket_width,  # Bucketize sequences by the same length for efficiency.
              maximum_features_length=maximum_length,
              maximum_labels_length=maximum_length))
  elif multi_lingual:
    for i,j, src,tgt in zip(src_langue, tgt_langue, source_file,target_file):
      train_datasets.append(model.examples_inputter.make_training_dataset(src, tgt,
              batch_size=batch_train_size,
              batch_type=batch_type,
              domain=(i,j),
              single_pass=single_pass,
              shuffle_buffer_size=shuffle_buffer_size,
              length_bucket_width=length_bucket_width,  # Bucketize sequences by the same length for efficiency.
              maximum_features_length=maximum_length,
              maximum_labels_length=maximum_length))
  else:
    for src,tgt in zip(source_file,target_file):
      train_datasets.append(model.examples_inputter.make_training_dataset(src, tgt,
              batch_size=batch_train_size,
              batch_type=batch_type,
              single_pass=single_pass,
              shuffle_buffer_size=shuffle_buffer_size,
              length_bucket_width=length_bucket_width,  # Bucketize sequences by the same length for efficiency.
              maximum_features_length=maximum_length,
              maximum_labels_length=maximum_length))
  
  if picking_prob=="Natural":
    datasets_size = [count_lines(src) for src in source_file]
    picking_prob = [data_size/sum(datasets_size) for data_size in datasets_size]
    picking_prob = [p ** temperature for p in picking_prob]
    print("picking probability: ", picking_prob)
    print("temperature: ", temperature)
  elif picking_prob=="Anneal":
    import itertools
    datasets_size = [count_lines(src) for src in source_file]
    picking_prob_ = [data_size/sum(datasets_size) for data_size in datasets_size]
    def anneal(i, end=200000 * strategy.num_replicas_in_sync):
      i = (end-i)/end
      prob_ = [p**i  for p in picking_prob_]
      return [p/sum(prob_) for p in prob_]
    tensor = tf.Variable(np.array([anneal(i) for i in range(200000)]))
    picking_prob = tf.data.Dataset.from_tensor_slices(tensor)
    print("picking probability: ", picking_prob)
  else:
    print("picking probability: ", picking_prob)
  if pick_in_order:
    print("pick_in_order")
    choice_dataset = tf.data.Dataset.range(len(train_datasets)).repeat()
    train_dataset = tf.data.experimental.choose_from_datasets(train_datasets, choice_dataset)
  else:
    print("random_pick")
    train_dataset = tf.data.experimental.sample_from_datasets(train_datasets, weights=picking_prob)
  
  
  with strategy.scope():
    base_dataset = train_dataset
    train_dataset = strategy.experimental_distribute_datasets_from_function(
          lambda _: base_dataset)  

  return train_dataset

def create_training_dataset_hvd(model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, num_input_pipelines, 
                                  input_pipeline_id, num_replicas_in_sync, maximum_length, single_pass=False, length_bucket_width=None,
                                  multi_domain=True, picking_prob=None):
  train_datasets = [] 
  if multi_domain:
    print(batch_type)
    for i,src,tgt in zip(domain,source_file,target_file):
      train_datasets.append(model.examples_inputter.make_training_dataset(src, tgt,
              batch_size=batch_train_size,
              batch_type=batch_type,
              domain=i,
              single_pass=single_pass,
              shuffle_buffer_size=shuffle_buffer_size,
              length_bucket_width=length_bucket_width,  # Bucketize sequences by the same length for efficiency.
              maximum_features_length=maximum_length,
              maximum_labels_length=maximum_length))
  else:
    for src,tgt in zip(source_file,target_file):
      train_datasets.append(model.examples_inputter.make_training_dataset(src, tgt,
              batch_size=batch_train_size,
              batch_type=batch_type,
              single_pass=single_pass,
              shuffle_buffer_size=shuffle_buffer_size,
              length_bucket_width=length_bucket_width,  # Bucketize sequences by the same length for efficiency.
              maximum_features_length=maximum_length,
              maximum_labels_length=maximum_length))
  train_datasets = [dataset.shard(num_input_pipelines, input_pipeline_id) for dataset in train_datasets]
  if picking_prob=="Natural":
    datasets_size = [count_lines(src) for src in source_file]
    picking_prob = [data_size/sum(datasets_size) for data_size in datasets_size]
    print("picking probability: ", picking_prob)
  else:
    print("picking probability: ", picking_prob)

  train_dataset = tf.data.experimental.sample_from_datasets(train_datasets, weights=picking_prob)
  train_dataset = train_dataset.apply(make_cardinality_multiple_of(num_replicas_in_sync))

  return train_dataset

def create_training_dataset_v1(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, maximum_length, multi_domain=True):

  train_datasets = [] 
  datasets_size = [count_lines(src) for src in source_file]
  datasets_numb = len(source_file)
  batch_size_ratios = [data_size/sum(datasets_size) for data_size in datasets_size]
  batches_size = [round(batch_train_size*datasets_numb*ratio) for ratio in batch_size_ratios]
  print("batch size per domain: ", batches_size)
  if multi_domain:
    for i,src,tgt in zip(domain,source_file,target_file):
      train_datasets.append(model.examples_inputter.make_training_dataset(src, tgt,
              batch_size=batches_size[i],
              batch_type=batch_type,
              domain=i,
              shuffle_buffer_size=shuffle_buffer_size,
              length_bucket_width=1,  # Bucketize sequences by the same length for efficiency.
              maximum_features_length=maximum_length,
              maximum_labels_length=maximum_length))
  else:
    for src,tgt in zip(source_file,target_file):
      train_datasets.append(model.examples_inputter.make_training_dataset(src, tgt,
              batch_size=batch_train_size,
              batch_type=batch_type,
              shuffle_buffer_size=shuffle_buffer_size,
              length_bucket_width=1,  # Bucketize sequences by the same length for efficiency.
              maximum_features_length=maximum_length,
              maximum_labels_length=maximum_length))
  
  train_dataset = tf.data.experimental.sample_from_datasets(train_datasets) #tf.data.Dataset.zip(tuple(train_datasets)).map(merge_map_fn)
  with strategy.scope():
    base_dataset = train_dataset
    train_dataset = strategy.experimental_distribute_datasets_from_function(
          lambda _: base_dataset)  

  return train_dataset

def create_training_dataset_v2(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, maximum_length, length_bucket_width, multi_domain=True):

  train_datasets = [] 
  if multi_domain:
    print("Using multi-domain inputter")
    for i,src,tgt in zip(domain,source_file,target_file):
      train_datasets.append(model.examples_inputter.make_training_dataset(src, tgt,
              batch_size=batch_train_size * strategy.num_replicas_in_sync,
              batch_type=batch_type,
              domain=i,
              shuffle_buffer_size=shuffle_buffer_size,
              length_bucket_width=length_bucket_width,  # Bucketize sequences by the same length for efficiency.
              maximum_features_length=maximum_length,
              maximum_labels_length=maximum_length))
  else:
    print("Using stardard inputter")
    for src,tgt in zip(source_file,target_file):
      train_datasets.append(model.examples_inputter.make_training_dataset(src, tgt,
              batch_size=batch_train_size * strategy.num_replicas_in_sync,
              batch_type=batch_type,
              shuffle_buffer_size=shuffle_buffer_size,
              length_bucket_width=length_bucket_width,  # Bucketize sequences by the same length for efficiency.
              maximum_features_length=maximum_length,
              maximum_labels_length=maximum_length))
  
  train_dataset = tf.data.experimental.sample_from_datasets(train_datasets) #tf.data.Dataset.zip(tuple(train_datasets)).map(merge_map_fn)
  with strategy.scope():
    train_dataset = strategy.experimental_distribute_dataset(train_dataset)

  return train_dataset

def create_multi_domain_meta_training_dataset_v2(strategy, model, domain, source_file, target_file, batch_meta_train_size, batch_meta_test_size, batch_type, shuffle_buffer_size, maximum_length, picking_prob=None):
  meta_train_datasets = [None] * len(source_file)
  meta_train_base_datasets = [None] * len(source_file)
  meta_train_data_flows = [None] * len(source_file)
  print("batch_type: ", batch_type)
  
  for i, src, tgt in zip(domain, source_file, target_file):
    meta_train_datasets[i] = model.examples_inputter.make_training_dataset(src, tgt,
              batch_size=batch_meta_train_size,
              batch_type=batch_type,
              batch_multiplier=1,
              domain=i,
              shuffle_buffer_size=shuffle_buffer_size,
              length_bucket_width=1,  # Bucketize sequences by the same length for efficiency.
              maximum_features_length=maximum_length,
              maximum_labels_length=maximum_length)  
                
    with strategy.scope():  
      meta_train_base_datasets[i] = meta_train_datasets[i]
      meta_train_data_flow = strategy.experimental_distribute_datasets_from_function(
          lambda _: meta_train_base_datasets[i])
      meta_train_data_flows[i] = meta_train_data_flow
  
  return meta_train_datasets

def create_training_dataset_with_domain_tag(strategy, model, domain, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, maximum_length, single_pass=False, length_bucket_width=None, multi_domain=True, picking_prob=None):

  train_datasets = [] 
  
  for i,src,tgt in zip(domain,source_file,target_file):
    train_datasets.append(model.examples_inputter.make_training_dataset(src, tgt,
            batch_size=batch_train_size,
            batch_type=batch_type,
            domain=i,
            single_pass=single_pass,
            shuffle_buffer_size=shuffle_buffer_size,
            length_bucket_width=length_bucket_width,  # Bucketize sequences by the same length for efficiency.
            maximum_features_length=maximum_length,
            maximum_labels_length=maximum_length))

  if picking_prob=="Natural":
    datasets_size = [count_lines(src) for src in source_file]
    picking_prob = [data_size/sum(datasets_size) for data_size in datasets_size]
    print("picking probability: ", picking_prob)
  elif picking_prob=="Anneal":
    import itertools
    datasets_size = [count_lines(src) for src in source_file]
    picking_prob_ = [data_size/sum(datasets_size) for data_size in datasets_size]
    def anneal(i, end=200000 * strategy.num_replicas_in_sync):
      i = (end-i)/end
      prob_ = [p**i  for p in picking_prob_]
      return [p/sum(prob_) for p in prob_]
    tensor = tf.Variable(np.array([anneal(i) for i in range(200000)]))
    picking_prob = tf.data.Dataset.from_tensor_slices(tensor)
    print("picking probability: ", picking_prob)
  else:
    print("picking probability: ", picking_prob)

  train_dataset = tf.data.experimental.sample_from_datasets(train_datasets, weights=picking_prob) #tf.data.Dataset.zip(tuple(train_datasets)).map(merge_map_fn)
  with strategy.scope():
    base_dataset = train_dataset
    train_dataset = strategy.experimental_distribute_datasets_from_function(
          lambda _: base_dataset)  

  return train_dataset

def meta_learning_function_on_next(metatrain_dataset, metatest_dataset, as_numpy=False):
    
  def decorator(func):
    def _fun():
      metatrain_iterator = iter(metatrain_dataset)
      metatest_iterator = iter(metatest_dataset)
      @tf.function
      def _tf_fun():
        return func(lambda: next(metatrain_iterator)+next(metatest_iterator))

      while True:
        try:
          outputs = _tf_fun()
          if as_numpy:
            outputs = tf.nest.map_structure(lambda x: x.numpy(), outputs)
          yield outputs
        except tf.errors.OutOfRangeError:
          break

    return _fun

  return decorator

def create_training_dataset_with_dprob(strategy, model, source_file, target_file, prob_file, batch_train_size, batch_type, shuffle_buffer_size, maximum_length, single_pass=False, length_bucket_width=None, multi_domain=True, picking_prob=None, temperature=1.0):

  print("maximum_length", maximum_length)
  train_datasets = [] 
  
  for src, tgt, prob in zip(source_file,target_file, prob_file):
    train_datasets.append(model.examples_inputter.make_training_dataset(src, tgt, prob,
            batch_size=batch_train_size,
            batch_type=batch_type,
            single_pass=single_pass,
            shuffle_buffer_size=shuffle_buffer_size,
            length_bucket_width=length_bucket_width,  # Bucketize sequences by the same length for efficiency.
            maximum_features_length=maximum_length,
            maximum_labels_length=maximum_length))
  
  if picking_prob=="Natural":
    datasets_size = [count_lines(src) for src in source_file]
    picking_prob = [data_size/sum(datasets_size) for data_size in datasets_size]
    picking_prob = [p ** temperature for p in picking_prob]
    print("picking probability: ", picking_prob)
    print("temperature: ", temperature)
  elif picking_prob=="Anneal":
    import itertools
    datasets_size = [count_lines(src) for src in source_file]
    picking_prob_ = [data_size/sum(datasets_size) for data_size in datasets_size]
    def anneal(i, end=200000 * strategy.num_replicas_in_sync):
      i = (end-i)/end
      prob_ = [p**i  for p in picking_prob_]
      return [p/sum(prob_) for p in prob_]
    tensor = tf.Variable(np.array([anneal(i) for i in range(200000)]))
    picking_prob = tf.data.Dataset.from_tensor_slices(tensor)
    print("picking probability: ", picking_prob)
  else:
    print("picking probability: ", picking_prob)

  train_dataset = tf.data.experimental.sample_from_datasets(train_datasets, weights=picking_prob)
  with strategy.scope():
    base_dataset = train_dataset
    train_dataset = strategy.experimental_distribute_datasets_from_function(
          lambda _: base_dataset)  

  return train_dataset

def create_training_dataset_DRO(strategy, model, source_file, target_file, prob_file, domain, batch_train_size, batch_type, shuffle_buffer_size, maximum_length, single_pass=False, length_bucket_width=None, multi_domain=True, picking_prob=None, temperature=1.0):

  print("maximum_length", maximum_length)
  train_datasets = [] 
  
  for src, tgt, prob, d in zip(source_file,target_file, prob_file, domain):
    train_datasets.append(model.examples_inputter.make_training_dataset(src, tgt, prob,
            batch_size=batch_train_size,
            domain=d,
            batch_type=batch_type,
            single_pass=single_pass,
            shuffle_buffer_size=shuffle_buffer_size,
            length_bucket_width=length_bucket_width,  # Bucketize sequences by the same length for efficiency.
            maximum_features_length=maximum_length,
            maximum_labels_length=maximum_length))
  
  if picking_prob=="Natural":
    datasets_size = [count_lines(src) for src in source_file]
    picking_prob = [data_size/sum(datasets_size) for data_size in datasets_size]
    picking_prob = [p ** temperature for p in picking_prob]
    print("picking probability: ", picking_prob)
    print("temperature: ", temperature)
  elif picking_prob=="Anneal":
    import itertools
    datasets_size = [count_lines(src) for src in source_file]
    picking_prob_ = [data_size/sum(datasets_size) for data_size in datasets_size]
    def anneal(i, end=200000 * strategy.num_replicas_in_sync):
      i = (end-i)/end
      prob_ = [p**i  for p in picking_prob_]
      return [p/sum(prob_) for p in prob_]
    tensor = tf.Variable(np.array([anneal(i) for i in range(200000)]))
    picking_prob = tf.data.Dataset.from_tensor_slices(tensor)
    print("picking probability: ", picking_prob)
  else:
    print("picking probability: ", picking_prob)

  train_dataset = tf.data.experimental.sample_from_datasets(train_datasets, weights=picking_prob)
  with strategy.scope():
    base_dataset = train_dataset
    train_dataset = strategy.experimental_distribute_datasets_from_function(
          lambda _: base_dataset)  

  return train_dataset

def create_training_dataset_robustness(strategy, model, domain, is_noisy, source_file, target_file, batch_train_size, batch_type, shuffle_buffer_size, maximum_length, single_pass=False, length_bucket_width=None, multi_domain=True, picking_prob=None, temperature=1.0):

  print("maximum_length", maximum_length)
  print(batch_type)
  train_datasets=[]
  for i,is_noisy_,src,tgt in zip(domain,is_noisy,source_file,target_file):
    
    train_datasets.append(model.examples_inputter.make_training_dataset(src, tgt,
            batch_size=batch_train_size,
            batch_type=batch_type,
            domain=i,
            is_noisy=is_noisy_,
            single_pass=single_pass,
            shuffle_buffer_size=shuffle_buffer_size,
            length_bucket_width=length_bucket_width,  # Bucketize sequences by the same length for efficiency.
            maximum_features_length=maximum_length,
            maximum_labels_length=maximum_length))
  if picking_prob=="Natural":
    datasets_size = [count_lines(src) for src in source_file]
    picking_prob = [data_size/sum(datasets_size) for data_size in datasets_size]
    picking_prob = [p ** temperature for p in picking_prob]
    print("picking probability: ", picking_prob)
    print("temperature: ", temperature)
  elif picking_prob=="Anneal":
    import itertools
    datasets_size = [count_lines(src) for src in source_file]
    picking_prob_ = [data_size/sum(datasets_size) for data_size in datasets_size]
    def anneal(i, end=200000 * strategy.num_replicas_in_sync):
      i = (end-i)/end
      prob_ = [p**i  for p in picking_prob_]
      return [p/sum(prob_) for p in prob_]
    tensor = tf.Variable(np.array([anneal(i) for i in range(200000)]))
    picking_prob = tf.data.Dataset.from_tensor_slices(tensor)
    print("picking probability: ", picking_prob)
  else:
    print("picking probability: ", picking_prob)

  train_dataset = tf.data.experimental.sample_from_datasets(train_datasets, weights=picking_prob)
  with strategy.scope():
    base_dataset = train_dataset
    train_dataset = strategy.experimental_distribute_datasets_from_function(
          lambda _: base_dataset)  

  return train_dataset

def function_on_next(dataset, as_numpy=False):  

  def decorator(func):
    def _fun():
      iterator = iter(dataset)

      def _tf_fun():
        return func(lambda: next(iterator))

      while True:
        try:
          outputs = _tf_fun()
          if as_numpy:
            outputs = tf.nest.map_structure(lambda x: x.numpy(), outputs)
          yield outputs
        except tf.errors.OutOfRangeError:
          break

    return _fun

  return decorator

def create_priming_training_dataset(strategy, model, source_file, target_file, pre_file, batch_train_size, batch_type, shuffle_buffer_size, maximum_length, single_pass=False, length_bucket_width=None, multi_domain=True, picking_prob=None, temperature=1.0, pick_in_order=False, window_size=None):

  print("maximum_length: ", maximum_length)
  print("shuffle_buffer_size: ", shuffle_buffer_size)
  if not isinstance(target_file, List):
    train_dataset = model.examples_inputter.make_training_dataset([source_file, pre_file], target_file,
              batch_size=batch_train_size,
              batch_type=batch_type,
              single_pass=single_pass,
              shuffle_buffer_size=shuffle_buffer_size,
              length_bucket_width=length_bucket_width,  # Bucketize sequences by the same length for efficiency.
              maximum_features_length=maximum_length,
              maximum_labels_length=maximum_length)
  else:
    assert isinstance(source_file, List)
    assert isinstance(pre_file, List)
    assert len(source_file) == len(target_file)
    assert len(pre_file) == len(target_file)
    print("detected multiple input files")
    train_datasets = []
    for src, pre, tgt in zip(source_file, pre_file, target_file):
      train_datasets.append(model.examples_inputter.make_training_dataset([src, pre], tgt,
              batch_size=batch_train_size,
              batch_type=batch_type,
              single_pass=single_pass,
              shuffle_buffer_size=shuffle_buffer_size,
              length_bucket_width=length_bucket_width,  # Bucketize sequences by the same length for efficiency.
              maximum_features_length=maximum_length,
              maximum_labels_length=maximum_length))
    train_dataset = tf.data.experimental.sample_from_datasets(train_datasets, weights=picking_prob)
    
  with strategy.scope():
    base_dataset = train_dataset
    train_dataset = strategy.experimental_distribute_datasets_from_function(
          lambda _: base_dataset)  

  return train_dataset








































































