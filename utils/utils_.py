import sys

import tensorflow as tf
import numpy as np
import subprocess
import io
import re
import os
import six
import json
import utils.misc as misc

def make_domain_mask(num_domains, num_units, num_domain_units=8, domain_region_sizes=None, dtype=tf.float32):
  if not domain_region_sizes:
    M = np.zeros((num_domains+1, num_domains * num_domain_units))
    M_ = np.ones((num_domains+1, num_units - num_domains * num_domain_units))
    for i in range(num_domains):
      for j in range(i*num_domain_units, (i+1)*num_domain_units):
        M[i,j] = 1
    M = np.concatenate([M_,M],1)
  else:
    total_domain_units = sum(domain_region_sizes)
    M = np.zeros((num_domains+1, total_domain_units))
    M_ = np.ones((num_domains+1, num_units - total_domain_units))
    for i in range(num_domains):
      for j in range(sum(domain_region_sizes[:i]),sum(domain_region_sizes[:i+1])):
        M[i,j] = 1
    M = np.concatenate([M_,M],1)
  
  return tf.constant(M, dtype=dtype)

def masking(ids, mask_id, noise_percentage=0.15):
  return 

def _compute_fans(shape):
  
  if len(shape) < 1:  
    fan_in = fan_out = 1
  elif len(shape) == 1:
    fan_in = fan_out = shape[0]
  elif len(shape) == 2:
    fan_in = shape[0]
    fan_out = shape[1]
  else:
    receptive_field_size = 1
    for dim in shape[:-2]:
      receptive_field_size *= dim
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
  return int(fan_in), int(fan_out)

def variance_scaling_initialier(shape, scale=1.0, mode="fan_in", distribution="uniform"):
  assert mode in ["fan_in","fan_out","fan_avg"]
  assert distribution in ["uniform","truncated_normal","untruncated_normal"]
  initializer = tf.keras.initializers.VarianceScaling(scale=scale, mode=mode, distribution=distribution)
  return initializer(shape)

def var_spec(var):
  print("var inspector:_____")
  if isinstance(var,list):
    print("list contains:  %d elements"%len(var))
  else:
    print(var)

class MultiBLEUScorer(object):

  def __init__(self, bleu_script="multi-bleu.perl"):
    
    self._bleu_script = bleu_script

  def __call__(self, labels_file, predictions_path):
    utils_dir = os.path.dirname(__file__)
    project_dir = os.path.dirname(utils_dir)
    script_path = os.path.join(project_dir, "scripts")
    try:
      with io.open(predictions_path, encoding="utf-8", mode="r") as predictions_file:
        bleu_out = subprocess.check_output(
            [os.path.join(script_path, self._bleu_script), labels_file],
            stdin=predictions_file,
            stderr=subprocess.STDOUT)
        bleu_out = bleu_out.decode("utf-8")
        bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
        return float(bleu_score)
    except subprocess.CalledProcessError:      
      return None

def load_and_update_if_needed_from_ckpt(model_dir,   
                        checkpoint_path,
                        trackables,
                        vocab_update=False,
                        model_key="model"):

  model = trackables.get(model_key)
  if model is None:
    raise ValueError("%s not found in trackables %s" % (model_key, trackables))
  if not model.built:
    raise ValueError("The model should be built before calling this function")

  checkpoint = tf.train.Checkpoint(**trackables)
  
  tf.get_logger().info("Reading checkpoint %s...", checkpoint_path)
  reader = tf.train.load_checkpoint(checkpoint_path)
  for path in six.iterkeys(reader.get_variable_to_shape_map()):
    if not path.startswith(model_key) or ".OPTIMIZER_SLOT" in path:
      continue
    variable_path = path.replace("/.ATTRIBUTES/VARIABLE_VALUE", "")
    #print(variable_path)
    variable = variable_which(trackables, variable_path)
    value = reader.get_tensor(path)
    if variable:
      if "_domain_classification" in variable.name:
        continue
      elif "ADAP" in variable.name:
        continue
      elif vocab_update and "_embedding" in variable.name:
        #print("vocab_update", vocab_update)
        #print(variable.name)
        new_value = np.concatenate((value, np.zeros((1,512))),axis=0)
        variable.assign(new_value)
      elif vocab_update and "dense_96/bias" in variable.name:
        print(variable.name)
        new_value = np.concatenate((value, np.zeros((1))),axis=0)
        variable.assign(new_value)
      elif vocab_update and "dense_96/kernel" in variable.name:
        print(variable.name)
        new_value = np.concatenate((value, np.zeros((512,1))),axis=1)
        variable.assign(new_value)
      else:
        #print(variable.name)
        #print(value)
        variable.assign(value)

def average_checkpoints(model_dir,
                        output_dir,
                        trackables,
                        max_count=8,
                        model_key="model"):
  """Averages object-based checkpoints.

  Args:
    model_dir: The directory containing checkpoints.
    output_dir: The directory that will contain the averaged checkpoint.
    trackables: A dictionary containing the trackable objects included in the
      checkpoint.
    max_count: The maximum number of checkpoints to average.
    model_key: The key in :obj:`trackables` that references the model.

  Returns:
    The path to the directory containing the averaged checkpoint.

  Raises:
    ValueError: if :obj:`output_dir` is the same as :obj:`model_dir`.
    ValueError: if a model is not found in :obj:`trackables` or is not already
      built.
    ValueError: if no checkpoints are found in :obj:`model_dir`.
  """
  if model_dir == output_dir:
    raise ValueError("Model and output directory must be different")
  model = trackables.get(model_key)
  if model is None:
    raise ValueError("%s not found in trackables %s" % (model_key, trackables))
  if not model.built:
    raise ValueError("The model should be built before calling this function")

  checkpoint = tf.train.Checkpoint(**trackables)
  checkpoint_manager = tf.train.CheckpointManager(checkpoint, model_dir, max_to_keep=None)

  checkpoints_path = checkpoint_manager.checkpoints
  if not checkpoints_path:
    raise ValueError("No checkpoints found in %s" % model_dir)
  if len(checkpoints_path) > max_count:
    checkpoints_path = checkpoints_path[-max_count:]
  num_checkpoints = len(checkpoints_path)
  last_step = int(checkpoints_path[-1].split("-")[-1])

  tf.get_logger().info("Averaging %d checkpoints...", num_checkpoints)
  for i, checkpoint_path in enumerate(reversed(checkpoints_path)):
    tf.get_logger().info("Reading checkpoint %s...", checkpoint_path)
    if i == 0:
      checkpoint.restore(checkpoint_path).assert_existing_objects_matched()
      for variable in model.variables:
        variable.assign(variable / num_checkpoints)
        #tf.print("variable:___", variable.name, tf.shape(variable),sep="|")
    else:
      reader = tf.train.load_checkpoint(checkpoint_path)
      for path in six.iterkeys(reader.get_variable_to_shape_map()):
        if not path.startswith(model_key) or ".OPTIMIZER_SLOT" in path:
          continue

        value = reader.get_tensor(path)
        variable_path = path.replace("/.ATTRIBUTES/VARIABLE_VALUE", "")
        variable = variable_which(trackables, variable_path)
        #tf.print("variable:___", variable.name, tf.shape(value), variable_path, sep="|")
        variable.assign_add(value / num_checkpoints)

  new_checkpoint_manager = tf.train.CheckpointManager(checkpoint, output_dir, max_to_keep=None)
  new_checkpoint_manager.save(checkpoint_number=last_step)
  return new_checkpoint_manager

def variable_which(structure, path):
  """Follows :obj:`path` in a nested structure of objects, lists, and dicts."""
  for key in path.split("/")[:-1]:
    if isinstance(structure, list):
      try:
        index = int(key)
        structure = structure[index] if index < len(structure) else None
      except ValueError:
        raise ValueError("Expected a list index, got %s instead" % key)
    elif isinstance(structure, dict):
      structure = structure.get(key)
    else:
      structure = getattr(structure, key, None)
  if structure:
    name = path.split("/")[-1]  
    if sum([name in v.name for v in structure.trainable_variables]):
      #print([v.name for v in structure.trainable_variables])
      for v in structure.trainable_variables:
        v_name = v.name.split("/")[-1].split(":")[0]
        if name == v_name:
          return v
    else:
      
      raise ValueError("Invalid path in structure: %s" % path)
  return structure

def create_slurm_strategy():
  SLURM_VARIABLES = [
            'SLURM_JOB_ID',
            'SLURM_JOB_NODELIST', 'SLURM_JOB_NUM_NODES', 'SLURM_NTASKS', 'SLURM_TASKS_PER_NODE',
            'SLURM_NODEID', 'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_TASK_PID']
  print(" ".join(["%s: %s"%(v, os.environ[v]) for v in SLURM_VARIABLES]))
  hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
  print("hostnames: %s"%hostnames)
  master_addr = hostnames.split()[0].decode('utf-8')
  print("master_addr: %s"%master_addr)
  port=12345
  host_addrs = []
  ntasks = int(os.environ["SLURM_NTASKS"])
  nnodes = int(os.environ["SLURM_JOB_NUM_NODES"])
  for name in hostnames.split():
    name = name.decode('utf-8')
    for i in range(ntasks//nnodes):
      host_addrs.append("%s:%d"%(name, port+i))
  os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': host_addrs
    },
    'task': {'type': 'worker', 'index': os.environ["SLURM_PROCID"]}
  })
  #jobs = {"worker": 2}
  #strategy = tf.distribute.experimental.ParameterServerStrategy(cluster_resolver=tf.distribute.cluster_resolver.SlurmClusterResolver(jobs, gpus_per_node=4, gpus_per_task=1))
  #strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(communication=tf.distribute.experimental.CollectiveCommunication.NCCL,cluster_resolver=tf.distribute.cluster_resolver.SlurmClusterResolver(jobs, gpus_per_node=4, gpus_per_task=4))
  strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
  return strategy
  
def get_checkpoint_variables(checkpoint_path):
  """Returns variables included in a checkpoint.
  Args:
    checkpoint_path: Path to the checkpoint.
  Returns:
    A dictionary mapping variables name to value.
  """
  reader = tf.train.load_checkpoint(checkpoint_path)
  return {
      name:reader.get_tensor(name)
      for name in reader.get_variable_to_shape_map().keys()}

def average_checkpoints_tf2_3(model_dir,
                        output_dir,
                        trackables,
                        max_count=8,
                        model_key="model"):
  """Averages object-based checkpoints.
  Args:
    model_dir: The directory containing checkpoints.
    output_dir: The directory that will contain the averaged checkpoint.
    trackables: A dictionary containing the trackable objects included in the
      checkpoint.
    max_count: The maximum number of checkpoints to average.
    model_key: The key in :obj:`trackables` that references the model.
  Returns:
    The path to the directory containing the averaged checkpoint.
  Raises:
    ValueError: if :obj:`output_dir` is the same as :obj:`model_dir`.
    ValueError: if a model is not found in :obj:`trackables` or is not already
      built.
    ValueError: if no checkpoints are found in :obj:`model_dir`.
  See Also:
    :func:`opennmt.utils.average_checkpoints_into_layer`
  """
  if model_dir == output_dir:
    raise ValueError("Model and output directory must be different")
  model = trackables.get(model_key)
  if model is None:
    raise ValueError("%s not found in trackables %s" % (model_key, trackables))

  checkpoint_state = tf.train.get_checkpoint_state(model_dir)
  if checkpoint_state is None:
    raise ValueError("No checkpoints found in %s" % model_dir)
  checkpoints_path = checkpoint_state.all_model_checkpoint_paths
  if len(checkpoints_path) > max_count:
    checkpoints_path = checkpoints_path[-max_count:]

  average_checkpoints_into_layer(checkpoints_path, model, model_key)

  last_step = get_step_from_checkpoint_prefix(checkpoints_path[-1])
  checkpoint = tf.train.Checkpoint(**trackables)
  new_checkpoint_manager = tf.train.CheckpointManager(checkpoint, output_dir, max_to_keep=None)
  new_checkpoint_manager.save(checkpoint_number=last_step)
  return new_checkpoint_manager

def average_checkpoints_into_layer(checkpoints, layer, layer_prefix):
  """Updates the layer weights with their average value in the checkpoints.
  Args:
    checkpoints: A non empty list of checkpoint paths.
    layer: A ``tf.keras.layers.Layer`` instance.
    layer_prefix: The name/scope that prefixes the layer variables names in the
      checkpoints.
  Raises:
    ValueError: if :obj:`checkpoints` is empty.
    ValueError: if :obj:`layer` is not already built.
  See Also:
    :func:`opennmt.utils.average_checkpoints`
  """
  if not checkpoints:
    raise ValueError("There should be at least one checkpoint")
  if not layer.built:
    raise ValueError("The layer should be built before calling this function")

  # Reset the layer variables to 0.
  for variable in layer.variables:
    variable.assign(tf.zeros_like(variable))

  # Get a map from variable names in the checkpoint to variables in the layer.
  _, names_to_variables = misc.get_variables_name_mapping(layer, root_key=layer_prefix)

  num_checkpoints = len(checkpoints)
  tf.get_logger().info("Averaging %d checkpoints...", num_checkpoints)
  for checkpoint_path in checkpoints:
    tf.get_logger().info("Reading checkpoint %s...", checkpoint_path)
    reader = tf.train.load_checkpoint(checkpoint_path)
    for path in reader.get_variable_to_shape_map().keys():
      if not path.startswith(layer_prefix) or ".OPTIMIZER_SLOT" in path:
        continue
      variable = names_to_variables[path]
      value = reader.get_tensor(path)
      variable.assign_add(value / num_checkpoints)

def get_step_from_checkpoint_prefix(prefix):
  """Extracts the training step from the checkpoint file prefix."""
  return int(prefix.split("-")[-1])

_V1_OPTIM_SCOPE = "optim"
_V1_SLOTS_MAPPING = {
    "Adam": "m",
    "Adam_1": "v"
}
