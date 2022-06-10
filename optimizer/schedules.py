import tensorflow as tf
class NGDDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Defines the decay function described in https://arxiv.org/abs/1706.03762."""

  def __init__(self, scale, model_dim, warmup_steps):
    """Initializes the decay function.

    Args:
      scale: The scale constant.
      model_dim: The model dimension.
      warmup_steps: The number of warmup steps.
    """
    self.scale = tf.cast(scale, tf.float32)
    self.model_dim = tf.cast(model_dim, tf.float32)
    self.warmup_steps = tf.cast(warmup_steps, tf.float32)

  def __call__(self, step):
    step = tf.cast(step + 1, tf.float32)
    return (self.scale
            * tf.pow(self.model_dim, -0.5)
            * tf.cond(tf.math.less(step, self.warmup_steps), lambda: step * tf.pow(self.warmup_steps,-1.5), lambda: tf.pow(self.warmup_steps, -0.3)*tf.pow(step,-0.2)))