
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.LogicalDeviceConfiguration(
    memory_limit=None, experimental_priority=None
)