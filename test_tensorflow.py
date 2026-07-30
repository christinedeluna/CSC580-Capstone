import tensorflow as tf

print("TensorFlow Version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("Devices:", tf.config.list_physical_devices())

hello = tf.constant("Hello, TensorFlow!")
print(hello)