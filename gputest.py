import tensorflow as tf
with tf.device('/cpu:0'):
    a = tf.constant([1.0,2.0,3.0],shape=[3],name='a')
    b = tf.constant([1.0,2.0,3.0],shape=[3],name='b')
with tf.device('/gpu:1'):
    c = a+b

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.global_variables_initializer())
print(sess.run(c))
print(tf.test.is_built_with_cuda()) # 结果是True
print(tf.test.is_gpu_available()) # 结果是True
# tf.config.list_physical_devices('GPU') # 输出可使用的GPU列表