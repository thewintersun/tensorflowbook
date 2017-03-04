import tensorflow as tf

v = tf.Variable([1,2,3,4,5,6,7,8,9])
reshaped_v = tf.reshape(v,[3,3])
init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  print("v's value is:")
  print(sess.run(v))
  print("reshaped value is:")
  print(sess.run(reshaped_v))
