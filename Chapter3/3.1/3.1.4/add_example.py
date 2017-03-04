import tensorflow as tf

v1 = tf.constant(1, name="value1")
v2 = tf.constant(1, name="value2")
add_op = tf.add(v1, v2, name="add_op_name")

with tf.Session() as sess:
  result = sess.run(add_op)
  print("1 + 1 = %.0f" % result)
