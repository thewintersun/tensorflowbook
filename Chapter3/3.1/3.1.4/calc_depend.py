#coding=utf-8
import tensorflow as tf

x = tf.Variable(0.0, name="x")
x_plus_1 = tf.assign_add(x, 1, name="x_plus")

with tf.control_dependencies([x_plus_1]):
    y = tf.identity(x, name="y")

init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  for i in range(5):
    print(y.eval())

summary_writer = tf.summary.FileWriter('./calc_graph' )
graph = tf.get_default_graph()
summary_writer.add_graph(graph)
summary_writer.flush()