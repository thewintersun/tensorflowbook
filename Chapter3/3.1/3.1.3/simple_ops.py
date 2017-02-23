#coding=utf-8
import tensorflow as tf

a = tf.Variable(1.0, name="a")
b = tf.add(a, 1, name="b")
c = tf.add(b, 1, name="c")
d = tf.add(b, 10, name="d")

summary_writer = tf.summary.FileWriter('./calc_graph' )
graph = tf.get_default_graph()
summary_writer.add_graph(graph)
summary_writer.flush()


    