#coding=utf-8
import tensorflow as tf

x = tf.constant(1.0, name="input")
w = tf.Variable(0.5, name="weight")
b = tf.Variable(0.1, name="biase")
y = tf.add(tf.mul(x,w, name="mul_op"), b, name="add_op")

summary_writer = tf.summary.FileWriter('./calc_graph' )
graph = tf.get_default_graph()
summary_writer.add_graph(graph)
summary_writer.flush()
    
