#coding=utf-8
import tensorflow as tf

v1 = tf.placeholder(tf.float32)
v2 = tf.placeholder(tf.float32)
v_mul = tf.multiply(v1,v2)

with tf.Session() as sess:
  while True:

    value1 = input("value1: ")
    value2 = input("value2: ")

    mul_result = sess.run(v_mul,feed_dict={v1:value1,v2:value2})
    print(mul_result)
