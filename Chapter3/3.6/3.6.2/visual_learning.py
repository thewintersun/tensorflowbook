#coding=utf-8
import numpy as np
import tensorflow as tf


# 构建图
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
weight = tf.get_variable("weight", [], tf.float32, initializer=tf.random_normal_initializer())
biase  = tf.get_variable("biase", [], tf.float32, initializer=tf.random_normal_initializer())
pred = tf.add(tf.mul(x, weight, name="mul_op"), biase, name="add_op")

#损失函数
loss = tf.square(y - pred, name="loss")
#优化函数
optimizer = tf.train.GradientDescentOptimizer(0.01)
#计算梯度，应用梯度操作
grads_and_vars = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(grads_and_vars)

#收集值的操作
tf.summary.scalar("weight", weight)
tf.summary.scalar("loss", loss[0])
tf.summary.scalar("biase", biase)

merged_summary = tf.summary.merge_all()

summary_writer = tf.summary.FileWriter('./log_graph' )

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for i in range(500):
        train_x = np.random.randn(1)
        train_y = 2 * train_x + np.random.randn(1) * 0.01  + 10
        _, summary = sess.run([train_op, merged_summary], feed_dict={x:train_x, y:train_y})
        summary_writer.add_summary(summary, i)
        
