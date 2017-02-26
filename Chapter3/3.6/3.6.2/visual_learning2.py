#coding=utf-8
import numpy as np
import tensorflow as tf

# 构建图
x = tf.placeholder(tf.float32, shape=[3,1])
y = tf.placeholder(tf.float32, shape=[2,1])
weight = tf.get_variable("weight", [3,2], tf.float32, initializer=tf.random_normal_initializer())
biase  = tf.get_variable("biase", [2], tf.float32, initializer=tf.random_normal_initializer())
x_trans = tf.transpose(x)
mul_op = tf.matmul(x_trans, weight, name="mul_op")
pred = tf.add(mul_op, biase, name="add_op")

#损失函数
loss = tf.square(y - pred, name="loss")
#优化函数
optimizer = tf.train.GradientDescentOptimizer(0.01)
#计算梯度，应用梯度操作
grads_and_vars = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(grads_and_vars)

#收集值的操作
tf.summary.histogram("weight", weight)
#tf.summary.scalar("weight", weight)
#tf.summary.scalar("biase", biase)
#tf.summary.scalar("loss", loss[0])
tf.summary.scalar("loss", 1)

merged_summary = tf.summary.merge_all()

summary_writer = tf.summary.FileWriter('./log_graph2' )
summary_writer.add_graph(tf.get_default_graph())
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for step in range(10):
        #print(step)
        train_x = [np.random.randn(1),np.random.randn(1),np.random.randn(1)]
        y1 = train_x[0] * 1 + train_x[1] * 2 + train_x[2] * 3 + 10 + np.random.randn(1) * 0.01
        y2 = train_x[0] * 4 + train_x[1] * 5 + train_x[2] * 6 + 20 + np.random.randn(1) * 0.01

        train_y = [ y1, y2]
        _, summary,w = sess.run([train_op, merged_summary,weight], feed_dict={x:train_x, y:train_y})
        for i in w:
          print(i)
        print("")
        #_ = sess.run([train_op], feed_dict={x:train_x, y:train_y})
        summary_writer.add_summary(summary, step)
        
