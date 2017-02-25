#coding=utf-8
import math
import tensorflow as tf


def inference(x):
  with tf.name_scope('hidden1'):
    weights = tf.get_variable("weight", [5], tf.float32, initializer=tf.random_normal_initializer())
    biases  = tf.get_variable("biase", [5], tf.float32, initializer=tf.random_normal_initializer())
    hidden1 = tf.nn.relu(tf.matmul(x, weights) + biases)

  with tf.name_scope('hidden2'):
    weights = tf.get_variable("weight", [5,5], tf.float32, initializer=tf.random_normal_initializer())
    biases  = tf.get_variable("biase", [5], tf.float32, initializer=tf.random_normal_initializer())
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

  with tf.name_scope('linear'):
    weights = tf.get_variable("weight", [5,1], tf.float32, initializer=tf.random_normal_initializer())
    biases  = tf.get_variable("biase", [1], tf.float32, initializer=tf.random_normal_initializer())
    logits = tf.matmul(hidden2, weights) + biases

  return logits

# 构建图
x = tf.placeholder(tf.float32, shape=[5])
label = tf.placeholder(tf.float32, shape=[1])

y = inference(x)

#损失函数
loss = tf.square(y - label, name="loss")
#优化函数
optimizer = tf.train.GradientDescentOptimizer(0.01)
#计算梯度，应用梯度操作
grads_and_vars = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(grads_and_vars)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for step in range(500):
        train_x = np.random.randn(1)
        train_y = math.sin(train_x)
        loss_value, _ = sess.run([loss,train_op], feed_dict={x:train_x, label:train_y})
        print("step: %d loss %f" %(step, loss_value))
        


