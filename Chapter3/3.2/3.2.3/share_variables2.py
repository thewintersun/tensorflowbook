import numpy as np
import tensorflow as tf

def get_data(number):
    list_x = []
    list_label = []
    for i in range(number):
        x = np.random.randn(1)
        label = 2 * x + np.random.randn(1) * 0.01  + 10
        list_x.append(x)
        list_label.append(label)
    return list_x, list_label

def inference(x):
    weight = tf.get_variable("weight",[1])
    biase = tf.get_variable("biase",[1])
    y = x * weight + biase
    return y

train_x = tf.placeholder(tf.float32)
train_label = tf.placeholder(tf.float32)
test_x = tf.placeholder(tf.float32)
test_label = tf.placeholder(tf.float32)

with tf.variable_scope("inference"):
    train_y = inference(train_x)
    tf.get_variable_scope().reuse_variables()
    test_y = inference(test_x)

train_loss = tf.square(train_y - train_label)
test_loss = tf.square(test_y - test_label)
opt = tf.train.GradientDescentOptimizer(0.002)
train_op = opt.minimize(train_loss)

init = tf.global_variables_initializer()

train_data_x, train_data_label = get_data(1000) #读取训练数据的函数
test_data_x, test_data_label = get_data(1)

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train_op, feed_dict={train_x: train_data_x[i],
                                      train_label:train_data_label[i]})
        if i % 10 == 0:
            test_loss_value = sess.run(test_loss, feed_dict={test_x:test_data_x[0],
                                                             test_label: test_data_label[0]})
            print("step %d eval loss is %.3f" %(i,test_loss_value))
