import tensorflow as tf

value_shape0 = tf.Variable(8672)
value_shape1 = tf.Variable([1.1,2.2,3.3])
value_shape2 = tf.Variable([[1,2],[3,4],[5,6]])
value_shape3 = tf.Variable([[[1],[2],[3]],[[4],[5],[6]]])

print(value_shape0.get_shape())
print(value_shape1.get_shape())
print(value_shape2.get_shape())
print(value_shape3.get_shape())
