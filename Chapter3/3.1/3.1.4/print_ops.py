import tensorflow as tf

v1 = tf.constant(1, name="value1")
v2 = tf.constant(1, name="value2")
add_op = tf.add(v1, v2, name="add_op_name")

graph = tf.get_default_graph()
operations = graph.get_operations()
print("number of operations: %d" % len(operations))
print("operations:")
for op in operations:
  print(op)
