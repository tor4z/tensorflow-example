### via google TensorFlow example 
# Make up some data in tow dimensions and then fits a line to fits

import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

### random_uniform
## tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32,
#                seed=None, name=None)
#
# The generated values follow a uniform distribution 
# in the range [minval, maxval). The lower bound minval
# is included in the range, while the upper bound maxval is excluded.


### zeros
## tf.zeros(shape, dtype=tf.float32, name=None)
#
# This operation returns a tensor of type dtype
# with shape shape and all elements set to zero.

w = tf.Variable(tf.random_uniform([1], -1, 1))
b = tf.Variable(tf.zeros([1]))
y = x_data * w + b

### reduce_mean
## tf.reduce_mean(input_tensor, axis=None, keep_dims=False,
#               name=None, reduction_indices=None)
#
# Computes the mean of elements across dimensions of a tensor.

loss = tf.reduce_mean(tf.square(y - y_data))

### class GradientDescentOptimizer
# tf.train.GradientDescentOptimizer
## __init__(learning_rate, use_locking=False, name='GradientDescent')
#
# Construct a new gradient descent optimizer.
#
## minimize(loss, global_step=None, var_list=None, gate_gradients=GATE_OP, 
#          aggregation_method=None, colocate_gradients_with_ops=False, 
#          name=None, grad_loss=None)
#
# Add operations to minimize loss by updating var_list.
#
# var_list: Optional list of Variable objects to update to minimize loss. 
#           Defaults to the list of variables collected in the graph under 
#           the key GraphKeys. TRAINABLE_VARIABLES.

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

### global_variables_initializer
## tf.global_variables_initializer()
#
# Returns an Op that initializes global variables.

init = tf.global_variables_initializer()

### class tf.Session
## __init__(target='', graph=None, config=None)
#
# Creates a new TensorFlow session.
# If no graph argument is specified when constructing the session, the default 
# graph will be launched in the session. If you are using more than one 
# graph (created with tf.Graph() in the same process, you will have to use 
# different sessions for each graph, but each graph can be used in multiple 
# sessions. In this case, it is often clearer to pass the graph to be launched 
# explicitly to the session constructor.
#
## close()
#
# Closes this session.
# Calling this method frees all resources associated with the session.
#
## run(fetches, feed_dict=None, options=None, run_metadata=None)
# 
# Runs operations and evaluates tensors in fetches.

sess = tf.Session()

sess.run(init)

for step in range(200):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(w), sess.run(b))

sess.close()

############# output ############
#    0 [ 0.81312418] [-0.13055398]
#    20 [ 0.27587447] [ 0.20475118]
#    40 [ 0.14227426] [ 0.27710542]
#    60 [ 0.1101613] [ 0.29449692]
#    80 [ 0.10244245] [ 0.29867727]
#    100 [ 0.1005871] [ 0.29968205]
#    120 [ 0.10014114] [ 0.29992357]
#    140 [ 0.10003392] [ 0.29998165]
#    160 [ 0.10000816] [ 0.2999956]
#    180 [ 0.10000196] [ 0.29999894]
###################################