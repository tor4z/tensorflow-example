# via https://github.com/lexfridman/deepcars.git
#
# bias
#     \
#      \ w1
#       \
#         y 
#       /
#      / 
#     / w2
#   x/
###############

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


num_examples = 50
losses = []
training_step = 100
learning_rate = 0.001

### numpy.linspace
## numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
#
# Return evenly spaced numbers over a specified interval. Returns num evenly spaced samples, 
# calculated over the interval [start, stop]. The endpoint of the interval can optionally be excluded.
#

### numpy.random.randn
## numpy.random.randn(d0, d1, ..., dn)
#
# Return a sample (or samples) from the "standard norma" distribution.
#
# If positive, int_like or int-convertible arguments are provided, randn generates an array of 
# shape (d0, d1, ..., dn), filled with random floats sampled from a univariate “normal” (Gaussian) 
# distribution of mean 0 and variance 1 (if any of the d_i are floats, they are first converted to 
# integers by truncation). A single float randomly sampled from the distribution is returned if no argument is provided.

data = np.array([np.linspace(-2, 4, num_examples), np.linspace(-6, 6, num_examples)])
data += np.random.randn(2, num_examples)

x_data, y_data = data
x_with_bias = np.array([(1., a) for a in x_data]).astype(np.float32)


### random_normal
## tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
#
# Outputs random values from a normal distribution.

xhat  = tf.constant(x_with_bias)
target = tf.constant(np.transpose([y_data]).astype(np.float32))
weights = tf.Variable(tf.random_normal([2, 1], 0, 0.1))

init = tf.global_variables_initializer()

### l2_loss
## tf.nn.l2_loss(t, name=None)
#
# Computes half the L2 norm of a tensor without the sqrt:
# output = sum(t ** 2) / 2
# 
# t:    A Tensor
# name: A name for the operation

yhat = tf.matmul(xhat, weights)
yerr = tf.subtract(yhat, target)
loss = tf.nn.l2_loss(yerr)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train     = optimizer.minimize(loss)

sess = tf.Session()
sess.run(init)

for _ in range(training_step):
    sess.run(train)
    losses.append(sess.run(loss))

best_weights = sess.run(weights)
y_results    = sess.run(yhat)
sess.close()


fig, (ax1, ax2) = plt.subplots(1, 2)
plt.subplots_adjust(wspace=.3)
fig.set_size_inches(10, 4)
line_x_range = (1, 6)

ax1.scatter(x_data, y_data, alpha=0.7, c = "b")
ax1.scatter(x_data, np.transpose(y_results)[0], c="g", alpha=0.6)
ax1.plot(line_x_range, [best_weights[0] + a * best_weights[1] for a in line_x_range], "g", alpha=0.6)

ax2.plot(range(0, training_step), losses)
ax2.set_ylabel("Loss")
ax2.set_xlabel("Training steps")

plt.show()

## graph
# https://github.com/tor4z/tensorflow-example/blob/master/images/snn.png