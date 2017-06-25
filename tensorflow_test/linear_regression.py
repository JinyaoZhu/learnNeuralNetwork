import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os

#ignore tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Model parameters
W = tf.Variable([.99], dtype=tf.float32)
b = tf.Variable([-.1], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
# loss
loss = tf.reduce_mean(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# training data
x_train = [1,2,3,4,5,6,7,8]
y_train = [0,2.3,3,3.8,5.2,6.1,7,8]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

plt.figure(1)
x_plot = np.arange(x_train[0]-1,x_train[-1]+1,0.1)
y_plot = curr_W[0]*x_plot + curr_b[0]
plt.plot(x_plot,y_plot,color="b")
plt.scatter(x_train,y_train,color="r")
plt.grid()
plt.xlabel("x")
plt.ylabel("y")
plt.legend(['model','data'])
plt.show()
