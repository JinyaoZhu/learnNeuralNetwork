import tensorflow as tf   
import numpy as np

import os

#ignore tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

D_input = 2
D_label = 1
D_hidden = 2
lr = 1e-2

x = tf.placeholder(tf.float32,[None,D_input],name="x")
t = tf.placeholder(tf.float32,[None,D_label],name="t")

w_h1 = tf.Variable(tf.truncated_normal([D_input,D_hidden],stddev=0.1),name="W_h")

b_h1 = tf.Variable(tf.constant(0.1,shape=[D_hidden]),name="b_h")

pre_act_h1 = tf.matmul(x,w_h1) + b_h1
act_h1 = tf.nn.relu(pre_act_h1,name="act_h")

w_o = tf.Variable(tf.truncated_normal([D_hidden,D_label],stddev=1.0),name="W_o")
b_o = tf.Variable(tf.constant(0.1,shape=[D_label]),name="b_o")

pre_act_o = tf.matmul(act_h1,w_o) + b_o

y = tf.nn.relu(pre_act_o,name="act_y")

loss = tf.reduce_mean((t - y)**2)

train = tf.train.GradientDescentOptimizer(lr).minimize(loss)

X = [[0,0],[0,1],[1,0],[1,1]]
Y = [[0],[0],[0],[1]]
X = np.array(X).astype('float32')
Y = np.array(Y).astype('float32')

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(5000):
    sess.run(train,feed_dict={x:X,t:Y})

print("predict:\n",sess.run(y,feed_dict={x:X}))
print("loss:",sess.run(loss,feed_dict={x:X,t:Y}))
# print(sess.run(w_h1))
