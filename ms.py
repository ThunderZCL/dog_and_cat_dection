#coding=utf-8
#保存soft.ph和soft.ckpt
#created by tengxing on 2017.2.22
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets("/home/thunder/桌面/MNIST_data/", one_hot=True)

#create model
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32,[None,784],name='x_input')
    y_ = tf.placeholder(tf.float32,[None,10],name='y_input')
with tf.name_scope('layer'):
    with tf.name_scope('W'):
        #tf.zeros([3, 4], tf.int32) ==> [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        W = tf.Variable(tf.zeros([784,10]),name='Weights')
    with tf.name_scope('b'):
        b = tf.Variable(tf.zeros([10]),name='biases')
    with tf.name_scope('W_p_b'):
        Wx_plus_b = tf.add(tf.matmul(x, W), b, name='Wx_plus_b')

    y = tf.nn.softmax(Wx_plus_b, name='final_result')
    print y

#define loss and optimizer
with tf.name_scope('loss'):
    loss = -tf.reduce_sum(y_ * tf.log(y))
with tf.name_scope('train_step'):
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    print train_step
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
sess.run(init)
writer = tf.summary.FileWriter("logs/", sess.graph)
#train
for step in range(100):
    batch_xs,batch_ys =mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys})
    print step
    variables = tf.all_variables()
    saver = tf.train.Saver(variables)
    print len(variables)
    print sess.run(b)
    #print W.get_shape(),b.get_shape()
    saver.save(sess, "data/soft.ckpt")
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
a = accuracy.eval({x:mnist.test.images,y_:mnist.test.labels})
print '最终的测试正确率：{0}'.format(a)
tf.train.write_graph(sess.graph_def,'graph','soft.pb',False)
