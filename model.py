import os
import glob
import time
import numpy as np

import tensorflow as tf

class DataLoader:
    def __init__(self, batch_size=4000):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        self.x_full = x_train[:batch_size]/255
        self.y_full = y_train[:batch_size]
        self.x_test = x_test[:(batch_size//10)] /255
        self.y_test = y_test[:(batch_size//10)]
        print('DataLoader Initialized')
        print('Batch_size: {} \t'.format(batch_size))

    def reset(self):
        '''Initialize data sets and session'''
        self.x_labeled = self.x_full[:0]
        self.y_labeled = self.y_full[:0]
        self.x_unlabeled = self.x_full
        self.y_unlabeled = self.y_full
    
    def label_manually(self, n):
        '''Human powered labeling (actually copying from the prelabeled MNIST dataset).'''
        self.x_labeled = np.concatenate([self.x_labeled, self.x_unlabeled[:n]])
        self.y_labeled = np.concatenate([self.y_labeled, self.y_unlabeled[:n]])
        self.x_unlabeled = self.x_unlabeled[n:]
        self.y_unlabeled = self.y_unlabeled[n:]

class Dense:
    def __init__(self, config):
        self.learning_rate = config.learning_rate
        self.dataLoader = DataLoader(config.batch_size)

        tf_config = tf.compat.v1.ConfigProto()
        self.sess = tf.compat.v1.Session(config=tf_config)

    def build(self):
        tf.compat.v1.disable_eager_execution()

        self.x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28])
        self.x_flat = tf.reshape(self.x, [-1, 28 * 28])
        self.y_ = tf.compat.v1.placeholder(tf.int32, [None])
        self.W = tf.Variable(tf.zeros([28 * 28, 10]), tf.float32)
        self.b = tf.Variable(tf.zeros([10]), tf.float32)
        self.y = tf.matmul(self.x_flat, self.W) + self.b
        self.y_sm = tf.nn.softmax(self.y)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_, tf.cast(tf.argmax(self.y, 1), tf.int32)), tf.float32))

    def resetData(self):
        self.dataLoader.reset()
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def train(self, manual_label_num=400):
        '''Train current labeled dataset until overfit.'''

        x_test = self.dataLoader.x_test
        y_test = self.dataLoader.y_test

        x_labeled = self.dataLoader.x_labeled
        y_labeled = self.dataLoader.y_labeled

        trial_count = 10
        acc = self.sess.run(self.accuracy, feed_dict={self.x:x_test, self.y_:y_test})
        weights = self.sess.run([self.W, self.b])
        while trial_count > 0:
            self.sess.run(self.optimizer, feed_dict={self.x:x_labeled, self.y_:y_labeled})
            acc_new = self.sess.run(self.accuracy, feed_dict={self.x:x_test, self.y_:y_test})
            if acc_new <= acc:
                trial_count -= 1
            else:
                trial_count = 10
                weights = self.sess.run([self.W, self.b])
                acc = acc_new

        self.sess.run([self.W.assign(weights[0]), self.b.assign(weights[1])])    
        acc = self.sess.run(self.accuracy, feed_dict={self.x:x_test, self.y_:y_test})
        print('Labels:', x_labeled.shape[0], '\tAccuracy:', acc)
    

    def active_learning(self, manual_label_num=10):
        for i in range(40):
            # pass unlabeled rest 3990 through the early model
            self.dataLoader.label_manually(manual_label_num)
            self.train()
            res = self.sess.run(self.y_sm, feed_dict={self.x: self.dataLoader.x_unlabeled})
            #find less confident samples
            pmax = np.amax(res, axis=1)
            pidx = np.argsort(pmax)
            #sort the unlabeled corpus on the confidency
            self.dataLoader.x_unlabeled = self.dataLoader.x_unlabeled[pidx]
            self.dataLoader.y_unlabeled = self.dataLoader.y_unlabeled[pidx]
        
        # 나머지 라벨링
        for i in range(320):
            res = self.sess.run(self.y_sm, feed_dict={self.x: self.dataLoader.x_unlabeled})
            self.y_autolabeled = res.argmax(axis=1)
            pmax = np.amax(res, axis=1)
            pidx = np.argsort(pmax)
            self.y_autolabeled = self.y_autolabeled[pidx]
            self.dataLoader.x_unlabeled = self.dataLoader.x_unlabeled[pidx]
            # 상위 10%만 옮기기
            self.dataLoader.x_labeled = np.concatenate([self.dataLoader.x_labeled, self.dataLoader.x_unlabeled[-10:]])
            self.dataLoader.y_labeled = np.concatenate([self.dataLoader.y_labeled, self.y_autolabeled[-10:]])
            self.dataLoader.x_unlabeled = self.dataLoader.x_unlabeled[:-10]
            self.train()
        print('Active Learning')

    def test(self, input_list):
        print('test')
