import numpy as np
import tensorflow as tf

class GaussianProcess(object):
    def __init__(self, learning_rate=0.001, ls=.1, sigma_f=.1, sigma_n=.01, ):
        self.session = tf.Session()

        self.ls = tf.Variable(np.array(ls))
        self.sigma_f = tf.Variable(np.array(sigma_f))
        self.sigma_n = tf.Variable(np.array(sigma_n))

        self.X = tf.placeholder(tf.float64, [None, None])
        self.y = tf.placeholder(tf.float64, [None, None])
        self.X_ = tf.placeholder(tf.float64, [None, None])
        self.y_, self.var_ = self._contruct_predictor()

        self.L = self._construct_loss()
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.optimizer = opt.minimize(-self.L)
        self.init = tf.global_variables_initializer()

    def _construct_K(self, X, X_=None):
        X = self.X / self.ls
        Xs = tf.reduce_sum(tf.square(X), 1)
        if (X_ is None):
            Sq_dist = -2 * tf.matmul(X, tf.transpose(X)) + \
                tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
        else:
            X_ = X_ / self.ls
            X_s = tf.reduce_sum(tf.square(X_), 1)
            Sq_dist = -2 * tf.matmul(X, tf.transpose(X_)) + \
                tf.reshape(Xs, (-1, 1)) + tf.reshape(X_s, (1, -1))
        K = tf.square(self.sigma_f) * tf.exp(-0.5 * Sq_dist)
        return K

    def _contruct_predictor(self):
        I = tf.diag(tf.ones([tf.shape(self.X)[0]], dtype=tf.float64))
        self.K = self._construct_K(self.X) + \
             I * (1e-7 + tf.square(self.sigma_n))
        self.Ki = tf.matrix_inverse(self.K)
        K_ = self._construct_K(self.X, self.X_)
        K_T = tf.transpose(K_)
        K_TKi = tf.matmul(K_T, self.Ki)
        y_ = tf.matmul(K_TKi, self.y)
        var_ = - tf.reduce_sum(K_T * K_TKi, 1) + tf.square(self.sigma_f)
        return y_, var_

    def _construct_loss(self):
        yTKiy = tf.matmul(tf.transpose(self.y), tf.matmul(self.Ki, self.y))
        logK = tf.log(tf.matrix_determinant(self.K))
        nlog2pi = tf.cast(tf.shape(self.X)[0], tf.float64) * \
            tf.log(2 * tf.cast(np.pi, tf.float64))
        L = - 0.5 * (yTKiy + logK + nlog2pi)
        return L

    def fit(self, X, y, train=True, iterations=10000):
        self.dataX = X
        self.datay = y
        self.session.run(self.init)

        if (train):
            fd = {self.X: self.dataX, self.y: self.datay}
            for i in range(0, iterations):
                (_, L) = self.session.run((self.optimizer, self.L), fd)
                print(i, L)
        
    
    def predict(self, X_):
        fd = {self.X: self.dataX, self.y: self.datay, self.X_: X_}
        (y_, var_) = self.session.run((self.y_, self.var_), feed_dict=fd)
        return y_, var_


