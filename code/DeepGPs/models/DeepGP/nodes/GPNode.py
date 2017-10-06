from .BaseNode import *
from ..kernel.GaussianKernel import SquaredExponential as SE

import tensorflow as tf
import numpy as np

# Log determinant of a positive semi-definite matrix
def getLogDet(M):
    return 2.0 * tf.reduce_sum(tf.log(tf.diag_part(tf.cholesky(M))), 0)

# Inverse of a positive semi-definite matrix
def getInversePSD(M):
    n = M.get_shape().as_list()[0]
    return tf.cholesky_solve(tf.cholesky(M), tf.eye(n))

class GPNode(BaseNode):
    def __init__(self,
                 input_means,
                 input_vars,
                 n_points,
                 n_inducing_points,
                 set_for_training,
                 initial = None):
        BaseNode.__init__(self, input_means, input_vars)
        self.input_means = input_means
        self.input_vars = input_vars
        self.n_inducing_points = n_inducing_points
        self.input_d = input_means.get_shape().as_list()[1]
        self.batch_size = tf.shape(input_means)[0]
        self.n_points = n_points
        self.set_for_training = set_for_training

        # Covariance parameters of the cavities 
        self.LParamPost = tf.Variable(
          tf.random_normal(((self.n_inducing_points, self.n_inducing_points))))
        # Mean parameters of the cavities
        self.mParamPost = tf.Variable(
            tf.random_normal((self.n_inducing_points, 1)))
        self.lls = tf.Variable(tf.zeros([1, self.input_d], dtype=tf.float32))
        self.lsf = tf.Variable(0.0, dtype=tf.float32)
        if (initial is None):
            self.z = tf.Variable(
                tf.random_uniform([self.n_inducing_points, self.input_d], -1, 1))
        else:
            self.z = tf.Variable(initial, dtype=tf.float32)
        jitter = tf.cast(1e-3, tf.float32)

        # Below is based on the equations from page 8
        # Expectation of Kxz w.r.t the input
        EKxz = SE.get_psi1(self.lls,    
                           self.lsf,
                           self.input_means,
                           self.input_vars,
                           self.z)
        Kzz = SE.get_kernel(self.lls, self.lsf, self.z, self.z)
        Kzz += tf.eye(self.n_inducing_points) * jitter * tf.exp(self.lsf)
        KzzInv = getInversePSD(Kzz)
        Lu = tf.matrix_band_part(self.LParamPost, 0, -1)
        LParamPost_tri = Lu + tf.diag(tf.exp(tf.diag_part(self.LParamPost)) \
                                      - tf.diag_part(self.LParamPost))
        LtL = tf.matmul(tf.transpose(LParamPost_tri), LParamPost_tri)
        scalar = (self.n_points - self.set_for_training) / self.n_points
        covCavityInv = KzzInv + LtL * scalar

        covCavity = getInversePSD(covCavityInv)
        meanCavity = tf.matmul(covCavity, scalar * self.mParamPost)
        KzzInvcovCavity = tf.matmul(KzzInv, covCavity)
        KzzInvmeanCavity = tf.matmul(KzzInv, meanCavity)
        self.output_means = tf.matmul(EKxz, KzzInvmeanCavity)

        Kxz = SE.get_kernel(self.lls, self.lsf, self.input_means, self.z)
        B1 = tf.matmul(KzzInvcovCavity, KzzInv) - KzzInv 
        v_out = tf.exp(self.lsf) + tf.reduce_sum(Kxz * tf.matmul(Kxz, B1),
                                                 1,
                                                 keep_dims = True)
        B2 = tf.matmul(KzzInvmeanCavity, tf.transpose(KzzInvmeanCavity))

        # Below is based on the equation (35)
        # L is the expectation of Kzz
        # B1 is Kinv
        # B2 is betabetaT
        L = SE.get_L(self.lls,    
                     self.lsf,
                     self.z,
                     self.input_means,
                     self.input_vars)
        k = tf.expand_dims(Kxz, 2)
        kT = tf.expand_dims(Kxz, 1)
        kkT = tf.matmul(k, kT)
        l = tf.expand_dims(EKxz, 2)
        lT = tf.expand_dims(EKxz, 1)
        llT = tf.matmul(l, lT)
        L_kk = L - kkT 
        L_ll = L - llT
        # Calculating the traces for the two terms
        v1 = tf.reduce_sum(tf.expand_dims(B2, 0) \
                           * tf.transpose(L_ll, [0, 2, 1]), [1, 2])
        v2 = tf.reduce_sum(tf.expand_dims(B1, 0) \
                           * tf.transpose(L_kk, [0, 2, 1]), [1, 2])
        v1 = tf.abs(tf.expand_dims(v1, 1))
        v2 = tf.abs(tf.expand_dims(v2, 1))
        
        self.output_vars = v_out + v2 + v1

        # Finally calculate the energy (page 9)
        logZpost = self.getLogNormalizerPosterior(KzzInv, LtL)
        logZprior = self.getLogNormalizerPrior(KzzInv)
        logZcav = self.getLogNormalizerCavity(meanCavity,
                                              covCavity,
                                              covCavityInv)

        # We multiply by the minibatch size and normalize terms
        # according to the total number of points (n_points)
        self.v1 = v1
        self.v2 = v2
        self.vout = v_out
        self.energy = (logZcav - logZpost) * self.n_points + logZpost \
            - logZprior

    def getLogNormalizerCavity(self, mean, cov, covInv):
        logDet = getLogDet(cov)
        return 0.5 * self.n_inducing_points * np.log(2.0 * np.pi) \
            + 0.5 * logDet \
            + 0.5 * tf.matmul(tf.matmul(tf.transpose(mean), covInv), mean)

    def getLogNormalizerPrior(self, KzzInv):
        logDet = getLogDet(KzzInv)
        return 0.5 * self.n_inducing_points * np.log(2.0 * np.pi) \
               - 0.5 * logDet

    def getLogNormalizerPosterior(self, KzzInv, LtL):
        covInv = KzzInv + LtL
        cov = getInversePSD(covInv)
        mean = tf.matmul(cov, self.mParamPost)
        self.z_mean = mean
        logDet = getLogDet(cov)
        return 0.5 * self.n_inducing_points * np.log(2 * np.pi) \
            + 0.5 * logDet \
            + 0.5 * tf.matmul(tf.matmul(tf.transpose(mean), covInv), mean)

    def getEnergyContribution(self):
        return self.energy

    def getOutput(self):
        return self.output_means, self.output_vars


