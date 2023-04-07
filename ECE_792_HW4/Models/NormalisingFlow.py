'''
Created on 06-Apr-2023

@author: EZIGO
'''
from tensorflow import keras
from Models.AffineCoupling import AffineCoupling
import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf

class NormalisingFlow(keras.Model):
    def __init__(self, num_layers,neurons,reg = 0.01):
        super().__init__()

        self.num_layers = num_layers
        self.neurons=neurons
        self.reg=reg
        # Distribution of the latent space.
        self.distribution = tfp.distributions.MultivariateNormalDiag(loc=[0.0, 0.0], scale_diag=[1.0, 1.0])
        self.masks = np.array([[0, 1], [1, 0]] * (num_layers), dtype="float32")
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.layers_list = [AffineCoupling.build(2,neurons,reg=reg) for i in range(num_layers)]

    @property
    def metrics(self):
        """List of the model's metrics.
        the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]

    def call(self, x, reg = 0.02, training=True):
        log_detJ_inv = 0
        direction = 1
        if training:
            direction = -1
        for i in range(self.num_layers)[::direction]:
            x_masked = x * self.masks[i]
            reversed_mask = 1 - self.masks[i]
            s, t = self.layers_list[i](x_masked)
            s *= reversed_mask
            t *= reversed_mask
            gate = (direction - 1) / 2
            x = (reversed_mask * (x * tf.exp(direction * s) + direction * t * tf.exp(gate * s)) + x_masked)
            log_detJ_inv += gate * tf.reduce_sum(s, [1])
        return x, log_detJ_inv

    # Log likelihood of the normal distribution plus the log determinant of the jacobian.
    def log_loss(self, x):
        y, logdetJ = self(x)
        log_likelihood = self.distribution.log_prob(y) + logdetJ
        return -tf.reduce_mean(log_likelihood)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self.log_loss(data)

        g = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self.log_loss(data)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
    
    
    
    