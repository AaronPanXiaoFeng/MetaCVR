import tensorflow as tf
import math

class SearchAdagrad():
    def __init__(self, conf):
        self.conf = conf

    def get_optimizer(self, global_step):
        learning_rate = self.get_learning_rate(global_step)
        tf.summary.scalar(name="Optimize/learning_rate", tensor=learning_rate)

        return tf.train.AdagradOptimizer(learning_rate)

    def get_learning_rate(self, global_step):
        learning_rate = tf.train.exponential_decay(self.conf['learning_rate'],
                                                   global_step=global_step,
                                                   decay_steps=self.conf['decay_step'],
                                                   decay_rate=self.conf['decay_rate'])
        return learning_rate