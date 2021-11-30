import tensorflow as tf
import math

class SearchAdagradDecay():
    def __init__(self, conf):
        self.conf = conf

    def get_optimizer(self, global_step):
        learning_rate = self.get_learning_rate(global_step)
        decay_step = self.get_decay_step()
        decay_rate = self.get_decay_rate()
        tf.summary.scalar(name="Optimize/learning_rate", tensor=learning_rate)

        return tf.train.AdagradDecayOptimizer(learning_rate, global_step,
                                              accumulator_decay_step=decay_step,
                                              accumulator_decay_rate=decay_rate)

    def get_learning_rate(self, global_step):
        def lr_cold_start(lr_tensor, global_step, lrcs_init_lr, lrcs_step):
            """
            Linear  interpolation increase learning rate from "lrcs_init_lr" to "lr_tensor"
            when global step increase from 0 to "lrcs_step"
            :param lr_tensor:
            :param globalstep:
            :param lrcs_init_lr:
            :param lrcs_step:
            :return: Tensor represents new learning rate
            """
            lrcs_init_lr = tf.convert_to_tensor(lrcs_init_lr, dtype=tf.float32)
            lrcs_step = tf.convert_to_tensor(lrcs_step, dtype=tf.float32)
            tlr = lrcs_init_lr + (lr_tensor - lrcs_init_lr) * tf.cast(global_step, tf.float32) / lrcs_step
            tlr = tf.minimum(lr_tensor, tf.maximum(lrcs_init_lr, tlr))
            return tlr

        lr = self.conf["learning_rate"]
        if 'lr_func' in self.conf and \
            self.conf['lr_func'] == 'cold_start':
            learning_rate_func = lambda lr, gs: lr_cold_start(
                lr,
                gs,
                self.conf['lrcs_init_lr'],
                self.conf['lrcs_init_step'])
        else:
            learning_rate_func = lambda lr, gs: lr

        return learning_rate_func(lr, global_step)

    def get_decay_step(self):
        return self.conf["decay_step"]

    def get_decay_rate(self):
        return self.conf["decay_rate"]

    def get_use_locking(self):
        if self.conf["use_locking"] == "True":
            return True
        else:
            return False