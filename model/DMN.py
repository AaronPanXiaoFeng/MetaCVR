# -*- coding: utf-8 -*-
# @Time    : 2021/11/29 9:09 下午
# @Author  : hongming
# @File    : DMN.py
# @Software: PyCharm

import sys
import os
import tensorflow as tf

cur_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.abspath(os.path.join(cur_path, '..')))
from tensorflow.python.ops import partitioned_variables
from tensorflow.contrib import layers
from tensorflow.python.training import training_util
from util.optimizer import optimizer_ops as myopt
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework.errors_impl import OutOfRangeError, ResourceExhaustedError
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from util.util import *
from FRN import FRN

from util.attention import attention as atten_func
import json


class DMN():
    def __init__(self):
        with open("config/algo_conf.json","r") as fp:
            self.config = json.load(fp)
        self.model_conf=self.config["CVR"]["modelx"]
        self.model_name ="DMN"
        self.partitioner=partitioned_variables.min_max_variable_partitioner(
                                   max_partitions=self.config["CVR"]['ps_num'],
                                   min_slice_size=self.config["CVR"]['embedding_min_slice_size'])
        self.layer_dict={}
        try:
            self.is_training = tf.get_default_graph().get_tensor_by_name("training:0")
        except KeyError:
            self.is_training = tf.placeholder(tf.bool, name="training")
    def gen_features(self,mode="Train"):
        # this function is used to generate input data flow!
        if mode=="Train":
            features=None
            feature_columns=None
            labels=None
            return features,feature_columns,labels
        elif mode=="eval":
            features = None
            feature_columns = None
            labels = None
            return features, feature_columns,labels

    def inference(self,features,feature_columns):
        FRN_net=FRN()
        FRN_net.is_training = False
        self.base_logits,self.sample_rep = FRN_net.inference(features,feature_columns)
        self.metric_logits =self.DMN_net(self.sample_rep) #list
        logits = self.EPN_net(self.base_logits,self.metric_logits)
        return logits

    def build_graph(self):
        self.feature,self.feature_columns,self.labels = self.gen_features()
        self.logits = self.inference(self.feature,self.feature_columns)
        self.loss_op = self.loss(self.logits)
        self.train_op = self.optimizer(self.loss_op)
        self.set_global_step()


    def optimizer(self, loss_op):
        '''
        return train_op
        '''
        with tf.variable_scope(
                name_or_scope="Optimize",
                partitioner=partitioned_variables.min_max_variable_partitioner(
                    max_partitions=self.config.get_job_config("ps_num"),
                    min_slice_size=self.config.get_job_config("embedding_min_slice_size")
                ),
                reuse=tf.AUTO_REUSE):

            global_opt_name = None
            global_optimizer = None
            global_opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=None)

            if len(global_opt_vars) == 0:
                raise ValueError("no trainable variables")

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            train_ops = []
            for opt_name, opt_conf in self.opts_conf.items():
                optimizer = self.get_optimizer(opt_name, opt_conf, self.global_step)
                if 'scope' not in opt_conf or opt_conf["scope"] == "Global":
                    global_opt_name = opt_name
                    global_optimizer = optimizer
                else:
                    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=opt_conf["scope"])
                    if len(vars) != 0:
                        for var in vars:
                            if var in global_opt_vars:
                                global_opt_vars.remove(var)
                        train_op, _, _ = myopt.optimize_loss(
                            loss=loss_op,
                            global_step=self.global_step,
                            learning_rate=opt_conf.get("learning_rate", 0.01),
                            optimizer=optimizer,
                            # update_ops=update_ops,
                            clip_gradients=opt_conf.get('clip_gradients', 5),
                            variables=vars,
                            increment_global_step=False,
                            summaries=myopt.OPTIMIZER_SUMMARIES)
                        train_ops.append(train_op)
            if global_opt_name is not None:
                train_op, self.out_gradient_norm, self.out_var_norm = myopt.optimize_loss(
                    loss=loss_op,
                    global_step=self.global_step,
                    learning_rate=self.opts_conf[global_opt_name].get("learning_rate", 0.01),
                    optimizer=global_optimizer,
                    # update_ops=update_ops,
                    clip_gradients=self.opts_conf[global_opt_name].get('clip_gradients', 5.0),
                    variables=global_opt_vars,
                    increment_global_step=False,
                    summaries=myopt.OPTIMIZER_SUMMARIES,
                )
                train_ops.append(train_op)

            with tf.control_dependencies(update_ops):
                with ops.control_dependencies([train_op_vec]):
                    with ops.colocate_with(self.global_step):
                        last_train_op = state_ops.assign_add(self.global_step, 1).op
                        return last_train_op

    def loss(self,logits,labels):
        with tf.name_scope("{}_Loss_Op".format(self.model_name)):
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        return loss

    def DMN_net(self, base_rep):
            # config
            relation_config = self.model_conf['model_hyperparameter']['relation_net']
            relation_type = relation_config['relation_type']
            merge_type = relation_config['merge_type']
            relation_units = relation_config['relation_units']
            occasion_set = relation_config['occasion_set']
            len_occasion_set = len(occasion_set)
            respectively_collect = relation_config['collect_type']['respectively']
            collect_units = relation_config['collect_type']['collect_units']

            # init
            collect_logits = []
            collect_logits_pos = []
            collect_logits_neg = []

            with tf.variable_scope(name_or_scope="{}_DMN_Network".format(self.model_name),
                                     partitioner=self.partitioner):

                def define_support_vector_pair(ttype):
                    # Saved weights
                    p_layer_w = contrib_variables.model_variable(
                        name='RN_%s_P_layer' % ttype,
                        shape=[relation_units, 1],
                        trainable=True,
                        initializer=tf.zeros_initializer())
                    n_layer_w = contrib_variables.model_variable(
                        name='RN_%s_n_layer' % ttype,
                        shape=[relation_units, 1],
                        trainable=True,
                        initializer=tf.zeros_initializer())

                    # Train cond
                    def relation_net_train(p_layer_w, p_layer, n_layer_w, n_layer):
                        p_layer_w = self.partition_assign(p_layer_w, p_layer)
                        n_layer_w = self.partition_assign(n_layer_w, n_layer)
                        return p_layer_w, n_layer_w

                    def relation_net_valid(p_layer_w, p_layer, n_layer_w, n_layer):
                        return p_layer_w, n_layer_w

                    if self.model_conf['run_mode'] == 'Train':
                        p_layer = tf.reshape(self.layer_dict['_'.join(['pos', ttype])][0, :], [-1, 1])
                        n_layer = tf.reshape(self.layer_dict['_'.join(['neg', ttype])][0, :], [-1, 1])
                        p_layer_w, n_layer_w = tf.cond(
                            self.is_training,
                            lambda: relation_net_train(p_layer_w, p_layer, n_layer_w, n_layer),
                            lambda: relation_net_valid(p_layer_w, p_layer, n_layer_w, n_layer)
                        )
                        p_layer_w = tf.stop_gradient(p_layer_w)
                        n_layer_w = tf.stop_gradient(n_layer_w)
                    return p_layer_w, n_layer_w

                # CONCAT
                if relation_type == 'concat':
                    # occasion logits
                    for ttype in occasion_set:
                        with arg_scope(
                                model_arg_scope(weight_decay=self.model_conf['model_hyperparameter']['dnn_l2_reg'])):
                            with self.variable_scope(name_or_scope="{}_part".format(ttype)) as relation_part_scope:
                                p_layer_w, n_layer_w = define_support_vector_pair(ttype)
                                tile_vector = tf.ones_like(tf.transpose(tf.reduce_sum(base_rep, axis=1)))
                                tile_vector = tf.reshape(tile_vector, [1,
                                                                       -1])  # tile_vector=[1, batch_size], base_rep=[batch_size, 128]
                                p_layer_w = tf.transpose(tf.matmul(p_layer_w, tile_vector))
                                n_layer_w = tf.transpose(tf.matmul(n_layer_w, tile_vector))
                                if respectively_collect:
                                    p_rep = layers.fully_connected(
                                        tf.concat([p_layer_w, base_rep], axis=1),
                                        collect_units,
                                        getActivationFunctionOp(relation_config['activation']),
                                        scope='RN_linear_%s_P' % ttype,
                                        normalizer_fn=layers.batch_norm if self.model_conf['model_hyperparameter'].get(
                                            'batch_norm',
                                            True) else None,
                                        normalizer_params={"scale": True, "is_training": self.is_training})
                                    n_rep = layers.fully_connected(
                                        tf.concat([n_layer_w, base_rep], axis=1),
                                        collect_units,
                                        getActivationFunctionOp(relation_config['activation']),
                                        scope='RN_linear_%s_N' % ttype,
                                        normalizer_fn=layers.batch_norm if self.model_conf['model_hyperparameter'].get(
                                            'batch_norm',
                                            True) else None,
                                        normalizer_params={"scale": True, "is_training": self.is_training})
                                    if self.model_conf['model_hyperparameter']['need_dropout'] and collect_units > 1:
                                        p_rep = tf.layers.dropout(
                                            p_rep,
                                            rate=relation_config['dropout_rate'],
                                            noise_shape=None,
                                            seed=None,
                                            training=self.is_training,
                                            name=None)
                                        n_rep = tf.layers.dropout(
                                            n_rep,
                                            rate=relation_config['dropout_rate'],
                                            noise_shape=None,
                                            seed=None,
                                            training=self.is_training,
                                            name=None)
                                    collect_logits_pos.append(p_rep)
                                    collect_logits_neg.append(n_rep)
                                else:
                                    _rep = layers.fully_connected(
                                        tf.concat([p_layer_w, n_layer_w, base_rep], axis=1),
                                        collect_units,
                                        getActivationFunctionOp(relation_config['activation']),
                                        scope='RN_linear_%s' % ttype,
                                        normalizer_fn=layers.batch_norm if self.model_conf['model_hyperparameter'].get(
                                            'batch_norm',
                                            True) else None,
                                        normalizer_params={"scale": True, "is_training": self.is_training})
                                    if self.model_conf['model_hyperparameter']['need_dropout'] and collect_units > 1:
                                        _rep = tf.layers.dropout(
                                            _rep,
                                            rate=relation_config['dropout_rate'],
                                            noise_shape=None,
                                            seed=None,
                                            training=self.is_training,
                                            name=None)
                                    collect_logits.append(_rep)

                # MULTI_RELA_NET
                elif relation_type == 'mulrelanet':
                    collect_units = 1
                    # occasion logits
                    for ttype in occasion_set:
                        with arg_scope(
                                model_arg_scope(weight_decay=self.model_conf['model_hyperparameter']['dnn_l2_reg'])):
                            with tf.variable_scope(name_or_scope="{}_part".format(ttype)):
                                # Input
                                p_layer_w, n_layer_w = define_support_vector_pair(ttype)
                                # Positive
                                p_base_rep = layers.linear(base_rep, relation_units,
                                                           scope='RN_%s_P_linear' % ttype,
                                                           biases_initializer=None)
                                # Negative
                                n_base_rep = layers.linear(base_rep, relation_units,
                                                           scope='RN_%s_N_linear' % ttype,
                                                           biases_initializer=None)
                                if self.model_conf['model_hyperparameter']['need_dropout']:
                                    p_base_rep = tf.layers.dropout(
                                        p_base_rep,
                                        rate=relation_config['dropout_rate'],
                                        noise_shape=None,
                                        seed=None,
                                        training=self.is_training,
                                        name=None)
                                    n_base_rep = tf.layers.dropout(
                                        n_base_rep,
                                        rate=relation_config['dropout_rate'],
                                        noise_shape=None,
                                        seed=None,
                                        training=self.is_training,
                                        name=None)
                                # Bias
                                _bias = contrib_variables.model_variable(
                                    name='RN_%s_bias' % ttype,
                                    shape=[1],
                                    initializer=tf.zeros_initializer(), trainable=True)
                                p_rep = getActivationFunctionOp(relation_config['activation'])(
                                    tf.matmul(p_base_rep, p_layer_w))
                                n_rep = getActivationFunctionOp(relation_config['activation'])(
                                    tf.matmul(n_base_rep, n_layer_w))
                                if respectively_collect:
                                    collect_logits_pos.append(p_rep)
                                    collect_logits_neg.append(n_rep)
                                else:
                                    _logits = p_rep - n_rep + _bias
                                    collect_logits.append(_logits)
                                    self.debug_tensor_collector['%s_rep' % ttype] = _logits

                if respectively_collect:
                    if merge_type == 'fc':
                        p_logits = layers.fully_connected(
                            tf.concat(collect_logits_pos, axis=-1),
                            relation_config['merge_type_fc']['hidden_units'],
                            getActivationFunctionOp(relation_config['activation']),
                            scope='relation_reduce_p',
                            normalizer_fn=layers.batch_norm if self.model_conf['model_hyperparameter'].get(
                                'batch_norm',
                                True) else None,
                            normalizer_params={"scale": True, "is_training": self.is_training})
                        n_logits = layers.fully_connected(
                            tf.concat(collect_logits_neg, axis=-1),
                            relation_config['merge_type_fc']['hidden_units'],
                            getActivationFunctionOp(relation_config['activation']),
                            scope='relation_reduce_n',
                            normalizer_fn=layers.batch_norm if self.model_conf['model_hyperparameter'].get(
                                'batch_norm',
                                True) else None,
                            normalizer_params={"scale": True, "is_training": self.is_training})
                        _logits = layers.linear(
                            tf.concat([p_logits, n_logits], axis=-1),
                            1,
                            scope="relation_reduce",
                            biases_initializer=None)
                        _bias = contrib_variables.model_variable(
                            'bias_weight',
                            shape=[1],
                            collections=[ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.MODEL_VARIABLES],
                            initializer=tf.zeros_initializer(),
                            trainable=True)
                        _logits = tf.reshape(nn_ops.bias_add(_logits, _bias), [-1, 1])
                elif not respectively_collect and collect_units == 1:
                    if merge_type == 'max':
                        _logits = tf.reshape(tf.reduce_max(tf.concat(collect_logits, axis=-1), axis=1), [-1, 1])
                    elif merge_type == 'mean':
                        _logits = tf.reshape(tf.reduce_mean(tf.concat(collect_logits, axis=-1), axis=1), [-1, 1])
                    elif merge_type == 'occaware':
                        # occasion aware
                        for layer_id, num_hidden_units in enumerate(
                                relation_config['merge_type_occaware']['hidden_units']):
                            occ_out = base_rep
                            with tf.variable_scope(
                                    name_or_scope="occ_aware_{}".format(layer_id)) as dnn_hidden_layer_scope:
                                occ_out = layers.fully_connected(
                                    occ_out,
                                    num_hidden_units,
                                    getActivationFunctionOp(relation_config['activation']),
                                    scope=dnn_hidden_layer_scope,
                                    normalizer_fn=layers.batch_norm if self.model_conf['model_hyperparameter'].get(
                                        'batch_norm',
                                        True) else None,
                                    normalizer_params={"scale": True, "is_training": self.is_training})
                        if self.model_conf['model_hyperparameter']['need_dropout']:
                            occ_out = tf.layers.dropout(
                                occ_out,
                                rate=relation_config['dropout_rate'],
                                noise_shape=None,
                                seed=None,
                                training=self.is_training,
                                name=None)
                        # [batch_size*4]
                        occ_aware = layers.linear(occ_out, len_occasion_set,
                                                  scope='occ_aware_last',
                                                  biases_initializer=None)
                        # occ_aware = tf.reshape(occ_aware, [len_occasion_set, 1])
                        occ_aware = tf.nn.softmax(tf.concat(occ_aware, axis=1))  # [batch_size*4]
                        _logits = tf.reshape(
                            tf.reduce_sum(tf.multiply(tf.concat(collect_logits, axis=-1), occ_aware), axis=1), [-1, 1])
                    elif merge_type == 'nomerge':
                        _logits = tf.reshape(tf.concat(collect_logits, axis=-1), [-1, 4])  # [batch_size, 4]
                elif not respectively_collect and collect_units > 1:
                    if merge_type == 'fc':
                        _logits = layers.fully_connected(
                            tf.concat(collect_logits, axis=-1),
                            relation_config['merge_type_fc']['hidden_units'],
                            getActivationFunctionOp(relation_config['activation']),
                            scope='relation_reduce_p',
                            normalizer_fn=layers.batch_norm if self.model_conf['model_hyperparameter'].get(
                                'batch_norm',
                                True) else None,
                            normalizer_params={"scale": True, "is_training": self.is_training})
                        _logits = layers.linear(
                            _logits,
                            1,
                            scope="relation_reduce",
                            biases_initializer=None)
                        _bias = contrib_variables.model_variable(
                            'bias_weight',
                            shape=[1],
                            collections=[ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.MODEL_VARIABLES],
                            initializer=tf.zeros_initializer(),
                            trainable=True)
                        _logits = tf.reshape(nn_ops.bias_add(_logits, _bias), [-1, 1])

                if relation_config['merge_activate']:
                    _logits = getActivationFunctionOp(self.model_conf['model_hyperparameter']['activation'])(_logits)
                return _logits

    def EPN_net(self,base_logits,metric_logits):
        with tf.variable_scope(name_or_scope="{}_EPN_Network".format(self.model_name),
                               partitioner=self.partitioner) as EPN_net_scope:
            output = metric_logits
            output.append(base_logits)
            output = tf.concat(output)
            logits=layers.fully_connected(
                output,
                1,
                None,
                scope=EPN_net_scope,
                normalizer_fn=layers.batch_norm if self.model_conf['model_hyperparameter'].get('batch_norm',
                                                                                               True) else None,
        return logits

    def partition_assign(self, partitioned_var, data):
                for idx, part in enumerate(partitioned_var):
                    return tf.assign(part, data)

    def run_train(self, mon_session, task_index, thread_index):
        localcnt = 0
        while True:
            localcnt += 1
            run_ops = [self.model.global_step, self.model.loss_op, self.model.metrics, self.labels]
            try:
                if task_index == 0:
                    feed_dict = {'training:0': False}
                    global_step, loss, metrics, labels = mon_session.run(run_ops, feed_dict=feed_dict)
                else:
                    feed_dict = {'training:0': True}
                    run_ops.append(self.model.train_ops)
                    global_step, loss, metrics, labels, _ = mon_session.run(run_ops, feed_dict=feed_dict)

                auc, totalauc = metrics['scalar/auc'], metrics['scalar/total_auc']
                self.logger.info(
                    'Global_Step:{}, poslabel:{}, loss={}, auc={}, totalauc={} thread={}'.format(
                        str(global_step),
                        str(labels.sum()),
                        str(loss),
                        str(auc),
                        str(totalauc),
                        str(thread_index)))

            except (ResourceExhaustedError, OutOfRangeError) as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
                break  # release all
            except ConnectionError as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
            except Exception as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))

