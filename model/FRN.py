# -*- coding: utf-8 -*-
# @Time    : 2021/11/29 9:18 下午
# @Author  : hongming
# @File    : FRN.py
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
from util.attention import attention as atten_func
import json

class FRN():
    def _init_(self,user_features,item_features,context_features,interaction_feature,ubq_features):
        with open("config/algo_conf.json","r") as fp:
            self.config = json.load(fp)
        self.model_conf=self.config["CVR"]["modelx"]
        self.model_name ="FRN"
        self.partitioner=partitioned_variables.min_max_variable_partitioner(
                                   max_partitions=self.config["CVR"]['ps_num'],
                                   min_slice_size=self.config["CVR"]['embedding_min_slice_size'])

        self.main_column_blocks = [interaction_feature,item_features,user_features]
        self.bias_column_blocks = [user_features,context_features]
        self.seq_column_blocks = [ubq_features]

        try:
            self.is_training = tf.get_default_graph().get_tensor_by_name("training:0")
        except KeyError:
            self.is_training = tf.placeholder(tf.bool, name="training")

        self.layer_dict = {}
        self.sequence_layer_dict = {}
        self.seq_column_blocks = []
        self.seq_column_len = {}
        self.seq_column_atten = {}
        arr_blocks = ubq_features
        for block in arr_blocks:
            arr = block.split(':', -1)
            if self.model_conf['model_hyperparameter']['atten_param']['atten_type'] == 'parallel':
                if len(arr) != 4: continue
                if len(arr[0]) > 0:
                    self.seq_column_blocks.append(arr[0])
                if len(arr[1]) > 0:
                    self.seq_column_len[arr[0]] = arr[1]
                if len(arr[2]) > 0:
                    self.seq_column_atten[arr[0] + '_user'] = arr[2]
                if len(arr[3]) > 0:
                    self.seq_column_atten[arr[0] + '_item'] = arr[3]
            elif self.model_conf['model_hyperparameter']['atten_param']['atten_type'] == 'traditional':
                if len(arr) != 3: continue
                if len(arr[0]) > 0:
                    self.seq_column_blocks.append(arr[0])
                if len(arr[1]) > 0:
                    self.seq_column_len[arr[0]] = arr[1]
                if len(arr[2]) > 0:
                    self.seq_column_atten[arr[0]] = arr[2]
            elif self.model_conf['model_hyperparameter']['atten_param']['atten_type'] == 'simple':
                # TODO
                pass

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
        self.embedding_layer(features, feature_columns)
        self.sequence_layer()
        dnn_output = self.dnn_layer()  # list
        logits_b, base_rep = self.logits_layer(dnn_output)
        return logits_b,base_rep

    def build_graph(self,mode):
        self.features,self.feature_columns,self.labels=self.gen_features(mode)
        self.logits_b,self.base_rep=self.inference(self.features,self.feature_columns)
        self.loss_op = self.loss(self.logits_b,self.labels)
        self.train_op = self.optimizer(loss_op=self.loss_op)
        self.set_global_step()

    def sequence_layer(self):
            for block_name in self.sequence_layer_dict.keys():
                with tf.variable_scope(name_or_scope="Share_Sequence_Layer_{}".format(block_name),
                                           partitioner=self.partitioner,
                                           reuse=tf.AUTO_REUSE) as scope:

                        max_len = self.config["CVR"]["modelx"]["model_hyperparameter"]["atten_param"]["seq_len"]

                        sequence = self.sequence_layer_dict[block_name]
                        if block_name not in self.seq_column_len or self.seq_column_len[
                            block_name] not in self.layer_dict:
                            sequence_mask = tf.sequence_mask(tf.ones_like(sequence[:, 0, 0], dtype=tf.int32), 1)
                            sequence_mask = tf.tile(sequence_mask, [1, max_len])
                        else:
                            sequence_length = self.layer_dict[self.seq_column_len[block_name]]
                            sequence_mask = tf.sequence_mask(tf.reshape(sequence_length, [-1]), max_len)

                        # sequence self attention
                        # vec shape: (batch, seq_len, dim)
                        dec = []

                        if self.model_conf['model_hyperparameter']['atten_param'].get('self', True):
                            print('add self attention.')
                            vec, self_atten_weight = atten_func(query_masks=sequence_mask,
                                                                key_masks=sequence_mask,
                                                                queries=sequence,
                                                                keys=sequence,
                                                                num_units=
                                                                self.model_conf['model_hyperparameter']['atten_param'][
                                                                    'sa_num_units'],
                                                                num_output_units=
                                                                self.model_conf['model_hyperparameter']['atten_param'][
                                                                    'sa_num_output_units'],
                                                                scope=block_name + "self_attention",
                                                                atten_mode=
                                                                self.model_conf['model_hyperparameter']['atten_param'][
                                                                    'atten_mode'],
                                                                reuse=tf.AUTO_REUSE,
                                                                num_heads=
                                                                self.model_conf['model_hyperparameter']['atten_param'][
                                                                    'num_heads'],
                                                                residual_connection=
                                                                self.model_conf['model_hyperparameter'][
                                                                    'atten_param'].get('residual_connection', False),
                                                                attention_normalize=
                                                                self.model_conf['model_hyperparameter'][
                                                                    'atten_param'].get('attention_normalize', False))
                        else:
                            vec = sequence

                        # must be given user attention blocks or item attention blocks
                        if self.seq_column_atten[block_name + '_user'] not in self.layer_dict and self.seq_column_atten[
                            block_name + '_item'] not in self.layer_dict:
                            raise RuntimeError("No existing attention layer.")

                        # user attention
                        if self.seq_column_atten[block_name + '_user'] in self.layer_dict:
                            self.logger.info('add user attention.')
                            attention_layer = tf.concat(self.layer_dict[self.seq_column_atten[block_name + '_user']],
                                                        axis=1)
                            attention = tf.expand_dims(attention_layer, 1)

                            # sequence X user intent attention
                            # user_vec shape: (batch, 1, att_out)
                            user_vec, user_atten_weight = atten_func(queries=attention,
                                                                     keys=vec,
                                                                     key_masks=sequence_mask,
                                                                     query_masks=tf.sequence_mask(
                                                                         tf.ones_like(attention[:, 0, 0],
                                                                                      dtype=tf.int32), 1),
                                                                     num_units=self.model_conf['model_hyperparameter'][
                                                                         'atten_param']['ma_num_units'],
                                                                     num_output_units=
                                                                     self.model_conf['model_hyperparameter'][
                                                                         'atten_param']['ma_num_output_units'],
                                                                     scope=block_name + "user_multihead_attention",
                                                                     atten_mode=self.model_conf['model_hyperparameter'][
                                                                         'atten_param']['atten_mode'],
                                                                     reuse=tf.AUTO_REUSE,
                                                                     num_heads=self.model_conf['model_hyperparameter'][
                                                                         'atten_param']['num_heads'],
                                                                     residual_connection=
                                                                     self.model_conf['model_hyperparameter'][
                                                                         'atten_param'].get('residual_connection',
                                                                                            False),
                                                                     attention_normalize=
                                                                     self.model_conf['model_hyperparameter'][
                                                                         'atten_param'].get('attention_normalize',
                                                                                            False))

                            if self.model_conf['model_hyperparameter']['atten_param'].get('residual_connection', False):
                                ma_num_output_units = attention.get_shape().as_list()[-1]
                            else:
                                ma_num_output_units = self.model_conf['model_hyperparameter']['atten_param'][
                                    'ma_num_output_units']
                            dec.append(tf.reshape(user_vec, [-1, ma_num_output_units]))
                        # item attention
                        if self.seq_column_atten[block_name + '_item'] in self.layer_dict:
                            print('add item attention.')
                            attention_layer = tf.concat(self.layer_dict[self.seq_column_atten[block_name + '_item']],
                                                        axis=1)
                            attention = tf.expand_dims(attention_layer, 1)

                            # sequence X target attention
                            # item_vec shape: (batch, seq_len, att_out)
                            item_vec, item_atten_weight = atten_func(queries=attention,
                                                                     keys=vec,
                                                                     key_masks=sequence_mask,
                                                                     query_masks=tf.sequence_mask(
                                                                         tf.ones_like(attention[:, 0, 0],
                                                                                      dtype=tf.int32), 1),
                                                                     num_units=self.model_conf['model_hyperparameter'][
                                                                         'atten_param']['ma_num_units'],
                                                                     num_output_units=
                                                                     self.model_conf['model_hyperparameter'][
                                                                         'atten_param']['ma_num_output_units'],
                                                                     scope=block_name + "item_multihead_attention",
                                                                     atten_mode=self.model_conf['model_hyperparameter'][
                                                                         'atten_param']['atten_mode'],
                                                                     reuse=tf.AUTO_REUSE,
                                                                     num_heads=self.model_conf['model_hyperparameter'][
                                                                         'atten_param']['num_heads'],
                                                                     residual_connection=
                                                                     self.model_conf['model_hyperparameter'][
                                                                         'atten_param'].get('residual_connection',
                                                                                            False),
                                                                     attention_normalize=
                                                                     self.model_conf['model_hyperparameter'][
                                                                         'atten_param'].get('attention_normalize',
                                                                                            False))

                            if self.model_conf['model_hyperparameter']['atten_param'].get('residual_connection', False):
                                ma_num_output_units = attention.get_shape().as_list()[-1]
                            else:
                                ma_num_output_units = self.model_conf['model_hyperparameter']['atten_param'][
                                    'ma_num_output_units']
                            dec.append(tf.reshape(item_vec, [-1, ma_num_output_units]))
                            self.debug_tensor_collector[block_name + 'item_attention'] = item_atten_weight
                        self.layer_dict[block_name] = tf.concat(dec, axis=1)

    def input_embedding_layer(self, features, feature_columns):
        with tf.variable_scope(name_or_scope="Embedding_Layer",
                               partitioner=self.partitioner,reuse=tf.AUTO_REUSE) as scope:
            block_list = (self.main_column_blocks + self.bias_column_blocks)
            for block_name in block_list:
                if block_name not in feature_columns or len(feature_columns[block_name]) <= 0:
                    raise ValueError("block_name:(%s) not in feature_columns for embed" % block_name)
                print("block_name:%s, len(feature_columns[block_name])=%d" %(block_name, len(feature_columns[block_name])))
                self.layer_dict[block_name] = layers.input_from_feature_columns(features,feature_columns=feature_columns[block_name], scope=scope)

        # use behavior sequence feature
        with tf.variable_scope(name_or_scope="seq_input_from_feature_columns",
                               partitioner=self.partitioner,reuse=tf.AUTO_REUSE) as scope:
            if len(self.seq_column_blocks) > 0:
                for block_name in self.seq_column_blocks:
                    if block_name not in feature_columns or len(feature_columns[block_name]) <= 0:
                        raise ValueError("block_name:(%s) not in feature_columns for seq" % block_name)
                    seq_len = self.config["CVR"]["modelx"]["model_hyperparameter"]["atten_param"]["seq_len"]
                    '''
                    sequence_layer.shape=(batch*max_seq_len, dimension_sum)
                    eg. batch_size=2, seq_len=2, dimension=3
                        item_id.shape=(2, 2, 3)
                        brand_id.shape=(2, 2, 3)
                        item_id = [[[0, 1, 2], [12, 13, 14]], 
                                   [[6, 7, 8], [18, 19, 20]]]
                        brand_id = [[[3, 4, 5], [15, 16, 17]],
                                    [[9, 10, 11], [21, 22, 23]]]

                        ===> sequence_layer = [[ 0,  1,  2,  3,  4,  5],
                                               [ 6,  7,  8,  9, 10, 11],
                                               [12, 13, 14, 15, 16, 17],
                                               [19, 20, 21, 22, 23, 24]]
                    '''
                    sequence_layer = layers.input_from_feature_columns(features, feature_columns[block_name],scope=scope)

                    if self.model_conf['model_hyperparameter']['atten_param']['seq_type'] == 'sum':
                        '''
                        sequence_layer.shape=(batch*max_seq_len, dimension_sum)
                        sequence_layer = [[ 3,  5,  7],
                                        [15, 17, 19],
                                        [27, 29, 31],
                                        [39, 41, 43]]
                        '''
                        sequence_split = tf.split(sequence_layer, len(feature_columns[block_name]), axis=1)
                        sequence_stack = tf.stack(values=sequence_split)
                        sequence_layer = tf.reduce_sum(sequence_stack, axis=0)
                    sequence = tf.split(sequence_layer, seq_len, axis=0)
                    sequence_stack = tf.stack(values=sequence, axis=1)
                    sequence_2d = tf.reshape(sequence_stack, [-1, tf.shape(sequence_stack)[2]])

                    if block_name in self.seq_column_len and self.seq_column_len[block_name] in self.layer_dict:
                        sequence_length = self.layer_dict[self.seq_column_len[block_name]]
                        sequence_mask = tf.sequence_mask(tf.reshape(sequence_length, [-1]), seq_len)
                        sequence_stack = tf.reshape(tf.where(tf.reshape(sequence_mask, [-1]),
                                                             sequence_2d, tf.zeros_like(sequence_2d)),
                                                    tf.shape(sequence_stack))
                    else:
                        sequence_stack = tf.reshape(sequence_2d, tf.shape(sequence_stack))
                    # (B,N,d)
                    self.sequence_layer_dict[block_name] = sequence_stack

        with tf.variable_scope(name_or_scope="atten_input_from_feature_columns",
                               partitioner=partitioned_variables.min_max_variable_partitioner(
                                   max_partitions=self.ps_num,
                                   min_slice_size=self.embedding_min_slice_size
                               ),
                               reuse=tf.AUTO_REUSE) as scope:
            for atten_block_name in self.seq_column_atten.values():
                if len(atten_block_name) <= 0: continue
                if atten_block_name not in feature_columns or len(feature_columns[atten_block_name]) <= 0:
                    raise ValueError("block_name:(%s) not in feature_columns for atten" % atten_block_name)
                self.layer_dict[atten_block_name] = layers.input_from_feature_columns(features,
                                                                                      feature_columns[
                                                                                          atten_block_name],
                                                                                      scope=scope)
                # self.debug_tensor_collector[atten_block_name] = self.layer_dict[atten_block_name]

        with tf.variable_scope(name_or_scope="relation_network_input",
                               partitioner=partitioned_variables.min_max_variable_partitioner(
                                   max_partitions=self.ps_num,
                                   min_slice_size=self.embedding_min_slice_size
                               ),
                               reuse=tf.AUTO_REUSE) as scope:
            for block_name in self.relation_column_blocks:
                if block_name not in feature_columns or len(feature_columns[block_name]) <= 0:
                    raise ValueError("block_name:(%s) not in feature_columns for embed" % block_name)
                self.layer_dict[block_name] = layers.input_from_feature_columns(features,
                                                                                feature_columns=feature_columns[
                                                                                    block_name], scope=scope)
                self.logger.info(
                    "block_name:%s, shape=%s" % (block_name, str(tf.shape(self.layer_dict[block_name]))))

    def main_net(self):
        main_net_layer = []
        for block_name in (self.main_column_blocks + self.seq_column_blocks):
            if not self.layer_dict.has_key(block_name):
                raise ValueError('[Main net, layer dict] does not has block : {}'.format(block_name))
            main_net_layer.append(self.layer_dict[block_name])
            print ('[main_net] add %s' % block_name)
        with tf.variable_scope(name_or_scope="{}_Main_Score_Network".format(self.model_name),
                                 partitioner=self.partitioner):
            main_out = tf.concat(values=main_net_layer, axis=1)
            # self.debug_tensor_collector['main_net_in'] = main_out
            for layer_id, num_hidden_units in enumerate(
                    self.model_conf['model_hyperparameter']['main_dnn_hidden_units']):
                with self.variable_scope(name_or_scope="hiddenlayer_{}".format(layer_id)) as dnn_hidden_layer_scope:
                    main_out = layers.fully_connected(
                        main_out,
                        num_hidden_units,
                        tf.nn.relu,
                        scope=dnn_hidden_layer_scope,
                        normalizer_fn=layers.batch_norm if self.model_conf['model_hyperparameter'].get('batch_norm',
                                                                                                       True) else None,
                        normalizer_params={"scale": True, "is_training": self.is_training})

                if self.model_conf['model_hyperparameter']['need_dropout']:
                    main_out = tf.layers.dropout(
                        main_out,
                        rate=self.model_conf['model_hyperparameter']['dropout_rate'],
                        noise_shape=None,
                        seed=None,
                        training=self.is_training,
                        name=None)
            return main_out

    def bias_net(self):
            if len(self.bias_column_blocks) <= 0:
                return
            bias_net_layer = []
            for block_name in self.bias_column_blocks:
                if not self.layer_dict.has_key(block_name):
                    raise ValueError('[Bias net, layer dict] does not has block : {}'.format(block_name))
                bias_net_layer.append(self.layer_dict[block_name])
                print('[bias_net] add %s' % block_name)
            with tf.variable_scope(name_or_scope="{}_Bias_Score_Network".format(self.model_name),
                                     partitioner=self.partitioner
                                     ):
                bias_out = tf.concat(values=bias_net_layer, axis=1)
                for layer_id, num_hidden_units in enumerate(
                        self.model_conf['model_hyperparameter']['bias_dnn_hidden_units']):
                    with self.variable_scope(
                            name_or_scope="hiddenlayer_{}".format(layer_id)) as dnn_hidden_layer_scope:
                        bias_out = layers.fully_connected(
                            bias_out,
                            num_hidden_units,
                            tf.nn.relu,
                            scope=dnn_hidden_layer_scope,
                            normalizer_fn=layers.batch_norm if self.model_conf['model_hyperparameter'].get(
                                'batch_norm',
                                True) else None,
                            normalizer_params={"scale": True, "is_training": self.is_training})

                    if self.model_conf['model_hyperparameter']['need_dropout']:
                        bias_out = tf.layers.dropout(
                            bias_out,
                            rate=self.model_conf['model_hyperparameter']['dropout_rate'],
                            noise_shape=None,
                            seed=None,
                            training=self.is_training,
                            name=None)
                return bias_out

    def dnn_layer(self):
        main_out = self.main_net()
        bias_out = self.bias_net()
        collect = [main_out, bias_out]
        return collect

    def logits_layer(self, dnn_output, logits_name=None):
        with self.variable_scope(name_or_scope="{}_Logits".format(self.model_name if logits_name is None else logits_name),
                                 partitioner=self.partitioner):
            base_rep = tf.concat(dnn_output, axis=1)
            for layer_id, num_hidden_units in enumerate(
                    self.model_conf['model_hyperparameter']['logits_dnn_hidden_units']):
                with self.variable_scope(name_or_scope="hiddenlayer_{}".format(layer_id)) as dnn_hidden_layer_scope:
                    base_rep = layers.fully_connected(
                        base_rep,
                        num_hidden_units,
                        tf.nn.relu,
                        scope=dnn_hidden_layer_scope,
                        normalizer_fn=layers.batch_norm if self.model_conf['model_hyperparameter'].get('batch_norm', True) else None,
                        normalizer_params={"scale": True, "is_training": self.is_training})
            _logits = layers.linear(
                base_rep,
                1,
                scope="logits_net",
                variables_collections=["output_layer"],
                outputs_collections=["output"],
                biases_initializer=None)
            _bias = contrib_variables.model_variable(
                'bias_weight',
                shape=[1],
                collections=[ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.MODEL_VARIABLES],
                initializer=tf.zeros_initializer(),
                trainable=True)
            _logits = nn_ops.bias_add(_logits, _bias)
        return _logits, base_rep

    def loss(self,logits,labels):
        with tf.name_scope("{}_Loss_Op".format(self.model_name)):
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        return loss

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

    def set_global_step(self):
        """Sets up the global step Tensor."""
        self.global_step = training_util.get_or_create_global_step()
        self.global_step_reset = tf.assign(self.global_step, 0)
        self.global_step_add = tf.assign_add(self.global_step, 1, use_locking=True)
        tf.summary.scalar('global_step/' + self.global_step.name, self.global_step)

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

