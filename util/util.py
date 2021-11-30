from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import init_ops
from tensorflow.contrib.layers.python.layers import regularizers
import tensorflow as tf
import re
from tensorflow.python.ops import math_ops
import traceback
import global_var as gl
import tensorflow.contrib.opt as opt


def model_arg_scope(weight_decay=0.0005, weights_initializer=initializers.xavier_initializer(),
                    biases_initializer=init_ops.zeros_initializer()):
  with arg_scope(
      [layers.fully_connected, layers.conv2d],
      weights_initializer=weights_initializer,
      weights_regularizer=regularizers.l2_regularizer(weight_decay),
      biases_initializer=biases_initializer) as arg_sc:
    return arg_sc


def gelu(input_tensor):
  """Gaussian Error Linear Unit.
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    input_tensor: float Tensor to perform activation.
  Returns:
    `input_tensor` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
  return input_tensor * cdf


def getActivationFunctionOp(act_name="relu"):
  if type(act_name) != str and type(act_name) != unicode:
    return act_name
  if act_name.lower() == 'relu':
    return tf.nn.relu
  elif act_name.lower() == 'tanh':
    return tf.nn.tanh
  elif act_name.lower() == 'lrelu':
    return lambda x: tf.nn.leaky_relu(x, alpha=0.01)
  elif act_name.lower() == 'llrelu':
    return lambda x: tf.nn.leaky_relu(x, alpha=0.1)
  elif act_name.lower() == 'gelu':
    return lambda x: gelu(x)
  else:
    return tf.nn.relu

def combine_parted_variables(vs, combine_postfix='/combine'):
  mmap = {}
  for x in vs:
    mkey = re.sub('/part_[0-9]*:', '/', x.name)
    if mkey in mmap:
      mmap[mkey].append(x)
    else:
      mmap[mkey] = [x]
  return [tf.concat(t, 0, name=k.replace(':', '_') + combine_postfix) for k, t in mmap.items()]

def value_percentile_summary(name, value, percentile):
  tf.summary.scalar(name=name + '-p' + str(percentile),
                    tensor=tf.contrib.distributions.percentile(value, percentile))
  tf.summary.scalar(name=name + '-p' + str(100 - percentile),
                    tensor=tf.contrib.distributions.percentile(value, 100 - percentile))

def value_percentile_summary_many(name, value):
  value_percentile_summary(name, value, 0.1)
  value_percentile_summary(name, value, 0.05)
  value_percentile_summary(name, value, 0.01)

def add_norm2_summary(collection_name, summary_prefix="Norm2/", contain_string=""):
  variables = tf.get_collection(collection_name)
  vv = combine_parted_variables(variables)
  logger = gl.get_value('logger')
  for x in vv:
    if contain_string in x.name:
      logger.info("add norm2 %s to summary with shape %s" % (str(x.name), str(x.shape)))
      try:
        tf.summary.scalar(name=summary_prefix + x.name.replace(":", "_"), tensor=tf.norm(x))
      except:
        tf.summary.scalar(name=summary_prefix + x.name.replace(":", "_"), tensor=tf.norm(x, axis=[-2, -1]))


def greater_zero_fraction(value, name=None):
  with tf.name_scope(name, "greater_fraction", [value]):
    value = tf.convert_to_tensor(value, name="value")
    zero = tf.constant(0, dtype=value.dtype, name="zero")
    return math_ops.reduce_mean(
      math_ops.cast(math_ops.greater(value, zero), tf.float32))


def add_weight_summary(collection_name, summary_prefix="Weight/", contain_string=""):
  variables = tf.get_collection(collection_name)
  vv = combine_parted_variables(variables)
  logger = gl.get_value('logger')
  for x in vv:
    if contain_string in x.name:
      try:
        name = x.name.replace(":", "_")
        x = tf.reshape(x, [-1])
        logger.info("add weight %s to summary with shape %s" % (str(x.name), str(x.shape)))

        tf.summary.scalar(name=summary_prefix + "Norm2/" + name,
                          tensor=tf.norm(x, axis=-1))
        tf.summary.histogram(name=summary_prefix + "Hist/" + name,
                             values=x)
        #value_percentile_summary_many(name, x)
        mean, variance = tf.nn.moments(x, axes=0)
        tf.summary.scalar(name=summary_prefix + "Mean/" + name,
                          tensor=mean)
        tf.summary.scalar(name=summary_prefix + "Variance/" + name,
                          tensor=variance)
        tf.summary.scalar(name=summary_prefix + "PosRatio/" + name, tensor=greater_zero_fraction(x))
      except Exception as e:
        logger.warn('Got exception run : %s | %s' % (e, traceback.format_exc()))
        logger.warn("add_dense_output_summary with rank not 2: [%s],shape=[%s]" % (str(x.name), str(x.shape)))


# histogram of each sample's zero fraction
def greater_zero_histogram(value, name=None):
  with tf.name_scope(name, "greater_histogram", [value]):
    value = tf.convert_to_tensor(value, name="value")
    zero = tf.constant(0, dtype=value.dtype, name="zero")
    return math_ops.reduce_mean(
      math_ops.cast(math_ops.greater(value, zero), tf.float32), axis=-1)


def add_dense_output_summary(collection_name, summary_prefix="DenseOutput/", contain_string=""):
  variables = tf.get_collection(collection_name)
  logger = gl.get_value('logger')
  for x in variables:
    if contain_string in x.name:
      try:
        logger.info("add dense_output %s to summary with shape %s" % (str(x.name), str(x.shape)))
        tf.summary.histogram(name=summary_prefix + "Hist/" + x.name, values=x)
        #value_percentile_summary_many(x.name, x)
        if len(x.shape) == 3:
          tf.summary.scalar(name=summary_prefix + "Norm2/" + x.name.replace(":", "_"),
                            tensor=tf.reduce_mean(tf.norm(x)))
          tf.summary.histogram(name=summary_prefix + "AbsHistMax/" + x.name.replace(":", "_"),
                               values=tf.reduce_max(tf.abs(x), axis=-1))
          mean, variance = tf.nn.moments(x, axes=[0, 1])
          tf.summary.scalar(name=summary_prefix + "Mean/" + x.name.replace(":", "_"),
                            tensor=tf.reduce_mean(mean))
          tf.summary.scalar(name=summary_prefix + "Variance/" + x.name.replace(":", "_"),
                            tensor=tf.reduce_mean(variance))
          tf.summary.histogram(name=summary_prefix + "PosR/" + x.name.replace(":", "_"),
                               values=greater_zero_histogram(x))
        elif x.shape[1] > 1:
          tf.summary.scalar(name=summary_prefix + "Norm2/" + x.name.replace(":", "_"),
                            tensor=tf.reduce_mean(tf.norm(x, axis=-1)))
          tf.summary.histogram(name=summary_prefix + "AbsHistMax/" + x.name.replace(":", "_"),
                               values=tf.reduce_max(tf.abs(x), axis=-1))
          mean, variance = tf.nn.moments(x, axes=-1)
          tf.summary.scalar(name=summary_prefix + "Mean/" + x.name.replace(":", "_"),
                            tensor=tf.reduce_mean(mean))
          tf.summary.scalar(name=summary_prefix + "Variance/" + x.name.replace(":", "_"),
                            tensor=tf.reduce_mean(variance))
          tf.summary.histogram(name=summary_prefix + "PosR/" + x.name.replace(":", "_"),
                               values=greater_zero_histogram(x))
        else:
          tf.summary.scalar(name=summary_prefix + "Norm2/" + x.name.replace(":", "_"),
                            tensor=tf.reduce_mean(tf.norm(x, axis=-1)))
          tf.summary.histogram(name=summary_prefix + "AbsHistMax/" + x.name.replace(":", "_"),
                               values=tf.reduce_max(tf.abs(x)))
          mean, variance = tf.nn.moments(x, axes=0)
          tf.summary.scalar(name=summary_prefix + "Mean/" + x.name.replace(":", "_"),
                            tensor=mean[0])
          tf.summary.scalar(name=summary_prefix + "Variance/" + x.name.replace(":", "_"),
                            tensor=variance[0])
        tf.summary.scalar(name=summary_prefix + "PosRatio/" + x.name.replace(":", "_"), tensor=greater_zero_fraction(x))
      except Exception as e:
        logger.warn('Got exception run : %s | %s' % (e, traceback.format_exc()))
        logger.warn("add_dense_output_summary with rank not 2: [%s],shape=[%s]" % (str(x.name), str(x.shape)))


def add_embed_layer_norm(layer_tensor, columns, omit=None):
  if layer_tensor is None:
    return
  i = 0
  for column in sorted(set(columns), key=lambda x: x.key):
    try:
      dim = column.dimension
    except:
      dim = column.embedding_dimension
    if omit is not None and column.name in omit:
      i += dim
      continue
    elif omit is not None:
      omit.add(column.name)
    tf.summary.scalar(name=column.name, tensor=tf.reduce_mean(tf.norm(layer_tensor[:, i:i + dim], axis=-1)))
    i += dim


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


def getOptimizer(optimizer, global_step=None, learning_rate=None, learning_rate_decay_fn=None):
  if optimizer == "AdagradDecay":
    if learning_rate == None:
      learning_rate = 0.01
    if learning_rate is not None and learning_rate_decay_fn is not None:
      if global_step is None:
        raise ValueError("global_step is required for learning_rate_decay_fn.")
      learning_rate = learning_rate_decay_fn(learning_rate, global_step)
    return tf.train.AdagradDecayOptimizer(learning_rate, global_step,
                                          accumulator_decay_step=5000000,
                                          accumulator_decay_rate=0.95)
  if optimizer == "AdamAsync":
    if learning_rate == None:
      learning_rate = 0.01
    if learning_rate is not None and learning_rate_decay_fn is not None:
      if global_step is None:
        raise ValueError("global_step is required for learning_rate_decay_fn.")
      learning_rate = learning_rate_decay_fn(learning_rate, global_step)
    return tf.train.AdamAsyncOptimizer(learning_rate, beta1=0.9,
                                       beta2=0.999,
                                       epsilon=1e-8)
  if optimizer.lower() == "ftrl":
    opti = lambda lr: tf.train.FtrlOptimizer(
      learning_rate=lr,
      initial_accumulator_value=0.1,  # more less , more sparse
      l1_regularization_strength=0.1,  # more large, more sparse
      l2_regularization_strength=0,
      use_locking=False
    )
    return opti
  else:
    return optimizer

def getInitOp(init_para, act_name="zero"):
  if type(act_name) != str and type(act_name) != unicode:
    return act_name

  if act_name.lower() == 'zero':
    return tf.zeros_initializer
  elif act_name.lower() == 'constant':
    return tf.constant_initializer(init_para)
  elif act_name.lower() == 'xavier':
    return initializers.xavier_initializer()
  else:
    return tf.zeros_initializer


from log import logger
import os, random, time
import socket
import json
import tensorflow as tf
from tensorflow.python.ops import array_ops


class Util:
  @staticmethod
  def getHostSpec():
    start, end = [61001, 65000]
    port = 0
    while True:
      time.sleep(2)
      pscmd = "netstat -ntl |grep -v Active| grep -v Proto|awk '{print $4}'|awk -F: '{print $NF}'"
      cont = os.popen(pscmd).read().strip()
      portList = [int(one) for one in cont.split('\n')]
      port = random.randint(start, end)
      if port not in portList:
        break
    hostIp = socket.gethostbyname(socket.gethostname())
    return (hostIp, port)

  @staticmethod
  def getHostIp():
    hostIp = socket.gethostbyname(socket.gethostname())
    return hostIp

  @staticmethod
  def getExternalHostIp():
    hostIp = 'none'
    if os.path.isfile('/etc/hostinfo'):
      with open('/etc/hostinfo') as f:
        hostIp = f.read().strip().split('\n')[-1].strip()
    return hostIp

  @staticmethod
  def set_metrics_env(context):
    properties = context.get_properties()
    os.environ['metric_open'] = properties.get('SYS:metric_open', 'yes')
    logger.info('set metric_open default %s', os.environ['metric_open'])
    os.environ['metric_sample'] = properties.get('SYS:metric_sample', '60')
    logger.info('set metric_sample default %s', os.environ['metric_sample'])

  @staticmethod
  def exportTFConfig(clusterJson, task_index):
    os.environ['TF_CONFIG'] = json.dumps({
      'cluster': {'ps': clusterJson['ps'], 'worker': clusterJson['worker']},
      'task': {'type': 'worker', 'index': task_index},
    })

  # @staticmethod
  # def toSparseMapping(clusterJson, task_index):
  #     return {
  #         'ps': clusterJson['ps'],
  #         'worker': {task_index: str(task_index) + ".zk.rpc:0"}
  #     }

  @staticmethod
  def toSparseMapping(clusterJson, task_index, workerInfo):
    logger.info("sparse " + str(task_index) + " " + workerInfo)
    return {
      'ps': clusterJson['ps'],
      'worker': {task_index: workerInfo}
    }

  @staticmethod
  def toSeastarSparseMapping(clusterJson, task_index, workerInfo, seastarWorkerInfo):
    logger.info("sparse seastar" + str(task_index) + " " + workerInfo + " " + seastarWorkerInfo)
    return {
      'ps': clusterJson['ps'],
      'worker': {task_index: workerInfo},
      'ps_seastar': clusterJson['ps_seastar'],
      'worker_seastar': {task_index: seastarWorkerInfo}
    }

  @staticmethod
  def getInputFiles(fileName):
    with open(fileName, 'r') as load_f:
      fileInfo = json.load(load_f)
    input_files = []
    for i in fileInfo:
      input_files.append(i["resourcePath"])
    return input_files

  @staticmethod
  def getCurrentWorkerFileList(fileList, taskIndex, workerNum):
    currentFileList = []
    if len(fileList) < workerNum:
      logger.info("file num less than task num %s %s" % (str(len(fileList)), str(workerNum)))
    for i in range(len(fileList)):
      if taskIndex == i % workerNum:
        currentFileList.append(fileList[i])
    return currentFileList

  @staticmethod
  def string2kv(s, d1, d2):
    kv = {}
    if type(s) == type(None) or s == '':
      return kv
    for ele in s.split(d1):
      pair = ele.split(d2)
      if len(pair) != 2:
        continue
      kv[pair[0]] = pair[1]
    return kv

  @staticmethod
  def get_value_by_default(kv, key, default_value):
    if kv.has_key(key):
      return kv[key]
    return default_value

  @staticmethod
  def reset_variables(collection_key=tf.GraphKeys.LOCAL_VARIABLES, matchname='auc/'):
    localv = tf.get_collection(collection_key)
    localv = [x for x in localv if matchname in x.name]
    retvops = [tf.assign(x, array_ops.zeros(shape=x.get_shape(), dtype=x.dtype)) for x in localv]
    if len(retvops) == 0:
      return None, None
    retvops = tf.tuple(retvops)
    return retvops, localv

  @staticmethod
  def getFileListFromInfo(fileInfo):
    input_files = []
    for i in fileInfo:
      input_files.append(i["resourcePath"])
    return input_files

  @staticmethod
  def getFileInfoFromPath(fileName):
    with open(fileName, 'r') as load_f:
      fileInfo = json.load(load_f)
    return fileInfo

  @staticmethod
  def parse_application_id(work_path):
    for s in work_path.split('/'):
      if s.startswith('application_'):
        return s
    return "unknown_{}".format(int(round(time.time())))

  @staticmethod
  def toRealClusterSpec(clusterJson):
    return {
      'ps': clusterJson['ps'],
      'worker': clusterJson['worker']
    }

  @staticmethod
  def getCurrentTime():
    import datetime
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
