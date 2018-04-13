#Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
import os
import time
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_data
import c3d_model
import fcn_model
import math
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 1
#flags.DEFINE_float('learning_rate',1e-4, 'Initial learning rate.')
flags.DEFINE_integer('max_steps',50000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size',16 , 'Batch size.')
FLAGS = flags.FLAGS
MOVING_AVERAGE_DECAY = 0.9999
model_save_dir = './models'
#model_save_dir =os.path.join('./models', 'c3d_ucf_model')
def placeholder_inputs(batch_size):
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         c3d_model.NUM_FRAMES_PER_CLIP,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CHANNELS))
  labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size,c3d_model.NUM_FRAMES_PER_CLIP))
  keep_pro = tf.placeholder(tf.float32)
  return images_placeholder, labels_placeholder,keep_pro


def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x 

def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def tower_loss(name_scope, logit, labels):
  cross_entropy_mean = tf.reduce_mean(
                  tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logit)
                  )
  tf.summary.scalar(
                  name_scope + '_cross_entropy',
                  cross_entropy_mean
                  )
  weight_decay_loss = tf.get_collection('weightdecay_losses')
  tf.summary.scalar(name_scope + '_weight_decay_loss', tf.reduce_mean(weight_decay_loss) )

  # Calculate the total loss for the current tower.
  total_loss = cross_entropy_mean + weight_decay_loss 
  tf.summary.scalar(name_scope + '_total_loss', tf.reduce_mean(total_loss) )
  return total_loss

def fcn_model_loss(name_scope, test, labels,batch_size):
  
  cross_entropy_mean = 0
  print labels
  print test
  #time for sum
  for i in range(batch_size):
     cross_entropy_mean += tf.reduce_sum(
                      tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels[i],logits=test[i])
                       )
  #batch_size for average
  cross_entropy_mean =cross_entropy_mean/batch_size

  tf.summary.scalar(
                  name_scope + '_cross_entropy2',
                  cross_entropy_mean
                  )

  weight_decay_loss = tf.get_collection('weightdecay_losses2')
  tf.summary.scalar(name_scope + '_weight_decay_loss2', tf.reduce_mean(weight_decay_loss) )

  # Calculate the total loss for the current tower.
  total_loss2 = (cross_entropy_mean + weight_decay_loss) 
  tf.summary.scalar(name_scope + '_total_loss2', tf.reduce_mean(total_loss2) )
  return total_loss2


def tower_acc(logit, labels,batch_size):
   accuracy=0
   print logit.shape
   for i in range(batch_size):
      correct_pred = tf.equal(tf.argmax(logit[i], 1), labels[i])
      accuracy += tf.reduce_mean(tf.cast(correct_pred, tf.float32))
   return accuracy/batch_size

def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, wd):
  var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
  if wd is not None:
    weight_decay = tf.nn.l2_loss(var)*wd
    tf.add_to_collection('weightdecay_losses', weight_decay)
  return var

def _variable_with_weight_decay2(name, shape, wd):
  var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
  if wd is not None:
    weight_decay = tf.nn.l2_loss(var)*wd
    tf.add_to_collection('weightdecay_losses2', weight_decay)
  return var


def run_training():
  # Get the sets of images and labels for training, validation, and
  # Tell TensorFlow that the model will be built into the default Graph.

  # Create model directory
  if not os.path.exists(model_save_dir):
      os.makedirs(model_save_dir)
  use_pretrained_model = True
  loss_list=[]
  acc_list=[]
  model_filename = "./sports1m_finetuning_ucf101.model"
  gpu_num=1
  with tf.Graph().as_default():
    global_step = tf.get_variable(
                    'global_step',
                    [],
                    initializer=tf.constant_initializer(0),
                    trainable=False
                    )
    images_placeholder, labels_placeholder ,keep_pro = placeholder_inputs(
                    FLAGS.batch_size * gpu_num
                    )
    tower_grads1 = []
    tower_grads2 = []
    logits = []

    learning_rate_stable=tf.train.exponential_decay(1e-5,global_step,decay_steps=FLAGS.max_steps/FLAGS.batch_size,decay_rate=0.99,staircase=True)
    learning_rate_finetuning=tf.train.exponential_decay(1e-6,global_step,decay_steps=FLAGS.max_steps/FLAGS.batch_size,decay_rate=0.99,staircase=True)
    tf.summary.scalar('learning_rate_stable', learning_rate_stable)
    tf.summary.scalar('learning_rate_finetuning', learning_rate_finetuning)
    opt_stable = tf.train.AdamOptimizer(learning_rate_stable)
    opt_finetuning = tf.train.AdamOptimizer(learning_rate_finetuning)
    with tf.variable_scope('var_name') as var_scope:
      weights = {
              'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.0005),
              'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.0005),
              'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.0005),
              'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.0005),
              'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.0005),
              'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.0005),
              'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.0005),
              'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.0005),
              'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.0005),
              'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.0005),
              'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.0005)
              }
      biases = {
              'bc1': _variable_with_weight_decay('bc1', [64], 0.000),
              'bc2': _variable_with_weight_decay('bc2', [128], 0.000),
              'bc3a': _variable_with_weight_decay('bc3a', [256], 0.000),
              'bc3b': _variable_with_weight_decay('bc3b', [256], 0.000),
              'bc4a': _variable_with_weight_decay('bc4a', [512], 0.000),
              'bc4b': _variable_with_weight_decay('bc4b', [512], 0.000),
              'bc5a': _variable_with_weight_decay('bc5a', [512], 0.000),
              'bc5b': _variable_with_weight_decay('bc5b', [512], 0.000),
              'bd1': _variable_with_weight_decay('bd1', [4096], 0.000),
              'bd2': _variable_with_weight_decay('bd2', [4096], 0.000),
              'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.000),
              }
      fcn_weights = {
              'wconv5': _variable_with_weight_decay2('conv5', [3, 7, 7, 512, 512], 0.0005),
              'wdown5': _variable_with_weight_decay2('down5', [1, 7, 7, 512, 512], 0.0005),
              'wup5': _variable_with_weight_decay2('up5', [2, 1, 1, 512, 512], 0.0005),
              'wconv4': _variable_with_weight_decay2('conv4', [3, 7, 7, 512, 512], 0.0005),
              'wdown4': _variable_with_weight_decay2('down4', [1, 7, 7, 512, 512], 0.0005),
              'wup4': _variable_with_weight_decay2('up4', [2, 1, 1, 4096, 512], 0.0005),
              'wup3': _variable_with_weight_decay2('up3', [2, 1, 1, fcn_model.NUM_CLASSES, 4096], 0.0005),
              #'wconv3': _variable_with_weight_decay2('dconv3', [3, 7, 7, 256,256 ], 0.0005),
              #'wdown3': _variable_with_weight_decay2('down3', [1, 7, 7, 256,4096 ], 0.0005),
              #'wup3': _variable_with_weight_decay2('up3', [2, 1, 1, fcn_model.NUM_CLASSES, 4096], 0.0005),
              }
      fcn_biases = {
              'bconv5': _variable_with_weight_decay2('bconv5', [512], 0.000),
              'bdown5': _variable_with_weight_decay2('bdown5', [512], 0.000),
              'bup5': _variable_with_weight_decay2('bup5', [512], 0.000),
              'bconv4': _variable_with_weight_decay2('bconv4', [512], 0.000),
              'bdown4': _variable_with_weight_decay2('bdown4', [512], 0.000),
              'bup4': _variable_with_weight_decay2('bup4', [4096], 0.000),
              'bup3': _variable_with_weight_decay2('bup3', [fcn_model.NUM_CLASSES], 0.000),
              #'bconv3': _variable_with_weight_decay2('bconv3', [256], 0.000),
              #'bdown3': _variable_with_weight_decay2('bdown3', [4096], 0.000),
              #'bup3': _variable_with_weight_decay2('bup3', [fcn_model.NUM_CLASSES], 0.000),
              }
    for gpu_index in range(0, gpu_num):
      with tf.device('/gpu:%d' % gpu_index):

        varlist1 = list( set(fcn_weights.values() + fcn_biases.values()) )
        varlist2 = [weights['wc3b'],weights['wc4b'],weights['wc5b'],biases['bc3b'],biases['bc4b'],biases['bc5b']]

        feature_map = c3d_model.inference_c3d(
                        images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size,:,:,:,:],
                        keep_pro,
                        FLAGS.batch_size,
                        weights,
                        biases
                        )
        
        logit=fcn_model.inference_fcn(
                        feature_map,
                        keep_pro,
                        FLAGS.batch_size,
                        fcn_weights,
                        fcn_biases
                        )
        
        loss_name_scope = ('gpud_%d_loss' % gpu_index)
       
        loss = fcn_model_loss(
                        loss_name_scope,
                        logit,
                        labels_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size],
                        FLAGS.batch_size
                        )
        grads1 = opt_stable.compute_gradients(loss, varlist1)
        grads2 = opt_finetuning.compute_gradients(loss, varlist2)

        tower_grads1.append(grads1)
        tower_grads2.append(grads2)

    accuracy = tower_acc(logit, labels_placeholder,FLAGS.batch_size)
    tf.summary.scalar('accuracy', accuracy)
    grads1 = average_gradients(tower_grads1)
    grads2 = average_gradients(tower_grads2)
    
    apply_gradient_stable = opt_stable.apply_gradients(grads1, global_step=global_step)    
    apply_gradient_finetuning = opt_finetuning.apply_gradients(grads2, global_step=global_step)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    train_op = tf.group(apply_gradient_stable,apply_gradient_finetuning, variables_averages_op)
    null_op = tf.no_op()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(weights.values() + biases.values())
    new_saver = tf.train.Saver(weights.values() + biases.values()+  fcn_weights.values() + fcn_biases.values())
    init = tf.global_variables_initializer()
    # Create a session for running Ops on the Graph.
    sess = tf.Session(
                    config=tf.ConfigProto(allow_soft_placement=True)
                    )
    sess.run(init)
    merged = tf.summary.merge_all()
  if os.path.isfile(model_filename) and use_pretrained_model:
    print 'loading pretrained_model....'
    saver.restore(sess, model_filename)
  print 'complete!'
  # Create summary writter
  train_writer = tf.summary.FileWriter('./visual_logs/train2', sess.graph)
  test_writer = tf.summary.FileWriter('./visual_logs/test2', sess.graph)
  for step in xrange(FLAGS.max_steps+1):
    start_time = time.time()
    train_images, train_labels, _, _= input_data.read_clip_and_label(
                      filename='annotation/train.list',
                      batch_size=FLAGS.batch_size * gpu_num,
                      num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP,
                      crop_size=c3d_model.CROP_SIZE,
                      shuffle=True
                      )
      
    sess.run(train_op, feed_dict={
                      images_placeholder: train_images,
                      labels_placeholder: train_labels,
                      keep_pro: 0.5
                      })
    duration = time.time() - start_time
    print('Batchnum %d: %.3f sec' % (step, duration))
    
    if (step) %50 == 0 or (step + 1) == FLAGS.max_steps:
        
      print('Step %d/%d: %.3f sec' % (step,FLAGS.max_steps, duration))
      print('Training Data Eval:')
      summary,loss_train,acc = sess.run(
                        [merged, loss,accuracy],
                        feed_dict={
                                      images_placeholder: train_images,
                                      labels_placeholder: train_labels,
                                      keep_pro: 1
                            })
      print 'loss: %f'%np.mean(loss_train)
      print ("accuracy: " + "{:.5f}".format(acc))
      train_writer.add_summary(summary, step)
    
    if (step) %100 == 0 or (step + 1) == FLAGS.max_steps:
        
      print('Validation Data Eval:')
      val_images, val_labels, _, _= input_data.read_clip_and_label(
                        filename='annotation/test.list',
                        batch_size=FLAGS.batch_size * gpu_num,
                        num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP,
                        crop_size=c3d_model.CROP_SIZE,
                        shuffle=True
                        )
      summary,loss_val, acc = sess.run(
                        [merged, loss, accuracy],
                        feed_dict={
                                        images_placeholder: val_images,
                                        labels_placeholder: val_labels,
                                        keep_pro: 1
                                        })
      print 'loss: %f'%np.mean(loss_val)
      print ("accuracy: " + "{:.5f}".format(acc))
      test_writer.add_summary(summary, step)
    # Save the model checkpoint periodically.
    if step > 1 and step % 2000 == 0:
      checkpoint_path = os.path.join('./models', 'model.ckpt')
      new_saver.save(sess, checkpoint_path, global_step=global_step) 
  print("done")

def main(_):
  run_training()

if __name__ == '__main__':
  tf.app.run()
