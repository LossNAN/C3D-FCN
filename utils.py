import os
import time
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_train_data
import input_test_data
import c3d_model
import fcn_model
import numpy as np

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

def fcn_model_loss(logits, labels, batch_size):
    cross_entropy_mean = 0
    print labels
    print logits
    #time for sum
    for i in range(batch_size):
        cross_entropy_mean += tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels[i],logits=logits[i])
        )
    #batch_size for average
    cross_entropy_mean = cross_entropy_mean/batch_size

    tf.summary.scalar('loss', cross_entropy_mean)
    weight_decay_loss = tf.get_collection('weightdecay_losses')
    tf.summary.scalar('weight_decay_loss', tf.reduce_mean(weight_decay_loss) )
    # Calculate the total loss for the current tower.
    total_loss = (cross_entropy_mean + weight_decay_loss) 
    tf.summary.scalar('total_loss', tf.reduce_mean(total_loss) )
    return total_loss

def c3d_model_loss(logits, labels, batch_size):
    cross_entropy_mean = 0
    #time for sum
    cross_entropy_mean += tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
        )
    #batch_size for average
    cross_entropy_mean = cross_entropy_mean/batch_size
    tf.summary.scalar('loss', cross_entropy_mean)
    weight_decay_loss = tf.get_collection('weightdecay_losses')
    tf.summary.scalar('weight_decay_loss', tf.reduce_mean(weight_decay_loss) )
    # Calculate the total loss for the current tower.
    total_loss = (cross_entropy_mean + weight_decay_loss) 
    tf.summary.scalar('total_loss', tf.reduce_mean(total_loss) )
    return total_loss

def tower_acc(logit, labels,batch_size):
    accuracy = 0
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
