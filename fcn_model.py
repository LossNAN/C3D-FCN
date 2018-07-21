
# coding: utf-8

import tensorflow as tf

NUM_CLASSES = 21

def conv3d(name, l_input, w, b,padding=0):
    if padding==1:
        conv = tf.nn.bias_add( tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='VALID'), b)
        conv = tf.contrib.layers.batch_norm(conv, updates_collections=None, epsilon=1e-05, decay=0.9, scale=True, center=True)
        return conv
    else:
        conv = tf.nn.bias_add( tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'), b)
        conv = tf.contrib.layers.batch_norm(conv, updates_collections=None, epsilon=1e-05, decay=0.9, scale=True, center=True)
        return conv

def deconv3d(name,l_input, w, b, output_shape, padding=0):
    # Deconv layer
    if padding == 0:
        deconv = tf.nn.conv3d_transpose(l_input, w, output_shape=output_shape, strides=[1, 2, 1, 1, 1], padding="SAME")
    else:
        deconv = tf.nn.conv3d_transpose(l_input, w, output_shape=output_shape, strides=[1, 2, 1, 1, 1], padding="VALID")
    deconv = tf.nn.bias_add(deconv, b)
    deconv = tf.nn.relu(deconv)
    deconv = tf.contrib.layers.batch_norm(deconv, updates_collections=None, epsilon=1e-05, decay=0.9, scale=True, center=True)
    return deconv


def max_pool(name, l_input, k,m=1):
    return tf.nn.max_pool3d(l_input, ksize=[1, k, 2*m, 2*m, 1], strides=[1, k, 2*m, 2*m, 1], padding='SAME', name=name)


def inference_fcn5(list, _dropout, batch_size, _myweights, _mybiases):
    conv3 = list[0]#256*L/2*28*28
    conv4 = list[1]#512*L/4*14*14
    conv5 = list[2]#512*L/8*7*7 
    # Convolution Layer
    conv5 = conv3d('conv5', conv5, _myweights['wconv5'], _mybiases['bconv5'],padding=0)
    conv5 = tf.nn.relu(conv5, 'relu5')
    conv5 = conv3d('conv5', conv5, _myweights['wdown5'], _mybiases['bdown5'],padding=1)
    conv5 = tf.nn.relu(conv5, 'relu5')
    
    output_shape = conv5.get_shape().as_list()
    output_shape[1] *= 2
    output_shape[4] = _myweights['wup5'].get_shape().as_list()[3]
    conv5 = deconv3d('deconv5',conv5,_myweights['wup5'],_mybiases['bup5'],output_shape,padding=0)#512*L/4*1*1
    conv5 = tf.nn.relu(conv5, 'relu5')
    
    output_shape = conv5.get_shape().as_list()
    output_shape[1] *= 2
    output_shape[4] = _myweights['wup4'].get_shape().as_list()[3]
    conv5 = deconv3d('deconv5',conv5,_myweights['wup4'],_mybiases['bup4'],output_shape,padding=0)
    conv5 = tf.nn.relu(conv5, 'relu5')
    
    output_shape = conv5.get_shape().as_list()
    output_shape[1] *= 2
    output_shape[4] = _myweights['wup3'].get_shape().as_list()[3]
    conv5 = deconv3d('deconv5',conv5,_myweights['wup3'],_mybiases['bup3'],output_shape,padding=0)
    conv5 = tf.nn.relu(conv5, 'relu5')
    conv5 = tf.transpose(conv5, perm=[0,1,4,2,3])
    out = tf.reshape(conv5, [batch_size, conv5.get_shape().as_list()[1], conv5.get_shape().as_list()[2]]) 
    print out.shape
    return out