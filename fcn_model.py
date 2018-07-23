
# coding: utf-8

import tensorflow as tf

NUM_CLASSES = 21

def conv3d(name, l_input, w, b,padding=0):
    if padding==0:
        conv = tf.nn.bias_add( tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='VALID'), b)
        #conv = tf.contrib.layers.batch_norm(conv, updates_collections=None, epsilon=1e-05, decay=0.9, scale=True, center=True)
        return conv
    else:
        conv = tf.nn.bias_add( tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'), b)
        #conv = tf.contrib.layers.batch_norm(conv, updates_collections=None, epsilon=1e-05, decay=0.9, scale=True, center=True)
        return conv

def deconv3d(name, l_input, w, b, output_shape, padding=0):
    # Deconv layer
    if padding == 1:
        deconv = tf.nn.conv3d_transpose(l_input, w, output_shape=output_shape, strides=[1, 2, 1, 1, 1], padding="SAME")
    else:
        deconv = tf.nn.conv3d_transpose(l_input, w, output_shape=output_shape, strides=[1, 2, 1, 1, 1], padding="VALID")
    deconv = tf.nn.bias_add(deconv, b)
    deconv = tf.nn.relu(deconv)
    #deconv = tf.contrib.layers.batch_norm(deconv, updates_collections=None, epsilon=1e-05, decay=0.9, scale=True, center=True)
    return deconv


def max_pool(name, l_input, k,m=1):
    return tf.nn.max_pool3d(l_input, ksize=[1, k, 2*m, 2*m, 1], strides=[1, k, 2*m, 2*m, 1], padding='SAME', name=name)


def inference_pool5(list, _dropout, batch_size, _myweights, _mybiases):
    pool5 = list[0]
    pool4 = list[1]
    pool3 = list[2]
    # Convolution Layer
    conv6 = conv3d('conv6', pool5, _myweights['wconv6'], _mybiases['bconv6'],padding=0)
    conv6 = tf.nn.relu(conv6, 'relu5')
    
    output_shape = conv6.get_shape().as_list()
    output_shape[1] *= 2
    output_shape[4] = _myweights['wup6'].get_shape().as_list()[3]
    CDC6 = deconv3d('deconv6', conv6, _myweights['wup6'], _mybiases['bup6'], output_shape, padding=1)#512*L/4*1*1
    CDC6 = tf.nn.relu(CDC6, 'relu6')
    print CDC6.shape
    output_shape = CDC6.get_shape().as_list()
    output_shape[1] *= 2
    output_shape[4] = _myweights['wup7'].get_shape().as_list()[3]
    CDC7 = deconv3d('deconv7', CDC6, _myweights['wup7'], _mybiases['bup7'], output_shape, padding=1)
    CDC7 = tf.nn.relu(CDC7, 'relu7')
    print CDC7.shape
    output_shape = CDC7.get_shape().as_list()
    output_shape[1] *= 2
    output_shape[4] = _myweights['wup8'].get_shape().as_list()[3]
    CDC8 = deconv3d('deconv8', CDC7, _myweights['wup8'], _mybiases['bup8'], output_shape, padding=1)
    CDC8 = tf.nn.relu(CDC8, 'relu8')
    print CDC8.shape
    CDC8 = tf.transpose(CDC8, perm=[0,1,4,2,3])
    out = tf.reshape(CDC8, [batch_size, CDC8.get_shape().as_list()[1], CDC8.get_shape().as_list()[2]]) 
    print out.shape
    return out

def inference_pool54(list, _dropout, batch_size, _myweights, _mybiases):
    pool5 = list[0]
    pool4 = list[1]
    pool3 = list[2]
    # Convolution Layer
    conv6 = conv3d('conv6', pool5, _myweights['wconv6'], _mybiases['bconv6'],padding=0)
    conv6 = tf.nn.relu(conv6, 'relu4')
   
    conv7 = conv3d('conv7', pool4, _myweights['wconv7'], _mybiases['bconv7'],padding=0)
    conv7 = tf.nn.relu(conv7, 'relu5')
    
    conv7 = tf.add_n([conv6, conv7], name='add')
    output_shape = conv7.get_shape().as_list()
    output_shape[1] *= 2
    output_shape[4] = _myweights['wup6'].get_shape().as_list()[3]
    CDC6 = deconv3d('deconv6', conv7, _myweights['wup6'], _mybiases['bup6'], output_shape, padding=1)#512*L/4*1*1
    CDC6 = tf.nn.relu(CDC6, 'relu6')
    print CDC6.shape
    output_shape = CDC6.get_shape().as_list()
    output_shape[1] *= 2
    output_shape[4] = _myweights['wup7'].get_shape().as_list()[3]
    CDC7 = deconv3d('deconv7', CDC6, _myweights['wup7'], _mybiases['bup7'], output_shape, padding=1)
    CDC7 = tf.nn.relu(CDC7, 'relu7')
    print CDC7.shape
    output_shape = CDC7.get_shape().as_list()
    output_shape[1] *= 2
    output_shape[4] = _myweights['wup8'].get_shape().as_list()[3]
    CDC8 = deconv3d('deconv8', CDC7, _myweights['wup8'], _mybiases['bup8'], output_shape, padding=1)
    CDC8 = tf.nn.relu(CDC8, 'relu8')
    print CDC8.shape
    CDC8 = tf.transpose(CDC8, perm=[0,1,4,2,3])
    out = tf.reshape(CDC8, [batch_size, CDC8.get_shape().as_list()[1], CDC8.get_shape().as_list()[2]]) 
    print out.shape
    return out