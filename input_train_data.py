# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import PIL.Image as Image
import random
import numpy as np
import cv2
import time

def _process_data(data):
    # normalization
    data = np.clip(np.fabs(data), -np.inf, np.inf)
    data -= np.amin(data)
    data /= np.amax(data)
    return data

def get_frames_data(filename, num_frames_per_clip, start_time, end_time):
    ret_arr = []
    for parent, dirnames, filenames in os.walk(filename):
        '''
        if(len(filenames)<num_frames_per_clip):
            filenames = sorted(filenames)
            for i in range(num_frames_per_clip):
                if i>=len(filenames):
                    i=i%len(filenames)
                image_name = str(filename) + '/' + str(filenames[i])
                img = Image.open(image_name)
                img_data = np.array(img)
                ret_arr.append(img_data)
            return ret_arr, s_index
        '''
        s_index = int(max(0, ((start_time + end_time) * 5 - num_frames_per_clip)/2))
        if ((start_time + end_time) * 5 + num_frames_per_clip)/2 >= len(filenames):
            s_index = int(min(len(filenames) - num_frames_per_clip - 1, ((start_time + end_time) * 5 - num_frames_per_clip)/2))
        filenames = sorted(filenames)
        for i in range(s_index, s_index + num_frames_per_clip):
            image_name = str(filename) + '/' + str(filenames[i])
            img = Image.open(image_name)
            img_data = np.array(img)
            ret_arr.append(img_data)
    return ret_arr, s_index

def read_clip_and_label(filename, batch_size, start_pos=-1, num_frames_per_clip=16, crop_size=112, video_list=[],position=-1):
    lines = open(filename,'r')
    read_dirnames = []
    data = []
    label = []
    batch_index = 0
    next_batch_start = -1
    lines = list(lines)
    strat_pos = start_pos
    if start_pos < 0 or len(video_list)< 32:
        shuffle = True
    else:
        shuffle = False
    if shuffle :
        video_indices = range(len(lines))
        random.seed(time.time())
        random.shuffle(video_indices)
    else:
        video_indices = video_list[start_pos+1:]
    for index in video_indices:
        if(batch_index>=batch_size):
            #next_batch_start = index
            break
        line = lines[index].strip('\n').split(',')
        frame_label = []
        position += 1
        if len(line) == 3:
            dirname = line[0]
            tmp_label = line[1]
            start_time = float(line[2].split()[0])
            end_time = float(line[2].split()[1])
            tmp_data, s_index = get_frames_data(dirname, num_frames_per_clip, start_time, end_time)
            img_datas = []
            #if shuffle:
                #print("Loading a video clip from {}...".format(dirname))
                #print ('strat_frame: {}'.format(s_index))
            if(len(tmp_data)!=0):
                for j in xrange(len(tmp_data)):
                    img = Image.fromarray(tmp_data[j].astype(np.uint8))
                    if(img.width>img.height):
                        scale = float(crop_size)/float(img.height)
                        img = np.array(cv2.resize(np.array(img),(int(img.width * scale + 1), crop_size))).astype(np.float32)
                    else:
                        scale = float(crop_size)/float(img.width)
                        img = np.array(cv2.resize(np.array(img),(crop_size, int(img.height * scale + 1)))).astype(np.float32)
                    crop_x = int((img.shape[0] - crop_size)/2)
                    crop_y = int((img.shape[1] - crop_size)/2)
                    img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size,:]
                    img = _process_data(img)
                    img_datas.append(img)
                data.append(img_datas)

                for i in range(0,num_frames_per_clip):
                    if s_index + i >= int(start_time * 5) and s_index + i <= int(end_time * 5)-1:
                        frame_label.append(int(tmp_label))
                    else:
                        frame_label.append(20)

                label.append(frame_label)
                batch_index = batch_index + 1
                read_dirnames.append(dirname)
    # pad (duplicate) data/label if less than batch_size
    valid_len = len(data)
    pad_len = batch_size - valid_len
    if pad_len:
        for i in range(pad_len):
            data.append(img_datas)
            label.append(int(tmp_label))
    np_arr_data = np.array(data).astype(np.float32)
    np_arr_label = np.array(label).astype(np.int64)

    return np_arr_data, np_arr_label.reshape(batch_size,num_frames_per_clip), next_batch_start, read_dirnames, video_indices, position