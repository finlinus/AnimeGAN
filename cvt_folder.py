import argparse
from glob import glob
import os
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from net import generator
from utils import preprocessing

def parse_args():
    desc = "Tensorflow implementation of AnimeGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--input', type=str, default=None,
                        help='image folder')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint/net_model',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--output', type=str, default=None,
                        help='output path')
    return parser.parse_args()
    
def getdirloc(initialdir='/', method='input'):
    root = tk.Tk()
    windowtitle = f'Please select {method} folder'
    dirloc = filedialog.askdirectory(parent=root, initialdir=initialdir, title=windowtitle)
    root.withdraw()
    return dirloc

def convert_image(img, img_size):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocessing(img, img_size)
    img = np.expand_dims(img, axis=0)
    img = np.asarray(img)
    return img
    
def inverse_image(img):
    img = (img.squeeze()+1.) / 2 * 255
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def stats_graph(graph):
    profiler_option = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(graph, options=profiler_option)
    print('FLOPs: {}'.format(flops.total_float_ops))

def cvt2anime_dir(inputdir, outputdir, checkpoint_dir, show_stats=False, img_size=(256,256)):

    gpu_stat = bool(len(tf.config.experimental.list_physical_devices('GPU')))
    if gpu_stat:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=gpu_stat)
    
    test_real = tf.compat.v1.placeholder(tf.float32, [1, None, None, 3], name='test')

    with tf.compat.v1.variable_scope("generator", reuse=False):
        test_generated = generator.G_net(test_real).fake
    
    # get image list
    img_files = glob('{}/*.*'.format(inputdir))
    
    tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    with tf.compat.v1.Session(config=tfconfig) as sess:
        # tf.global_variables_initializer().run()
        # load model
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  # checkpoint file information
        saver = tf.compat.v1.train.Saver()
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)  # first line
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
        else:
            print(" [*] Failed to find a checkpoint")
            return
        
        # show FLOPs
        if show_stats:
            stats_graph(tf.compat.v1.get_default_graph())

        # convert
        for imgfile in tqdm(img_files):
            img = cv2.imread(imgfile).astype(np.float32)
            img_cvt = convert_image(img, img_size)
            fake_img = sess.run(test_generated, feed_dict={test_real: img_cvt})
            fake_img_inv = inverse_image(fake_img)
            output_path = os.path.join(outputdir, '{0}'.format(os.path.basename(imgfile)))
            cv2.imwrite(output_path, fake_img_inv)
        
if __name__ == '__main__':
    arg = parse_args()
    if not arg.input:
        arg.input = getdirloc(initialdir='input/', method='input')
    if not arg.output:
        arg.output = getdirloc(initialdir='output/', method='output')
    cvt2anime_dir(arg.input, arg.output, arg.checkpoint_dir)
