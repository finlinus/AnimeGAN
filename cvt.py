import argparse
import os
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
import tensorflow as tf

from net import generator
from utils import preprocessing

def parse_args():
    desc = "Tensorflow implementation of AnimeGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--input', type=str, default=None,
                        help='image file')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint/net_model',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--output', type=str, default=None,
                        help='output file include file extension')
    return parser.parse_args()
    
def getfileloc(initialdir='/', method='open', filetypes=(("image files", ".jpg .png"), ("all files","*.*"))):
    root = tk.Tk()
    windowtitle = f'Please select a file to {method}'
    if method == 'open':
        fileloc = filedialog.askopenfilename(parent=root, initialdir=initialdir, title=windowtitle, filetypes=filetypes)
    elif method == 'save':
        fileloc = filedialog.asksaveasfilename(parent=root, initialdir=initialdir, title=windowtitle, filetypes=filetypes)
    root.withdraw()
    return fileloc

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

def cvt2anime(imgfile, output, checkpoint_dir, show_stats=False, img_size=(256,256)):

    gpu_stat = bool(len(tf.config.experimental.list_physical_devices('GPU')))
    if gpu_stat:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=gpu_stat)
    
    test_real = tf.compat.v1.placeholder(tf.float32, [1, None, None, 3], name='test')

    with tf.compat.v1.variable_scope("generator", reuse=False):
        test_generated = generator.G_net(test_real).fake
    
    # load image
    img = cv2.imread(imgfile).astype(np.float32)
    img_cvt = convert_image(img, img_size)
    
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
        fake_img = sess.run(test_generated, feed_dict={test_real: img_cvt})
        fake_img_inv = inverse_image(fake_img)
        cv2.imwrite(output, fake_img_inv)
        
if __name__ == '__main__':
    arg = parse_args()
    if not arg.input:
        arg.input = getfileloc(initialdir='input/')
    if not arg.output:
        arg.output = getfileloc(initialdir='output/', method='save')
    cvt2anime(arg.input, arg.output, arg.checkpoint_dir)
