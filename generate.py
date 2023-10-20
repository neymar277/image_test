# Use a trained DenseFuse Net to generate fused images

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imsave, imresize
from datetime import datetime
from os import listdir, mkdir, sep
from os.path import join, exists, splitext
from Generator import Generator
from Discriminator import Discriminator1, Discriminator2
import time
from PIL  import Image
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
def rgb2ihs(img_rgb):
    r = img_rgb[:,:,0]
    g = img_rgb[:,:,1]
    b = img_rgb[:,:,2]
    
    # rgb 转 ihs
    I = 1/np.sqrt(3)*r + 1/np.sqrt(3)*g + 1/np.sqrt(3)*b
    v1 = 1/np.sqrt(6)*r + 1/np.sqrt(6)*g - 2/np.sqrt(6)*b
    v2 = 1/np.sqrt(2)*r - 1/np.sqrt(2)*g
    
    
    return I, v1, v2

def ihs2rgb(I, V1, V2):
    
    r = 1/np.sqrt(3)*I + 1/np.sqrt(6)*V1 + 1/np.sqrt(2)*V2
    g = 1/np.sqrt(3)*I + 1/np.sqrt(6)*V1 - 1/np.sqrt(2)*V2
    b = 1/np.sqrt(3)*I - 2/np.sqrt(6)*V1
    
    img_rgb = np.stack((r, g, b), axis=-1)
    
    # 将RGB值的范围限制在[0, 1]
    img_rgb[img_rgb < 0] = 0
    img_rgb[img_rgb > 1] = 1
    
    return img_rgb


def generate(ir_path, vis_path, model_path, index=0, save_name=None):
    # ir_img = imread(ir_path, flatten=True, mode='YCbCr').astype(np.float) / 255.0
    # vis_img = imread(vis_path, flatten=True, mode='YCbCr').astype(np.float) / 255.0
    
    
    ir_img =  np.array(Image.open(ir_path).convert('RGB')) / 255.0
    vis_img = np.array(Image.open(vis_path).convert('L'))  / 255.0
    ir_img, ir_h, ir_s = rgb2ihs(ir_img)
    h, w =  vis_img.shape
    ir_img = np.resize(ir_img, (int(h / 4), int(w / 4)))
    print(ir_img.shape)
    ir_dimension = list(ir_img.shape)
    
    vis_dimension = list(vis_img.shape)
    ir_dimension.insert(0, 1)
    ir_dimension.append(1)
    vis_dimension.insert(0, 1)
    vis_dimension.append(1)
    ir_img = ir_img.reshape(ir_dimension)
    vis_img = vis_img.reshape(vis_dimension)

    with tf.Graph().as_default() as graph, tf.Session() as sess:
        SOURCE_VIS = tf.placeholder(tf.float32, shape = vis_dimension, name = 'SOURCE_VIS')
        SOURCE_ir = tf.placeholder(tf.float32, shape = ir_dimension, name = 'SOURCE_ir')
        start = time.time()
        G = Generator('Generator')
        output_image = G.transform(vis = SOURCE_VIS, ir = SOURCE_ir)
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        output = sess.run(output_image, feed_dict = {SOURCE_VIS: vis_img, SOURCE_ir: ir_img})
        end = time.time()
        output = output[0, :, :, 0]
        output = ihs2rgb(output, ir_h, ir_s)
        imsave(save_name, output)      
        return end - start

