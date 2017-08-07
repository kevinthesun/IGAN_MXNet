from __future__ import print_function

import os
import sys
sys.path.append('..')
import argparse
import mxnet as mx
import cv2
import numpy as np
from mxnet import autograd
from mxnet.gluon import Trainer, util, loss
from mxnet.gluon.model_zoo.vision import alexnet
from mxnet.gluon.model_zoo.model_store import get_model_file
from matplotlib import pyplot as plt
from datetime import datetime

from model_def.dcgan import DCGAN as dcgan
from train_dcgan import prep_data, visual

parser = argparse.ArgumentParser(description="Train DCGAN Predictor.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='celeba',
                        help='dataset used for training')
parser.add_argument('--num_layer', type=int, default=3,
                        help='number of intermediate layers')
parser.add_argument('--num_gpu', type=int, default='0',
                        help='number of gpus used. 0 means using cpu')
parser.add_argument('--num_epoch', type=int, default=25,
                        help='max num of epochs')
parser.add_argument('--alpha', type=float, default=0.002
                        help='the weight for feature loss. Loss=pixel_loss + alpha * feature_loss')
parser.add_argument('--lr', type=float, default=0.0002,
                        help='initial learning rate')
parser.add_argument('--beta1', type=float, default=0.5,
                        help='Exponential decay rate for the first moment estimates for adam')
parser.add_argument('--wd', type=float, default=0,
                        help='weight decay for adam')
parser.add_argument('--latent_vector_size', type=int, default=100,
                        help='Length of latent vector')
parser.add_argument('--num_image', type=int, default=-1,
                        help='Number of training images')
parser.add_argument('--batch_size', type=int, default=64,
                        help='The batch size.')
parser.add_argument('--img_dim', type=bool, default=True,
                        help='Whether save model.')
parser.add_argument('--use_hybrid', type=bool, default=True,
                        help='Whether to use hybrid mode.')
parser.add_argument('--save_model', type=bool, default=True,
                        help='Whether save model.')

img_dim = (3, 64, 64)
alexnet_input_dim = (3, 256, 256)

# Utility functions
class AlexNetFeature(HybridBlock):
    """AlexNet model from the `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
       Extract layers until the fourth convolutional layer.
    """
    def __init__(self, classes=1000, **kwargs):
        super(AlexNet, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            with self.features.name_scope():
                self.features.add(nn.Conv2D(64, kernel_size=11, strides=4, padding=2))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Conv2D(192, kernel_size=5, padding=2))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Conv2D(384, kernel_size=3, padding=1))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.Conv2D(256, kernel_size=3, padding=1))
                self.features.add(nn.Activation('relu'))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        return x

def transform_im(img, npxl=64, nc=3):
    if nc == 3:
        img_trans = (img + 1.0) * 127.5
    else:
        #img_trans = T.tile(img, [1,1,1,3]) * 255.0  #[hack] to-be-tested
        #TODO
        img_trans = 0
    mean_channel = np.load(os.path.join(pkg_dir, 'ilsvrc_2012_mean.npy')).mean(1).mean(1)
    mean_im = mean_channel[np.newaxis,:,np.newaxis,np.newaxis]
    mean_im = floatX(np.tile(mean_im, [1,1, npxl, npxl]))
    img_trans = img_trans[:, [2,1,0], :,:]
    img_trans = img_trans - mean_im
    return img_trans


if __name__ == '__main__':
    args = parser.parse_args()
    ctx = [mx.gpu(int(i)) for i in range(args.num_gpu)] if args.num_gpu > 0 else mx.cpu(0)

    print("Preprocessing data...")
    train_iter = prep_data(img_dim[1], img_dim[2])

    print("Building model...")
    dcgan_builder = dcgan(num_layer=args.num_layer)

    # Load generator pre-trained module
    gen_params_file = '../model_zoo/celeba_G.params'
    generator = dcgan_builder.make_generator()
    generator.load_params(gen_params_file, ctx)

    # Create alexnet conv4 feature network
    alexnet_feature = AlexNetFeature()
    alexnet_feature.load_params(get_model_file('alexnet'), ctx, ignore_extra=True)

    # Initialize parameters and set optimizer for predictor
    predictor = dcgan_builder.make_predictor()
    pred_loss_func = loss.L2Loss()
    predictor.collect_params().initialize(mx.init.Normal(0.02), ctx=ctx)
    pred_trainer = Trainer(predictor.collect_params(), 'adam',
        {
            'learning_rate': args.lr,
            'wd': args.wd,
            'beta1': args.beta1,
        })

    if use_hybrid:
        generator.hybridize()
        alexnet_feature.hybridize()
        predictor.hybridize()

    print('Training predictor...')
    stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')

    # =============train===============
    for epoch in range(args.num_epoch):
        train_iter.reset()
        for t, batch in enumerate(train_iter):
            real_img_slice = utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            loss_val = 0
            pred_loss = []
            with ag.record():
                for real_img in real_img_slice:
                    pred_latent_vector = predictor.forward(real_img)
                    gen_img = generator.forward(pred_latent_vector)
                    gen_img_upscale = mx.nd.UpSampling(data=gen_img, scale=4, sample_type='bilinear')
                    real_img_upscale = mx.nd.UpSampling(data=real_img, scale=4, sample_type='bilinear')
                    alexnet_gen_output = alexnet_feature.forward(gen_img_upscale)
                    alexnet_real_output = alexnet_feature.forward(real_img_upscale)
                    # Total loss is sum of pixel to pixel L2 loss and alexnet feature output L2 loss
                    pixel_loss = pred_loss(gen_img, real_img)
                    alexnet_feature_loss = pred_loss(alexnet_gen_output, alexnet_real_output)
                    loss = pixel_loss + alexnet_feature_loss
                    pred_loss.append(loss)
                    loss_val += loss.asnumpy()
                for loss in pred_loss:
                    loss.backward()
            pred_trainer.step(batch.data[0].shape[0])

            t += 1
            if t % 10 == 0:
                print('epoch:', epoch, 'iter:', t, 'metric:', loss_val.mean())

        if args.save_model:
            print('Saving model...')
            mod_gen.save_params('%s_P_%s-%04d.params' % (args.dataset, stamp, args.num_epoch))

