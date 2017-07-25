from __future__ import print_function

import os
import sys
sys.path.append('..')
import argparse
import mxnet as mx
import cv2
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime

from model_def.dcgan import DCGAN as dcgan
from train_dcgan import prep_data, visual

parser = argparse.ArgumentParser(description="Train DCGAN.",
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
parser.add_argument('--batch_size', type=int, default=32,
                        help='The batch size.')
parser.add_argument('--img_dim', type=bool, default=True,
                        help='Whether save model.')
parser.add_argument('--interactive', type=bool, default=False,
                        help='Whether show output images for every 10 batches.')
parser.add_argument('--save_model', type=bool, default=True,
                        help='Whether save model.')

img_dim = (3, 64, 64)
alexnet_input_dim = (3, 256, 256)

# Utility functions
def alexnet_feature(layer_name='inception_4e_output'):
    model_zoo_dir = "/../model_zoo/alexnet/"
    os.system('cp %s/bvlc_alexnet-symbol.json bvlc_alexnet-symbol.json' % (model_zoo_dir))
    os.system('cp %s/bvlc_alexnet-0000.params bvlc_alexnet-0000.params' % (model_zoo_dir))
    sym, arg_params, aux_params = mx.model.load_checkpoint('bvlc_alexnet', 0)
    os.system('rm bvlc_alexnet-symbol.json bvlc_alexnet-0000.params')

    all_layers = sym.get_internals()
    net = all_layers[layer_name + '_output']
    mod = mx.mod.Module(symbol=net, context=ctx, label_names=None)
    mod.bind(data_shapes=[('data', tuple(args.batch_size + list(alexnet_input_dim)))])
    return mod

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
    train_iter = prep_data()

    print("Building model...")
    dcgan_builder = dcgan(num_layer=args.num_layer)

    #Load generator module
    img_dim = (3, 64, 64)
    prefix = args.dataset
    model_zoo_dir = "/../model_zoo/%s/" % (prefix)
    os.system('cp %s/%s_G-symbol.json %s_G.json' % (model_zoo_dir, prefix, prefix))
    os.system('cp %s/%s_G.params %s_G-0000.params' % (model_zoo_dir, prefix, prefix))
    gen_sym, arg_params, aux_params = mx.model.load_checkpoint('%s_G' % (prefix), 0)
    os.system('rm %s_G.json %s_G-0000.params' % (prefix, prefix))

    mod_gen = mx.mod.Module(symbol=sym, context=ctx, data_names=('latent_vector'), label_names=None)
    mod_gen.bind(data_shapes=[('latent_vector', (args.batch_size, args.latent_vector_size))],
                 inputs_need_grad=True)
    mod_gen.set_params(arg_params, aux_params)

    #Create predictor module
    r_img = mx.sym.Variable('r_img')
    pred_out = dcgan_builder.make_predictor(r_img, args.latent_vector_size)
    mod_pred = mx.mod.Module(symbol=pred_out, context=ctx, data_names=('r_img'), label_names=None)
    mod_pred.bind(data_shapes=[('r_img', train_iter.provide_data)])
    mod_pred.init_params(initializer=mx.init.Normal(0.02))
    mod_pred.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': args.lr,
            'wd': args.wd,
            'beta1': args.beta1,
        })


    print('Training predictor...')
    stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')

    # =============train===============
    for epoch in range(args.num_epoch):
        train_iter.reset()
        for t, batch in enumerate(train_iter):
            mod_pred.forward(batch)
            pred_z = mod_pred.get_outputs()
            mod_gen.forward(mx.io.DataBatch(pred_z))
            gen_img = mod_gen.get_outputs()[0]
            real_img = batch.data[0]

            #Calculate pixel loss and alexnet loss
            pixel_loss = mx.nd.LinearRegressionOutput(data=gen_img, label=real_img)
            mod_alexnet = alexnet_feature()
            gen_img_upscale = mx.nd.UpSampling(data=gen_img, scale=4, sample_type='bilinear')
            real_img_upscale = mx.nd.UpSampling(data=real_img, scale=4, sample_type='bilinear')
            gen_img_trans = transform_im(gen_img_upscale)
            real_img_trans = transform_im(real_img_upscale)
            mod_alexnet.forward(mx.io.DataBatch([gen_img_trans]), is_train=False)
            gen_img_output = mod_alexnet.get_outputs()
            mod_alexnet.forward(mx.io.DataBatch([real_img_trans]), is_train=False)
            real_img_output = mod_alexnet.get_outputs()
            alexnet_loss = mx.nd.LinearRegressionOutput(data=gen_img_output, label=real_img_out)
            total_loss = [pixel + alexnet * args.alpha for pixel, alexnet in zip(pixel_loss, alexnet_loss)]

            #Train predictor
            mod_gen.backward(total_loss)
            gen_input_grad = mod_gen.get_input_grads()
            mod_pred.backward(gen_input_grad)
            mod_pred.update()

            t += 1
            if t % 10 == 0:
                print('epoch:', epoch, 'iter:', t, 'metric:', total_loss[0])

                if args.interactive:
                    visual('gen_image', gen_img.asnumpy())
                    visual('real_image', batch.data[0].asnumpy())

        if args.save_model:
            print('Saving model...')
            mod_gen.save_params('%s_P_%s-%04d.params' % (args.dataset, stamp, args.num_epoch))

