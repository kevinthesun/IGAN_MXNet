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

# Utility functions
def prep_data():
    parent_folder = '../datasets/'
    data_path = parent_folder + args.dataset

    #Resize images to 64x64 and concatenate to ndarray
    data=[]
    img_files = [os.path.join(data_path, fname) for fname in os.listdir(data_path)]
    num_img = len(img_files) if args.num_image == -1 else args.num_image
    for i, image in enumerate(img_files):
        if i == num_img:
            break
        if not image.endswith('jpg'):
            continue
        img_arr = cv2.imread(image)
        resized = cv2.resize(img_arr, (img_dim[1], img_dim[2]), interpolation=cv2.INTER_AREA)
        normalized = np.array(resized/127.5 - 1)
        data.append(np.rollaxis(normalized, 2, 0))
    train_iter = mx.io.NDArrayIter(data=mx.nd.array(data), batch_size=args.batch_size,
                                   label_name='logistic_label')
    return train_iter

def alexnet_feature(layer_name='conv4'):
    model_zoo_dir = "/../model_zoo/alexnet/"
    os.system('cp %s/bvlc_alexnet-symbol.json bvlc_alexnet-symbol.json' % (model_zoo_dir))
    os.system('cp %s/bvlc_alexnet-0000.params bvlc_alexnet-0000.params' % (model_zoo_dir))
    sym, arg_params, aux_params = mx.model.load_checkpoint('bvlc_alexnet', 0)
    os.system('rm bvlc_alexnet-symbol.json bvlc_alexnet-0000.params')

    all_layers = sym.get_internals()
    net = all_layers[layer_name]
    mod = mx.mod.Module(symbol=net, context=ctx, label_names='r_img')



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
    sym, arg_params, aux_params = mx.model.load_checkpoint('%s_G' % (prefix), 0)
    os.system('rm %s_G.json %s_G-0000.params' % (prefix, prefix))

    mod_gen = mx.mod.Module(symbol=sym, context=ctx, label_names='r_img')
    mod_gen.bind(data_shapes=[('latent_vector', (args.batch_size, args.latent_vector_size))],
                 label_shapes=['r_img', tuple([args.batch_size] + list(img_dim))])
    mod_gen.set_params(arg_params, aux_params)

    #Create predictor module
    pred_out = dcgan_builder.make_predictor(r_img, args.latent_vector_size)
    g_img = mx.sym.Variable('g_img')
    dist_loss.save('%s_D-symbol.json'%args.dataset)

    mod_dist = mx.mod.Module(symbol=dist_sym, data_names=('data',),
                             label_names=('logistic_label',), context=ctx)
    mod_dist.bind(data_shapes=train_iter.provide_data,
              label_shapes=[('logistic_label', (args.batch_size,))],
              inputs_need_grad=True)
    mod_dist.init_params(initializer=mx.init.Normal(0.02))
    mod_dist.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': args.lr,
            'wd': args.wd,
            'beta1': args.beta1,
        })
    mods.append(mod_dist)

    # Printing utility function
    def norm_stat(d):
        return mx.nd.norm(d) / np.sqrt(d.size)

    mon = mx.mon.Monitor(10, norm_stat, pattern=".*output|d1_backward_data", sort=True)
    if mon is not None:
        for mod in mods:
            mod.install_monitor(mon)

    def facc(label, pred):
        pred = pred.ravel()
        label = label.ravel()
        return ((pred > 0.5) == label).mean()

    def fentropy(label, pred):
        pred = pred.ravel()
        label = label.ravel()
        return -(label * np.log(pred + 1e-12) + (1. - label) * np.log(1. - pred + 1e-12)).mean()

    mG = mx.metric.CustomMetric(fentropy)
    mD = mx.metric.CustomMetric(fentropy)
    mACC = mx.metric.CustomMetric(facc)

    print('Training generator and discriminator...')
    stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')

    # =============train===============
    for epoch in range(args.num_epoch):
        train_iter.reset()
        for t, batch in enumerate(train_iter):
            # Draw z_latent vector from normal distribution
            rbatch = rand_iter.next()

            if mon is not None:
                mon.tic()

            mod_gen.forward(rbatch, is_train=True)
            out_gen = mod_gen.get_outputs()

            # Train discriminator on generated fake images
            label[:] = 0
            mod_dist.forward(mx.io.DataBatch(out_gen, [label]), is_train=True)
            mod_dist.backward()

            grad_dist = [[grad.copyto(grad.context) for grad in grads] for grads in mod_dist._exec_group.grad_arrays]
            mod_dist.update_metric(mD, [label])
            mod_dist.update_metric(mACC, [label])

            # Train discriminator on real images
            label[:] = 1
            batch.label = [label]
            mod_dist.forward(batch, is_train=True)
            mod_dist.backward()
            for gradsr, gradsf in zip(mod_dist._exec_group.grad_arrays, grad_dist):
                for gradr, gradf in zip(gradsr, gradsf):
                    gradr += gradf
            mod_dist.update()
            mod_dist.update_metric(mD, [label])
            mod_dist.update_metric(mACC, [label])

            # Train generator
            label[:] = 1
            mod_dist.forward(mx.io.DataBatch(out_gen, [label]), is_train=True)
            mod_dist.backward()
            diff_dist = mod_dist.get_input_grads()
            mod_gen.backward(diff_dist)
            mod_gen.update()

            mG.update([label], mod_dist.get_outputs())

            if mon is not None:
                mon.toc_print()

            t += 1
            if t % 10 == 0:
                print('epoch:', epoch, 'iter:', t, 'metric:', mACC.get(), mG.get(), mD.get())
                mACC.reset()
                mG.reset()
                mD.reset()

                if args.interactive:
                    visual('gout', out_gen[0].asnumpy())
                    diff = diff_dist[0].asnumpy()
                    diff = (diff - diff.mean()) / diff.std()
                    visual('diff', diff)
                    visual('data', batch.data[0].asnumpy())

        if args.save_model:
            print('Saving model...')
            mod_gen.save_params('%s_G_%s-%04d.params' % (args.dataset, stamp, args.num_epoch))
            mod_gen.save_params('%s_D_%s-%04d.params' % (args.dataset, stamp, args.num_epoch))

