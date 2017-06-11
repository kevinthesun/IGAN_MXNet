import mxnet as mx


class DCGAN(object):
    """DCGAN model definition
    """
    def __init__(self, num_g_filter=64, num_d_filter=64, num_channel=3, num_layer=3):
        self.num_g_filter = num_g_filter
        self.num_d_filter = num_d_filter
        self.num_channel = num_channel
        self.num_layer = num_layer

    def make_generator(self, z_latent, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12):
        ngf = self.num_g_filter
        nc = self.num_channel
        nlayer = self.num_layer

        def trans_conv_factory(data, index, kernel, stride=(1, 1), pad=(0, 0), num_filter=ngf, apply_bn=True,
                               act_type='relu'):
            gen = mx.sym.Deconvolution(data, name='g' + str(index), kernel=kernel, stride=stride,
                                       pad=pad, num_filter=num_filter, no_bias=no_bias)
            gen_bn = mx.sym.BatchNorm(gen, name='gbn' + str(index), fix_gamma=fix_gamma, eps=eps)
            gen_act = mx.sym.Activation(gen_bn if apply_bn else gen, name='gact' + str(index),
                                        act_type=act_type)
            return gen_act

        gin = trans_conv_factory(z_latent, index=1, kernel=(4, 4), num_filter=ngf * 2 ** nlayer)
        last_out = gin

        for idx in range(nlayer):
            last_out = trans_conv_factory(last_out, index=idx + 2, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                                     num_filter=ngf * 2 ** (nlayer - idx - 1))

        gout = trans_conv_factory(last_out, index=nlayer + 2, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                                  num_filter=nc, apply_bn=False, act_type='tanh')
        return gout

    def make_discriminator(self, image, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12):
        ndf = self.num_d_filter
        nlayer = self.num_layer

        def conv_factory(data, index, kernel, stride=(1, 1), pad=(0, 0), num_filter=ndf, apply_bn=True):
            disc = mx.sym.Convolution(data, name='d' + str(index), kernel=kernel, stride=stride,
                                       pad=pad, num_filter=num_filter, no_bias=no_bias)
            disc_bn = mx.sym.BatchNorm(disc, name='dbn' + str(index), fix_gamma=fix_gamma, eps=eps)
            disc_act = mx.sym.Activation(disc_bn if apply_bn else gen, name='dact' + str(index),
                                         act_type='leaky', slope=0.2)
            return disc_act

        din = conv_factory(image, index=1, kernel=(4, 4), stride=(2, 2), pad=(1, 1), apply_bn=False)
        last_out = din

        for idx in range(nlayer):
            last_out = conv_factory(last_out, index=idx + 2, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                                num_filter=ndf * 2 ** (idx + 1))

        dout = mx.sym.Convolution(last_out, name='d5', kernel=(4, 4), num_filter=1, no_bias=no_bias)
        dout = mx.sym.Flatten(dout)
        return dout

    def make_predictor(self, image, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12):
        ndf = self.num_d_filter
        nlayer = self.num_layer

        def conv_factory(data, index, kernel, stride=(1, 1), pad=(0, 0), num_filter=ndf, apply_bn=True):
            pred = mx.sym.Convolution(data, name='d' + str(index), kernel=kernel, stride=stride,
                                      pad=pad, num_filter=num_filter, no_bias=no_bias)
            pred_bn = mx.sym.BatchNorm(pred, name='dbn' + str(index), fix_gamma=fix_gamma, eps=eps)
            pred_act = mx.sym.Activation(pred_bn if apply_bn else gen, name='dact' + str(index),
                                         act_type='leaky', slope=0.2)
            return pred_act

        din = conv_factory(image, index=1, kernel=(4, 4), stride=(2, 2), pad=(1, 1), apply_bn=False)
        last_out = din

        for idx in range(nlayer):
            last_out = conv_factory(last_out, index=idx + 2, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                                    num_filter=ndf * 2 ** (idx + 1))

        pout = mx.sym.Convolution(last_out, name='d5', kernel=(4, 4), num_filter=1, no_bias=no_bias)
        pout = mx.sym.Flatten(pout)
        pout = mx.sym.Activation(pout, act_type='tanh')
        return pout

