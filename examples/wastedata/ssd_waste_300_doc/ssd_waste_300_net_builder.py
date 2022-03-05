import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

import math
import shutil

class NetBuilder:

    def __init__(self, data, batch_size, num_classes, share_location, background_label_id, 
            net_file, model_name, transform_param, label_map_file):
        self.net = caffe.NetSpec()

        self.data = data
        self.backend = P.Data.LMDB
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.share_location = share_location
        self.background_label_id = background_label_id
        self.net_file = net_file
        self.model_name = model_name
        self.transform_param = transform_param
        self.label_map_file = label_map_file

        self.definePriorParam()

    # Set parameters for generating priors.
    def definePriorParam(self):
        # minimum dimension of input image
        self.min_dim = 300
        # conv4_3 ==> 38 x 38
        # fc7 ==> 19 x 19
        # conv8_2 ==> 10 x 10
        # conv9_2 ==> 5 x 5
        # conv10_2 ==> 3 x 3
        # conv11_2 ==> 1 x 1
        self.source_layers = ['conv4_3', 'fc7', 'conv8_2', 'conv9_2', 'conv10_2', 'conv11_2']
        # in percent %
        self.min_ratio = 20
        self.max_ratio = 90
        step = int(math.floor((self.max_ratio - self.min_ratio) / (len(self.source_layers) - 2)))
        self.min_sizes = []
        self.max_sizes = []
        for ratio in range(self.min_ratio, self.max_ratio + 1, step):
          self.min_sizes.append(self.min_dim * ratio / 100.)
          self.max_sizes.append(self.min_dim * (ratio + step) / 100.)
        self.min_sizes = [self.min_dim * 10 / 100.] + self.min_sizes
        self.max_sizes = [self.min_dim * 20 / 100.] + self.max_sizes
        self.steps = [8, 16, 32, 64, 100, 300]
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        # L2 normalize conv4_3.
        self.normalizations = [20, -1, -1, -1, -1, -1]
        self.priorbox_codetype = P.PriorBox.CENTER_SIZE
        # variance used to encode/decode prior bboxes.
        self.prior_variance = [0.1, 0.1, 0.2, 0.2]
        self.offset = 0.5
        self.flip = True
        self.clip = False

    def unpackVariable(self, var, num):
        ret = []
        for i in range(0, num):
            ret.append(var)
        return ret

    def convLayer(self, from_layer, conv_name, use_relu, num_output,
            kernel_size, pad, stride, dilation=1):
        kwargs = {
            'param': [
                dict(lr_mult=1, decay_mult=1),
                dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)
            }

        [kernel_h, kernel_w] = self.unpackVariable(kernel_size, 2)
        [pad_h, pad_w] = self.unpackVariable(pad, 2)
        [stride_h, stride_w] = self.unpackVariable(stride, 2)

        if kernel_h == kernel_w:
            self.net[conv_name] = L.Convolution(self.net[from_layer], num_output=num_output,
                kernel_size=kernel_h, pad=pad_h, stride=stride_h, **kwargs)
        else:
            self.net[conv_name] = L.Convolution(self.net[from_layer], num_output=num_output,
                kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
                stride_h=stride_h, stride_w=stride_w, **kwargs)
        if dilation > 1:
            self.net.update(conv_name, {'dilation': dilation})
        if use_relu:
            relu_name = '{}_relu'.format(conv_name)
            self.net[relu_name] = L.ReLU(self.net[conv_name], in_place=True)

    # Add extra layers on top of a "base" network.
    def addExtraLayers(self):
        use_relu = True

        # 19 x 19
        from_layer = self.net.keys()[-1]

        # 10 x 10
        conv_name = "conv8_1"
        self.convLayer(from_layer, conv_name, use_relu, 256, 1, 0, 1)

        from_layer = conv_name
        conv_name = "conv8_2"
        self.convLayer(from_layer, conv_name, use_relu, 512, 3, 1, 2)

        # 5 x 5
        from_layer = conv_name
        conv_name = "conv9_1"
        self.convLayer(from_layer, conv_name, use_relu, 128, 1, 0, 1)

        from_layer = conv_name
        conv_name = "conv9_2"
        self.convLayer(from_layer, conv_name, use_relu, 256, 3, 1, 2)

        # 3 x 3
        from_layer = conv_name
        conv_name = "conv10_1"
        self.convLayer(from_layer, conv_name, use_relu, 128, 1, 0, 1)

        from_layer = conv_name
        conv_name = "conv10_2"  
        self.convLayer(from_layer, conv_name, use_relu, 256, 3, 0, 1)

        # 1 x 1
        from_layer = conv_name
        conv_name = "conv11_1"
        self.convLayer(from_layer, conv_name, use_relu, 128, 1, 0, 1)

        from_layer = conv_name
        conv_name = "conv11_2"
        self.convLayer(from_layer, conv_name, use_relu, 256, 3, 0, 1)

    def createAnnotatedDataLayer(self, source, batch_size=32, train=True, batch_sampler=[{}]):
        ntop = 2
        
        if train:
            kwargs = {
                    'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                    'transform_param': self.transform_param,
                    }
        else:
            kwargs = {
                    'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
                    'transform_param': self.transform_param,
                    }

        annotated_data_param = {
            'label_map_file': self.label_map_file,
            'batch_sampler': batch_sampler,
            }

        return L.AnnotatedData(name="data", annotated_data_param=annotated_data_param,
            data_param=dict(batch_size=batch_size, backend=self.backend, source=source),
            ntop=ntop, **kwargs)

    def VGGNetBody(self, from_layer):
        kwargs = {
                'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=0)],
                'weight_filler': dict(type='xavier'),
                'bias_filler': dict(type='constant', value=0)}

        assert from_layer in self.net.keys()

        self.net.conv1_1 = L.Convolution(self.net[from_layer], num_output=64, pad=1, kernel_size=3, **kwargs)
        self.net.relu1_1 = L.ReLU(self.net.conv1_1, in_place=True)
        self.net.conv1_2 = L.Convolution(self.net.relu1_1, num_output=64, pad=1, kernel_size=3, **kwargs)
        self.net.relu1_2 = L.ReLU(self.net.conv1_2, in_place=True)
        self.net.pool1 = L.Pooling(self.net.relu1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        self.net.conv2_1 = L.Convolution(self.net.pool1, num_output=128, pad=1, kernel_size=3, **kwargs)
        self.net.relu2_1 = L.ReLU(self.net.conv2_1, in_place=True)
        self.net.conv2_2 = L.Convolution(self.net.relu2_1, num_output=128, pad=1, kernel_size=3, **kwargs)
        self.net.relu2_2 = L.ReLU(self.net.conv2_2, in_place=True)
        self.net.pool2 = L.Pooling(self.net.relu2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        self.net.conv3_1 = L.Convolution(self.net.pool2, num_output=256, pad=1, kernel_size=3, **kwargs)
        self.net.relu3_1 = L.ReLU(self.net.conv3_1, in_place=True)
        self.net.conv3_2 = L.Convolution(self.net.relu3_1, num_output=256, pad=1, kernel_size=3, **kwargs)
        self.net.relu3_2 = L.ReLU(self.net.conv3_2, in_place=True)
        self.net.conv3_3 = L.Convolution(self.net.relu3_2, num_output=256, pad=1, kernel_size=3, **kwargs)
        self.net.relu3_3 = L.ReLU(self.net.conv3_3, in_place=True)
        self.net.pool3 = L.Pooling(self.net.relu3_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        self.net.conv4_1 = L.Convolution(self.net.pool3, num_output=512, pad=1, kernel_size=3, **kwargs)
        self.net.relu4_1 = L.ReLU(self.net.conv4_1, in_place=True)
        self.net.conv4_2 = L.Convolution(self.net.relu4_1, num_output=512, pad=1, kernel_size=3, **kwargs)
        self.net.relu4_2 = L.ReLU(self.net.conv4_2, in_place=True)
        self.net.conv4_3 = L.Convolution(self.net.relu4_2, num_output=512, pad=1, kernel_size=3, **kwargs)
        self.net.relu4_3 = L.ReLU(self.net.conv4_3, in_place=True)
        self.net.pool4 = L.Pooling(self.net.relu4_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        self.net.conv5_1 = L.Convolution(self.net.pool4, num_output=512, pad=1, kernel_size=3, **kwargs)
        self.net.relu5_1 = L.ReLU(self.net.conv5_1, in_place=True)
        self.net.conv5_2 = L.Convolution(self.net.relu5_1, num_output=512, pad=1, kernel_size=3, **kwargs)
        self.net.relu5_2 = L.ReLU(self.net.conv5_2, in_place=True)
        self.net.conv5_3 = L.Convolution(self.net.relu5_2, num_output=512, pad=1, kernel_size=3, **kwargs)
        self.net.relu5_3 = L.ReLU(self.net.conv5_3, in_place=True)
        self.net.pool5 = L.Pooling(self.net.relu5_3, pool=P.Pooling.MAX, pad=1, kernel_size=3, stride=1)

        dilation = 6
        kernel_size = 3
        num_output = 1024
        pad = int((kernel_size + (dilation - 1) * (kernel_size - 1)) - 1) // 2

        self.net.fc6 = L.Convolution(self.net.pool5, num_output=1024, pad=pad, kernel_size=kernel_size, dilation=dilation, **kwargs)
        self.net.relu6 = L.ReLU(self.net.fc6, in_place=True)

        self.net.fc7 = L.Convolution(self.net.relu6, num_output=1024, kernel_size=1, **kwargs)
        self.net.relu7 = L.ReLU(self.net.fc7, in_place=True)

    def createMultiBoxHead(self, data_layer="data", kernel_size=1, pad=0):
        assert self.num_classes, "must provide num_classes"
        assert self.num_classes > 0, "num_classes must be positive number"
        assert len(self.source_layers) == len(self.normalizations), "source_layers and normalizations should have same length"
        assert len(self.source_layers) == len(self.min_sizes), "source_layers and min_sizes should have same length"
        assert len(self.source_layers) == len(self.max_sizes), "source_layers and max_sizes should have same length"
        assert len(self.source_layers) == len(self.aspect_ratios), "source_layers and aspect_ratios should have same length"
        assert len(self.source_layers) == len(self.steps), "source_layers and steps should have same length"
        assert data_layer in self.net.keys(), "data_layer is not in net's layers"

        num = len(self.source_layers)
        priorbox_layers = []
        loc_layers = []
        conf_layers = []

        for i in range(0, num):
            from_layer = self.source_layers[i]

            # Get the normalize value.
            if self.normalizations[i] != -1:
                norm_name = "{}_norm".format(from_layer)
                self.net[norm_name] = L.Normalize(self.net[from_layer], scale_filler=dict(type="constant", value=self.normalizations[i]),
                    across_spatial=False, channel_shared=False)
                from_layer = norm_name

            # Estimate number of priors per location given provided parameters.
            min_size = self.min_sizes[i]
            if type(min_size) is not list:
                min_size = [min_size]
            aspect_ratio = []

            if len(self.aspect_ratios) > i:
                aspect_ratio = self.aspect_ratios[i]
                if type(aspect_ratio) is not list:
                    aspect_ratio = [aspect_ratio]
            max_size = []

            if len(self.max_sizes) > i:
                max_size = self.max_sizes[i]
                if type(max_size) is not list:
                    max_size = [max_size]
                assert len(max_size) == len(min_size), "max_size and min_size should have same length."

            num_priors_per_location = (2 + len(aspect_ratio)) * len(min_size)
            num_priors_per_location += len(aspect_ratio) * len(min_size)
            step = []

            if len(self.steps) > i:
                step = self.steps[i]

            # Create location prediction layer.
            name = "{}_mbox_loc".format(from_layer)
            num_loc_output = num_priors_per_location * 4;
            if not self.share_location:
                num_loc_output *= self.num_classes
            self.convLayer(from_layer, name, use_relu=False, num_output=num_loc_output, 
                kernel_size=kernel_size, pad=pad, stride=1)
            permute_name = "{}_perm".format(name)
            self.net[permute_name] = L.Permute(self.net[name], order=[0, 2, 3, 1])
            flatten_name = "{}_flat".format(name)
            self.net[flatten_name] = L.Flatten(self.net[permute_name], axis=1)
            loc_layers.append(self.net[flatten_name])

            # Create confidence prediction layer.
            name = "{}_mbox_conf".format(from_layer)
            num_conf_output = num_priors_per_location * self.num_classes;
            self.convLayer(from_layer, name, use_relu=False, num_output=num_conf_output, 
                kernel_size=kernel_size, pad=pad, stride=1)
            permute_name = "{}_perm".format(name)
            self.net[permute_name] = L.Permute(self.net[name], order=[0, 2, 3, 1])
            flatten_name = "{}_flat".format(name)
            self.net[flatten_name] = L.Flatten(self.net[permute_name], axis=1)
            conf_layers.append(self.net[flatten_name])

            # Create prior generation layer.
            name = "{}_mbox_priorbox".format(from_layer)
            self.net[name] = L.PriorBox(self.net[from_layer], self.net[data_layer], min_size=min_size,
                    clip=self.clip, variance=self.prior_variance, offset=self.offset)
            if max_size:
                self.net.update(name, {'max_size': max_size})
            if aspect_ratio:
                self.net.update(name, {'aspect_ratio': aspect_ratio, 'flip': self.flip})
            if step:
                self.net.update(name, {'step': step})

            priorbox_layers.append(self.net[name])

        # Concatenate priorbox, loc, and conf layers.
        mbox_layers = []
        name = "mbox_loc"
        self.net[name] = L.Concat(*loc_layers, axis=1)
        mbox_layers.append(self.net[name])
        name = "mbox_conf"
        self.net[name] = L.Concat(*conf_layers, axis=1)
        mbox_layers.append(self.net[name])
        name = "mbox_priorbox"
        self.net[name] = L.Concat(*priorbox_layers, axis=2)
        mbox_layers.append(self.net[name])

        return mbox_layers

    def exportNet(self, job_dir, name_suffix):
        with open(self.net_file, 'w') as f:
            print('name: "{}_{}"'.format(self.model_name, name_suffix), file=f)
            print(self.net.to_proto(), file=f)
            shutil.copy(self.net_file, job_dir)