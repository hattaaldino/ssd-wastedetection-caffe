from __future__ import print_function
import caffe
from caffe.ssd_custom_layer import *
from google.protobuf import text_format

import math
import os
import shutil
import stat
import subprocess
import sys

# Add extra layers on top of a "base" network.
def AddExtraLayers(net, lr_mult=1):
    use_relu = True

    # Add additional convolutional layers.
    # 19 x 19
    from_layer = net.keys()[-1]

    # 10 x 10
    conv_name = "conv8_1"
    ConvLayer(net, from_layer, conv_name, use_relu, 256, 1, 0, 1,
        lr_mult=lr_mult)

    from_layer = conv_name
    conv_name = "conv8_2"
    ConvLayer(net, from_layer, conv_name, use_relu, 512, 3, 1, 2,
        lr_mult=lr_mult)

    # 5 x 5
    from_layer = conv_name
    conv_name = "conv9_1"
    ConvLayer(net, from_layer, conv_name, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = conv_name
    conv_name = "conv9_2"
    ConvLayer(net, from_layer, conv_name, use_relu, 256, 3, 1, 2,
      lr_mult=lr_mult)

    # 3 x 3
    from_layer = conv_name
    conv_name = "conv10_1"
    ConvLayer(net, from_layer, conv_name, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = conv_name
    conv_name = "conv10_2"
    ConvLayer(net, from_layer, conv_name, use_relu, 256, 3, 0, 1,
      lr_mult=lr_mult)

    # 1 x 1
    from_layer = conv_name
    conv_name = "conv11_1"
    ConvLayer(net, from_layer, conv_name, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = conv_name
    conv_name = "conv11_2"
    ConvLayer(net, from_layer, conv_name, use_relu, 256, 3, 0, 1,
      lr_mult=lr_mult)

    return net


# The directory which contains the caffe code.
caffe_root = os.getcwd()

# Set true if you want to start training right after generating all files.
run_soon = True

# The database file
test_data = "examples\\wastedata\\wastedata_test_lmdb"
# Specify the batch sampler.
resize_width = 300
resize_height = 300
resize = "{}x{}".format(resize_width, resize_height)
test_transform_param = {
        'mean_value': [104, 117, 123],
        'force_color': True,
        'resize_param': {
                'prob': 1,
                'resize_mode': P.Resize.WARP,
                'height': resize_height,
                'width': resize_width,
                'interp_mode': [P.Resize.LINEAR],
                },
        }

# The job name should be same as the name used in examples/ssd/ssd_pascal.py.
job_name = "SSD_{}".format(resize)
# The name of the model. Modify it if you want.
model_name = "VGG_wastedata_{}".format(job_name)

# Directory which stores the model .prototxt file.
save_dir = "models\\VGGNet\\wastedata\\{}_score".format(job_name)
# Directory which stores the snapshot of trained models.
snapshot_dir = "models\\VGGNet\\wastedata\\{}".format(job_name)
# Directory which stores the job script and log file.
job_dir = "jobs\\VGGNet\\wastedata\\{}_score".format(job_name)
# Directory which stores the detection results.
output_result_dir = "data\\wastedata\\results\\{}_score\\Main".format(job_name)

# model definition files.
train_net_file = "models\\VGGNet\\wastedata\\{}\\train.prototxt".format(job_name)
test_net_file = "{}\\test.prototxt".format(save_dir)
solver_file = "{}\\solver.prototxt".format(save_dir)
# snapshot prefix.
snapshot_prefix = "{}\\{}".format(snapshot_dir, model_name)
# job script path.
job_file = "{}\\{}.cmd".format(job_dir, model_name)

# Find most recent snapshot.
max_iter = 0
for file in os.listdir(snapshot_dir):
  if file.endswith(".caffemodel"):
    basename = os.path.splitext(file)[0]
    iter = int(basename.split("{}_iter_".format(model_name))[1])
    if iter > max_iter:
      max_iter = iter

if max_iter == 0:
  print("Cannot find snapshot in {}".format(snapshot_dir))
  sys.exit()

# Stores the test image names and sizes. Created by data/wastedata/create_list.sh
name_size_file = "data\\wastedata\\test_name_size.txt"
# The resume model.
pretrain_model = "{}_iter_{}.caffemodel".format(snapshot_prefix, max_iter)
# Stores LabelMapItem.
label_map_file = "data\\wastedata\\labelmap_waste.prototxt"

# Detection parameters
lr_mult = 1
num_classes = 3
share_location = True
background_label_id = 0
code_type = P.PriorBox.CENTER_SIZE

# parameters for generating priors.
# minimum dimension of input image
min_dim = 300
# conv4_3 ==> 38 x 38
# fc7 ==> 19 x 19
# conv8_2 ==> 10 x 10
# conv9_2 ==> 5 x 5
# conv10_2 ==> 3 x 3
# conv11_2 ==> 1 x 1
mbox_source_layers = ['conv4_3', 'fc7', 'conv8_2', 'conv9_2', 'conv10_2', 'conv11_2']
# in percent %
min_ratio = 20
max_ratio = 90
step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2)))
min_sizes = []
max_sizes = []
for ratio in range(min_ratio, max_ratio + 1, step):
  min_sizes.append(min_dim * ratio / 100.)
  max_sizes.append(min_dim * (ratio + step) / 100.)
min_sizes = [min_dim * 10 / 100.] + min_sizes
max_sizes = [min_dim * 20 / 100.] + max_sizes
steps = [8, 16, 32, 64, 100, 300]
aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
# L2 normalize conv4_3.
normalizations = [20, -1, -1, -1, -1, -1]
# variance used to encode/decode prior bboxes.
if code_type == P.PriorBox.CENTER_SIZE:
  prior_variance = [0.1, 0.1, 0.2, 0.2]
else:
  prior_variance = [0.1]
flip = True
clip = False

# Solver parameters.
# Defining which GPUs to use.
gpus = "0"
gpulist = gpus.split(",")
num_gpus = len(gpulist)
solver_mode = P.Solver.CPU
if num_gpus > 0:
  solver_mode = P.Solver.GPU

# Evaluate on whole test set.
num_test_image = 1000
test_batch_size = 8
test_iter = int(math.ceil(float(num_test_image) / test_batch_size))

solver_param = {
    # Train parameters
    'base_lr': 0.001,
    'weight_decay': 0.0005,
    'lr_policy': "multistep",
    'stepvalue': [30000, 50000, 60000],
    'gamma': 0.1,
    'momentum': 0.9,
    'iter_size': 0,
    'max_iter': 0,
    'snapshot': 0,
    'display': 10,
    'average_loss': 10,
    'type': "SGD",
    'solver_mode': solver_mode,
    'device_id': 0,
    'debug_info': False,
    'snapshot_after_train': False,
    # Test parameters
    'test_iter': [test_iter],
    'test_interval': 5000,
    'eval_type': "detection",
    'ap_version': "11point",
    'test_initialization': True,
    }

# parameters for generating detection output.
det_out_param = {
    'num_classes': num_classes,
    'share_location': share_location,
    'background_label_id': background_label_id,
    'nms_param': {'nms_threshold': 0.45, 'top_k': 400},
    'save_output_param': {
        'output_directory': output_result_dir,
        'output_name_prefix': "comp4_det_test_",
        'output_format': "VOC",
        'label_map_file': label_map_file,
        'name_size_file': name_size_file,
        'num_test_image': num_test_image,
        },
    'keep_top_k': 200,
    'confidence_threshold': 0.01,
    'code_type': code_type,
    }

# parameters for evaluating detection results.
det_eval_param = {
    'num_classes': num_classes,
    'background_label_id': background_label_id,
    'overlap_threshold': 0.5,
    'evaluate_difficult_gt': False,
    'name_size_file': name_size_file,
    }

### Hopefully you don't need to change the following ###
# Check file.
check_if_exist(test_data)
check_if_exist(label_map_file)
check_if_exist(pretrain_model)
make_if_not_exist(save_dir)
make_if_not_exist(job_dir)
make_if_not_exist(snapshot_dir)

# Create test net.
net = caffe.NetSpec()
net.data, net.label = CreateAnnotatedDataLayer(test_data, batch_size=test_batch_size,
        train=False, output_label=True, label_map_file=label_map_file,
        transform_param=test_transform_param)

VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
    dropout=False)

AddExtraLayers(net)

mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
        min_sizes=min_sizes, max_sizes=max_sizes, aspect_ratios=aspect_ratios,
        steps=steps, normalizations=normalizations, num_classes=num_classes,
        flip=flip, clip=clip, kernel_size=3, pad=1, lr_mult=lr_mult,
        prior_variance=prior_variance, share_location=share_location)

conf_name = "mbox_conf"
reshape_name = "{}_reshape".format(conf_name)
net[reshape_name] = L.Reshape(net[conf_name], shape=dict(dim=[0, -1, num_classes]))
softmax_name = "{}_softmax".format(conf_name)
net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
flatten_name = "{}_flatten".format(conf_name)
net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
mbox_layers[1] = net[flatten_name]

net.detection_out = L.DetectionOutput(*mbox_layers,
    detection_output_param=det_out_param,
    include=dict(phase=caffe_pb2.Phase.Value('TEST')))
net.detection_eval = L.DetectionEvaluate(net.detection_out, net.label,
    detection_evaluate_param=det_eval_param,
    include=dict(phase=caffe_pb2.Phase.Value('TEST')))

with open(test_net_file, 'w') as f:
    print('name: "{}_test"'.format(model_name), file=f)
    print(net.to_proto(), file=f)
shutil.copy(test_net_file, job_dir)

# Create solver.
solver = caffe_pb2.SolverParameter(
        train_net=train_net_file,
        test_net=[test_net_file],
        snapshot_prefix=snapshot_prefix,
        **solver_param)

with open(solver_file, 'w') as f:
    print(solver, file=f)
shutil.copy(solver_file, job_dir)

# Create job file.
with open(job_file, 'w') as f:
  f.write('cd {}\n'.format(caffe_root))
  f.write('build\\tools\\caffe train ^\n')
  f.write('--solver="{}" ^\n'.format(solver_file))
  f.write('--weights="{}" ^\n'.format(pretrain_model))
  if solver_param['solver_mode'] == P.Solver.GPU:
    f.write('--gpu {} 2>&1 | tee {}\\{}_score_test_{}.log\n'.format(gpus, job_dir, model_name, max_iter))
  else:
    f.write('2>&1 | tee {}\\{}.log\n'.format(job_dir, model_name))

# Copy the python script to job_dir.
py_file = os.path.abspath(__file__)
shutil.copy(py_file, job_dir)

# Run the job.
os.chmod(job_file, stat.S_IRWXU)
if run_soon:
  subprocess.Popen(job_file, shell=True)
