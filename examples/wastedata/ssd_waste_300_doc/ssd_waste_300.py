from ssd_waste_300_train_builder import *
from ssd_waste_300_test_builder import *
from ssd_waste_300_solver_builder import * 

import os
import shutil
import stat
import subprocess

class SSDWaste300:

    def __init__(self):
        # The directory which contains the caffe code.
        self.caffe_root = os.getcwd()

        # If true, load from most recently saved snapshot.
        # Otherwise, load from the pretrain_model defined below.
        self.resume_training = True

        # The last iteration that has been snapshoted.
        self.max_iter = 0

        # Input dimension
        self.resize_width = 300
        self.resize_height = 300
        self.resize = "{}x{}".format(self.resize_width, self.resize_height)

        # Architecture name.
        self.arch_name = "SSD_{}".format(self.resize)
        # The name of the model.
        self.model_name = "VGG_wastedata_{}".format(self.arch_name)

        # Set path for directories and files.
        self.definePath()

        # Defining which GPUs to use.
        self.gpus = "0"
        self.device_id = int(self.gpus)

        #Learning rate
        self.base_lr = 0.001

        # Number of class include the background.
        self.num_classes = 3

        # Batch for training phase.
        self.train_batch_size = 8
        self.accum_batch_size = 32

        # Batch for testing phase.
        self.test_batch_size = 2
        self.num_test_image = 1000

        # snapshot prefix.
        self.snapshot_prefix = "{}\\{}".format(self.snapshot_dir, self.model_name)

    # Specify path for directories and files used as source and storage destination.  
    def definePath(self):
        # The database file
        self.train_data = "examples\\wastedata\\wastedata_trainval_lmdb"
        self.test_data = "examples\\wastedata\\wastedata_test_lmdb"

        # Directory which stores the model .prototxt file.
        self.save_dir = "models\\VGGNet\\wastedata\\{}".format(self.arch_name)
        # Directory which stores the snapshot of models.
        self.snapshot_dir = "models\\VGGNet\\wastedata\\{}".format(self.arch_name)
        # Directory which stores the job script and log file.
        self.job_dir = "jobs\\VGGNet\\wastedata\\{}".format(self.arch_name)
        # Directory which stores the detection results.
        self.output_result_dir = "data\\wastedata\\results\\{}\\Main".format(self.arch_name)

        # model definition files.
        self.train_net_file = "{}\\train.prototxt".format(self.save_dir)
        self.test_net_file = "{}\\test.prototxt".format(self.save_dir)
        self.solver_file = "{}\\solver.prototxt".format(self.save_dir)
        self.deploy_net_file = "{}\\deploy.prototxt".format(self.save_dir)

        # job script path.
        self.job_file = "{}\\{}.cmd".format(self.job_dir, self.model_name)

        # Stores the test image names and sizes. Created by data\wastedata\create_list.sh
        self.name_size_file = "data\\wastedata\\test_name_size.txt"
        # The pretrained model: Fully convolutional reduced (atrous) VGGNet.
        self.pretrain_model = "models\\VGGNet\\VGG_ILSVRC_16_layers_fc_reduced.caffemodel"
        # Stores LabelMapItem.
        self.label_map_file = "data\\wastedata\\labelmap_waste.prototxt"


    def check_if_exist(self, path):
        return os.path.exists(path)

    def make_if_not_exist(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    # Count the iteration that occur for one process.
    def countIter(self, batch_scope, batch_size):
        return int(batch_scope / batch_size)

    # Check file existence.
    def checkFile(self):
        self.check_if_exist(self.train_data)
        self.check_if_exist(self.test_data)
        self.check_if_exist(self.label_map_file)
        self.check_if_exist(self.pretrain_model)
        self.make_if_not_exist(self.save_dir)
        self.make_if_not_exist(self.job_dir)
        self.make_if_not_exist(self.snapshot_dir)

    # Specify the data transform param
    def defineTrainTransformParam(self):
        train_transform_param = {
            'mirror': True,
            'mean_value': [104, 117, 123],
            'force_color': True,
            'resize_param': {
                    'prob': 1,
                    'resize_mode': P.Resize.WARP,
                    'height': self.resize_height,
                    'width': self.resize_width,
                    'interp_mode': [
                            P.Resize.LINEAR,
                            P.Resize.AREA,
                            P.Resize.NEAREST,
                            P.Resize.CUBIC,
                            P.Resize.LANCZOS4,
                            ],
                    },
            'distort_param': {
                    'brightness_prob': 0.5,
                    'brightness_delta': 32,
                    'contrast_prob': 0.5,
                    'contrast_lower': 0.5,
                    'contrast_upper': 1.5,
                    'hue_prob': 0.5,
                    'hue_delta': 18,
                    'saturation_prob': 0.5,
                    'saturation_lower': 0.5,
                    'saturation_upper': 1.5,
                    'random_order_prob': 0.0,
                    },
            'expand_param': {
                    'prob': 0.5,
                    'max_expand_ratio': 4.0,
                    },
            'emit_constraint': {
                'emit_type': caffe_pb2.EmitConstraint.CENTER,
                }
            }

        return train_transform_param

    # Specify the data transform param
    def defineTestTransformParam(self):
        test_transform_param = {
            'mean_value': [104, 117, 123],
            'force_color': True,
            'resize_param': {
                    'prob': 1,
                    'resize_mode': P.Resize.WARP,
                    'height': self.resize_height,
                    'width': self.resize_width,
                    'interp_mode': [P.Resize.LINEAR],
                    },
            }

        return test_transform_param

    # Find most recent snapshot.
    def findRecentSS(self):
        for file in os.listdir(self.snapshot_dir):
          if file.endswith(".solverstate"):
            basename = os.path.splitext(file)[0]
            iter = int(basename.split("{}_iter_".format(self.model_name))[1])
            if iter > self.max_iter:
              self.max_iter = iter

    # Create job file.
    def createJob(self):
        train_src_param = '--weights="{}" ^\n'.format(self.pretrain_model)
        if self.max_iter > 0:
          train_src_param = '--snapshot="{}_iter_{}.solverstate" ^\n'.format(self.snapshot_prefix, self.max_iter)

        with open(self.job_file, 'w') as f:
          f.write('cd {}\n'.format(self.caffe_root))
          f.write('build\\tools\\caffe train ^\n')
          f.write('--solver="{}" ^\n'.format(self.solver_file))
          f.write(train_src_param)
          f.write('--gpu {} 2>&1 | tee {}\\{}.log\n'.format(self.gpus, self.job_dir, self.model_name))

    # Copy the python script to job_dir.
    def scriptToJobDir(self):
        py_file = os.path.abspath(__file__)
        shutil.copy(py_file, self.job_dir)

    # Run the job.
    def runJob(self):
        os.chmod(self.job_file, stat.S_IRWXU)
        subprocess.Popen(self.job_file, shell=True)

    def main(self):
        self.checkFile()

        # Define train network
        train_transform_param = self.defineTrainTransformParam()
        train_net = TrainBuilder(self.train_data, self.train_batch_size, self.num_classes,
            share_location=True, background_label_id=0, net_file=self.train_net_file, 
            model_name=self.model_name, transform_param=train_transform_param, 
            label_map_file=self.label_map_file)
        train_net.createNet()
        train_net.exportNet(self.job_dir, "train")

        # Define test network
        test_transform_param = self.defineTestTransformParam()
        test_net = TestBuilder(self.test_data, self.test_batch_size, self.num_test_image,
            self.num_classes, share_location=True, background_label_id=0, 
            net_file=self.test_net_file, model_name=self.model_name, 
            transform_param=test_transform_param, label_map_file=self.label_map_file, 
            name_size_file=self.name_size_file, output_result_dir=self.output_result_dir)
        test_net.createNet()
        test_net.exportNet(self.job_dir, "test")
        test_net.createDeploy(self.deploy_net_file, self.job_dir, 
            self.resize_height, self.resize_width)

        # Define solver
        iter_size = self.countIter(self.accum_batch_size, self.train_batch_size)
        test_iter = self.countIter(self.num_test_image, self.test_batch_size)
        solver = SolverBuilder(self.base_lr, iter_size, test_iter, self.device_id,
            self.train_net_file, self.test_net_file, self.snapshot_prefix, 
            self.solver_file)
        solver.createSolver()
        solver.exportSolver(self.job_dir)

        if self.resume_training:
            self.findRecentSS()

        self.createJob()
        self.scriptToJobDir()
        self.runJob()

if __name__ == '__main__':
    SSDWaste300().main()