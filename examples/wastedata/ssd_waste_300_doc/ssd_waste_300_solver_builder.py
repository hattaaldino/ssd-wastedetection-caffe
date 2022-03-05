import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

import shutil

class SolverBuilder:

    def __init__(self, base_lr, iter_size, test_iter, device_id,
            train_net_file, test_net_file, snapshot_prefix, solver_file):
        self.train_net_file = train_net_file
        self.test_net_file = test_net_file
        self.snapshot_prefix = snapshot_prefix
        self.solver_file = solver_file

        self.solver_mode = P.Solver.GPU

        # Solver parameters
        self.solver_param = {
            # Train parameters
            'base_lr': base_lr,
            'weight_decay': 0.0005,
            'lr_policy': "multistep",
            'stepvalue': [30000, 50000, 60000],
            'gamma': 0.1,
            'momentum': 0.9,
            'iter_size': iter_size,
            'max_iter': 60000,
            'snapshot': 10000,
            'display': 10,
            'average_loss': 10,
            'type': "SGD",
            'solver_mode': self.solver_mode,
            'device_id': device_id,
            'debug_info': False,
            'snapshot_after_train': True,
            # Test parameters
            'test_iter': [test_iter],
            'test_interval': 5000,
            'eval_type': "detection",
            'ap_version': "11point",
            'test_initialization': False,
            'show_per_class_result': True,
            }

    def createSolver(self):
        self.solver = caffe_pb2.SolverParameter(
                train_net=self.train_net_file,
                test_net=[self.test_net_file],
                snapshot_prefix=self.snapshot_prefix,
                **self.solver_param)

    def exportSolver(self, job_dir):
        with open(self.solver_file, 'w') as f:
            assert self.solver, "Solver not defined!"
            print(self.solver, file=f)
        shutil.copy(self.solver_file, job_dir)