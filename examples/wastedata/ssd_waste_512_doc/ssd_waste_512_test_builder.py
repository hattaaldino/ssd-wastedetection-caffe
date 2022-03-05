from ssd_waste_300_net_builder import *

class TestBuilder(NetBuilder):

    def __init__(self, data, batch_size, num_test_image, num_classes, share_location, 
            background_label_id, net_file, model_name, transform_param, label_map_file, 
            name_size_file, output_result_dir):
        self.num_test_image = num_test_image
        self.name_size_file = name_size_file
        self.output_result_dir = output_result_dir

        super().__init__(data, batch_size, num_classes, share_location, background_label_id, 
                        net_file, model_name, transform_param, label_map_file)

        self.defineDetectionParam()

    # Set detection layers parameters.
    def defineDetectionParam(self):
        self.det_out_param = {
            'num_classes': self.num_classes,
            'share_location': self.share_location,
            'background_label_id': self.background_label_id,
            'nms_param': {'nms_threshold': 0.45, 'top_k': 400},
            'save_output_param': {
                'output_directory': self.output_result_dir,
                'output_name_prefix': "comp4_det_test_",
                'output_format': "VOC",
                'label_map_file': self.label_map_file,
                'name_size_file': self.name_size_file,
                'num_test_image': self.num_test_image,
                },
            'keep_top_k': 200,
            'confidence_threshold': 0.01,
            'code_type': self.priorbox_codetype,
            }

        # parameters for evaluating detection results.
        self.det_eval_param = {
            'num_classes': self.num_classes,
            'background_label_id': self.background_label_id,
            'overlap_threshold': 0.5,
            'evaluate_difficult_gt': False,
            'name_size_file': self.name_size_file,
            }

    # Create test net.
    def createNet(self):
        self.net.data, self.net.label = self.createAnnotatedDataLayer(self.data, batch_size=self.batch_size,
        train=False)

        self.VGGNetBody(from_layer='data')

        self.addExtraLayers()

        mbox_layers = self.createMultiBoxHead(data_layer='data', kernel_size=3, pad=1)

        conf_name = "mbox_conf"
        reshape_name = "{}_reshape".format(conf_name)
        self.net[reshape_name] = L.Reshape(self.net[conf_name], shape=dict(dim=[0, -1, self.num_classes]))
        softmax_name = "{}_softmax".format(conf_name)
        self.net[softmax_name] = L.Softmax(self.net[reshape_name], axis=2)
        flatten_name = "{}_flatten".format(conf_name)
        self.net[flatten_name] = L.Flatten(self.net[softmax_name], axis=1)
        mbox_layers[1] = self.net[flatten_name]

        self.net.detection_out = L.DetectionOutput(*mbox_layers,
            detection_output_param=self.det_out_param,
            include=dict(phase=caffe_pb2.Phase.Value('TEST')))
        self.net.detection_eval = L.DetectionEvaluate(self.net.detection_out, self.net.label,
            detection_evaluate_param=self.det_eval_param,
            include=dict(phase=caffe_pb2.Phase.Value('TEST')))

    # Create deploy net.
    def createDeploy(self, deploy_net_file, job_dir, resize_height, resize_width):
        # Remove the first and last layer from test net.
        deploy_net = self.net
        with open(deploy_net_file, 'w') as f:
            deploy_net_param = deploy_net.to_proto()

            # Remove the first (AnnotatedData) and last (DetectionEvaluate) layer from test net.
            del deploy_net_param.layer[0]
            del deploy_net_param.layer[-1]
            deploy_net_param.name = '{}_deploy'.format(self.model_name)
            deploy_net_param.input.extend(['data'])
            deploy_net_param.input_shape.extend([
                caffe_pb2.BlobShape(dim=[1, 3, resize_height, resize_width])])
            print(deploy_net_param, file=f)
        shutil.copy(deploy_net_file, job_dir)
