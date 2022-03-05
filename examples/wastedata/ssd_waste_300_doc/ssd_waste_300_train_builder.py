from ssd_waste_300_net_builder import *

class TrainBuilder(NetBuilder):

    def __init__(self, data, batch_size, num_classes, share_location, background_label_id, 
            net_file, model_name, transform_param, label_map_file):
        super().__init__(data, batch_size, num_classes, share_location, background_label_id, 
                        net_file, model_name, transform_param, label_map_file)

        self.defineSamplerParam()
        self.defineLossParam()

    # Specify the batch sampler parameters.
    def defineSamplerParam(self):
        self.batch_sampler = [
                {
                        'sampler': {
                                },
                        'max_trials': 1,
                        'max_sample': 1,
                },
                {
                        'sampler': {
                                'min_scale': 0.3,
                                'max_scale': 1.0,
                                'min_aspect_ratio': 0.5,
                                'max_aspect_ratio': 2.0,
                                },
                        'sample_constraint': {
                                'min_jaccard_overlap': 0.1,
                                },
                        'max_trials': 50,
                        'max_sample': 1,
                },
                {
                        'sampler': {
                                'min_scale': 0.3,
                                'max_scale': 1.0,
                                'min_aspect_ratio': 0.5,
                                'max_aspect_ratio': 2.0,
                                },
                        'sample_constraint': {
                                'min_jaccard_overlap': 0.3,
                                },
                        'max_trials': 50,
                        'max_sample': 1,
                },
                {
                        'sampler': {
                                'min_scale': 0.3,
                                'max_scale': 1.0,
                                'min_aspect_ratio': 0.5,
                                'max_aspect_ratio': 2.0,
                                },
                        'sample_constraint': {
                                'min_jaccard_overlap': 0.5,
                                },
                        'max_trials': 50,
                        'max_sample': 1,
                },
                {
                        'sampler': {
                                'min_scale': 0.3,
                                'max_scale': 1.0,
                                'min_aspect_ratio': 0.5,
                                'max_aspect_ratio': 2.0,
                                },
                        'sample_constraint': {
                                'min_jaccard_overlap': 0.7,
                                },
                        'max_trials': 50,
                        'max_sample': 1,
                },
                {
                        'sampler': {
                                'min_scale': 0.3,
                                'max_scale': 1.0,
                                'min_aspect_ratio': 0.5,
                                'max_aspect_ratio': 2.0,
                                },
                        'sample_constraint': {
                                'min_jaccard_overlap': 0.9,
                                },
                        'max_trials': 50,
                        'max_sample': 1,
                },
                {
                        'sampler': {
                                'min_scale': 0.3,
                                'max_scale': 1.0,
                                'min_aspect_ratio': 0.5,
                                'max_aspect_ratio': 2.0,
                                },
                        'sample_constraint': {
                                'max_jaccard_overlap': 1.0,
                                },
                        'max_trials': 50,
                        'max_sample': 1,
                },
            ]

    # Set multiBoxLoss parameters.
    def defineLossParam(self):
        self.multibox_loss_param = {
            'loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
            'conf_loss_type': P.MultiBoxLoss.SOFTMAX,
            'loc_weight': 1.0,
            'num_classes': self.num_classes,
            'share_location': self.share_location,
            'match_type': P.MultiBoxLoss.PER_PREDICTION,
            'overlap_threshold': 0.5,
            'use_prior_for_matching': True,
            'background_label_id': self.background_label_id,
            'use_difficult_gt': True,
            'mining_type': P.MultiBoxLoss.MAX_NEGATIVE,
            'neg_pos_ratio': 3.0,
            'neg_overlap': 0.5,
            'code_type': self.priorbox_codetype,
            'ignore_cross_boundary_bbox': False,
            }

        self.loss_param = {
            'normalization': P.Loss.VALID,
            }

    # Create train net.
    def createNet(self):
        self.net.data, self.net.label = self.createAnnotatedDataLayer(self.data, batch_size=self.batch_size,
                train=True, batch_sampler=self.batch_sampler)

        self.VGGNetBody(from_layer='data')

        self.addExtraLayers()

        mbox_layers = self.createMultiBoxHead(data_layer='data', kernel_size=3, pad=1)

        # Create the MultiBoxLossLayer.
        name = "mbox_loss"
        mbox_layers.append(self.net.label)
        self.net[name] = L.MultiBoxLoss(*mbox_layers, multibox_loss_param=self.multibox_loss_param,
                loss_param=self.loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                propagate_down=[True, True, False, False])