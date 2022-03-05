cd D:\Riset\WasteSSD
build\tools\caffe train ^
--solver="models\VGGNet\wastedata\SSD_300x300\solver.prototxt" ^
--weights="models\VGGNet\VGG_ILSVRC_16_layers_fc_reduced.caffemodel" ^
--gpu 1 2>&1 | tee jobs\VGGNet\wastedata\SSD_300x300\VGG_wastedata_SSD_300x300.log
