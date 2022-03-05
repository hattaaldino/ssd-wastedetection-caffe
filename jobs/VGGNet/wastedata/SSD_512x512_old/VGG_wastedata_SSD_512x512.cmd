cd D:\Riset\WasteSSD
build\tools\caffe train ^
--solver="models\VGGNet\wastedata\SSD_512x512\solver.prototxt" ^
--weights="models\VGGNet\VGG_ILSVRC_16_layers_fc_reduced.caffemodel" ^
--gpu 0 2>&1 | tee jobs\VGGNet\wastedata\SSD_512x512\VGG_wastedata_SSD_512x512.log
