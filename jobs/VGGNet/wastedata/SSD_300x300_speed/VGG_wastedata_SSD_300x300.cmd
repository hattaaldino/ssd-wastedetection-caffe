cd D:\Riset\WasteSSD
build\tools\caffe train ^
--solver="models\VGGNet\wastedata\SSD_300x300_speed\solver.prototxt" ^
--weights="models\VGGNet\wastedata\SSD_300x300\VGG_wastedata_SSD_300x300_iter_60000.caffemodel" ^
--gpu 0 2>&1 | tee jobs\VGGNet\wastedata\SSD_300x300_speed\VGG_wastedata_SSD_300x300_speed_test_60000.log
