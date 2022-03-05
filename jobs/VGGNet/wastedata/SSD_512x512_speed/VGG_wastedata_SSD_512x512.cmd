cd D:\Riset\WasteSSD
build\tools\caffe train ^
--solver="models\VGGNet\wastedata\SSD_512x512_speed\solver.prototxt" ^
--weights="models\VGGNet\wastedata\SSD_512x512\VGG_wastedata_SSD_512x512_iter_60000.caffemodel" ^
--gpu 1 2>&1 | tee jobs\VGGNet\wastedata\SSD_512x512_speed\VGG_wastedata_SSD_512x512_speed_test_60000.log
