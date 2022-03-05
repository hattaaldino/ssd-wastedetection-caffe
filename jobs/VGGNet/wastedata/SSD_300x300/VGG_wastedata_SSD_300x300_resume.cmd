cd D:\Riset\WasteSSD
build\tools\caffe train ^
--solver="models\VGGNet\wastedata\SSD_300x300\solver.prototxt" ^
--snapshot="D:\Riset\WasteSSD\models\VGGNet\wastedata\SSD_300x300\VGG_wastedata_SSD_300x300_iter_20000.solverstate" ^
--gpu 1 2>&1 | tee jobs\VGGNet\wastedata\SSD_300x300\VGG_wastedata_SSD_300x300.log