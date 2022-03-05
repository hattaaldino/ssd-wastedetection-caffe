import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib import colors as col

########
#TODO
########
# Histogram of image dimensions /done
# Histogram of image aspect ratios /done
# Histogram of image color channels /done
# Histogram of object labels distribution /done
# Histogram of bounding box aspect ratios
# Histogram of bounding box size areas by image dimension


# source path
root_path = ".."
imageset_dir = root_path + "\data\wastedata\ImageSets"
annotation_dir = root_path + "\data\wastedata\Annotations"
train_list = imageset_dir + "\\trainval.txt"
test_list = imageset_dir + "\\test.txt"

# histogram values
w_list = list()
h_list = list()
ch_list = list()
ar_list = list()
bbox_ar_list = list()
bbox_sz_list = list()
lbl_list = {
    'trainval': {
        'organik': 0,
        'anorganik': 0
        },
    'test': {
        'organik': 0,
        'anorganik': 0
        }
    }

# collect dimension information on image level
def ImgDimInfo(size_tag):
    width = int(size_tag.find('width').text)
    height = int(size_tag.find('height').text)
    depth = int(size_tag.find('depth').text)
    
    w_list.append(width)
    h_list.append(height)
    ch_list.append(depth)
    
    if width != 0 and height != 0:
        ar_list.append(round(width / height, 1))
    else:
        return False
    
    return True
    
# collect object label
def collectLabel(setId, label_tag):
    label = label_tag.text
    
    if setId <= 2000:
        if label == "organik":
            lbl_list['trainval']['organik'] += 1
        else:
            lbl_list['trainval']['anorganik'] += 1
    elif setId > 2000 and setId < 2500:
        if label == "organik":
            lbl_list['test']['organik'] += 1
        else:
            lbl_list['test']['anorganik'] += 1

# collect bbox dimension information
def bboxInfo(bbox_tag):
    xmin = int(bbox_tag.find('xmin').text)
    xmax = int(bbox_tag.find('xmax').text)
    ymin = int(bbox_tag.find('ymin').text)
    ymax = int(bbox_tag.find('ymax').text)
    
    box_w = xmax - xmin
    box_h = ymax - ymin
    box_area = box_w * box_h
    boxscale_to_img = float(box_area / (w_list[-1] * h_list[-1]))
    
    if box_w != 0 and box_h != 0:
        bbox_ar_list.append(round(box_w / box_h, 1))
        bbox_sz_list.append(round(boxscale_to_img, 5))
    else:
        bbox_ar_list.append(0.0)
        bbox_sz_list.append(0.0)
        
def imgInfoGrf():
    fig, ax1 = plt.subplots()
    ax1.hist2d(w_list, h_list, bins=(60, 50) , norm=col.LogNorm())
    ax1.set_ylabel("Tinggi")
    ax1.set_xlabel("Lebar")
    
    fig, ax2 = plt.subplots()
    ax2.hist(ar_list, bins=50)
    ax2.set_xlabel("Aspek Rasio (lebar/tinggi)")
    
    fig, ax3 = plt.subplots()
    ax3.hist(ch_list, bins=15, range=(1, 6), align='left')
    ax3.set_xlabel("Dimensi Warna")
    
    plt.show()
    
def labelGrf():
    set_data = ['organik', 'anorganik']
    trainval_bar = [lbl_list['trainval'][set_] for set_ in set_data]
    test_bar = [lbl_list['test'][set_] for set_ in set_data]
    X_ = np.arange(len(set_data))
    
    fig, ax = plt.subplots()
    ax.bar(X_ - 0.15, trainval_bar, 0.3, label="trainval")
    ax.bar(X_ + 0.15, test_bar, 0.3, label="test")
    ax.legend()
    ax.set_xticks(X_)
    ax.set_xticklabels(set_data)
    plt.show()

def bboxInfoGrf():
    fig, ax1 = plt.subplots()
    ax1.hist(bbox_ar_list, bins=50)
    ax1.set_xlabel("Aspek Rasio (lebar/tinggi)")
    
    fig, ax2 = plt.subplots()
    ax2.hist(bbox_sz_list, bins=50)
    ax2.set_xlabel("Skala")
    
    plt.show()

# collect information from every annotation in trainval and test set
for set_list in train_list, test_list:
    with open(set_list, 'r') as f:
        for name in f.readlines():
            name = name.strip("\n")
            imgIdNum = int(name.split('_')[1])
            filepath = annotation_dir + "\\" + name + ".xml"
            
            tree = ET.parse(filepath)
            root = tree.getroot()
            if ImgDimInfo(root.find('size')):
                for obj in root.findall('object'):
                    collectLabel(imgIdNum, obj.find('name'))
                    bboxInfo(obj.find('bndbox'))

small_obj = 0
for bbox in bbox_sz_list:
    if bbox < 0.04:
        small_obj+=1
print(len(bbox_sz_list))
small_obj_per = (small_obj / len(bbox_sz_list)) * 100
print(f"Small Object: {small_obj}")
print(f"Small Object Percentage: {small_obj_per}")
       
# visualize all information into graph
# imgInfoGrf()
# labelGrf()
# bboxInfoGrf()
