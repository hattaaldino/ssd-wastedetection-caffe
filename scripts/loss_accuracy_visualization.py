# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 11:29:34 2021

@author: hattaldino
"""

import pandas as pd
import matplotlib.pyplot as plt

# load data
root_path = ".."
train_log_300 = pd.read_csv(root_path + 
                            "\jobs\VGGNet\wastedata\SSD_300x300\VGG_wastedata_SSD_300x300.log.train")
test_log_300 = pd.read_csv(root_path + 
                           "\jobs\VGGNet\wastedata\SSD_300x300\VGG_wastedata_SSD_300x300.log.test")
train_log_512 = pd.read_csv(root_path + 
                            "\jobs\VGGNet\wastedata\SSD_512x512\VGG_wastedata_SSD_512x512.log.train")
test_log_512 = pd.read_csv(root_path + 
                           "\jobs\VGGNet\wastedata\SSD_512x512\VGG_wastedata_SSD_512x512.log.test")

#visualize train log
#ssd_300
_, ax1 = plt.subplots()
ax1.plot(train_log_300["NumIters"], train_log_300["loss"], lw=0.3)
ax1.set_xlabel('iterasi')
ax1.set_ylabel('loss pelatihan')
plt.savefig(root_path + "\jobs\VGGNet\wastedata\SSD_300x300\VGG_wastedata_SSD_300x300_loss.png")

#ssd_512
_, ax2 = plt.subplots()
ax2.plot(train_log_512["NumIters"], train_log_512["loss"], lw=0.3)
ax2.set_xlabel('iterasi')
ax2.set_ylabel('loss pelatihan')
plt.savefig(root_path + "\jobs\VGGNet\wastedata\SSD_512x512\VGG_wastedata_SSD_512x512_loss.png")

#visualize test log
#ssd_300
test300_phase = list(range(1, len(test_log_300["NumIters"])+1))
_, ax3 = plt.subplots()
ax3.plot(test300_phase, test_log_300["detection_eval"])
ax3.set_xlabel('fase pengujian')
ax3.set_ylabel('akurasi (mAP)')
plt.savefig(root_path + "\jobs\VGGNet\wastedata\SSD_300x300\VGG_wastedata_SSD_300x300_acc.png")

#ssd_512
test512_phase = list(range(1, len(test_log_512["NumIters"])+1))
_, ax4 = plt.subplots()
ax4.plot(test512_phase, test_log_512["detection_eval"])
ax4.set_xlabel('fase pengujian')
ax4.set_ylabel('akurasi (mAP)')
plt.savefig(root_path + "\jobs\VGGNet\wastedata\SSD_512x512\VGG_wastedata_SSD_512x512_acc.png")

# plt.show()