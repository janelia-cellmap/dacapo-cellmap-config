import numpy as np
import matplotlib.pyplot as plt

def visualize_datasplit(datasplit, id =0):
    raw = datasplit.train_configs[id].raw_config.array()
    gt = datasplit.train_configs[id].gt_config.array()
    raw_data = raw.to_ndarray(gt.roi)
    gt_data = gt.to_ndarray(gt.roi)
    empty_channel = np.zeros((1, *gt_data.shape[1:]), dtype=gt_data.dtype)
    gt_data = np.concatenate((empty_channel, gt_data), axis=0)
    gt_data = np.argmax(gt_data, axis=0)
    plt.imshow(raw_data[0], cmap='gray')
    plt.imshow(gt_data[0], cmap='jet', alpha=0.5)
    plt.show()