import torch
import numpy as np


# Can read single slices from the MM-WHS dataset
class MmWhsDataset(torch.utils.data.Dataset):
    # takes a list of file ids and a directory root
    def __init__(self, file_list, file_root):
        self.file_list = file_list
        self.file_root = file_root
        self.binary_labels = True

    # Denotes the total number of samples
    def __len__(self):
        return len(self.file_list)

    # Generates one sample of data
    def __getitem__(self, index):
        # Select sample
        file_idx = self.file_list[index]
        # print(f"Getting index {index} : {file_idx} ")

        name_img_slice = f"{self.file_root}/MM-WHS-{file_idx}-image-has_heart.npy"
        name_img_label = f"{self.file_root}/MM-WHS-{file_idx}-label-has_heart.npy"

        one_slice = np.load(name_img_slice)
        labels = np.load(name_img_label)

        if self.binary_labels:
            labels = labels > 0
            # Can not compute loss using bools. So convert to floats.
            labels = np.float32(labels)

        # Adding one channel.
        # Pytorch would like the output to be (channels, width, height)
        one_slice = one_slice.reshape(1, one_slice.shape[0], one_slice.shape[1])
        labels = labels.reshape(1, labels.shape[0], labels.shape[1])

        return one_slice, labels
