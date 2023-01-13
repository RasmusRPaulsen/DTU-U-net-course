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
        print(f"Getting index {index} : {file_idx} ")

        name_img_slice = f"{self.file_root}/MM-WHS-{file_idx}-image-has_heart.npy"
        name_img_label = f"{self.file_root}/MM-WHS-{file_idx}-label-has_heart.npy"

        one_slice = np.load(name_img_slice)
        labels = np.load(name_img_label)

        if self.binary_labels:
            labels = labels > 0

        return one_slice, labels
