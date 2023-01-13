from numpy import loadtxt
from mm_whs_dataset import MmWhsDataset
import torch


def test_read_file_list():
    file_name = "slice_list.txt"
    file_ids = loadtxt(file_name, dtype=str)
    print(file_ids)


def test_dataset():
    file_root = 'C:/data/test/MM-WHS/'
    training_file_names = "slice_list.txt"
    file_ids = loadtxt(training_file_names, dtype=str)
    training_set = MmWhsDataset(file_ids, file_root)

    training_loader = torch.utils.data.DataLoader(training_set, batch_size=8, num_workers=12, shuffle=True)
    print(f"Number of batches in dataset: {len(training_loader)}")

    # lets see if we can use a GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # Not strictly necessary but can potentially speed up the network and improve memory footprint
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    # Start looping over batches
    for image_batch, label_batch in training_loader:
        print(f"Number of images/label images in current batch: {len(image_batch)}")
        # Transfer to GPU if available
        image_batch = image_batch.to(device)
        label_batch = label_batch.to(device)
        # Do fancy deep learning here


if __name__ == '__main__':
    # test_read_file_list()
    test_dataset()
