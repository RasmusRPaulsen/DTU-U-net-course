from numpy import loadtxt
from mm_whs_dataset import MmWhsDataset
import torch
import torch.nn as nn
import torch.optim as optim
from u_net_model import UnetModel
import numpy as np

def test_read_file_list():
    file_name = "slice_list.txt"
    file_ids = loadtxt(file_name, dtype=str)
    print(file_ids)


def do_train_model():
    file_root = 'C:/data/test/MM-WHS/'
    training_file_names = "slice_list.txt"
    file_ids = loadtxt(training_file_names, dtype=str)

    # lets see if we can use a GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # Not strictly necessary but can potentially speed up the network and improve memory footprint
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    u_net = UnetModel()
    u_net.to(device)

    n_epochs = 10000
    learning_rate = 0.005
    batch_size = 32
    alpha = 0.5

    training_set = MmWhsDataset(file_ids, file_root)
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, num_workers=12, shuffle=True)
    print(f"Number of batches in dataset: {len(training_loader)}")

    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(u_net.parameters(), lr=learning_rate)
    # model_ids = []

    print("~  B E G I N   T R A I N I N G  ~\n")
    losses, val_losses = [], []

    for e in range(n_epochs):
        u_net.train()

        losses = []

        # Start looping over batches
        for image_batch, label_batch in training_loader:
            # print(f"Number of images/label images in current batch: {len(image_batch)}")
            # The image batch will be in the format (batches, channels, width, height)
            # For example (8, 1, 64 ,64)
            # Transfer to GPU if available
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward prediction based on input image
            outputs = u_net(image_batch)
            predicted_heart, predicted_background = outputs[:, 0], outputs[:, 1]

            heart_label = label_batch[:, 0]
            background_label = 1 - label_batch[:, 0]

            # calculate weighted loss
            loss = alpha * criterion(predicted_heart, heart_label) \
                   + (1 - alpha) * criterion(predicted_background, background_label)

            # print(loss)
            losses.append(loss.item())

            # backpropagate errors
            loss.backward()

            # optimize
            optimizer.step()

        sum_loss = np.sum(losses)
        n_losses = len(losses)
        print(f"Epoch {e} average loss {sum_loss/n_losses}")


if __name__ == '__main__':
    do_train_model()
