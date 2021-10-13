import os, argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

from DiffNet.networks.autoencoders import AE

"""
Matsci data loader
"""
class MicrostructureDataset(Dataset):
    """
    Class to read the numpy dataset for the microstructure
    """

    def __init__(self, data_path, transform=None):
        self.tuple_data = np.load(data_path, allow_pickle=True)
        self.data = self.tuple_data[:, 0]
        self.label = self.tuple_data[:, 1]  # for J
        self.ff = self.tuple_data[:, 2]  # for ff
        self.transform = transform

    # return image and labels
    def __getitem__(self, index):
        x = torch.FloatTensor(np.float32(self.data[index]))
        if self.transform is not None:
            x = self.transform(x)
        y = torch.FloatTensor(np.expand_dims(np.float32(self.label[index]), axis=0))
        z = torch.FloatTensor(np.expand_dims(np.float32(self.ff[index]), axis=0))
        return x, y, z

    def __len__(self):
        return self.data.shape[0]






def training_epoch(model, optimizer, data):
    gen_data = model(torch.randn_like(real_data))

    # calculate losses
    l2_loss = F.mse_loss(input=gen_data, target=fake_data)

    optimizer.zero_grad()
    l2_loss.backward()
    optimizer.step()

    return l2_loss / data.shape(0)

def load_data(path_to_folder):
    dataset = MicrostructureDataset(path_to_folder)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True, pin_memory=True)
    return dataset_loader

def training_data_loader():
    return load_data('/data/Joshua/DARPA_data/augmented_JF_filtered_norm_balanced_train.npy')


def main(args):
    # training parameters
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = AE(in_channels=2, out_channels=2, dims=16, n_downsample=3)
    model = model.to(device)

    # data
    dataloader = training_data_loader()
    dataiter = iter(dataloader)
    # ouput path
    output_path = os.path.join('./pretrained_AE/')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # prints
    print("# of params in model: ", sum(a.numel() for a in model.parameters()))

    # training loop
    for i in range(args.epoch):
        try:
            real_data, _, _ = next(dataiter)
        except StopIteration:
            dataiter = iter(dataloader)
            real_data, _, _ = dataiter.next()
        real_data = real_data.to(device)
        print(training_epoch(model, optimizer, real_data))

    # save training outputs and model checkpoints
    torch.save(model.state_dict(), os.path.join(output_path, "microstructure_AE.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DEQ model for implicit topopt')
    parser.add_argument('-ep', '--epoch', default=500, type=int,
                        help='How many epochs would you like to train for?')
    hparams = parser.parse_args()
    main(hparams)
