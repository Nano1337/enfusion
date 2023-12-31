"""Implements dataloaders for the AVMNIST dataset.

Here, the data is assumed to be in a folder titled "avmnist".
"""
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.nn import functional as F


class Avmnistdataset(Dataset):
    def __init__(self, data, max_pad=False, max_pad_num=50):
        self.dataset = data
        self.max_pad = max_pad
        self.max_pad_num = max_pad_num
    def __getitem__(self, ind):
        vision = torch.tensor(self.dataset[ind][0])
        audio = torch.tensor(self.dataset[ind][1])
        label = torch.tensor(self.dataset[ind][2])
        if self.max_pad:
            tmp = [vision, audio, label]
            for i in range(len(tmp) - 1):
                tmp[i] = tmp[i][:self.max_pad_num]
                tmp[i] = F.pad(tmp[i], (0, 0, 0, self.max_pad_num - tmp[i].shape[0]))
        else:
            tmp = [vision, audio, ind, label]
        return tmp
    def __len__(self):
        return len(self.dataset)


def get_dataloader(data_dir, batch_size=40, num_workers=8, max_pad=False, max_seq_len=50, train_shuffle=True, flatten_audio=False, flatten_image=False, unsqueeze_channel=True, generate_sample=False, normalize_image=True, normalize_audio=True):
    """Get dataloaders for AVMNIST.

    Args:
        data_dir (str): Directory of data.
        batch_size (int, optional): Batch size. Defaults to 40.
        num_workers (int, optional): Number of workers. Defaults to 8.
        train_shuffle (bool, optional): Whether to shuffle training data or not. Defaults to True.
        flatten_audio (bool, optional): Whether to flatten audio data or not. Defaults to False.
        flatten_image (bool, optional): Whether to flatten image data or not. Defaults to False.
        unsqueeze_channel (bool, optional): Whether to unsqueeze any channels or not. Defaults to True.
        generate_sample (bool, optional): Whether to generate a sample and save it to file or not. Defaults to False.
        normalize_image (bool, optional): Whether to normalize the images before returning. Defaults to True.
        normalize_audio (bool, optional): Whether to normalize the audio before returning. Defaults to True.

    Returns:
        tuple: Tuple of (training dataloader, validation dataloader, test dataloader)
    """
    trains = [np.load(data_dir+"/image/train_data.npy"), np.load(data_dir +
                                                                 "/audio/train_data.npy"), np.load(data_dir+"/train_labels.npy")]
    tests = [np.load(data_dir+"/image/test_data.npy"), np.load(data_dir +
                                                               "/audio/test_data.npy"), np.load(data_dir+"/test_labels.npy")]
    if flatten_audio:
        trains[1] = trains[1].reshape(60000, 112*112)
        tests[1] = tests[1].reshape(10000, 112*112)
    if generate_sample:
        _saveimg(trains[0][0:100])
        _saveaudio(trains[1][0:9].reshape(9, 112*112))
    if normalize_image:
        trains[0] /= 255.0
        tests[0] /= 255.0
    if normalize_audio:
        trains[1] = trains[1]/255.0
        tests[1] = tests[1]/255.0
    if not flatten_image:
        trains[0] = trains[0].reshape(60000, 28, 28)
        tests[0] = tests[0].reshape(10000, 28, 28)
    if unsqueeze_channel:
        trains[0] = np.expand_dims(trains[0], 1)
        tests[0] = np.expand_dims(tests[0], 1)
        trains[1] = np.expand_dims(trains[1], 1)
        tests[1] = np.expand_dims(tests[1], 1)
    trains[2] = trains[2].astype(int)
    tests[2] = tests[2].astype(int)
    process = eval("_process_2") if max_pad else eval("_process_1")
    trainlist = [[trains[j][i] for j in range(3)] for i in range(60000)]
    testlist = [[tests[j][i] for j in range(3)] for i in range(10000)]
    valids = DataLoader(Avmnistdataset(trainlist[55000:60000], max_pad=max_pad, max_pad_num=max_seq_len,), shuffle=False, num_workers=num_workers, batch_size=batch_size, collate_fn=process)
    tests = DataLoader(Avmnistdataset(testlist, max_pad=max_pad, max_pad_num=max_seq_len,), shuffle=False, num_workers=num_workers, batch_size=batch_size, collate_fn=process)
    trains = DataLoader(Avmnistdataset(trainlist[0:55000], max_pad=max_pad, max_pad_num=max_seq_len,), shuffle=train_shuffle, num_workers=num_workers, batch_size=batch_size, collate_fn=process)
    return trains, valids, tests

# this function creates an image of 100 numbers in avmnist


def _saveimg(outa):
    from PIL import Image
    t = np.zeros((300, 300))
    for i in range(0, 100):
        for j in range(0, 784):
            imrow = i // 10
            imcol = i % 10
            pixrow = j // 28
            pixcol = j % 28
            t[imrow*30+pixrow][imcol*30+pixcol] = outa[i][j]
    newimage = Image.new('L', (300, 300))  # type, size
    
    newimage.putdata(t.reshape((90000,)))
    newimage.save("samples.png")


def _saveaudio(outa):
    
    from PIL import Image
    t = np.zeros((340, 340))
    for i in range(0, 9):
        for j in range(0, 112*112):
            imrow = i // 3
            imcol = i % 3
            pixrow = j // 112
            pixcol = j % 112
            t[imrow*114+pixrow][imcol*114+pixcol] = outa[i][j]
    newimage = Image.new('L', (340, 340))  # type, size
    
    newimage.putdata(t.reshape((340*340,)))
    newimage.save("samples2.png")

def _process_1(inputs):
    processed_input = []
    processed_input_lengths = []
    inds = []
    labels = []


    for i in range(len(inputs[0]) - 2):
        feature = []
        for sample in inputs:
            feature.append(sample[i])
        processed_input_lengths.append(torch.as_tensor([v.size(0) for v in feature]))
        pad_seq = pad_sequence(feature, batch_first=True)
        processed_input.append(pad_seq)

    for sample in inputs:
        
        inds.append(sample[-2])
        # if len(sample[-2].shape) > 2:
        #     labels.append(torch.where(sample[-2][:, 1] == 1)[0])
        # else:
        labels.append(sample[-1])

    return processed_input, processed_input_lengths, \
           torch.tensor(inds).view(len(inputs), 1), torch.tensor(labels).view(len(inputs), 1)


def _process_2(inputs):
    processed_input = []
    processed_input_lengths = []
    labels = []

    for i in range(len(inputs[0]) - 1):
        feature = []
        for sample in inputs:
            feature.append(sample[i])
        processed_input_lengths.append(torch.as_tensor([v.size(0) for v in feature]))
        # pad_seq = pad_sequence(feature, batch_first=True)
        processed_input.append(torch.stack(feature))

    for sample in inputs:
        labels.append(sample[-1])

    return processed_input[0], processed_input[1], torch.tensor(labels).view(len(inputs), 1)
