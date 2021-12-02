import numpy as np
import imageio

from torch.utils.data import Dataset, DataLoader
from . import dense_transforms

DATASET_PATH = 'raw_data'


def load_recording(recording):
    from pickle import load
    with open(recording, 'rb') as f:
        while True:
            try:
                yield load(f)
            except EOFError:
                break


def generateImages(dataset_path=DATASET_PATH, save_path='image_data'):
    i = 0
    for r in load_recording(dataset_path):
        imgs = [r['team1_images'][0], r['team1_images'][1]]
        for img in imgs:
            imageio.imwrite(f'{save_path}/{i:05d}.png', img)
            np.savetxt(f'{save_path}/{i:05d}.csv',
                       r['soccer_state']['ball']['location'], delimiter=',', fmt='%i')
            i += 1


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor()):
        from PIL import Image
        from glob import glob
        from os import path
        self.data = []
        for f in glob(path.join(dataset_path, '*.csv')):
            i = Image.open(f.replace('.csv', '.png'))
            i.load()
            self.data.append(
                (i, np.loadtxt(f, dtype=np.float32, delimiter=',')))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.transform(*data)
        return data


def load_data(dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path, transform=transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


if __name__ == '__main__':
    # l = load_recording('raw_data/data')
    # for i in l:
    #     print(len(i['team1_images']))
    #     print(i['team1_images'][0].shape)
    #     stop[0]
    generateImages('raw_data/data')
