import numpy as np
import imageio

from torch.utils.data import Dataset, DataLoader
from . import dense_transforms


def _to_image(x, proj, view):
    p = proj @ view @ np.array(list(x) + [1])
    print(np.array([p[0] / p[-1], -p[1] / p[-1]]))
    return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)

def load_recording(recording):
    from pickle import load
    with open(recording, 'rb') as f:
        while True:
            try:
                yield load(f)
            except EOFError:
                break

def generateImages(args):
    i = args.start_at
    for frame, r in enumerate(load_recording(args.dataset_path)):
        
        # Only collect one out of every args.skip_every frames
        if(frame % args.skip_every != 0):
            continue

        imgs = r['team1_images']
        players = r['team1_state']
        assert(len(imgs) == len(players))
        assert(r['team1_images'] is not None)
        
        for (img, player) in zip(imgs, players):
            imageio.imwrite(f'{args.save_path}/{i:05d}.png', img)
            
            puck_location = r['soccer_state']['ball']['location']
            puck_location_screen = _to_image(puck_location, player['camera']['projection'], player['camera']['view'])
            is_off_screen = 1 if abs(puck_location_screen[0]) == 1.0 or abs(puck_location_screen[1]) == 1.0 else 0
            print(player['camera'])
            if(is_off_screen == 1):
                print(i, puck_location_screen)
            np.savetxt(f'{args.save_path}/{i:05d}.csv',
                [np.concatenate((puck_location, [is_off_screen]))], delimiter=',', fmt='%.3f')
            i += 1

    print("Finished generating " + str(i) + " images\n")


# class SuperTuxDataset(Dataset):
#     def __init__(self, dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor()):
#         from PIL import Image
#         from glob import glob
#         from os import path
#         self.data = []
#         for f in glob(path.join(dataset_path, '*.csv')):
#             i = Image.open(f.replace('.csv', '.png'))
#             i.load()
#             self.data.append(
#                 (i, np.loadtxt(f, dtype=np.float32, delimiter=',')))
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         data = self.data[idx]
#         data = self.transform(*data)
#         return data


# def load_data(dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=128):
#     dataset = SuperTuxDataset(dataset_path, transform=transform)
#     return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--start_at', type=int, default=0)
    parser.add_argument('--dataset_path', default='raw_data/data')
    parser.add_argument('--save_path', default='image_data')
    parser.add_argument('--skip_every', type=int, default=21)

    args = parser.parse_args()
    generateImages(args)
