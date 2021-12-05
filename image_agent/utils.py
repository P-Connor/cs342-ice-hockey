import numpy as np
import imageio
from .model import PuckLocator


from torch.utils.data import Dataset, DataLoader
from . import dense_transforms


def _to_image(x, proj, view):
    proj, view = np.array(proj).T, np.array(view).T
    # print(proj)
    # print(view)
    p = proj.dot(view.dot(np.array(list(x) + [1])))
    # print(np.array([p[0] / p[-1], -p[1] / p[-1]]))
    # print(p)
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
        # print(r.keys())
        # print(r['soccer_state'].keys())
        # print(r['soccer_state']['ball'])
        # print(r['team1_instance'][0])
        # print(r['soccer_state']['ball']['id'] in r['team1_instance'][0])
        # stop[0]
        # Only collect one out of every args.skip_every frames
        if(frame % args.skip_every != 0):
            continue

        imgs = r['team1_images']
        players = r['team1_state']
        assert(len(imgs) == len(players))
        assert(r['team1_images'] is not None)

        for j, (img, player) in enumerate(zip(imgs, players)):
            imageio.imwrite(f'{args.save_path}/{i:05d}.png', img)

            puck_location = r['soccer_state']['ball']['location']
            player_location = player['kart']['location']
            player_facing = player['kart']['front'] - np.array(player_location)
            player_facing_u = player_facing / np.linalg.norm(player_facing)
            # puck_location_screen = _to_image(puck_location, player['camera']['projection'], player['camera']['view'])

            puck_on_screen = [r['soccer_state']['ball']
                              ['id'] in r['team1_instance'][j]]
            puck_location[1] = puck_location[1] if not puck_on_screen else -1
            np.savetxt(f'{args.save_path}/{i:05d}.csv',
                       [np.concatenate((puck_location, player_location, player_facing_u))], delimiter=',', fmt='%.3f')
            i += 1

    print("Finished generating " + str(i) + " images\n")


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform=dense_transforms.ToTensor()):
        from PIL import Image
        from glob import glob
        from os import path
        self.data = []
        for f in glob(path.join(dataset_path, '*.csv')):
            i = Image.open(f.replace('.csv', '.png'))

            i.load()
            lbl = np.loadtxt(f, dtype=np.float32, delimiter=',')
            self.data.append(
                (i, lbl[3:6], lbl[:3]))
        self.transform = transform
        print("DATASET LEN " + str(len(self.data)), dataset_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.transform(*data)
        return data


def load_data(dataset_path, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path, transform=transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def load_model():
    from torch import load
    from os import path
    r = PuckLocator()
    r.load_state_dict(load(path.join(path.dirname(
        path.abspath(__file__)), 'puck_locator.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--start_at', type=int, default=0)
    parser.add_argument('--dataset_path', default='raw_data/data')
    parser.add_argument('--save_path', default='image_data')
    parser.add_argument('--skip_every', type=int, default=21)

    args = parser.parse_args()
    generateImages(args)
