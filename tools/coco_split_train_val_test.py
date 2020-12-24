#!/usr/bin/env python
import os
import os.path as osp
import json
import random
from tqdm import tqdm
import shutil
import click

@click.command()
@click.option('--train', type=float, default=0.7)
@click.option('--val', type=float, default=0.2)
@click.option('--test', type=float, default=0.1)
@click.option('--no_shuffle', is_flag=True)
@click.option('--seed', default=7)
@click.option('--out_f', '-o', default='')
@click.argument('src_f')
def main(train, val, test, no_shuffle, seed, out_f, src_f):
    print (f"loading {src_f}")
    data = json.load(open(src_f))
    print (f"loaded {src_f}")

    imgs0 = data['images']
    if not no_shuffle:
        random.seed(seed)
        random.shuffle(imgs0)

    anns = data['annotations']

    N = len(imgs0)

    train_nr = int(N * train)
    val_nr = int(N * val)
    test_nr = N - train_nr - val_nr

    ranges = {
        'train': (0, train_nr),
        'val': (train_nr, train_nr + val_nr),
        'test': (train_nr + val_nr, N),
    }

    for tp, (start, end) in ranges.items():
        data = {i: data[i] for i in ['info', 'licenses', 'categories']}
        imgs = imgs0[start:end]
        data['images'] = imgs

        img_ids = set(i['id'] for i in imgs)
        data['annotations'] = [i for i in anns if i['image_id'] in img_ids]

        if out_f:
            dst_f = osp.splitext(out_f)[0]
        else:
            dst_f = osp.splitext(src_f)[0]

        json.dump(data, open(dst_f + f'_{tp}.json', 'w'), indent=2)

if __name__ == "__main__":
    main()
