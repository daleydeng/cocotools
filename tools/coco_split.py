#!/usr/bin/env python
import os
import os.path as osp
import json
from tqdm import tqdm
import shutil
import click

@click.command()
@click.option('--start', '-s', type=float, default=0)
@click.option('--end', '-e', type=float, default=0)
@click.argument('src')
@click.argument('dst')
def main(start, end, src, dst):
    if osp.exists(dst):
        print (f"{dst} exists!")
        return

    print (f"loading {src}")
    data = json.load(open(src))
    print (f"loaded {src}")

    out = {i: data[i] for i in ['info', 'licenses', 'categories']}

    imgs = data['images']
    anns = data['annotations']

    if end > 0:
        if end < 1 or 0 < start < 1: # ratio
            end = end * len(imgs)
            start = start * len(imgs)

        start = int(start)
        end = int(end)
        imgs = imgs[start:end]
        out['images'] = imgs

    img_ids = set(i['id'] for i in imgs)
    out['annotations'] = [i for i in anns if i['image_id'] in img_ids]
    json.dump(out, open(dst, 'w'), indent=2, sort_keys=True)

if __name__ == "__main__":
    main()
