#!/usr/bin/env python
import os
import os.path as osp
import json
import shutil
import click

@click.command()
@click.argument('ann_file')
@click.argument('img_dir')
@click.argument('out_dir')
def main(ann_file, img_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    data = json.load(open(ann_file))
    for i in data['images']:
        dst_f = osp.join(out_dir, i['file_name'])
        if osp.exists(dst_f):
            continue

        src_f = osp.join(img_dir, i['file_name'])
        shutil.copy2(src_f, dst_f)

if __name__ == "__main__":
    main()
