#!/usr/bin/env python
import os
import os.path as osp
from glob import glob
from PIL import Image
import json
import click

def file_name(f):
    return osp.splitext(osp.basename(f))[0]

@click.command()
@click.argument('src_d')
@click.argument('dst_f')
def main(src_d, dst_f):
    out = {}
    for i in sorted(os.listdir(src_d)):
        img_f = osp.join(src_d, i)
        img = Image.open(img_f)
        imw, imh = img.size

        name = file_name(img_f)

        out[name] = {
            "filename": osp.basename(img_f),
            'height': imh,
            'width': imw,
        }

    json.dump(out, open(dst_f, 'w'), indent=2, sort_keys=True)

if __name__ == "__main__":
    main()
