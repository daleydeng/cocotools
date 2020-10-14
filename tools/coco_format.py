#!/usr/bin/env python
import os
import os.path as osp
from copy import deepcopy
import json
import click

@click.command()
@click.option('--out', '-o', default='')
@click.argument('src')
def main(out, src):
    if not out:
        out = src

    data = json.load(open(src))

    valid_img_ids = set(i['image_id'] for i in data['annotations'])
    for i in data['images']:
        i['file_name'] = osp.basename(i['file_name'])

    out_data = deepcopy(data)
    out_data['images'] = []
    for i in data['images']:
        if i['id'] in valid_img_ids:
            out_data['images'].append(i)

    img_dic = {i['id']: i for i in data['images']}
    print ("empty images", [img_dic[i]['file_name'] for i in (img_dic.keys() - valid_img_ids)])
    json.dump(out_data, open(out, 'w'), indent=2, sort_keys=True)

if __name__ == "__main__":
    main()
