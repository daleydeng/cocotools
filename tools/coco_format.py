#!/usr/bin/env python
import os
import os.path as osp
from copy import deepcopy
import json
import click

@click.command()
@click.option('--out', '-o', default='')
@click.option('--ignore_id', default=128)
@click.argument('src')
def main(out, ignore_id, src):
    if not out:
        out = src

    data = json.load(open(src))
    cats = {i['id']: i for i in data['categories']}

    assert all(i > 0 for i in cats)
    ignore_ids = [k for k, v in cats.items() if 'ignore' in v['name']]
    assert len(ignore_ids) <= 1

    ignore_id_map = {i: 128 for i in ignore_ids}

    valid_img_ids = set(i['image_id'] for i in data['annotations'])
    for i in data['images']:
        i['file_name'] = osp.basename(i['file_name'])

    for i in data['annotations']:
        if i['category_id'] in ignore_ids:
            i['iscrowd'] = 1
            i['category_id'] = ignore_id_map[i['category_id']]

    for i in data['categories']:
        if i['id'] in ignore_ids:
            i['id'] = ignore_id_map[i['id']]

    data['categories'] = sorted(data['categories'], key=lambda x: x['id'])
    data['images'] = sorted(data['images'], key=lambda x: x['id'])
    data['annotations'] = sorted(data['annotations'], key=lambda x: x['id'])

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
