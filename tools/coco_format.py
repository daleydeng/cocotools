#!/usr/bin/env python
import os
import os.path as osp
from copy import deepcopy
import json
import click

def clamp(v, max_v, min_v=0):
    return max(min(v, max_v), min_v)

def clamp_coords(coords, img_size, stride=2):
    imw, imh = img_size
    out = []
    for i, x in enumerate(coords):
        if i % stride == 0:
            x = clamp(x, imw-1)
        elif i % stride == 1:
            x = clamp(x, imh-1)

        out.append(x)
    return out

def clamp_bbox(bbox, img_size):
    imw, imh = img_size
    x, y, w, h = bbox

    x0, x1 = clamp(x, imw-1), clamp(x + w, imw-1)
    y0, y1 = clamp(y, imh-1), clamp(y + h, imh-1)
    return [x0, y0, x1 - x0, y1 - y0]

@click.command()
@click.option('--out', '-o', default='')
@click.option('--ignore_id', default=128)
@click.argument('ann_f')
@click.argument('img_dir')
def main(out, ignore_id, ann_f, img_dir):
    if not out:
        out = ann_f

    data = json.load(open(ann_f))
    cats = {i['id']: i for i in data['categories']}

    assert all(i > 0 for i in cats)
    ignore_ids = [k for k, v in cats.items() if 'ignore' in v['name']]
    assert len(ignore_ids) <= 1

    ignore_id_map = {i: 128 for i in ignore_ids}

    valid_images = []
    for i in data['images']:
        i = deepcopy(i)
        i['file_name'] = osp.basename(i['file_name'])
        if osp.exists(osp.join(img_dir, i['file_name'])):
            valid_images.append(i)

    data['images'] = valid_images
    img_sizes = {i['id']: (i['width'], i['height']) for i in data['images']}

    valid_anns = []
    for i in data['annotations']:
        if i['image_id'] in img_sizes:
            valid_anns.append(i)
    data['annotations'] = valid_anns

    valid_img_ids = set(i['image_id'] for i in data['annotations'])
    for i in data['images']:
        i['file_name'] = osp.basename(i['file_name'])

    for ann in data['annotations']:
        if ann['category_id'] in ignore_ids:
            ann['iscrowd'] = 1
            ann['category_id'] = ignore_id_map[ann['category_id']]

        img_size = img_sizes[ann['image_id']]
        clamp_bbox(ann['bbox'], img_size)
        for i in ann['segmentation']:
            clamp_coords(i, img_size)
        if 'keypoints' in ann:
            clamp_coords(ann['keypoints'], img_size, stride=3)

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
    print ("image_ids - annotation_image_ids", [img_dic[i]['file_name'] for i in (img_dic.keys() - valid_img_ids)])
    os.makedirs(osp.dirname(out), exist_ok=True)
    json.dump(out_data, open(out, 'w'), indent=2, sort_keys=True)

if __name__ == "__main__":
    main()
