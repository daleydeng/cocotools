#!/usr/bin/env python
import os
import os.path as osp
from glob import glob
import json
from shapely.geometry import Polygon
from tqdm import tqdm
import click

def file_name(f):
    return osp.splitext(osp.basename(f))[0]

CLASSES = [
    'plane',
    'ship',
    'storage-tank',
    'baseball-diamond',
    'tennis-court',
    'basketball-court',
    'ground-track-field',
    'harbor',
    'bridge',
    'small-vehicle',
    'large-vehicle',
    'helicopter',
    'roundabout',
    'soccer-ball-field',
    'swimming-pool',
    'container-crane',
]

def id_generator():
    i = 1
    while True:
        yield i
        i += 1

id_gen = id_generator()

def clamp(v, max_v, min_v=0):
    return max(min(v, max_v), min_v)

def flatten(xs):
    out = []
    for i in xs:
        out += i
    return out

def parse_annotation(fname, class_ids, imgs_info):
    img_info = imgs_info[file_name(fname)]
    fp = open(fname)
    fp.readline()
    fp.readline()

    out = []

    for l in fp:
        if not l.strip():
            continue

        (*flatten_xs, cls_name, dif) = l.split()
        flatten_xs = [float(i) for i in flatten_xs]
        xs, ys = flatten_xs[::2], flatten_xs[1::2]

        imw, imh = img_info['width'], img_info['height']
        xs = [clamp(i, imw) for i in xs]
        ys = [clamp(i, imh) for i in ys]

        p = Polygon(list(zip(xs, ys)))

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        ann_id = next(id_gen)

        ann = {
            'id': ann_id,
            'image_id': img_info['id'],
            'category_id': class_ids[cls_name],
            'segmentation': [flatten_xs],
            'area': p.area,
            'bbox': [x_min, y_min, x_max - x_min, y_max - y_min],

            'num_keypoints': 4,
            'keypoints': flatten([[x, y, 2] for x, y in zip(xs, ys)]),

            'iscrowd': dif,
        }

        out.append(ann)

    return out

@click.command()
@click.option('--hbb', is_flag=True)
@click.argument('src')
@click.argument('img_sizes')
@click.argument('dst')
def main(hbb, src, img_sizes, dst):
    if osp.exists(dst):
        return

    imgs_info = json.load(open(img_sizes))

    for idx, v in enumerate(imgs_info.values()):
        v['id'] = idx + 1

    class_ids = {
        cls: idx + 1
        for idx, cls in enumerate(CLASSES)
    }

    anns = []
    for name in tqdm(sorted(os.listdir(src))):
        ann_f = osp.join(src, name)
        anns += parse_annotation(ann_f, class_ids, imgs_info)

    out = {
        'info': {
            "contributor": "",
            "date_created": "",
            "description": "",
            "url": "",
            "version": "",
            "year": "",
        },
        'licenses': [
            {
                "id": 0,
                "name": "",
                "url": "",
            }
        ],

        "categories": [
            {
                "id": idx + 1,
                "name": cls,
                "supercategory": "",
                'keypoints': ['tl', 'tr', 'br', 'bl'],
                'skeleton': [[1,2], [2,3], [3,4], [4,1]],
            } for idx, cls in enumerate(CLASSES)
        ],

        'images': [
            {
                'coco_url': "",
                "date_captured": 0,
                "file_name": v['filename'],
                "flickr_url": "",
                "height": v['height'],
                "id": v['id'],
                'license': 0,
                "width": v['width'],

            } for idx, v in enumerate(imgs_info.values())
        ],
        "annotations": anns,
    }

    os.makedirs(osp.dirname(dst), exist_ok=True)
    json.dump(out, open(dst, 'w'), indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
