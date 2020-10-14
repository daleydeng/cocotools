#!/usr/bin/env python
import os
import os.path as osp
from copy import deepcopy
import cv2
from functools import partial
from shapely.geometry import Polygon
import multiprocessing as mp
import json
from tqdm import tqdm
import click

def rescale_coords(coords, scale):
    sx, sy = scale
    out = []
    for i, x in enumerate(coords):
        if i % 2 == 0:
            s = sx
        else:
            s = sy

        out.append(x * s)

    return out

def rescale_keypoints(xs, scale):
    sx, sy = scale
    out = []
    for i, x in enumerate(xs):
        if i % 3 == 2:
            out.append(x)
            continue

        if i % 3 == 0:
            s = sx
        elif i % 3 == 1:
            s = sy

        out.append(x * s)

    return out

def rescale_bbox(bbox, scale):
    return [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]

def calc_polygon_area(p):
    return Polygon(list(zip(p[::2], p[1::2]))).area

def resize_img(d, src_d, dst_d):
    name, scale = d
    dst_f = osp.join(dst_d, name)
    if osp.exists(dst_f):
        return

    img = cv2.imread(osp.join(src_d, name))
    sx, sy = scale
    h, w = img.shape[:2]
    w1, h1 = int(sx * w), int(sy * h)
    img = cv2.resize(img, (w1, h1))
    cv2.imwrite(dst_f, img)

@click.command()
@click.option('--width', '-w', default=768)
@click.option('--jobs', default=-1)
@click.argument('ann_file')
@click.argument('img_dir')
@click.argument('out_dir')
def main(width, jobs, ann_file, img_dir, out_dir):
    if jobs < 0:
        jobs = mp.cpu_count()

    data = json.load(open(ann_file))

    img_dic_by_id = {i['id']: i for i in data['images']}
    img_dic_by_name = {i['file_name']: i for i in data['images']}
    scales = {k: (width / v['width'], width / v['width'])
              for k, v in img_dic_by_id.items()}

    for img in data['images']:
        sx, sy = scales[img['id']]
        img['width'] = int(img['width'] * sx)
        img['height'] = int(img['height'] * sy)

    for ann in data['annotations']:
        img_id = ann['image_id']
        scale = scales[img_id]

        ann['bbox'] = rescale_coords(ann['bbox'], scale)
        ann['segmentation'] = [rescale_coords(i, scale)
                               for i in ann['segmentation']]
        ann['area'] = sum(calc_polygon_area(i) for i in ann['segmentation'])

        if 'keypoints' in ann:
            ann['keypoints'] = rescale_keypoints(ann['keypoints'], scale)

    dst_json = osp.join(out_dir, 'annotations', osp.basename(ann_file))
    os.makedirs(osp.dirname(dst_json), exist_ok=True)
    json.dump(data, open(dst_json, 'w'), indent=2, sort_keys=True)

    out_img_dir = osp.join(out_dir, 'images')
    os.makedirs(out_img_dir, exist_ok=True)

    resize_worker = partial(resize_img, src_d=img_dir, dst_d=out_img_dir)

    with mp.Pool(jobs) as p:
        with tqdm(total=len(data['images'])) as pbar:
            for i, _ in enumerate(p.imap_unordered(resize_worker, [(i['file_name'], scales[i['id']]) for i in data['images']])):
                pbar.update()

if __name__ == "__main__":
    main()
