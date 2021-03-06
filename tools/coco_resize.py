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

def clamp(v, max_v, min_v=0):
    return int(max(min(v, max_v), min_v))

def rescale_coords(coords, scale, img_size, stride=2):
    imw, imh = img_size
    sx, sy = scale
    out = []
    for i, x in enumerate(coords):
        if i % stride == 0:
            x = clamp(sx * x, imw-1)
        elif i % stride == 1:
            x = clamp(sy * x, imh-1)

        out.append(x)
    return out

def rescale_bbox(bbox, scale, img_size):
    sx, sy = scale
    imw, imh = img_size
    x, y, w, h = bbox
    x, y, w, h = x * sx, y * sy, w * sx, h * sy

    x0, x1 = clamp(x, imw-1), clamp(x + w, imw-1)
    y0, y1 = clamp(y, imh-1), clamp(y + h, imh-1)
    return [x0, y0, x1 - x0, y1 - y0]

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
    scales = {k: (width / max(v['width'], v['height']),
                  width / max(v['width'], v['height']))
              for k, v in img_dic_by_id.items()}

    for img in data['images']:
        sx, sy = scales[img['id']]
        img['width'] = int(img['width'] * sx)
        img['height'] = int(img['height'] * sy)

    img_sizes = {i['id']: (i['width'], i['height']) for i in data['images']}

    for ann in data['annotations']:
        img_id = ann['image_id']
        scale = scales[img_id]
        img_size = img_sizes[img_id]

        ann['bbox'] = rescale_bbox(ann['bbox'], scale, img_size)
        ann['segmentation'] = [rescale_coords(i, scale, img_size)
                               for i in ann['segmentation']]
        ann['area'] = sum(calc_polygon_area(i) for i in ann['segmentation'])

        if 'keypoints' in ann:
            ann['keypoints'] = rescale_coords(ann['keypoints'], scale, img_size, stride=3)

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
