#!/usr/bin/env python
import numpy as np
import os
import os.path as osp
import cv2
from pycocotools.coco import COCO
import pycocotools.mask as coco_mask
from tqdm import tqdm
import click

def segm_to_rle(segm, img_shape):
    h, w = img_shape
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = coco_mask.frPyObjects(segm, h, w)
        return coco_mask.merge(rles)

    if type(segm['counts']) == list:
        # uncompressed RLE
        return coco_mask.frPyObjects(segm, h, w)

    return segm

def get_ann_mask(ann, img_shape):
    rle = segm_to_rle(ann['segmentation'], img_shape)
    m = coco_mask.decode(rle)
    return m

@click.command()
@click.option('-f', '--force', is_flag=True)
@click.argument('ann_file')
@click.argument('out_dir')
def main(force, ann_file, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    coco = COCO(ann_file)

    for img_id, v in tqdm(coco.imgs.items()):
        dst_f = osp.join(out_dir, osp.splitext(v['file_name'])[0]+'.png')
        if not force and osp.exists(dst_f):
            continue

        img_shape = (v['height'], v['width'])
        segmap = np.zeros(img_shape, dtype='u2')
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            cat_id = ann['category_id']
            mask = get_ann_mask(ann, img_shape)
            mask = mask > 0.5
            segmap[mask] = cat_id

        cv2.imwrite(dst_f, segmap)

if __name__ == "__main__":
    main()
