#!/usr/bin/env python
import os
import os.path as osp
import numpy as np
import numpy.random as npr
import multiprocessing as mp
from functools import partial
from pycocotools.coco import COCO
import pycocotools.mask as coco_mask
from skimage.io import imread, imsave
from skimage.morphology import binary_erosion, disk
from easydict import EasyDict as edict
import cv2
from pprint import pprint
from matplotlib import colors
import pylab as plt
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

def draw_color_bar(class_dic, color_map, shape, block_shape=(20, 50), **kws):
    ph, pw = block_shape
    canvas = np.ones((*shape, 3), dtype='u1') * 255

    off = 0
    for cat_id, cat_name in class_dic.items():
        color = color_map[cat_id]
        x0, y0 = 0, off
        x1, y1 = pw, off + ph
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, -1)
        cv2.putText(canvas, cat_name, (x1 + 10, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        off += ph

    return canvas

def process_one(d, img_dir, out_dir, seg_dir, color_map, class_dic, mask_thr=0.5, cbar_width=300, **cfg):
    cfg = edict(cfg)

    img, anns = d
    if not anns:
        return

    fname = img['file_name']
    src_f = osp.join(img_dir, fname)
    dst_f = osp.join(out_dir, fname)
    if osp.exists(dst_f):
        return

    I = imread(src_f)
    img0 = I.copy()

    for ann in anns:
        color = color_map[ann['category_id']]
        face_mask = get_ann_mask(ann, (img['height'], img['width']))
        face_mask = face_mask > mask_thr
        mask = np.logical_xor(face_mask, binary_erosion(face_mask, disk(cfg\
.inst_border_size)))

        I[mask, :] = color

    if not cfg.no_bbox:
        for ann in anns:
            x0, y0, w, h = map(int, ann['bbox'])
            x1, y1 = x0 + w, y0 + h

            cv2.rectangle(I, (x0, y0), (x1, y1), cfg.bbox_color, thickness=cfg.thickness)

            label_text = class_dic[ann['category_id']]
            cv2.putText(I, label_text, (x0, y0-2), cv2.FONT_HERSHEY_COMPLEX, cfg.font_scale, cfg.text_color)

    semseg_f = osp.join(seg_dir, osp.splitext(fname)[0]+'.png')
    if osp.exists(semseg_f):
        semseg = imread(semseg_f)
        for cat_id, cat_name in class_dic.items():
            mask = semseg == cat_id
            color = np.asarray(color_map[cat_id])
            I[mask, :] = (1 - cfg.blend_alpha) * I[mask, :] + cfg.blend_alpha * color
    else:
        cv2.putText(I, "NO SEMSEG!", (0, 10), cv2.FONT_HERSHEY_COMPLEX, cfg.warn_font_scale, cfg.warn_text_color)

    color_bar = draw_color_bar(class_dic, color_map, (img0.shape[0], cbar_width))

    canvas = np.hstack((img0, I, color_bar))
    if cfg.show:
        plt.title(fname)
        plt.imshow(canvas)
        plt.axis('off')
        plt.show()

    imsave(dst_f, canvas)

def get_color(v):
    r, g, b = colors.to_rgb(v)[:3]
    return int(r * 255), int(g * 255), int(b * 255)

def get_cmap_color(v, cmap):
    r, g, b = cmap(v)[:3]
    return int(r * 255), int(g * 255), int(b * 255)

@click.command()
@click.option('--no_bbox', is_flag=True)
@click.option('--thickness', default=1)
@click.option('--bbox_color', default='g')
@click.option('--text_color', default='g')
@click.option('--warn_text_color', default='r')
@click.option('--font_scale', default=0.5)
@click.option('--warn_font_scale', default=0.5)
@click.option('--cmap', default='hsv')
@click.option('--ignore_color', default='black')
@click.option('--inst_border_size', default=3)
@click.option('--blend_alpha', default=0.5)
@click.option('--show', is_flag=True)
@click.option('--jobs', default=-1)
@click.option('--semseg', default='')
@click.argument('ann_file')
@click.argument('img_dir')
@click.argument('out_dir')
def main(no_bbox, thickness, bbox_color, text_color, warn_text_color, font_scale, warn_font_scale, cmap, ignore_color, inst_border_size, blend_alpha, show, jobs, semseg, ann_file, img_dir, out_dir):
    if show:
        jobs = 1

    if jobs < 0:
        jobs = mp.cpu_count()

    os.makedirs(out_dir, exist_ok=True)

    coco = COCO(ann_file)
    cats = coco.cats
    print ("COCO categories:")
    pprint (cats)

    ignore_ids = [k for k, v in cats.items() if 'ignore' in v['name']]
    class_dic = {k: v['name'] for k, v in cats.items()}

    cmap = plt.get_cmap(cmap)
    color_map = {
        i: get_cmap_color(idx / len(class_dic), cmap)
        for idx, i in enumerate(class_dic)
    }

    for i in ignore_ids:
        color_map[i] = get_color(ignore_color)

    img_ids = coco.getImgIds()
    imgs = coco.loadImgs(img_ids)

    process_worker = partial(
        process_one,
        img_dir=img_dir,
        out_dir=out_dir,
        seg_dir=semseg,
        color_map=color_map,
        class_dic=class_dic,

        show=show,
        no_bbox=no_bbox,
        thickness=thickness,
        bbox_color=get_color(bbox_color),
        text_color=get_color(text_color),
        warn_text_color=get_color(warn_text_color),
        font_scale=font_scale,
        warn_font_scale=warn_font_scale,
        inst_border_size=inst_border_size,
        blend_alpha=blend_alpha,
    )

    works = [(i, coco.loadAnns(coco.getAnnIds(imgIds=[i['id']]))) for i in imgs]
    with mp.Pool(jobs) as p:
        with tqdm(total=len(imgs)) as pbar:
            for i, _ in enumerate(p.imap_unordered(process_worker, works)):
                pbar.update()

if __name__ == "__main__":
    main()
