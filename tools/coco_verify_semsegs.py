#!/usr/bin/env python
import os
import os.path as osp
import numpy as np
import multiprocessing as mp
from functools import partial
from pycocotools.coco import COCO
from skimage.io import imread, imsave
from easydict import EasyDict as edict
import cv2
from pprint import pprint
from matplotlib.colors import to_rgb
import pylab as plt
from tqdm import tqdm
import click

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

def process_one(img, img_dir, seg_dir, out_dir, color_map, class_dic, **cfg):
    cfg = edict(cfg)

    fname = img['file_name']
    src_f = osp.join(img_dir, fname)
    dst_f = osp.join(out_dir, fname)
    if osp.exists(dst_f):
        return

    I = imread(src_f)
    img0 = I.copy()

    semseg_f = osp.join(seg_dir, osp.splitext(fname)[0]+'.png')
    if osp.exists(semseg_f):
        semseg = imread(semseg_f)
        for cat_id, cat_name in class_dic.items():
            mask = semseg == cat_id
            color = np.asarray(color_map[cat_id])
            I[mask, :] = (1 - cfg.blend_alpha) * I[mask, :] + cfg.blend_alpha * color
    else:
        cv2.putText(I, "NO SEMSEG!", (0, 10), cv2.FONT_HERSHEY_COMPLEX, cfg.warn_font_scale, cfg.warn_text_color)

    color_bar = draw_color_bar(class_dic, color_map, (img0.shape[0], 300))

    canvas = np.hstack((img0, I, color_bar))
    if cfg.show:
        plt.title(fname)
        plt.imshow(canvas)
        plt.axis('off')
        plt.show()

    imsave(dst_f, canvas)

def get_color(v):
    return tuple(map(int, np.array(to_rgb(v)[:3]) * 255))

def get_cmap_color(v, cmap):
    return tuple(map(int, np.array(cmap(v)[:3]) * 255))

@click.command()
@click.option('--text_color', default='g')
@click.option('--font_scale', default=0.5)
@click.option('--warn_text_color', default='r')
@click.option('--warn_font_scale', default=1)
@click.option('--cmap', default='hsv')
@click.option('--ignore_color', default='black')
@click.option('--blend_alpha', default=0.5)
@click.option('--show', is_flag=True)
@click.option('--jobs', default=-1)
@click.argument('ann_file')
@click.argument('img_dir')
@click.argument('seg_dir')
@click.argument('out_dir')
def main(text_color, font_scale, warn_text_color, warn_font_scale, cmap, ignore_color, blend_alpha, show, jobs, ann_file, img_dir, seg_dir, out_dir):
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

    imgs = coco.imgs.values()

    process_worker = partial(
        process_one,
        img_dir=img_dir,
        seg_dir=seg_dir,
        out_dir=out_dir,
        color_map=color_map,
        class_dic=class_dic,

        show=show,
        text_color=get_color(text_color),
        font_scale=font_scale,
        warn_text_color=get_color(warn_text_color),
        warn_font_scale=warn_font_scale,
        blend_alpha=blend_alpha,
    )

    with mp.Pool(jobs) as p:
        with tqdm(total=len(imgs)) as pbar:
            for i, _ in enumerate(p.imap_unordered(process_worker, imgs)):
                pbar.update()

if __name__ == "__main__":
    main()
