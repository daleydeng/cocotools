#!/usr/bin/env python
import os
import os.path as osp
import numpy as np
import numpy.random as npr
from pycocotools.coco import COCO
from skimage.io import imread, imsave
import cv2
from pprint import pprint
import pylab as plt
import click

COLOR_TAB = {
    'green': (0, 255, 0),
}

@click.command()
@click.option('--no_bbox', is_flag=True)
@click.option('--thickness', default=1)
@click.option('--bbox_color', default='green')
@click.option('--text_color', default='green')
@click.option('--font_scale', default=0.5)
@click.option('--show', is_flag=True)
@click.argument('ann_file')
@click.argument('img_dir')
@click.argument('out_dir')
def main(no_bbox, thickness, bbox_color, text_color, font_scale, show, ann_file, img_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    coco = COCO(ann_file)

    cats = coco.loadCats(coco.getCatIds())
    print ("COCO categories:")
    pprint (cats)

    class_dic = {i['id']: i['name'] for i in cats}

    color_masks = {
        i: npr.randint(0, 256, (1, 3), dtype=np.uint8)
        for i in class_dic
    }

    cat_ids = coco.getCatIds()
    img_ids = coco.getImgIds(catIds=cat_ids)

    imgs = coco.loadImgs(img_ids)

    for i in imgs:
        fname = i['file_name']
        src_f = osp.join(img_dir, fname)
        dst_f = osp.join(out_dir, fname)

        I = imread(src_f)
        img0 = I.copy()

        ann_ids = coco.getAnnIds(imgIds=i['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            ann_mask = coco.annToMask(ann)
            ann_color_img = np.dstack([ann_mask]*3) * color_masks[ann['category_id']]
            ann_mask = ann_mask > 0.5

            I[ann_mask, :] = 0.5 * I[ann_mask, :] + 0.5 * ann_color_img[ann_mask, :]

        if not no_bbox:
            for ann in anns:
                x0, y0, w, h = map(int, ann['bbox'])
                x1, y1 = x0 + w, y0 + h

                cv2.rectangle(I, (x0, y0), (x1, y1), COLOR_TAB[bbox_color], thickness=thickness)

                label_text = class_dic[ann['category_id']]
                cv2.putText(I, label_text, (x0, y0-2), cv2.FONT_HERSHEY_COMPLEX, font_scale, COLOR_TAB[text_color])

        canvas = np.hstack((img0, I))
        if show:
            plt.title(fname)
            plt.imshow(canvas)
            plt.axis('off')
            plt.show()

        imsave(dst_f, canvas)

if __name__ == "__main__":
    main()
