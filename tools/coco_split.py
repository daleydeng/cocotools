#!/usr/bin/env python
import os
import os.path as osp
import json
from tqdm import tqdm
import shutil
import click

@click.command()
@click.option('--start', '-s', default=0)
@click.option('--end', '-e', default=1000)
@click.option('--no_img', is_flag=True)
@click.option('--src_img_dir', default='')
@click.option('--dst_img_dir', default='')
@click.argument('src')
@click.argument('dst')
def main(start, end, no_img, src_img_dir, dst_img_dir, src, dst):
    print (f"loading {src}")
    coco = json.load(open(src))
    print (f"loaded {src}")

    out = {i: coco[i] for i in ['info', 'licenses', 'categories']}

    imgs = coco['images']
    anns = coco['annotations']

    if end == -1:
        end = len(imgs)

    imgs = imgs[start:end]
    img_ids = set(i['id'] for i in imgs)

    anns = [i for i in anns if i['image_id'] in img_ids]

    out.update({
        'images': imgs,
        'annotations': anns,
    })

    base_d = osp.join(osp.dirname(src), '../')
    src_name = osp.basename(src)

    set_name = osp.splitext(src_name)[0].split('_')[-1]
    if not src_img_dir:
        src_img_dir = osp.join(base_d, set_name)

    if not dst_img_dir:
        dst_img_dir = osp.join(dst, set_name)

    dst_json = osp.join(dst, 'annotations', osp.basename(src))
    os.makedirs(osp.dirname(dst_json), exist_ok=True)
    json.dump(out, open(dst_json, 'w'), indent=2, sort_keys=True)

    if not no_img:
        os.makedirs(dst_img_dir, exist_ok=True)
        print ("copying image")
        for img in tqdm(imgs):
            shutil.copy2(osp.join(src_img_dir, img['file_name']), osp.join(dst_img_dir))

if __name__ == "__main__":
    main()
