#!/usr/bin/env python
from collections import defaultdict
from pycocotools.coco import COCO
import json
from pprint import pprint
import click

@click.command()
@click.argument('ann_file')
def main(ann_file):
    coco = COCO(ann_file)

    print ("info")
    print (coco.info())

    print ("categories:")
    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)
    pprint (cats)

    print ("annotations by category:")
    cat_anns = defaultdict(list)
    for k, v in coco.anns.items():
        cat_id = v['category_id']
        cat_anns[coco.cats[cat_id]['name']].append(v)
    stats = {k: len(v) for k, v in cat_anns.items()}
    pprint (stats)

    print ("images by category:")
    stats = {k: len(set(j['image_id'] for j in v))
             for k, v in cat_anns.items()}
    pprint (stats)

if __name__ == "__main__":
    main()
