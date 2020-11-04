import os.path as osp
import itertools
from PIL import Image
from shapely.geometry import Polygon

def clamp(v, max_v, min_v=0):
    return max(min(v, max_v), min_v)

def clamp(v, max_v, min_v=0):
    return max(min(v, max_v), min_v)

def clamp_coords(coords, img_size, stride=2):
    if img_size is None:
        return coords

    imw, imh = img_size
    out = []
    for i, x in enumerate(coords):
        if i % stride == 0:
            x = clamp(x, imw-1)
        elif i % stride == 1:
            x = clamp(x, imh-1)

        out.append(x)
    return out

def flatten(xs):
    out = []
    for i in xs:
        out += i
    return out

def make_coco_img(id, filename, img_size=None, **kws):
    if img_size is None:
        width, height = Image.open(filename).size
    else:
        width, height = img_size

    return {
        "id": id,
        "file_name": osp.basename(filename),
        "height": height,
        "width": width,

        'coco_url': '',
        "date_captured": 0,

        "flickr_url": '',

        'license': 0,

        **kws
    }

def _fill_segm(out, segms, img_size=None):
    if not isinstance(segms[0], list):
        segms = [segms]

    flat_segms = [itertools.chain(*i) for i in segms]
    flat_segms = [clamp_coords(i, img_size) for i in flat_segms]

    out['segmentation'] = flat_segms

    polys = [Polygon(list(zip(i[::2], i[1::2]))) for i in flat_segms]
    area = sum(i.area for i in polys)
    out['area'] = area

    if 'bbox' not in out:
        pts = list(itertools.chain(*flat_segms))
        xs = pts[::2]
        ys = pts[1::2]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        out['bbox'] = [x0, y0, x1 - x0, y1 - y0]

def make_coco_ann(id, img_id, cat_id, bbox=None, segms=None, kps=None, iscrowd=False, img_size=None):
    out = {}

    if bbox:
        x0, y0, x1, y1 = clamp_coords(bbox, img_size)
        out['bbox'] = [x0, y0, x1 - x0, y1 - y0]

    if segms is not None:
        _fill_segm(out, segms, img_size=img_size)

    if kps:
        kps = clamp_coords(kps, img_size=img_size, stride=3)
        out.update({
            'keypoints': kps,
            'num_keypoints': len(kps) // 3,
        })

        if 'segmentation' not in out:
            xs = list(zip(kps[::3], kps[1::3]))
            _fill_segm(out, xs, img_size=img_size)

    return {
        'id': id,
        'image_id': img_id,
        'category_id': cat_id,
        'iscrowd': iscrowd,

        **out,
    }

def make_coco(imgs, anns, class_ids, kp_rbbox=False, info={}, licenses=[]):
    cat_kws = {}
    if kp_rbbox:
        cat_kws.update({
            'keypoints': ['tl', 'tr', 'br', 'bl'],
            'skeleton': [[1,2], [2,3], [3,4], [4,1]],
        })

    return {
        'info': {
            "contributor": "",
            "date_created": "",
            "description": "",
            "url": "",
            "version": "",
            "year": "",

            **info,
        },

        'licenses': [
            {
                "id": 0,
                "name": "",
                "url": "",
            }
        ] + licenses,

        "categories": [
            {
                "id": cid,
                "name": cls,
                "supercategory": "",
                **cat_kws,
            } for cls, cid in class_ids.items()
        ],

        'images': imgs,
        'annotations': anns,
    }
