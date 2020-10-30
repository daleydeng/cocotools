#!/usr/bin/env python
import os
import os.path as osp
import click
import multiprocessing as mp
from tqdm import tqdm

def file_name(f):
    return osp.splitext(osp.basename(f))[0]

cmd_tpl = 'convert {} {}'

def convert_one(d):
    src_f, dst_f = d
    cmd = cmd_tpl.format(src_f, dst_f)
    os.system(cmd)

@click.command()
@click.option('--jobs', default=-1)
@click.option('--ext', default='.jpg')
@click.argument('src_d')
@click.argument('dst_d')
def main(jobs, ext, src_d, dst_d):
    if jobs < 0:
        jobs = mp.cpu_count()

    os.makedirs(dst_d, exist_ok=True)

    tasks = []
    for i in tqdm(sorted(os.listdir(src_d))):
        src_f = osp.join(src_d, i)
        dst_f = osp.join(dst_d, file_name(i) + ext)
        if osp.exists(dst_f):
            continue

        tasks.append((src_f, dst_f))

    with mp.Pool(jobs) as p:
        with tqdm(total=len(tasks)) as pbar:
            for i, _ in enumerate(p.imap_unordered(convert_one, tasks)):
                pbar.update()

if __name__ == "__main__":
    main()
