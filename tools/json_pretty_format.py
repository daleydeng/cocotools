#!/usr/bin/env python
import json
import click

@click.command()
@click.option('--out', '-o', default='')
@click.argument('src')
def main(out, src):
    if not out:
        out = src

    data = json.load(open(src))
    json.dump(data, open(out, 'w'), indent=2, sort_keys=True)

if __name__ == "__main__":
    main()
