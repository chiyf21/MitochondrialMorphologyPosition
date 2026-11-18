"""Stage only the 112 neuron masks near FISH2 study instances, without EM."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import re
import sys
import tarfile

SPACING = (8.,8.,30.)
TILE = (128,128,32)
RADIUS_NM = 1500.
NAME = re.compile(r"x(-?\d+)_y(-?\d+)_z(-?\d+)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest",type=Path)
    parser.add_argument("root",type=Path)
    args = parser.parse_args()
    bodies = defaultdict(list)
    with args.manifest.open() as file:
        for row in csv.DictReader(file):
            bodies[row["body_id"]].append(tuple(float(row[f"centroid_{axis}_voxel"]) for axis in "xyz"))
    with tarfile.open(fileobj=sys.stdout.buffer,mode="w|") as archive:
        for body,coords in sorted(bodies.items()):
            folder = args.root/body/"blocks"
            total, size = 0,0
            for block in folder.glob("x*_y*_z*"):
                hit = NAME.fullmatch(block.name)
                if not hit:
                    continue
                xyz = tuple(map(int,hit.groups()))
                if not any(all(b-RADIUS_NM/v-2 <= c < b+t+RADIUS_NM/v+2
                               for b,c,t,v in zip(xyz,p,TILE,SPACING)) for p in coords):
                    continue
                src = next((p for p in (block/"neuron_mask.npy",block/"neuron_mask.npz") if p.is_file()),None)
                if src:
                    archive.add(src,arcname=src.relative_to(args.root),recursive=False)
                    total += 1
                    size += src.stat().st_size
            print(body,total,size,file=sys.stderr,flush=True)


if __name__ == "__main__":
    main()
