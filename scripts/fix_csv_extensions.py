"""Make the 'file' column of test_labels.csv match the real file names on disk.

Usage (from repo root):  python scripts/fix_csv_extensions.py
Matches by number/name without extension: a CSV row '16.jpg' is changed to '16.png'
if the only file called 16.* in data/raw/test is 16.png. Saves a backup first.
Does not touch the images.
"""
import argparse
import csv
import os
import shutil
from collections import defaultdict

IMG_EXT = (".jpg", ".jpeg", ".png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join("data", "raw", "test_labels.csv"))
    ap.add_argument("--dir", default=os.path.join("data", "raw", "test"))
    a = ap.parse_args()

    by_stem = defaultdict(list)
    for f in os.listdir(a.dir):
        if f.lower().endswith(IMG_EXT):
            by_stem[os.path.splitext(f)[0].lower()].append(f)

    with open(a.csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)

    changed, problems = 0, []
    for r in rows:
        name = r["file"].strip()
        stem = os.path.splitext(name)[0].lower()
        matches = by_stem.get(stem, [])
        if name in matches:
            continue
        if len(matches) == 1:
            print(f"  {name}  ->  {matches[0]}")
            r["file"] = matches[0]
            changed += 1
        else:
            problems.append((name, matches))

    if changed:
        shutil.copy(a.csv, a.csv + ".bak")
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print(f"\nFixed {changed} file name(s)." + (f" Backup saved as {a.csv}.bak" if changed else ""))
    for name, m in problems:
        print(f"  CANNOT FIX {name}: matching files on disk = {m or 'none'}")


if __name__ == "__main__":
    main()