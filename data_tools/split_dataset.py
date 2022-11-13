import os
import json
import argparse
from glob import glob
import pandas as pd
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('dir', metavar='dir', type=str,
                    help='enter directory containing data')
# parser.add_argument('output', metavar='dir', type=str,
#                     help='enter directory containing data')

args = parser.parse_args()

os.makedirs(os.path.join(os.getcwd(), args.dir, 'val'), exist_ok=True)

df = pd.read_csv(os.path.join(args.dir, 'splited_label.csv'))

for row in df.itertuples():
    if row.split == 'val':
        basename = os.path.basename(row.fname).split('.')[0]
        print(basename)
        shutil.move(os.path.join(args.dir, 'train', basename),
                    os.path.join(args.dir, 'val'))
