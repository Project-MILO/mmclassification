import os
import json
import argparse
from glob import glob
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('dir', metavar='dir', type=str,
                    help='enter directory containing data')
parser.add_argument('output', metavar='output_dir', type=str,
                    help='enter directory containing data')

args = parser.parse_args()

anno = pd.read_csv(os.path.join(args.dir, 'raw_label.csv'))

os.makedirs(os.path.join(os.getcwd(), args.output, 'meta'), exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), args.output, 'train'), exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), args.output, 'val'), exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), args.output, 'test'), exist_ok=True)

train_anno_txt = open(os.path.join(
    os.getcwd(), args.output, 'meta/train.txt'), 'w')
val_anno_txt = open(os.path.join(
    os.getcwd(), args.output, 'meta/val.txt'), 'w')
for row in anno.itertuples():
    dir = row.fname.split('.')[0]
    if os.path.exists(os.path.join(args.dir, 'train', dir)):
        files = glob(os.path.join(args.dir, f'train/{dir}/*'))
        for file in files:
            file_path = os.path.join(dir, os.path.basename(file))
            train_anno_txt.write(f'{file_path} {row.liveness_score}')
            train_anno_txt.write('\n')

    elif os.path.exists(os.path.join(args.dir, 'val', dir)):
        files = glob(os.path.join(args.dir, f'val/{dir}/*'))
        for file in files:
            file_path = os.path.join(dir, os.path.basename(file))
            val_anno_txt.write(f'{file_path} {row.liveness_score}')
            val_anno_txt.write('\n')

train_anno_txt.close()
val_anno_txt.close()
