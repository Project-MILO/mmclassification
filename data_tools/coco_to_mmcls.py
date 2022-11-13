import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dir', metavar='dir', type=str,
                    help='enter directory containing data')

args = parser.parse_args()

os.makedirs(os.path.join(os.getcwd(), 'data/liveness_1fps/meta'), exist_ok=True)
os.makedirs(os.path.join(
    os.getcwd(), 'data/liveness_1fps/train'), exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), 'data/liveness_1fps/val'), exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), 'data/liveness_1fps/test'), exist_ok=True)

f = open(os.path.join(args.dir, 'val/annotations/instances_default.json'))

data = json.load(f)

images = data['images']
anno = data['annotations']

with open(os.path.join(os.getcwd(), 'data/liveness_1fps/meta/val.txt'), 'w') as anno_txt:
    for i, image in enumerate(images):
        label = int(anno[i]['attributes']['real'])
        anno_txt.write(f'{images[i]["file_name"]} {label}')
        anno_txt.write('\n')

anno_txt.close()
f.close()

f = open(os.path.join(args.dir, 'train/annotations/instances_default.json'))

data = json.load(f)

images = data['images']
anno = data['annotations']

with open(os.path.join(os.getcwd(), 'data/liveness_1fps/meta/train.txt'), 'w') as anno_txt:
    for i, image in enumerate(images):
        label = int(anno[i]['attributes']['real'])
        anno_txt.write(f'{images[i]["file_name"]} {label}')
        anno_txt.write('\n')

anno_txt.close()
f.close()
