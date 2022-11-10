import os
import json

os.makedirs(os.path.join(os.getcwd(), 'data/liveness_1fps/meta'))
os.makedirs(os.path.join(os.getcwd(), 'data/liveness_1fps/train'))
os.makedirs(os.path.join(os.getcwd(), 'data/liveness_1fps/val'))
os.makedirs(os.path.join(os.getcwd(), 'data/liveness_1fps/test'))

f = open('/home/kiennt54/data/milo/liveness/train_images_1fps/val/annotations/instances_default.json')

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

f = open('/home/kiennt54/data/milo/liveness/train_images_1fps/train/annotations/instances_default.json')

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
