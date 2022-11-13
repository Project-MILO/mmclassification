# Copyright (c) OpenMMLab. All rights reserved.
from mmcls.apis import inference_model, init_model, show_result_pyplot
import mmcv
from argparse import ArgumentParser
import os
import glob
import pandas as pd
import numpy as np

from datetime import timezone, datetime, timedelta
now = datetime.now()


def main():

    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Whether to show the predict results by matplotlib.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    df = pd.DataFrame(columns=['fname', 'liveness_score'])
    df_full = pd.DataFrame(
        columns=['fname', 'liveness_score', 'mean_f', 'mean_r'])

    model = init_model(args.config, args.checkpoint, device=args.device)
    dirs = glob.glob(f'{os.getcwd()}/{args.img}/*')

    for dir in dirs:
        real_counts = 0
        fake_counts = 0
        real_total = 0
        fake_total = 0
        video_name = f'{dir.split("/")[-1]}.mp4'
        row = {}
        row_full = {}
        for img in glob.glob(f'{dir}/*'):
            result = inference_model(model, img)
            # print(result)
            if result['pred_label'] == 1:
                real_counts += 1
                real_total += result['pred_score']
            else:
                fake_counts += 1
                fake_total += result['pred_score']
        real_mean = real_total / float(real_counts) if real_counts > 0 else 0
        fake_mean = fake_total / float(fake_counts) if fake_counts > 0 else 0
        m = max(real_mean, fake_mean)
        label = np.argmax([fake_mean, real_mean])

        row['fname'] = video_name
        row['liveness_score'] = 1 - m if label == 0 else m

        row_full['fname'] = video_name
        row_full['liveness_score'] = 1 - m if label == 0 else m
        row_full['mean_f'] = fake_mean
        row_full['mean_r'] = real_mean
        print(row_full)
        df = df.append(row, ignore_index=True)
        # df = pd.concat([df, row])
        df_full = df_full.append(row_full, ignore_index=True)
        # df_full = pd.concat([df_full, row_full])

    current_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    print(current_time)
    os.makedirs(os.path.join(os.getcwd(), f'results/{current_time}'))
    df.to_csv(f'results/{current_time}/result.csv', index=False)
    df_full.to_csv(f'results/{current_time}/result_full.csv', index=False)

    if args.show:
        show_result_pyplot(model, args.img, result)


if __name__ == '__main__':
    main()
