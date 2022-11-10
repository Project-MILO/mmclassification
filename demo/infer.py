# Copyright (c) OpenMMLab. All rights reserved.
from mmcls.apis import inference_model, init_model, show_result_pyplot
import mmcv
from argparse import ArgumentParser
import os
import glob
import pandas as pd

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

    model = init_model(args.config, args.checkpoint, device=args.device)
    dirs = glob.glob(f'{os.getcwd()}/{args.img}/*')
    results = {}
    for dir in dirs:
        real_counts = 0
        fake_counts = 0
        video_name = f'{dir.split("/")[-1]}.mp4'
        for img in glob.glob(f'{dir}/*'):
            result = inference_model(model, img)
            if result['pred_label'] == 1:
                real_counts += 1
            else:
                fake_counts += 1
        results[video_name] = 1 if real_counts > fake_counts else 0

    print(results)
    current_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    print(current_time)
    os.makedirs(os.path.join(os.getcwd(), f'results/{current_time}'))
    df = pd.DataFrame(list(results.items()), columns=[
                      "fname", "liveness_score"])
    df.to_csv(f'results/{current_time}/result.csv', index=False)

    # print(mmcv.dump(result, file_format='json', indent=4))
    if args.show:
        show_result_pyplot(model, args.img, result)


if __name__ == '__main__':
    main()
