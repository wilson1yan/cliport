import os
import os.path as osp
import numpy as np
import multiprocessing as mp
import argparse


def worker(id, n):
    dst_dir = osp.join(args.output_dir, str(id))
    video_path = osp.join(dst_dir, 'video')
    cmd = f'python cliport/demos.py n={n} task={args.task} mode=train data_dir={dst_dir} disp=False record.save_video=true record.video_height={args.resolution} record.video_width={args.resolution} record.save_video_path={video_path}'
    os.system(cmd)


def main():
    chunks = [args.n // args.num_workers + (i < (args.n % args.num_workers))
              for i in range(args.num_workers)]
    procs = [mp.Process(target=worker, args=(id, n)) for id, n in zip(range(args.num_workers), chunks)]
    [p.start() for p in procs]
    [p.join() for p in procs]
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default='packing-seen-google-objects-group')
    parser.add_argument('-n', '--n', type=int, default=40000)
    parser.add_argument('-r', '--resolution', type=int, default=256)
    parser.add_argument('-w', '--num_workers', type=int, default=48)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    args = parser.parse_args()

    main()
