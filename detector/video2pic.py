import os.path
import subprocess

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import utils.io


def v2p():
    oid = 104
    root_dir = os.path.join('/data/real_exp/i12P_video', f'{oid:>06}', 'xyz', 'rolling')

    cap = cv2.VideoCapture(os.path.join(root_dir, 'IMG_2230.MOV'))
    i = 0
    while cap.isOpened():
        ret, im = cap.read()
        if not ret:
            break
        plt.imsave(os.path.join(root_dir, 'orig', f'{i:>04}.png'), im[..., ::-1], vmin=0., vmax=1.)
        i += 1


def p2v():
    oid = 104
    root_dir = os.path.join('/data/real_exp/i12P_video', f'{oid:>06}', 'sa', 'rolling')
    sub_dir = 'pose'

    out = cv2.VideoWriter(os.path.join(root_dir, f'{sub_dir}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 59.94, (3840, 2160))
    img_path_list = utils.io.list_img_from_dir(os.path.join(root_dir, sub_dir), ext='png')
    for img_path in tqdm(img_path_list):
        im = utils.io.imread(img_path, opencv_bgr=True)
        out.write(im)
    out.release()


def p2v_cmd():
    oid = 104
    root_dir = os.path.join('/data/real_exp/i12P_video', f'{oid:>06}', 'sa', 'holding')
    sub_dir = 'orig'
    fps = 60. * .999

    # root_dir = os.path.join('/home/user/Desktop/texture_evolution')
    # sub_dir = f'{oid}'

    cmd = ['/usr/bin/ffmpeg', '-y', '-framerate', f'{fps:.3f}', '-i',
        os.path.join(root_dir, sub_dir, '%04d.png'),
        '-c:v', 'libx265',  # '-vtag', 'hvc1', '-crf', '20', '-preset', 'slower',
        os.path.join(root_dir, f'{sub_dir}.mp4')]
    print(' '.join(cmd))
    # os.system(' '.join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

    while True:
        nextline = p.stdout.readline()
        if nextline == '' and p.poll() is not None:
            break
        print(nextline, end='', flush=True)

    output = p.communicate()[0]
    exitCode = p.returncode

    if (exitCode == 0):
        return output
    else:
        a = 0
        # raise ProcessException(command, exitCode, output)


def merge():
    oid = 104
    root_dir = os.path.join('/data/real_exp/i12P_video', f'{oid:>06}', 'sa', 'rolling')

    img_path_list_o = utils.io.list_img_from_dir(os.path.join(root_dir, 'orig'), ext='png')
    img_path_list_d = utils.io.list_img_from_dir(os.path.join(root_dir, 'detect'), ext='png')
    img_path_list_p = utils.io.list_img_from_dir(os.path.join(root_dir, 'pose'), ext='png')
    im_margin = np.zeros([15, 3840, 3]).astype('uint8')
    for img_path_o, img_path_d, img_path_p in tqdm(zip(img_path_list_o, img_path_list_d, img_path_list_p)):
        im_o = utils.io.imread(img_path_o, opencv_bgr=False)[930:1640]
        im_d = utils.io.imread(img_path_d, opencv_bgr=False)[930:1640]
        im_p = utils.io.imread(img_path_p, opencv_bgr=False)[930:1640]
        im = np.concatenate([im_o, im_margin, im_d, im_margin, im_p], axis=0)
        plt.imsave(os.path.join(root_dir, 'merge', f'{img_path_o.split("/")[-1].split(".")[0]}.png'), im, vmin=0, vmax=255)


def merge2():
    oid = 104
    root_dir = os.path.join('/data/real_exp/i12P_video', f'{oid:>06}', 'sa', 'bottle')

    img_path_list_o = utils.io.list_img_from_dir(os.path.join(root_dir, 'orig'), ext='png')
    img_path_list_d = utils.io.list_img_from_dir(os.path.join(root_dir, 'detect'), ext='png')
    img_path_list_p = utils.io.list_img_from_dir(os.path.join(root_dir, 'pose'), ext='png')
    im_margin = np.zeros([5, 1440, 3]).astype('uint8')
    for img_path_o, img_path_d, img_path_p in tqdm(zip(img_path_list_o, img_path_list_d, img_path_list_p)):
        im_o = utils.io.imread(img_path_o, opencv_bgr=False)[:850, -1440:]
        im_d = utils.io.imread(img_path_d, opencv_bgr=False)[:850, -1440:]
        im_p = utils.io.imread(img_path_p, opencv_bgr=False)[:850, -1440:]
        im = np.concatenate([im_o, im_margin, im_d, im_margin, im_p][::-1], axis=0)
        im = im.transpose([1, 0, 2])[:, ::-1]
        plt.imsave(os.path.join(root_dir, 'merge', f'{img_path_o.split("/")[-1].split(".")[0]}.png'), im, vmin=0, vmax=255)


def mergev():
    oid = 104
    root_dir = os.path.join('/data/real_exp/i12P_video', f'{oid:>06}', 'sa', 'rolling')

    out = cv2.VideoWriter(os.path.join(root_dir, f'{"merge"}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 59.94, (3840, 2160))
    img_path_list_o = utils.io.list_img_from_dir(os.path.join(root_dir, 'orig'), ext='png')
    img_path_list_d = utils.io.list_img_from_dir(os.path.join(root_dir, 'detect'), ext='png')
    img_path_list_p = utils.io.list_img_from_dir(os.path.join(root_dir, 'pose'), ext='png')
    im_margin = np.zeros([15, 3840, 3]).astype('uint8')
    for img_path_o, img_path_d, img_path_p in tqdm(zip(img_path_list_o, img_path_list_d, img_path_list_p)):
        im_o = utils.io.imread(img_path_o, opencv_bgr=True)[930:1640]
        im_d = utils.io.imread(img_path_d, opencv_bgr=True)[930:1640]
        im_p = utils.io.imread(img_path_p, opencv_bgr=True)[930:1640]
        im = np.concatenate([im_o, im_margin, im_d, im_margin, im_p], axis=0)
        out.write(im)
    out.release()


if __name__ == '__main__':
    # v2p()
    # p2v()
    p2v_cmd()
    # merge2()
