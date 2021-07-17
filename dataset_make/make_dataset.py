import argparse
import copy
import os

import cv2
import h5py
import numpy as np
import torch


def convert_rgb_to_y(img):
    if type(img) == np.ndarray:
        return 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        return 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
    else:
        raise Exception('Unknown Type', type(img))


def convert_rgb_to_ycbcr(img):
    if type(img) == np.ndarray:
        y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
        cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
        cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
        cb = 128. + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :]) / 256.
        cr = 128. + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :]) / 256.
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def convert_ycbcr_to_rgb(img):
    if type(img) == np.ndarray:
        r = 298.082 * img[:, :, 0] / 256. + 408.583 * img[:, :, 2] / 256. - 222.921
        g = 298.082 * img[:, :, 0] / 256. - 100.291 * img[:, :, 1] / 256. - 208.120 * img[:, :, 2] / 256. + 135.576
        b = 298.082 * img[:, :, 0] / 256. + 516.412 * img[:, :, 1] / 256. - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - 222.921
        g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + 135.576
        b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def make_h5(data_path, h5_path, size_input=25, size_label=100, stride=10, scale=4):
    imgs_lr = []
    imgs_hr = []
    for image_name in os.listdir(data_path):
        img_label = cv2.imread(os.path.join(data_path, image_name))
        # print(img_label.shape)
        img = copy.deepcopy(img_label)
        img_shape = np.array(list(img.shape)[:-1]) / scale
        img = cv2.resize(img, tuple(img_shape.astype(np.int)[::-1]), cv2.INTER_CUBIC)
        # print(img.shape)

        for x in np.arange(0, img.shape[0] - size_input + 1, stride):
            for y in np.arange(0, img.shape[1] - size_input + 1, stride):
                img_lr = img[int(x): int(x + size_input),
                         int(y): int(y + size_input)]
                img_hr = img_label[int(x * scale): int(x * scale + size_label),
                         int(y * scale): int(y * scale + size_label)]
                imgs_lr.append(img_lr.transpose(2, 0, 1))
                imgs_hr.append(img_hr.transpose(2, 0, 1))

    print('begin to save h5 file to %s' % h5_path)
    # print(np.array(imgs_lr).shape)
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('lr', data=np.array(imgs_lr, dtype=np.float32))
        f.create_dataset('hr', data=np.array(imgs_hr, dtype=np.float32))
    print('saved')


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--valid_path', type=str, required=True)
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--valid_data', type=str, required=True)
    parser.add_argument('--size_input', type=int, default=25, help='the size of input image, default=25*25')
    parser.add_argument('--size_label', type=int, default=100, help='the size of output image, default=100*100')
    parser.add_argument('--stride', type=int, default=10, help='stride when making dataset, default=10')
    parser.add_argument('--scale', type=int, default=10, help='scale, default=4')
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = get_options()

    train_path = opt.train_path
    valid_path = opt.valid_path
    train_data = opt.train_path
    valid_data = opt.valid_data

    make_h5(train_data, train_path,
            size_input=opt.size_input, size_label=opt.size_label, stride=opt.stride, scale=opt.scale
            )
    make_h5(valid_data, valid_path,
            size_input=opt.size_input, size_label=opt.size_label, stride=opt.stride, scale=opt.scale
            )
