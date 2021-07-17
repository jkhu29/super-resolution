import torch
import numpy as np
import cv2
from model import SRResNet
from skimage.metrics import structural_similarity as ssim


path_pth = '/path/to/srgan_generator_pretrain.pth'
path_pth2 = '/path/to/srgan_generator_final.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model1 = SRResNet(within=False).to(device)
model2 = SRResNet(within=False).to(device)
model1.load_state_dict(torch.load(path_pth))
model2.load_state_dict(torch.load(path_pth2))
model1.to(device)
model2.to(device)
model1.eval()
model2.eval()

img = cv2.imread('test.jpg')
img_shape = np.array(list(img.shape)[:-1]) / 4
img0 = cv2.resize(img, tuple(img_shape.astype(np.int)[::-1]), cv2.INTER_CUBIC)
print(img.shape)

img1 = img0.transpose(2, 0, 1)
img1 = torch.from_numpy(img1.astype(np.float32)).to(device)
img1 = img1.unsqueeze(0)

img_out = model1(img1).detach().squeeze(0)
img_out2 = model2(img1).detach().squeeze(0)
img_out = img_out.cpu().numpy()
img_out2 = img_out2.cpu().numpy()
img_out = img_out.transpose(1, 2, 0)
img_out2 = img_out2.transpose(1, 2, 0)

img = cv2.resize(img, (500, 332))
print(img.shape, img_out.shape, img_out2.shape)

print('psnr: %.4f' % cv2.PSNR(img.astype(np.float32), img_out))
print('ssim: %.4f' % ssim(img, img_out, multichannel=True))

print('psnr: %.4f' % cv2.PSNR(img.astype(np.float32), img_out2))
print('ssim: %.4f' % ssim(img, img_out2, multichannel=True))

cv2.imshow('label', img)
# img_out = img_out.astype(np.int8)
print(img_out.shape)
cv2.imshow('sr', img_out / 255.)
cv2.imshow('gan', img_out2 / 255.)

cv2.imwrite('srresnet_pretrain.jpg', img_out / 255.)
cv2.imwrite('srgan_final.jpg', img_out2 / 255.)

cv2.waitKey(0)
cv2.destroyAllWindows()

