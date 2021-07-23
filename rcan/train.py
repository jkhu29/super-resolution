import copy
import random
import warnings

import torch
from torch import nn
from torch import optim
from torch.utils.data import dataloader

from tqdm import tqdm

from model import RCAN
import config
import dataset
import utils
from loss import L1_edge_W_loss


opt = config.get_options()

# deveice init
CUDA_ENABLE = torch.cuda.is_available()
if CUDA_ENABLE and opt.cuda:
    import torch.backends.cudnn as cudnn
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True
elif CUDA_ENABLE and not opt.cuda:
    warnings.warn("WARNING: You have CUDA device, so you should probably run with --cuda")
elif not CUDA_ENABLE and opt.cuda:
    assert CUDA_ENABLE, "ERROR: You don't have a CUDA device"

device = torch.device('cuda:0' if CUDA_ENABLE else 'cpu')

# seed init
manual_seed = opt.seed
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# models init
model = RCAN().to(device)

# criterion init
criterion = L1_edge_W_loss(device)

# dataset init, train file need .h5
train_dataset = dataset.TrainDataset(opt.train_file)
train_dataloader = dataloader.DataLoader(dataset=train_dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=opt.workers,
                                         pin_memory=True,
                                         drop_last=True)

valid_dataset = dataset.ValidDataset(opt.valid_file)
valid_dataloader = dataloader.DataLoader(dataset=valid_dataset, batch_size=1)

model.apply(utils.weights_init)

# optim init
if opt.adam:
    model_optimizer = optim.Adam(model.parameters(), lr=opt.lr, eps=1e-8, weight_decay=1)
else:
    model_optimizer = optim.RMSprop(model.parameters(), lr=opt.lr)

model_scheduler = optim.lr_scheduler.CosineAnnealingLR(model_optimizer, T_max=opt.niter)

if opt.save_model_pdf:
    from torchviz import make_dot
    sampleData = torch.rand(1, 3, 30, 30).to(device)
    out = model(sampleData)
    d = make_dot(out)
    d.render('modelviz', view=False)

# train rcan
for epoch in range(opt.niter):

    model.train()

    epoch_losses = utils.AverageMeter()

    with tqdm(total=(len(train_dataset) - len(train_dataset) % opt.batch_size)) as t:
        t.set_description('epoch: {}/{}'.format(epoch+1, opt.niter))

        for data in train_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)

            model_optimizer.zero_grad()

            loss = criterion(preds, labels)
            loss.backward()
            epoch_losses.update(loss.item(), len(inputs))

            model_optimizer.step()

            t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
            t.update(len(inputs))

    model_scheduler.step()

    # test
    model.eval()

    epoch_pnsr = utils.AverageMeter()
    epoch_ssim = utils.AverageMeter()

    for data in valid_dataloader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels[0].to(device)

        with torch.no_grad():
            preds = model(inputs)[0]
            epoch_pnsr.update(utils.calc_pnsr(preds, labels), len(inputs))
            epoch_ssim.update(utils.calc_ssim(preds, labels), len(inputs))

    print('eval psnr: {:.4f} eval ssim: {:.4f}'.format(epoch_pnsr.avg, epoch_ssim.avg))

torch.save(model.state_dict(), "%s/models/rcan.pth" % opt.output_dir)
