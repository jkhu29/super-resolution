import random
import warnings

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import dataloader

from tqdm import tqdm

import config
import dataset
import utils
import loss

from srresnet.model import SRCNN, FSRCNN, SRResNet
from edsr.model import EDSR, VDSR
from rcan.model import RCAN
from test_model.model import SRFNet


model_dict = {
    "srcnn": SRCNN(),
    "fsrcnn": FSRCNN(),
    "srresnet": SRResNet(),
    "edsr": EDSR(),
    "vdsr": VDSR(),
    "rcan": RCAN(),
    "test": SRFNet(),
}

criterion_dict = {
    "l1": nn.L1Loss(),
    "l2": nn.MSELoss(),
    "cl1": loss.L1_Charbonnier_loss(),
    "cl2": loss.L2_Charbonnier_loss(),
}


def model_pretrain(opt):
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
    model = model_dict[opt.model_name].to(device)

    # criterion init
    criterion = criterion_dict[opt.criterion_name]

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
        sample_data = torch.rand(1, 3, 30, 30).to(device)
        out = model(sample_data)
        d = make_dot(out)
        d.render('modelviz.pdf', view=False)

    # train edsr
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

    if opt.save_model:
        torch.save(model.state_dict(), "{}/models/{}.pth".format(opt.output_dir, opt.model_name))


if __name__ == "__main__":
    opt = config.get_options()
    model_pretrain(opt)
