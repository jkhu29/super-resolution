import random
import warnings

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import dataloader
from torch.autograd import Variable
import torchvision

from tqdm import tqdm

import config
import dataset
import utils
import loss

from srresnet.model import SRCNN, FSRCNN, SRResNet
from edsr.model import EDSR, VDSR
from rcan.model import RCAN
from ttsr.model import TTSR
from test_model.model import SRFNet

from discriminator import vgg19

model_dict = {
    "srcnn": SRCNN(),
    "fsrcnn": FSRCNN(),
    "srresnet": SRResNet(),
    "edsr": EDSR(),
    "vdsr": VDSR(),
    "rcan": RCAN(),
    # "ttsr": TTSR(), TODO: need a new train function
    "test": SRFNet(),
}

discriminator_dict = {
    "vgg19": vgg19.Vgg19Discriminator,
}

criterion_dict = {
    "l1": nn.L1Loss(),
    "l2": nn.MSELoss(),
    "cl1": loss.L1_Charbonnier_loss(),
    "cl2": loss.L2_Charbonnier_loss(),
}


def train_init():
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

    return device, train_dataloader, valid_dataloader, len(train_dataset)


def model_pretrain(opt, device, train_dataloader, valid_dataloader, length):
    # models init
    model = model_dict[opt.model_name].to(device)
    model.apply(utils.weights_init)

    # criterion init
    criterion = criterion_dict[opt.criterion_name]

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

        with tqdm(total=(length - length % opt.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch + 1, opt.niter))

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

    return model


def gan_train(opt, device, length, model_g):
    # models init
    model_d = discriminator_dict[opt.discriminator_name].to(device)

    # criterion init
    criterion_g = criterion_dict[opt.criterion_name]
    criterion_d = nn.BCELoss()

    # ature extractor init
    feature_extractor = vgg19.FeatureExtractor(torchvision.models.vgg19(pretrained=True)).to(device)

    # train gan
    if opt.adam:
        model_g_optimizer = optim.Adam(model_g.parameters(), lr=opt.lr * 0.1, eps=1e-8, weight_decay=1)
        model_d_optimizer = optim.Adam(model_d.parameters(), lr=opt.lr * 0.1, eps=1e-8, weight_decay=1)
    else:
        model_g_optimizer = optim.RMSprop(model_g.parameters(), lr=opt.lr * 0.1)
        model_d_optimizer = optim.RMSprop(model_d.parameters(), lr=opt.lr * 0.1)

    model_g_scheduler = optim.lr_scheduler.CosineAnnealingLR(model_g_optimizer, T_max=opt.niter)
    model_d_scheduler = optim.lr_scheduler.CosineAnnealingLR(model_d_optimizer, T_max=opt.niter)

    for epoch in range(opt.niter):
        model_g.train()
        model_d.train()

        epoch_losses_d = utils.AverageMeter()
        epoch_losses_total = utils.AverageMeter()

        with tqdm(total=(length - length % opt.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch + 1, opt.niter))

            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)

                h_fake = model_g(inputs)

                target_real = Variable(torch.rand(opt.batch_size) * 0.5 + 0.7)
                target_fake = Variable(torch.rand(opt.batch_size) * 0.3)

                h_fake = h_fake.to(device)
                h_real = Variable(labels).to(device)
                target_real = target_real.to(device)
                target_fake = target_fake.to(device)

                # srgan train
                model_d.zero_grad()

                loss_d = criterion_d(model_d(h_real), target_real) + \
                         criterion_d(model_d(Variable(h_fake)), target_fake)
                loss_d.backward()
                epoch_losses_d.update(loss_d.item(), len(inputs))

                model_d_optimizer.step()

                # srresnet train
                model_g.zero_grad()

                features_real = Variable(feature_extractor(h_real.data)).to(device)
                features_fake = feature_extractor(h_fake.data).to(device)

                content_loss = criterion_g(h_fake, h_real) + 0.006 * criterion_g(features_fake, features_real)
                adversarial_loss = criterion_d(model_d(h_fake), Variable(torch.ones(opt.batch_size)).to(device))
                total_loss = content_loss + 1e-3 * adversarial_loss
                total_loss.backward()
                epoch_losses_total.update(total_loss.item(), len(inputs))

                model_g_optimizer.step()

                t.set_postfix(total_loss='{:.6f}'.format(epoch_losses_total.avg))
                t.update(len(inputs))

        model_g_scheduler.step()
        model_d_scheduler.step()

        # test
        model_g.eval()
        model_d.eval()

        epoch_pnsr = utils.AverageMeter()
        epoch_ssim = utils.AverageMeter()

        for data in valid_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels[0].to(device)

            with torch.no_grad():
                preds = model_g(inputs)[0]
                epoch_pnsr.update(utils.calc_pnsr(preds, labels), len(inputs))
                epoch_ssim.update(utils.calc_ssim(preds, labels), len(inputs))

        print('eval psnr: {:.4f} eval ssim: {:.4f}'.format(epoch_pnsr.avg, epoch_ssim.avg))

    torch.save(model_g.state_dict(), '%s/models/srgan_generator_final.pth' % opt.output_dir)
    torch.save(model_d.state_dict(), '%s/models/srgan_discriminator_final.pth' % opt.output_dir)


if __name__ == "__main__":
    opt = config.get_options()
    device, train_dataloader, valid_dataloader, length = train_init()
    model = model_pretrain(opt, device, train_dataloader, valid_dataloader, length)
    if "gan" in opt.model_name.lower():
        gan_train(opt, device, length, model_g=model)
