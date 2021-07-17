import copy
import random
import warnings

import torch
from torch import nn
from torch import optim
from torch.utils.data import dataloader
from torch.autograd import Variable
import torchvision

from tqdm import tqdm

from model import SRResNet, SRGAN_D, FeatureExtractor
import config
import dataset
import utils


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
# models = [SRResNet().to(device)]
model_g = SRResNet(within=False).to(device)
model_d = SRGAN_D().to(device)

# criterion init
# criterions = [nn.MSELoss(), nn.L1Loss()]
criterion_g = nn.L1Loss()
criterion_d = nn.BCELoss()
feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True)).to(device)

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

model_g.apply(utils.weights_init)
model_d.apply(utils.weights_init)

# optim init
if opt.adam:
    model_g_optimizer = optim.Adam(model_g.parameters(), lr=opt.lr, eps=1e-8, weight_decay=1e-3)
else:
    model_g_optimizer = optim.RMSprop(model_g.parameters(), lr=opt.lr)

model_g_scheduler = optim.lr_scheduler.CosineAnnealingLR(model_g_optimizer, T_max=opt.niter)

if opt.save_model_pdf:
    from torchviz import make_dot
    sampleData = torch.rand(64, 3, 30, 30)
    out = model_g(sampleData)
    out_d = model_d(out)
    d = make_dot(out_d)
    d.render('modelviz', view=False)

# pre-train srresnet first
for epoch in range(opt.niter):

    model_g.train()

    epoch_losses = utils.AverageMeter()

    with tqdm(total=(len(train_dataset) - len(train_dataset) % opt.batch_size)) as t:
        t.set_description('epoch: {}/{}'.format(epoch+1, opt.niter))

        for data in train_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            preds_fake = model_g(inputs)

            model_g_optimizer.zero_grad()

            loss = criterion_g(preds_fake, labels)
            loss.backward()
            epoch_losses.update(loss.item(), len(inputs))

            model_g_optimizer.step()

            t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
            t.update(len(inputs))

    model_g_scheduler.step()

    # test
    model_g.eval()

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

torch.save(model_g.state_dict(), "%s/models/srgan_generator_pretrain.pth" % opt.output_dir)

# train srgan
if opt.adam:
    model_g_optimizer = optim.Adam(model_g.parameters(), lr=opt.lr*0.1, eps=1e-8, weight_decay=1e-3)
    model_d_optimizer = optim.Adam(model_d.parameters(), lr=opt.lr*0.1, eps=1e-8, weight_decay=1e-3)
else:
    model_g_optimizer = optim.RMSprop(model_g.parameters(), lr=opt.lr*0.1)
    model_d_optimizer = optim.RMSprop(model_d.parameters(), lr=opt.lr*0.1)

model_g_scheduler = optim.lr_scheduler.CosineAnnealingLR(model_g_optimizer, T_max=opt.niter)
model_d_scheduler = optim.lr_scheduler.CosineAnnealingLR(model_d_optimizer, T_max=opt.niter)

for epoch in range(opt.niter):

    model_g.train()
    model_d.train()

    epoch_losses_d = utils.AverageMeter()
    epoch_losses_total = utils.AverageMeter()

    with tqdm(total=(len(train_dataset) - len(train_dataset) % opt.batch_size)) as t:
        t.set_description('epoch: {}/{}'.format(epoch+1, opt.niter))

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
