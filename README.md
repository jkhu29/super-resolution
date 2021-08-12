# super-resolution

## non-deeplearning

- [ ] SCSR
- [ ] SelfExSR
- [ ] RFL

## deeplearning

### data

- [x] CUBIC(SRCNN) `./dataset_make/make_dataset.py`
- [x] DIV2K / Flickr2K / BSD / ... 
- [ ] TextZoom

### model

- [x] SRCNN / FSRCNN  `./srresnet/`
- [x] SRResNet / SRGAN `./srresnet/`
- [ ] SRDenseNet
- [x] VDSR / EDSR / MDSR `./edsr/`
- [ ] DBPN
- [x] RCAN `./rcan/`  rethinking
- [x] TTSR `./ttsr/`  
- [x] USRNet `./usrnet`
- [ ] RFDN

### loss

- [x] L1/L2 ls
- [x] Perceptual loss `./srresnet/`
- [x] Charbonnier loss `./loss.py`
- [x] sinkhorn loss `./loss.py` o with L1 & Charbonnier
- [ ] sinkhorn loss in edges

## develop

- [ ] TensorRT
- [ ] OpenVino
- [ ] TVM
- [ ] OpenPPL

## Usage

> please run `sudo apt install graphviz` first if you open the --save_model_pdf

```
python train.py \
    --model_name ttsr \
    --train_file ./dataset_make/train_data.h5 \
    --valid_file ./dataset_make/valid_data.h5 \
```

or you can see more attrs in `config.py`
