# super-resolution

> please run `sudo apt install graphviz` first if you open the --save_model_pdf

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
- [ ] USRNet

## develop

- [ ] TensorRT
- [ ] OpenVino
- [ ] TVM
- [ ] OpenPPL