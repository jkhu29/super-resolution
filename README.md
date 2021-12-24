# super-resolution

model zoos

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
- [x] VDSR / EDSR / MDSR `./edsr/`
- [ ] DBPN
- [x] RCAN `./rcan/`
- [x] TTSR `./ttsr/`      # NOTE(JiaKui Hu): rethinking
- [x] USRNet `./usrnet`   # NOTE(JiaKui Hu): rethinking
- [x] RFDN `./rfdn`
- [x] ZSSR `./zssr`
- [x] AcNet `./acnet`
- [x] PAN `./pan`
- [ ] EPSANet

### loss

- [x] L1/L2 ls
- [x] Perceptual loss `./srresnet/`
- [x] Charbonnier loss `./loss.py`
- [x] sinkhorn loss `./loss.py`  with L1 & Charbonnier
- [ ] sinkhorn loss in edges

## inference

- [ ] TensorRT
- [ ] TVM
- [ ] vitis-ai

## Usage

to make sure all the models could be run on your machine
```
pytest test.py
```

> please run `sudo apt install graphviz` first if you open the --save_model_pdf

```
python train.py \
    --model_name ttsr \
    --train_file ./dataset_make/train_data.h5 \
    --valid_file ./dataset_make/valid_data.h5 \
```

or see more attrs in `config.py`
