import pytest
from torchsummary import summary


def summary_model(m):
    summary(m, (3, 64, 64))


def test_srcnn():
    from srresnet.model import SRCNN
    model = SRCNN().cuda()
    summary_model(model)


def test_srresnet():
    from srresnet.model import SRResNet
    model = SRResNet().cuda()
    summary_model(model)


def test_rcan():
    from rcan.model import RCAN
    model = RCAN().cuda()
    summary_model(model)


def test_edsr():
    from edsr.model import EDSR, VDSR
    model1 = EDSR().cuda()
    model2 = VDSR().cuda()
    summary_model(model1)
    summary_model(model2)


def test_acnet():
    from acnet.model import AcNet
    model = AcNet().cuda()
    summary_model(model)


def test_rfdn():
    from rfdn.model import RFDN
    model = RFDN().cuda()
    summary_model(model)


def test_pan():
    from pan.model import PAN
    model = PAN().cuda()
    summary_model(model)


# def test_dbpn():
#     from dbpn.model import DBPN


if __name__ == "__main__":
    pytest.main(["-s", "test.py"])
