from otter.test_files import test_case

OK_FORMAT = False

import torch
import torch.nn as nn
import numpy as np
from itertools import product

name = "Exercise 6.1"
points = 6

@test_case(points=4)
def test_1(Encoder, env):    
    img_size = [32, 64, 128]
    emb_size = [128, 512]
    patch_size = [4, 8, 16]
    for cfg in product(img_size, patch_size, emb_size):
        w = h = cfg[0]
        p = cfg[1]
        d = cfg[2]
        img = torch.rand(16,3,w,h)
        enc = Encoder(in_channels=3, patch_size=p, emb_size=d)
        out0, _ = enc(img)
        assert out0.shape == (16, w//p * h//p, d), "Output shape is wrong: {}".format(out0.shape)

@test_case(points=2)
def test_2(Encoder, env):    
    img_size = [32, 64, 128]
    emb_size = [128, 512]
    patch_size = [4, 8, 16]
    for cfg in product(img_size, patch_size, emb_size):
        w = h = cfg[0]
        p = cfg[1]
        d = cfg[2]
        img = torch.rand(16,3,w,h)
        enc = Encoder(in_channels=3, patch_size=p, emb_size=d)
        _, out1 = enc(img)
        assert out1.shape == (16, w//p * h//p + 1, d), "Output shape is wrong: {}".format(out1.shape)
        assert isinstance(enc.token, nn.Parameter), "Added vector in Step 2 is not a nn.Parameter!"
