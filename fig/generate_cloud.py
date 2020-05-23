#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hugo Ladret
This file can be used to generate the MC.png image 
MotionClouds library can be installed with a simple  ' pip install MotionClouds '
paper : https://journals.physiology.org/doi/full/10.1152/jn.00737.2011
"""

import MotionClouds as mc
import numpy as np
import imageio

def generate_cloud(theta, b_theta, sf_0,
                   N_X, N_Y,
                   seed, contrast=1):
    fx, fy, ft = mc.get_grids(N_X, N_Y, 1)

    mc_i = mc.envelope_gabor(fx, fy, ft,
                             V_X=0., V_Y=0., B_V=0.,
                             sf_0=sf_0, B_sf=sf_0,
                             theta=theta, B_theta=b_theta)

    im_ = mc.rectif(mc.random_cloud(mc_i, seed=seed),
                    contrast=contrast)
    return im_[:, :, 0]

if __name__ == "__main__" :
    im = generate_cloud(np.pi/4, np.pi/36, .1, 512, 512, 42) 
    imageio.imsave('./MC.png', im)