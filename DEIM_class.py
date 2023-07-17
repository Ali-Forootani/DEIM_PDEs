#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 08:25:47 2023

@author: forootani
"""



import numpy as np
import sys
import os
from pathlib import Path
import torch
from scipy import linalg
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
import warnings
import time
import numpy as np
from scipy.linalg import svd, qr
import itertools
############################################


cwd = os.getcwd()
#sys.path.append(cwd + '/my_directory')
sys.path.append(cwd)
warnings.filterwarnings("ignore")
np.random.seed(1234)
torch.manual_seed(7)
# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
data = scipy.io.loadmat('data/AC.mat')
t_o = data["tt"].flatten()[0:201, None]
x_o = data["x"].flatten()[:, None]
Exact = np.real(data["uu"])


###################################################
###################################################
###################################################



class DEIM:
    def __init__(self, X, n_d, t_o, x_o, dec_rate = 0, num_basis=20):
        self.X = X
        self.n_d = n_d
        self.num_basis = num_basis
        self.i_t = []
        self.i_x = []
        self.u_selected = []
        self.t_sampled = []
        self.x_sampled = []
        self.X_sampled = []
        self.T_sampled = []
        self.S_star = []
        self.T_star = []
        self.U_star = []
        self.coords = None
        self.t_o = t_o
        self.x_o = x_o
        self.dec_rate = dec_rate

    def deim(self, X, i):
        U, Sigma, Vt = svd(X, full_matrices=False)

        # Step 2: Select the basis functions
        k = (self.num_basis - self.dec_rate * 2)  # Number of basis functions to retain

        # Step 3: Compute the SVD-based approximation
        Uk = U[:, :k]
        Sigma_k = np.diag(Sigma[:k])
        Vk_t = Vt[:k, :]

        X_k = Uk @ Sigma_k @ Vk_t

        left = Uk @ np.sqrt(Sigma_k)
        right = np.sqrt(Sigma_k) @ Vk_t

        q_x, r_x, p_x = qr(Uk.T, mode='economic', pivoting=True)
        i_x = p_x[:k]

        q_t, r_t, p_t = qr(Vk_t, mode='economic', pivoting=True)
        i_t = p_t[:k]

        return i_t, i_x

    def execute(self):
        n_k = self.X.shape[1]
        n_s = int(n_k / self.n_d)

        for i in range(self.n_d):
            
            i_tf, i_xf = self.deim(self.X[:, i * n_s: (i+1) * n_s], i)
            i_tf = i_tf + i * n_s
            self.i_t.append([i_tf])
            self.i_x.append([i_xf])
            space_o, T_o = np.meshgrid(self.x_o, self.t_o, indexing="ij")
            
            self.X_sampled.append(space_o)
            self.T_sampled.append(T_o)
            #########################
            t, space = np.meshgrid(i_tf, i_xf, indexing="ij")
            self.u_selected.append(self.X[space, t])
            self.t_sampled.append(T_o[space,t])
            self.x_sampled.append(space_o[space, t])
            
            X_star = np.hstack((t.flatten()[:, None], space.flatten()[:, None]))
            #plt.scatter(X_star[:,0], X_star[:,1])
            plt.scatter(X_star[:,0], X_star[:,1], c=self.X[space, t])
            #plt.ylim([-50,600])
            ############################
            
            self.S_star.append(self.x_sampled[i].flatten())
            self.T_star.append(self.t_sampled[i].flatten())
            self.U_star.append(self.u_selected[i].flatten())


        S_s = np.concatenate(self.S_star, axis=0).reshape(-1, 1)
        T_s = np.concatenate(self.T_star, axis=0).reshape(-1, 1)
        U_s = np.concatenate(self.U_star, axis=0).reshape(-1, 1)

        self.coords = np.hstack((S_s,T_s))
        
        return S_s, T_s, U_s



deim_instance = DEIM(Exact, 2, t_o, x_o, num_basis=40)
S_s, T_s, U_s = deim_instance.execute()

#S_star = deim_instance.S_star
#T_star = deim_instance.T_star
#U_star = deim_instance.U_star
#coords = deim_instance.coords

#print(U_s)
#print(T_s)
#print(S_s)
