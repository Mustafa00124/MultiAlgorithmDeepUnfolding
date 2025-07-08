import torch
import torch.nn as nn
import torch.nn.functional as F
from madu_layers import Madu_Layer

   
class Madu(nn.Module):
    def __init__(self, params):
        super(Madu, self).__init__()
        self.params = params
        self.h = params['H']
        if params['mode'] == 'median':
            self.branch1 = 'median'
        if params['mode'] == 'mean':
            self.branch1 = 'mean'
        if params['mode'] == 'dft':
            self.branch1 = 'dft'
        if params['mode'] == 'svd':
            self.branch1 = 'svd'
        if params['mode'] == 'Madu1':
            self.branch1 = 'Madu1'
        if params['mode'] == 'Madu2':
            self.branch1 = 'Madu2'
        if params['mode'] == 'Madu3':
            self.branch1 = 'Madu3'
        if params['mode'] == 'Ensemble1':
            self.branch1 = 'mean'
            self.branch2 = 'dft'
            self.branch3 = 'svd'
        if params['mode'] == 'Ensemble2':
            self.branch1 = 'median'
            self.branch2 = 'dft'
            self.branch3 = 'svd'
        if params['mode'] == 'Ensemble3':
            self.branch1 = 'mean'
            self.branch2 = 'median'
            self.branch3 = 'dft'
        if params['mode'] == 'Series1':
            self.branch1 = 'mean'
            self.branch2 = 'dft'
            self.branch3 = 'svd'
        if params['mode'] == 'Series2':
            self.branch1 = 'median'
            self.branch2 = 'dft'
            self.branch3 = 'svd'
        if params['mode'] == 'Series3':
            self.branch1 = 'median'
            self.branch2 = 'mean'
            self.branch3 = 'dft'



        if params['mode'] in ['svd', 'mean', 'median', 'dft', 'Madu1', 'Madu2', 'Madu3']:
            self.branch1 = self.make_Madu_layers()
        else:
            self.branch1 = self.make_Madu_layers()
            self.branch2 = self.make_Madu_layers()
            self.branch3 = self.make_Madu_layers()

            # Low rank weights
            self.u_M_branch1 = nn.Parameter(torch.ones(self.h, 1))
            self.v_M_branch1 = nn.Parameter(torch.ones(128, 1) / 3.0)

            self.u_M_branch2 = nn.Parameter(torch.ones(self.h, 1))
            self.v_M_branch2 = nn.Parameter(torch.ones(128, 1) / 3.0)

            self.u_M_branch3 = nn.Parameter(torch.ones(self.h, 1))
            self.v_M_branch3 = nn.Parameter(torch.ones(128, 1) / 3.0)

            self.w_branch1 = nn.Parameter(torch.randn(1, 1))
            self.w_branch2 = nn.Parameter(torch.randn(1, 1))
            self.w_branch3 = nn.Parameter(torch.randn(1, 1))

        self.foregrounds1 = []
        self.foregrounds2 = []
        self.foregrounds3 = []

        # Define forward modes based on the mode
        mode_map = {
            'median': ('median', 'median', 'median'),
            'mean': ('mean', 'mean', 'mean'),
            'dft': ('dft', 'dft', 'dft'),
            'svd': ('svd', 'svd', 'svd'),
            'Madu1': ('Madu1', 'Madu1', 'Madu1'),
            'Madu2': ('Madu2', 'Madu2', 'Madu2'),
            'Madu3': ('Madu3', 'Madu3', 'Madu3'),
            'Ensemble1': ('mean', 'dft', 'svd'),
            'Ensemble2': ('median', 'dft', 'svd'),
            'Ensemble3': ('mean', 'median', 'dft'),
            'Series1': ('mean', 'dft', 'svd'),
            'Series2': ('median', 'dft', 'svd'),
            'Series3': ('median', 'mean', 'dft'),
            'serialx': ('median', 'dft', 'svd')
        }
        self.forward_mode1, self.forward_mode2, self.forward_mode3 = mode_map.get(params['mode'], ('default', 'default', 'default'))

    def depthwise_softmax(self,mean,median,uv):
        # Stack matrices along a new dimension (0) to apply softmax depthwise
        stacked = torch.stack([mean, median, uv], dim=0)  # Shape becomes [3, H, W] for 3 matrices
        # Apply softmax across the first dimension (models)
        softmaxed = F.softmax(stacked, dim=0)
        return softmaxed

    def make_Madu_layers(self):
        layers = []
        params = self.params
        for _ in range( self.params['layers']):
            layers.append(Madu_Layer(kernel=params['kernel'][0],
            coef_L_initializer=params['coef_L'],
            coef_S_initializer=params['coef_S'],
            coef_S_side_initializer=params['coef_S_side'],
            l1_l1=params['l1_l1'],
            reweightedl1_l1=params['reweightedl1_l1'],
            hidden_dim=params['hidden_filters'],
            l1_l2=params['l1_l2']))
        return nn.Sequential(*layers)
    
    def forward_branch1(self, x):
        D = x
        B, T, H, W = x.shape
        L = torch.median(D, dim=1, keepdim=True).values.repeat(1, D.shape[1], 1, 1).cuda()
        M = torch.where(torch.abs(x - L) > 0.05, torch.ones_like(x), torch.zeros_like(x))
        for layer in self.branch1: 
            (D, L, M),foreground = layer((D, L, M),self.forward_mode1)
            if not self.training:
                self.foregrounds1.append(foreground)
        return L, None, M
     
    def forward_branch2(self, x):
        D = x
        B, T, H, W = x.shape
        L = torch.median(D, dim=1, keepdim=True).values.repeat(1, D.shape[1], 1, 1).cuda()
        M = torch.where(torch.abs(x - L) > 0.05, torch.ones_like(x), torch.zeros_like(x))
        for layer in self.branch2: 
            (D, L, M),foreground = layer((D, L, M),self.forward_mode2)
            if not self.training:
                self.foregrounds2.append(foreground)
        return L, None, M
    
    def forward_branch3(self, x):
        D = x
        B, T, H, W = x.shape
        L = torch.median(D, dim=1, keepdim=True).values.repeat(1, D.shape[1], 1, 1).cuda()
        M = torch.where(torch.abs(x - L) > 0.05, torch.ones_like(x), torch.zeros_like(x))
        for layer in self.branch3: 
            (D, L, M),foreground = layer((D, L, M),self.forward_mode3)
            if not self.training:
                self.foregrounds3.append(foreground)
        return L, None, M
    
    def forward_serial(self, x):
        D = x
        B, T, H, W = x.shape
        L = torch.median(D, dim=1, keepdim=True).values.repeat(1, D.shape[1], 1, 1).cuda()
        M = torch.where(torch.abs(x - L) > 0.05, torch.ones_like(x), torch.zeros_like(x))
        forward_modes = [self.forward_mode1, self.forward_mode2, self.forward_mode3]
        for i, layer in enumerate(self.branch1):
            (D, L, M), foreground = layer((D, L, M), forward_modes[i])
            if not self.training:
                self.foregrounds1.append(foreground)
        return L, None, M
    
    def forward_serialx(self, x):
        D = x
        B, T, H, W = x.shape
        L = torch.median(D, dim=1, keepdim=True).values.repeat(1, T, 1, 1).cuda()
        M = torch.where(torch.abs(x - L) > 0.05, torch.ones_like(x), torch.zeros_like(x))

        # Pass through branch1
        for layer in self.branch1:
            (D, L, M), foreground = layer((D, L, M), self.forward_mode1)
            if not self.training:
                self.foregrounds1.append(foreground)

        # Pass through branch2
        for layer in self.branch2:
            (D, L, M), foreground = layer((D, L, M), self.forward_mode2)
            if not self.training:
                self.foregrounds2.append(foreground)

        # Pass through branch3
        for layer in self.branch3:
            (D, L, M), foreground = layer((D, L, M), self.forward_mode3)
            if not self.training:
                self.foregrounds3.append(foreground)

        return L, None, M

    
    def forward_ensemble(self, x):
        L_branch1, _, M_branch1 = self.forward_branch1(x)
        L_branch2, _, M_branch2 = self.forward_branch2(x)
        L_branch3, _, M_branch3 = self.forward_branch3(x)

        M_weighted = self.w_branch1.unsqueeze(0).unsqueeze(0) * M_branch1 + self.w_branch2.unsqueeze(0).unsqueeze(0) * M_branch2 + self.w_branch3.unsqueeze(0).unsqueeze(0) * M_branch3
        M_weighted = M_weighted.clamp(min=0, max=1)
        return L_branch1, None, M_weighted

    def forward_ensemble2(self, x):
        L_branch1, _, M_branch1 = self.forward_branch1(x)
        L_branch2, _, M_branch2 = self.forward_branch2(x)
        L_branch3, _, M_branch3 = self.forward_branch3(x)

        weight_matrix_M_branch1 = torch.matmul(self.u_M_branch1, self.v_M_branch1.T)
        weight_matrix_M_branch2 = torch.matmul(self.u_M_branch2, self.v_M_branch2.T)
        weight_matrix_M_branch3 = torch.matmul(self.u_M_branch3, self.v_M_branch3.T)


        # Apply depthwise softmax to normalize the weight matrices
        softmaxed_weights_M = self.depthwise_softmax(weight_matrix_M_branch1, weight_matrix_M_branch2, weight_matrix_M_branch3)

        # Use the softmaxed weight matrices to weight the L outputs
        M_weighted = softmaxed_weights_M[0].unsqueeze(0).unsqueeze(0) * M_branch1 + softmaxed_weights_M[1].unsqueeze(0).unsqueeze(0) * M_branch2 + softmaxed_weights_M[2].unsqueeze(0).unsqueeze(0) * M_branch3
        M_weighted = M_weighted.clamp(min=0, max=1)
        return L_branch1, None, M_weighted
    
    def freeze_madu_layers(self):
        # Freeze weights in all madu layer instances
        for layer in self.branch1:
            layer.freeze_weights()
        for layer in self.branch2:
            layer.freeze_weights()
        for layer in self.branch3:
            layer.freeze_weights()

    def unfreeze_madu_layers(self):
        # Unfreeze weights in all madu layer instances
        for layer in self.branch1:
            layer.unfreeze_weights()
        for layer in self.branch2:
            layer.unfreeze_weights()
        for layer in self.branch3:
            layer.unfreeze_weights()




