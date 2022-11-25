import torch
import torch.nn as nn
from torch.nn import functional as F

from ..xutilities.nn import Module as MainModule

class GRU(MainModule):
    def __init__(self, num_layers=None, feat_dim=None, inp_dim=None, out_dim=None, meta_enabled=False, meta_out_dim=None):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.feat_dim = feat_dim
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.meta_enabled = meta_enabled
        self.meta_out_dim = meta_out_dim
        
        self.in_linear = nn.Linear(inp_dim, feat_dim)
        self.gru = nn.GRU(feat_dim, feat_dim, num_layers = num_layers, batch_first=True, bidirectional=True)
        self.out_linear = nn.Linear(feat_dim*2, out_dim)

        if meta_enabled:
            self.meta_gru = nn.GRU(input_size=out_dim, hidden_size=256, bidirectional=True, batch_first=True)
            self.meta_linear = nn.Linear(256*2, meta_out_dim)
        
    def forward(self, inp, *args, **kwargs):
        result = {}
        # inp : B D L
        inp = inp.permute(0, 2, 1) # B L D
        inp = self.in_linear(inp)
        out, _ = self.gru(inp)
        out = self.out_linear(out) # B L D
        out = out.permute(0, 2, 1) # B D L
        
        result['output'] = out

        if self.meta_enabled:
            out_, _ = self.meta_gru(out.permute(0, 2, 1))
            result['meta_output'] = self.meta_linear(out_[:,-1,:])
        
        return result