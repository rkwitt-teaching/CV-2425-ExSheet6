"""Submission for exercise sheet 5

SUBMIT this file as submission_<STUDENTID>.py where
you replace <STUDENTID> with your student ID, e.g.,
submission_1234567.py
"""

import torch
import torch.nn as nn
from einops import repeat, rearrange
import numpy as np


# Exercise 6.1
class Encoder(nn.Module):
    def __init__(self, 
        in_channels: int = 3, 
        patch_size: int = 16, 
        emb_size: int = 768):
        
        super().__init__()
        
        self.conv_layer = # <--- YOUR CODE HERE --->
        
        
        # Uncomment and complete the line in case you solve Step 2
        #self.token = # <--- YOUR CODE HERE --->

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # <--- YOUR CODE HERE --->
       
       
        """
        Return the output 
            (1) AFTER convolution + reshaping, and 
            (2) AFTER concatenating the token to the reshaped output
            
        For instance:
        
        return out0, out1
        
        or 
        
        return out0, None
        """
        pass