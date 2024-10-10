#!/usr/bin/env python
# A wrapper that wraps the model with block-recurrence

# coding=utf8
# File: models.py
import sys
import random
import torch
import numpy
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from utils import ce_loss_mask, mse_loss_mask, img_pro, img_post

class BlockRecurrentWrapper(nn.Module):
    """
    Wrapping a temporal modeler with a memory cache to make it block-recurrent
    """
    def __init__(self, temporal_module, memory_length, memory_type='kv'):
        """
        Memory_Type: "kv", "mem"
        """
        super().__init__()

        self.clear_memory()
        self.temporal_module = temporal_module
        self.mem_len = memory_length
        self.memory_type = memory_type.lower()

    def clear_memory(self):
        self.memory = None
        
    def merge_memory_in_cache(self, cache):
        if(self.memory_type == "kv"):
            if(cache is not None and self.memory is not None):
                new_cache = []
                for mem, ca in zip(self.memory, cache):
                    new_cache.append(torch.cat((mem, ca), dim=1))
            elif(self.memory is not None):
                new_cache = self.memory
            elif(cache is not None):
                new_cache = cache
            else:
                new_cache = None
            return new_cache
        elif(self.memory_type == "mem"):
            if(cache is not None):
                new_cache = cache
            else:
                new_cache = self.memory
            return new_cache

    def update_memory_cache(self, cache):
        # Updates the Memory and Cache
        # For KV cache, in case the memory + cache > 2 * memory_length, we update the memory
        # Else, we keep the cache and the memory
        if(self.memory_type == "kv"):
            new_cache = self.merge_memory_in_cache(cache)
            self.memory = [c[:, -self.mem_len:] for c in new_cache]
            new_cache = None
        elif(self.memory_type == "mem"):
            # Just update the memory and the cache
            self.memory = []
            for l_cache in cache:
                mem = ()
                for lt_cache in l_cache:
                    mem += (lt_cache.detach(),)
                self.memory.append(mem)
            new_cache = cache
        else:
            raise Exception(f"No such memory type: {self.memory_type}")
    
        return new_cache
            
    def forward(self, src, cache=None, need_cache=False, verbose=True, checkpoints_density=-1):
        output, new_cache = self.temporal_module.forward(
                src, 
                cache=self.merge_memory_in_cache(cache), 
                need_cache=need_cache, 
                checkpoints_density=checkpoints_density)
        new_cache = self.update_memory_cache(new_cache)
        return output, new_cache

if __name__=="__main__":
    from utils import Configure
    config=Configure()
    config.from_yaml(sys.argv[1])

    inputs = torch.randint(0, 1024, (4, 64))
    ART2 = ARTransformerEncoder()
    out_nlp, cache = ART2(inputs, need_cache=True)
    out_nlp2, cache = ART2(inputs, cache=cache, need_cache=True)
    print(out_nlp.shape, out_nlp2.shape)
