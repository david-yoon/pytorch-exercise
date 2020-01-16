#-*- coding: utf-8 -*-

import torch
import torch.nn as nn

'''
desc : apply luong attention to target vector with given condition

input :
   - key          : [batch, seq, embed]
   - query        : [batch, embed, 1],  if batch==1, it will be broadcasted
   - seq_mask     : [batch, seq]  valid:1, mask:0

output : 
   - weighted_sum : [batch, embed], weighted_sum of the key
   - norm_dot     : [batch, seq], attention weights
'''
def luong_attention( key, query, seq_mask ) :

    # compute similarity
    b_sim = torch.matmul(key, query)
    b_sim = torch.squeeze(b_sim, -1)   # [batch, max_seq]
    

    # weighted sum by using similarity (normalized)
    norm_b_sim = masked_softmax(b_sim, seq_mask)
    key_mul_norm = torch.mul(norm_b_sim.unsqueeze(-1), key)
    weighted_sum = torch.sum(key_mul_norm, dim=1)
    
    return weighted_sum, norm_b_sim
    
    
def masked_softmax(
    vector: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
    memory_efficient: bool = False,
    mask_fill_value: float = -1e32,
) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).to(dtype=torch.bool), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result
