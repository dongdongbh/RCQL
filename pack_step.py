#!/usr/bin/env python3


import torch
from torch import nn
import copy
from torch.nn import DataParallel
import torch.nn.functional as F
import numpy as np


def pack_step(modules, state, h_caches, problem_params):
    actor_modules = modules['actor']

    actor_encoder_out, h_caches[0] = actor_modules['encoder'](state.packed_state, h_caches[0])
    if not state.online:
        # (batch, block, 1)
        s_out = actor_modules['s_decoder'](state.boxes, actor_encoder_out)

        select_mask = state.get_mask()
#         print(state.boxes, state.packed_state)
        s_log_p, selected = _select_step(s_out.squeeze(-1), select_mask)

    else:
        selected = torch.zeros(state.packed_state.size(0), device=state.packed_state.device)
        s_log_p = 0

    # select (batch)
    state.update_select(selected)
    # (batch, 2)
    q_rotation = state.action.get_shape().unsqueeze(1)

    r_out = actor_modules['r_decoder'](q_rotation, actor_encoder_out).squeeze(1)

    r_log_p, rotation = _rotate_step(r_out.squeeze(-1))

    # rotation
    state.update_rotate(rotation)

    

    if problem_params['problem_type'] == 'pack2d':
        p_position = state.action.get_shape().unsqueeze(1)
        
        if not problem_params['no_query']:
    
            p_out = actor_modules['p_decoder'](p_position, actor_encoder_out).squeeze(1)

        else:
            
            p_out = actor_modules['p_decoder'](q_rotation, actor_encoder_out).squeeze(1)

        x_log_p, box_xs = _drop_step(p_out.squeeze(-1), state.get_boundx())

        value, h_caches[1] = modules['critic'](state.boxes, state.packed_state, h_caches[1])
        value = value.squeeze(-1)
        # update location and finish one step packing
        state.update_pack(box_xs)

        return s_log_p, r_log_p, x_log_p, value, h_caches
    else:
        
        p_position = state.action.get_shape().unsqueeze(1)
        q_position = state.action.get_shape().unsqueeze(1)

        if not problem_params['no_query']:

            p_out = actor_modules['p_decoder'](p_position, actor_encoder_out).squeeze(1)
            q_out = actor_modules['q_decoder'](q_position, actor_encoder_out).squeeze(1)
        else:

            p_out = actor_modules['p_decoder'](q_rotation, actor_encoder_out).squeeze(1)
            q_out = actor_modules['q_decoder'](q_rotation, actor_encoder_out).squeeze(1)

        x_log_p, box_xs = _drop_step(p_out.squeeze(-1), state.get_boundx())
        y_log_p, box_ys = _drop_step(q_out.squeeze(-1), state.get_boundy())

        value, h_caches[1] = modules['critic'](state.boxes, state.packed_state, h_caches[1])
        value = value.squeeze(-1)

        state.update_pack(box_xs, box_ys)

        return s_log_p, r_log_p, x_log_p, y_log_p, value, h_caches




def _select_step(s_logits, mask):

    s_logits = s_logits.masked_fill(mask, -np.inf)

    s_log_p = F.log_softmax(s_logits, dim=-1)

    # (batch)
    selected = _select(s_log_p.exp()).unsqueeze(-1)

    # do not reinforce masked and avoid entropy become nan
    s_log_p = s_log_p.masked_fill(mask, 0)

    return s_log_p, selected


def _rotate_step(r_logits):

    r_log_p = F.log_softmax(r_logits, dim=-1)

    # rotate (batch, 1)
    rotate = _select(r_log_p.exp()).unsqueeze(-1)
    
    return r_log_p, rotate

def _drop_step(p_logits, right_bound):

    batch_size, p_options = p_logits.size()
    # (-1, 1) ---->(0, DISCRETE_XNUM) (batch, 1)
    right_b = ((right_bound + 1.0) * (p_options/2)).floor().long()

    bound_range = torch.arange(p_options, device=p_logits.device).unsqueeze(0)
    bound_range = bound_range.repeat(batch_size, 1)

    # bound_mask (batch, DISCRETE_XNUM)
    bound_mask = bound_range.gt(right_b.unsqueeze(-1))

    x_logits_masked = p_logits.masked_fill(bound_mask, -np.inf)
    # (batch, DISCRETE_XNUM)
    x_log_p = F.log_softmax(x_logits_masked, dim=-1)
    assert not torch.isnan(x_log_p).any()

    # (batch, 1)
    x_selects = _select(x_log_p.exp()).unsqueeze(1)

    # do not reinforce masked
    x_log_p = x_log_p.masked_fill(bound_mask, 0)
    
    box_xs = x_selects.float()/(p_options/2) - 1.0
    
    # test continuous and discrete conversion
    # test = ((box_xs + 1.0) * (p_options/2)).round().long()
    # assert test.eq(x_selects).all(), "conversion error!"
    
    return x_log_p, box_xs


def _select(probs, decode_type="sampling"):
    assert (probs == probs).all(), "Probs should not contain any nans"
    
    if decode_type == "greedy":
        _, selected = probs.max(-1)
    elif decode_type == "sampling":
        selected = probs.multinomial(1).squeeze(1)
    
    else:
        assert False, "Unknown decode type"
        
    return selected


