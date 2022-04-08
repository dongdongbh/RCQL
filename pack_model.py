#!/usr/bin/env python3


import torch
from torch import nn
from models import EncoderSeq, QDecoder



def get_tgt_entropy(problem_type, block_size, tgt_entropy, p_options):
    s_tgt_entropy = block_size * tgt_entropy / p_options
    if problem_type=='pack2d':
        r_tgt_entropy = 2 * tgt_entropy / p_options
        target_entropy = torch.tensor([s_tgt_entropy, r_tgt_entropy, tgt_entropy])
    elif problem_type=='pack3d':
        r_tgt_entropy = 6 * tgt_entropy / p_options
        target_entropy = torch.tensor([s_tgt_entropy, r_tgt_entropy, tgt_entropy, tgt_entropy])
    else:
        raise ValueError('Invalided problem type')

    print('target_entropy: ', target_entropy)
    return target_entropy



class PackDecoder(nn.Module):
    def __init__(self, head_hidden_size, res_size, state_size, hidden_size, decoder_layers, **kargs):
        nn.Module.__init__(self)

        self.att_decoder = QDecoder(state_size, hidden_size, decoder_nb_layers=decoder_layers, **kargs)

        self.head = nn.Sequential(
                            nn.Linear(hidden_size, head_hidden_size),
                            nn.ReLU(),
                            nn.Linear(head_hidden_size, res_size)
                            )


    def forward(self, x, embedding):
        h = self.att_decoder(x, embedding)
        out = self.head(h)
        return out

class Cirtic(nn.Module):
    def __init__(self, head_hidden_size, res_size, packed_state_size, box_state_size, hidden_size, c_encoder_layers, c_decoder_layers, **kargs):
        nn.Module.__init__(self)

        # add this parameter for entropy temp
        # 3D
        if packed_state_size==6:
            self.log_alpha = nn.Parameter(torch.tensor([-2.0,-2.0,-2.0, -2.0]))
        elif packed_state_size==4:
            self.log_alpha = nn.Parameter(torch.tensor([-2.0,-2.0,-2.0]))
        else:
            raise ValueError('Invalided problem type')


        self.att_encoder = EncoderSeq(
        state_size=packed_state_size,
        hidden_size=hidden_size,
        encoder_nb_layers=c_encoder_layers,
        **kargs)

        self.att_decoder = QDecoder(
            box_state_size,
            hidden_size,
            decoder_nb_layers=c_decoder_layers,
            **kargs)

        self.head = nn.Sequential(
                            nn.Linear(hidden_size, head_hidden_size),
                            nn.ReLU(),
                            nn.Linear(head_hidden_size, res_size)
                            )


    def forward(self, q, x, h_cache):

        embedding, h_cache = self.att_encoder(x, h_cache)
        h = self.att_decoder(q, embedding)  # B x Q x H
        h = h.mean(dim=1)                     # B x H
        out = self.head(h)
        return out, h_cache


def set_model(model, device, parallel=True):
    if parallel:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    return model


def get_ac_parameters(modules):

    critic_params = modules['critic'].parameters()

    actor_params = modules['actor'].parameters()

    return actor_params, critic_params





def build_model(
    device,
    problem_params,
    model_params,
    adapt_span_params):


    if problem_params['problem_type']=='pack2d':
        packed_state_size = 4
        box_state_size = 2
        rotate_out_size = 2
    elif problem_params['problem_type']=='pack3d':
        packed_state_size = 6
        box_state_size = 3
        rotate_out_size = 6
    else:
        raise ValueError('Invalided problem type')


    encoder = EncoderSeq(
        state_size=packed_state_size,
        encoder_nb_layers=model_params['encoder_layers'],
        **model_params,
        adapt_span_params=adapt_span_params)


    s_decoder = PackDecoder(head_hidden_size=model_params['head_hidden'],
        res_size=1,
        state_size=box_state_size,
        **model_params,
         adapt_span_params=adapt_span_params)

    r_decoder = PackDecoder(head_hidden_size=model_params['head_hidden'],
        res_size=rotate_out_size,
        state_size=box_state_size,
        **model_params,
        adapt_span_params=adapt_span_params)

    p_decoder = PackDecoder(head_hidden_size=model_params['head_hidden_pos'],
        res_size=problem_params['p_options'],
        state_size=box_state_size,
        **model_params,
        adapt_span_params=adapt_span_params)

    q_decoder = PackDecoder(head_hidden_size=model_params['head_hidden_pos'],
        res_size=problem_params['p_options'],
        state_size=box_state_size,
        **model_params,
        adapt_span_params=adapt_span_params)

    critic = Cirtic(head_hidden_size=model_params['head_hidden'],
        res_size=1,
        packed_state_size=packed_state_size,
        box_state_size=box_state_size,
        **model_params,
         adapt_span_params=adapt_span_params)


    encoder = set_model(encoder, device)
    s_decoder = set_model(s_decoder, device)
    r_decoder = set_model(r_decoder, device)
    p_decoder = set_model(p_decoder, device)
    q_decoder = set_model(q_decoder, device)
    critic = set_model(critic, device)

    if problem_params['problem_type'] == 'pack2d':

        actor_modules = nn.ModuleDict({
                    'encoder': encoder,
                    's_decoder': s_decoder,
                    'r_decoder': r_decoder,
                    'p_decoder': p_decoder}
                    )
    else:
        actor_modules = nn.ModuleDict({
                    'encoder': encoder,
                    's_decoder': s_decoder,
                    'r_decoder': r_decoder,
                    'p_decoder': p_decoder,
                    'q_decoder': q_decoder}
                    )


    packing_modules = nn.ModuleDict({
                'actor': actor_modules,
                'critic': critic
    })



    return packing_modules






