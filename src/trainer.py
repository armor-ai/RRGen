import torch
import torch.nn as nn
from torch.autograd import Variable
from embedding import BOS
import torch.nn.functional as F
from parameter import opts
import pandas as pd
from util import detach_hidden, masked_cross_entropy, USE_CUDA


def compute_grad_norm(parameters, norm_type=2):
    """ Ref: http://pytorch.org/docs/0.3.0/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm
    """
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    return total_norm


def train(src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens, rate_sents, cate_sents, senti_sents, encoder, decoder, encoder_optim, decoder_optim, opts):
    # -------------------------------------
    # Prepare input and output placeholders
    # -------------------------------------
    # Last batch might not have the same size as we set to the `batch_size`
    batch_size = src_seqs.size(1)
    assert (batch_size == tgt_seqs.size(1))

    # Pack tensors to variables for neural network inputs (in order to autograd)
    src_seqs = Variable(src_seqs)
    tgt_seqs = Variable(tgt_seqs)
    # Categorize seq lengths
    if opts.use_sent_len:
        bins = range(0,opts.max_seq_len+10,opts.use_sent_len)
        src_len_cates = list(pd.cut(src_lens, bins, labels=range(len(bins)-1)))
        src_len_cates = torch.tensor(src_len_cates, dtype=torch.long)
    else:
        src_len_cates = None
    src_lens = Variable(torch.LongTensor(src_lens))
    tgt_lens = Variable(torch.LongTensor(tgt_lens))

    # Decoder's input
    input_seq = Variable(torch.LongTensor([BOS] * batch_size))
    rate_sents = torch.tensor(rate_sents, dtype=torch.long)
    cate_sents = torch.tensor(cate_sents, dtype=torch.long)
    senti_sents = torch.tensor(senti_sents, dtype=torch.long)

    # Decoder's output sequence length = max target sequence length of current batch.
    max_tgt_len = tgt_lens.data.max()

    # Store all decoder's outputs.
    # **CRUTIAL**
    # Don't set:
    # >> decoder_outputs = Variable(torch.zeros(max_tgt_len, batch_size, decoder.vocab_size))
    # Varying tensor size could cause GPU allocate a new memory causing OOM,
    # so we intialize tensor with fixed size instead:
    # `opts.max_seq_len` is a fixed number, unlike `max_tgt_len` always varys.
    decoder_outputs = Variable(torch.zeros(opts.max_seq_len, batch_size, decoder.vocab_size))

    # Move variables from CPU to GPU.
    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        tgt_seqs = tgt_seqs.cuda()
        src_lens = src_lens.cuda()
        tgt_lens = tgt_lens.cuda()
        input_seq = input_seq.cuda()
        rate_sents = rate_sents.cuda()
        cate_sents = cate_sents.cuda()
        if opts.use_sent_len:
            src_len_cates = src_len_cates.cuda()
        senti_sents = senti_sents.cuda()
        decoder_outputs = decoder_outputs.cuda()

    # -------------------------------------
    # Training mode (enable dropout)
    # -------------------------------------
    encoder.train()
    decoder.train()

    # -------------------------------------
    # Zero gradients, since optimizers will accumulate gradients for every backward.
    # -------------------------------------
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()

    # -------------------------------------
    # Forward encoder
    # -------------------------------------
    encoder_outputs, encoder_hidden = encoder(src_seqs, src_lens.data.tolist())

    # -------------------------------------
    # Forward decoder
    # -------------------------------------
    # Initialize decoder's hidden state as encoder's last hidden state.
    decoder_hidden = encoder_hidden
    if USE_CUDA:
        decoder_hidden = decoder_hidden.cuda()

    # Run through decoder one time step at a time.
    for t in range(max_tgt_len):
        # decoder returns:
        # - decoder_output   : (batch_size, vocab_size)
        # - decoder_hidden   : (num_layers, batch_size, hidden_size)
        # - attention_weights: (batch_size, max_src_len)
        decoder_output, decoder_hidden, attention_weights = decoder(input_seq, decoder_hidden, encoder_outputs, src_lens, rate_sents, cate_sents, src_len_cates, senti_sents)

        # Store decoder outputs.
        decoder_outputs[t] = decoder_output

        # Next input is current target
        input_seq = tgt_seqs[t]

        # Detach hidden state:
        detach_hidden(decoder_hidden)

    # -------------------------------------
    # Compute loss
    # -------------------------------------
    loss, pred_seqs, num_corrects, num_words = masked_cross_entropy(
        decoder_outputs[:max_tgt_len].transpose(0, 1).contiguous(),
        tgt_seqs.transpose(0, 1).contiguous(),
        tgt_lens
    )

    pred_seqs = pred_seqs[:max_tgt_len]

    # -------------------------------------
    # Backward and optimize
    # -------------------------------------
    # Backward to get gradients w.r.t parameters in model.
    loss.backward()

    # Clip gradients
    encoder_grad_norm = nn.utils.clip_grad_norm(encoder.parameters(), opts.max_grad_norm)
    decoder_grad_norm = nn.utils.clip_grad_norm(decoder.parameters(), opts.max_grad_norm)
    clipped_encoder_grad_norm = compute_grad_norm(encoder.parameters())
    clipped_decoder_grad_norm = compute_grad_norm(decoder.parameters())

    # Update parameters with optimizers
    encoder_optim.step()
    decoder_optim.step()

    return loss.data.item(), pred_seqs, attention_weights, num_corrects, num_words, \
           encoder_grad_norm, decoder_grad_norm, clipped_encoder_grad_norm, clipped_decoder_grad_norm