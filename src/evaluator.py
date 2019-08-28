import torch
from util import masked_cross_entropy, detach_hidden, USE_CUDA
from parameter import opts
from embedding import BOS
from torch.autograd import Variable
import pandas as pd

def evaluate(src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens, rate_sents, cate_sents, senti_sents, encoder, decoder):
    # -------------------------------------
    # Prepare input and output placeholders
    # -------------------------------------
    # Last batch might not have the same size as we set to the `batch_size`
    batch_size = src_seqs.size(1)
    assert (batch_size == tgt_seqs.size(1))

    # Pack tensors to variables for neural network inputs (in order to autograd)
    src_seqs = Variable(src_seqs, volatile=True)
    tgt_seqs = Variable(tgt_seqs, volatile=True)
    # Categorize seq lengths
    if opts.use_sent_len:
        bins = range(0, opts.max_seq_len + 10, opts.use_sent_len)
        src_len_cates = list(pd.cut(src_lens, bins, labels=range(len(bins) - 1)))
        src_len_cates = Variable(torch.LongTensor(src_len_cates), volatile=True)
    else:
        src_len_cates = None

    src_lens = Variable(torch.LongTensor(src_lens), volatile=True)
    tgt_lens = Variable(torch.LongTensor(tgt_lens), volatile=True)
    rate_sents = Variable(torch.LongTensor(rate_sents), volatile=True)
    cate_sents = Variable(torch.LongTensor(cate_sents), volatile=True)
    senti_sents = Variable(torch.LongTensor(senti_sents), volatile=True)

    # Decoder's input
    input_seq = Variable(torch.LongTensor([BOS] * batch_size), volatile=True)

    # Decoder's output sequence length = max target sequence length of current batch.
    max_tgt_len = tgt_lens.data.max()

    # Store all decoder's outputs.
    # **CRUTIAL**
    # Don't set:
    # >> decoder_outputs = Variable(torch.zeros(max_tgt_len, batch_size, decoder.vocab_size))
    # Varying tensor size could cause GPU allocate a new memory causing OOM,
    # so we intialize tensor with fixed size instead:
    # `opts.max_seq_len` is a fixed number, unlike `max_tgt_len` always varys.
    decoder_outputs = Variable(torch.zeros(opts.max_seq_len, batch_size, decoder.vocab_size), volatile=True)

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
    # Evaluation mode (disable dropout)
    # -------------------------------------
    encoder.eval()
    decoder.eval()

    # -------------------------------------
    # Forward encoder
    # -------------------------------------
    encoder_outputs, encoder_hidden = encoder(src_seqs, src_lens.data.tolist())

    # -------------------------------------
    # Forward decoder
    # -------------------------------------
    # Initialize decoder's hidden state as encoder's last hidden state.
    decoder_hidden = encoder_hidden

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

        # Detach hidden state (may not need this, since no BPTT)
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

    return loss.data.item(), pred_seqs, attention_weights, num_corrects, num_words