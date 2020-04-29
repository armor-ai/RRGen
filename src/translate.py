import torch
from torch.autograd import Variable
from embedding import BOS, UNK, EOS, args
from util import USE_CUDA, detach_hidden
from parameter import opts
import pandas as pd


def translate(src_text, rate_sents, cate_sents, senti_sents, train_dataset, encoder, decoder, max_seq_len, replace_unk=True):
    # -------------------------------------
    # Prepare input and output placeholders
    # -------------------------------------
    # Like dataset's `__getitem__()` and dataloader's `collate_fn()`.
    src_sent = src_text.split()
    src_seqs = torch.LongTensor([train_dataset.tokens2ids(tokens=src_text.split(),token2id=train_dataset.src_vocab.token2id,append_BOS=False, append_EOS=True)]).transpose(0, 1)
    src_lens = [len(src_seqs)]
    if opts.use_sent_len:
        bins = range(0, opts.max_seq_len + 10, opts.use_sent_len)
        src_len_cates = list(pd.cut(src_lens, bins, labels=range(len(bins) - 1)))
        src_len_cates = torch.tensor(src_len_cates, dtype=torch.long)
    else:
        src_len_cates = None
    # Last batch might not have the same size as we set to the `batch_size`
    batch_size = src_seqs.size(1)

    # Pack tensors to variables for neural network inputs (in order to autograd)
    src_seqs = Variable(src_seqs, volatile=True)
    src_lens = Variable(torch.LongTensor(src_lens), volatile=True)


    # Decoder's input
    input_seq = Variable(torch.LongTensor([BOS] * batch_size), volatile=True)
    rate_sents = torch.tensor([rate_sents], dtype=torch.long)
    cate_sents = torch.tensor(cate_sents, dtype=torch.long)
    senti_sents = torch.tensor(senti_sents, dtype=torch.long)

    # Store output words and attention states
    out_sent = []
    all_attention_weights = torch.zeros(max_seq_len, len(src_seqs))

    # Move variables from CPU to GPU.
    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        src_lens = src_lens.cuda()
        rate_sents = rate_sents.cuda()
        input_seq = input_seq.cuda()
        if opts.use_sent_len:
            src_len_cates = src_len_cates.cuda()
        cate_sents = cate_sents.cuda()
        senti_sents = senti_sents.cuda()

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
    for t in range(max_seq_len):

        # decoder returns:
        # - decoder_output   : (batch_size, vocab_size)
        # - decoder_hidden   : (num_layers, batch_size, hidden_size)
        # - attention_weights: (batch_size, max_src_len)
        decoder_output, decoder_hidden, attention_weights = decoder(input_seq, decoder_hidden, encoder_outputs, src_lens, rate_sents, cate_sents, src_len_cates, senti_sents)

        # Store attention weights.
        # .squeeze(0): remove `batch_size` dimension since batch_size=1
        all_attention_weights[t] = attention_weights.squeeze(0).cpu().data

        # Choose top word from decoder's output
        prob, token_id = decoder_output.data.topk(1)
        token_id = token_id[0][0].item()  # get value
        if token_id == EOS:
            break
        else:
            token = ''
            if token_id == UNK and replace_unk:
                # Replace unk by selecting the source token with the highest attention score.
                score, idx = all_attention_weights[t].max(0)
                try:
                    # get value of scalar and minus 1 for zero-based indexing
                    token = src_sent[idx.item()-1]
                    # token = src_sent[idx[0]]
                except IndexError:
                    print('[!] Could not retrieve idx {} from {}'.format(idx.item()-1, str(src_sent)))
            else:
                # <UNK>
                try:
                    token = train_dataset.tgt_vocab.id2token[token_id]
                except KeyError:
                    print('token_id is ', token_id)

            if token:
                out_sent.append(token)

        # Next input is chosen word
        input_seq = Variable(torch.LongTensor([token_id]), volatile=True)
        if USE_CUDA: input_seq = input_seq.cuda()

        # Repackage hidden state (may not need this, since no BPTT)
        detach_hidden(decoder_hidden)

    src_text = ' '.join([train_dataset.src_vocab.id2token[token_id] for token_id in src_seqs.data.squeeze(1).tolist()])
    out_text = ' '.join(out_sent)

    # all_attention_weights: (out_len, src_len)
    return src_text, out_text, all_attention_weights[:len(out_sent)]

