import torch
import torch.nn as nn
import torch.nn.functional as F
from util import sequence_mask


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, encoder, embedding=None, attention=True, bias=True, tie_embeddings=False, dropout=0.3, tie_ext_feature=False, ext_rate_embedding=None, ext_appcate_embedding=None, ext_seqlen_embedding=None, ext_senti_embedding=None):
        """ General attention in `Effective Approaches to Attention-based Neural Machine Translation`
            Ref: https://arxiv.org/abs/1508.04025

            Share input and output embeddings:
            Ref:
                - "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
                   https://arxiv.org/abs/1608.05859
                - "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
                   https://arxiv.org/abs/1611.01462
        """
        super(LuongAttnDecoderRNN, self).__init__()

        self.hidden_size = encoder.hidden_size * encoder.num_directions
        self.num_layers = encoder.num_layers
        self.dropout = dropout
        self.embedding = embedding
        self.attention = attention
        self.tie_embeddings = tie_embeddings
        self.tie_ext_feature = tie_ext_feature
        self.ext_rate_embedding = ext_rate_embedding
        self.ext_appcate_embedding = ext_appcate_embedding
        self.ext_seqlen_embedding = ext_seqlen_embedding
        self.ext_senti_embedding = ext_senti_embedding

        ### rate_size
        if self.ext_rate_embedding:
            self.ext_rate_size = self.ext_rate_embedding.embedding_dim
        else:
            self.ext_rate_size = 0
        ### appcate_size
        if self.ext_appcate_embedding:
            self.ext_appcate_size = self.ext_appcate_embedding.embedding_dim
        else:
            self.ext_appcate_size = 0
        ### seq_len
        if self.ext_seqlen_embedding:
            self.ext_seqlen_size = self.ext_seqlen_embedding.embedding_dim
        else:
            self.ext_seqlen_size = 0
        ### sentiment
        if self.ext_senti_embedding:
            self.ext_senti_size = self.ext_senti_embedding.embedding_dim
        else:
            self.ext_senti_size = 0
        self.vocab_size = self.embedding.num_embeddings
        self.word_vec_size = self.embedding.embedding_dim

        self.rnn_type = encoder.rnn_type
        self.rnn = getattr(nn, self.rnn_type)(
            input_size=self.word_vec_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout)

        if self.attention:
            self.W_a = nn.Linear(encoder.hidden_size * encoder.num_directions,
                                 self.hidden_size, bias=bias)
            self.W_c = nn.Linear(encoder.hidden_size * encoder.num_directions + self.hidden_size,
                                 self.hidden_size, bias=bias)

        if self.tie_embeddings:
            self.W_proj = nn.Linear(self.hidden_size, self.word_vec_size, bias=bias)
            self.W_s = nn.Linear(self.word_vec_size, self.vocab_size, bias=bias)
            self.W_s.weight = self.embedding.weight
        else:
            self.W_s = nn.Linear(self.hidden_size, self.vocab_size, bias=bias)

        input_size = self.hidden_size
        if self.tie_ext_feature:
            if self.ext_rate_embedding:
                input_size += self.ext_rate_size
            if self.ext_senti_embedding:
                input_size += self.ext_senti_size
            if self.ext_appcate_embedding:
                input_size += self.ext_appcate_size
            if self.ext_seqlen_embedding:
                input_size += self.ext_seqlen_size
        self.W_r = nn.Linear(input_size, self.hidden_size, bias=bias)
        # self.ext_rate_size+self.hidden_size+self.ext_appcate_size+self.ext_seqlen_size+self.ext_senti_size

    def forward(self, input_seq, decoder_hidden, encoder_outputs, src_lens, rate_sents, cate_sents, srclen_cates, senti_sents):
        """ Args:
            - input_seq      : (batch_size)
            - decoder_hidden : (t=0) last encoder hidden state (num_layers * num_directions, batch_size, hidden_size)
                               (t>0) previous decoder hidden state (num_layers, batch_size, hidden_size)
            - encoder_outputs: (max_src_len, batch_size, hidden_size * num_directions)

            Returns:
            - output           : (batch_size, vocab_size)
            - decoder_hidden   : (num_layers, batch_size, hidden_size)
            - attention_weights: (batch_size, max_src_len)
        """
        # (batch_size) => (seq_len=1, batch_size)
        input_seq = input_seq.unsqueeze(0)

        # (seq_len=1, batch_size) => (seq_len=1, batch_size, word_vec_size)
        emb = self.embedding(input_seq)

        # Add external embeddings: (batch_size, feature_size) => (num_layers, batch_size, feature_size
        if self.ext_rate_embedding:
            ext_rate_embedding = self.ext_rate_embedding(rate_sents)
            ext_rate_embedding = ext_rate_embedding.unsqueeze(0).repeat(self.num_layers, 1, 1)
        if self.ext_appcate_embedding:
            ext_appcate_embedding = self.ext_appcate_embedding(cate_sents)
            ext_appcate_embedding = ext_appcate_embedding.unsqueeze(0).repeat(self.num_layers, 1, 1)
        if self.ext_seqlen_embedding:
            ext_seqlen_embedding = self.ext_seqlen_embedding(srclen_cates)
            ext_seqlen_embedding = ext_seqlen_embedding.unsqueeze(0).repeat(self.num_layers, 1, 1)
        if self.ext_senti_embedding:
            ext_senti_embedding = self.ext_senti_embedding(senti_sents)
            ext_senti_embedding = ext_senti_embedding.unsqueeze(0).repeat(self.num_layers, 1, 1)

        # rnn returns:
        # - decoder_output: (seq_len=1, batch_size, hidden_size)
        # - decoder_hidden: (num_layers, batch_size, hidden_size)

        if self.tie_ext_feature:
            if self.ext_rate_embedding:
                decoder_hidden = torch.cat((decoder_hidden, ext_rate_embedding), 2)
            if self.ext_appcate_embedding:
                decoder_hidden = torch.cat((decoder_hidden, ext_appcate_embedding), 2)
            if self.ext_seqlen_embedding:
                decoder_hidden = torch.cat((decoder_hidden, ext_seqlen_embedding), 2)
            if self.ext_senti_embedding:
                decoder_hidden = torch.cat((decoder_hidden, ext_senti_embedding), 2)
            # decoder_hidden = torch.cat((decoder_hidden, ext_rate_embedding, ext_appcate_embedding, ext_seqlen_embedding, ext_senti_embedding), 2)
        decoder_hidden = F.tanh(self.W_r(decoder_hidden))
        decoder_output, decoder_hidden = self.rnn(emb, decoder_hidden)

        # (seq_len=1, batch_size, hidden_size) => (batch_size, seq_len=1, hidden_size)
        decoder_output = decoder_output.transpose(0, 1)

        """ 
        ------------------------------------------------------------------------------------------
        Notes of computing attention scores
        ------------------------------------------------------------------------------------------
        # For-loop version:

        max_src_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        attention_scores = Variable(torch.zeros(batch_size, max_src_len))

        # For every batch, every time step of encoder's hidden state, calculate attention score.
        for b in range(batch_size):
            for t in range(max_src_len):
                # Loung. eq(8) -- general form content-based attention:
                attention_scores[b,t] = decoder_output[b].dot(attention.W_a(encoder_outputs[t,b]))

        ------------------------------------------------------------------------------------------
        # Vectorized version:

        1. decoder_output: (batch_size, seq_len=1, hidden_size)
        2. encoder_outputs: (max_src_len, batch_size, hidden_size * num_directions)
        3. W_a(encoder_outputs): (max_src_len, batch_size, hidden_size)
                        .transpose(0,1)  : (batch_size, max_src_len, hidden_size) 
                        .transpose(1,2)  : (batch_size, hidden_size, max_src_len)
        4. attention_scores: 
                        (batch_size, seq_len=1, hidden_size) * (batch_size, hidden_size, max_src_len) 
                        => (batch_size, seq_len=1, max_src_len)
        """

        if self.attention:
            # attention_scores: (batch_size, seq_len=1, max_src_len)
            attention_scores = torch.bmm(decoder_output, self.W_a(encoder_outputs).transpose(0, 1).transpose(1, 2))

            # attention_mask: (batch_size, seq_len=1, max_src_len)
            attention_mask = sequence_mask(src_lens).unsqueeze(1)

            # Fills elements of tensor with `-float('inf')` where `mask` is 1.
            attention_scores.data.masked_fill_(1 - attention_mask.data, -float('inf'))

            # attention_weights: (batch_size, seq_len=1, max_src_len) => (batch_size, max_src_len) for `F.softmax`
            # => (batch_size, seq_len=1, max_src_len)
            try:  # torch 0.3.x
                attention_weights = F.softmax(attention_scores.squeeze(1), dim=1).unsqueeze(1)
            except:
                attention_weights = F.softmax(attention_scores.squeeze(1)).unsqueeze(1)

            # context_vector:
            # (batch_size, seq_len=1, max_src_len) * (batch_size, max_src_len, encoder_hidden_size * num_directions)
            # => (batch_size, seq_len=1, encoder_hidden_size * num_directions)
            context_vector = torch.bmm(attention_weights, encoder_outputs.transpose(0, 1))

            # concat_input: (batch_size, seq_len=1, encoder_hidden_size * num_directions + decoder_hidden_size)
            concat_input = torch.cat([context_vector, decoder_output], -1)

            # (batch_size, seq_len=1, encoder_hidden_size * num_directions + decoder_hidden_size) => (batch_size, seq_len=1, decoder_hidden_size)
            concat_output = F.tanh(self.W_c(concat_input))

            # Prepare returns:
            # (batch_size, seq_len=1, max_src_len) => (batch_size, max_src_len)
            attention_weights = attention_weights.squeeze(1)
        else:
            attention_weights = None
            concat_output = decoder_output

        # If input and output embeddings are tied,
        # project `decoder_hidden_size` to `word_vec_size`.
        if self.tie_embeddings:
            output = self.W_s(self.W_proj(concat_output))
        else:
            # (batch_size, seq_len=1, decoder_hidden_size) => (batch_size, seq_len=1, vocab_size)
            output = self.W_s(concat_output)

            # Prepare returns:
        # (batch_size, seq_len=1, vocab_size) => (batch_size, vocab_size)
        output = output.squeeze(1)

        del src_lens

        return output, decoder_hidden, attention_weights