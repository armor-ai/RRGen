# from annoy import AnnoyIndex     ## package to remove noise
import numpy as np
import torch
from tqdm import tqdm ## package to show progress
from argparse import Namespace
from collections import Counter
from ast import literal_eval
import json
from torch.autograd import Variable


args = Namespace(
    glove_filename='/home/cuiyun1/tasks/review_reply/data/glove.6B.100d.txt',
    test_filename='/home/cuiyun1/tasks/review_reply/data/test_label_n.txt',
    train_filename='/home/cuiyun1/tasks/review_reply/data/train_label_n.txt',
    valid_filename='/home/cuiyun1/tasks/review_reply/data/valid_label_n.txt',
    cate_filename='/home/cuiyun1/tasks/review_reply/data/app_category.txt',
    keyword_filename='/home/cuiyun1/tasks/review_reply/data/keyword_dict.json',
    feature_vec_size = 20
)


PAD = 0
BOS = 1
EOS = 2
UNK = 3

class AttrDict(dict):
    """ Access dictionary keys like attribute
        https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
    """
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self

class Dataset():
    def __init__(self, data_file, max_vocab_size=50, src_vocab=None, tgt_vocab=None, rate_vocab=None, cate_vocab=None, senti_vocab=None):
        self.max_vocab_size = max_vocab_size
        self.data_file = data_file
        self.app_cate_dict = self.load_cates(args.cate_filename)
        self.src_sents, self.tgt_sents, self.rate_sents, self.app_cates, self.senti_sents = self.load_sents(self.data_file)
        self.src_counter = self.build_counter(self.src_sents)
        self.tgt_counter = self.build_counter(self.tgt_sents)
        self.rate_counter = self.build_counter(self.rate_sents)
        self.cate_counter = self.build_cate_counter(self.app_cates)
        self.senti_counter = self.build_cate_counter(self.senti_sents)
        if not src_vocab:
            self.src_vocab = self.build_vocab(self.src_counter, self.max_vocab_size)
            self.tgt_vocab = self.build_vocab(self.tgt_counter, self.max_vocab_size)
            self.rate_vocab = self.build_rate_vocab(self.rate_counter)
            self.cate_vocab = self.build_rate_vocab(self.cate_counter)
            self.senti_vocab = self.build_rate_vocab(self.senti_counter)
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab
            self.rate_vocab = rate_vocab
            self.cate_vocab = cate_vocab
            self.senti_vocab = senti_vocab


    def __len__(self):
        return len(self.src_sents)

    def __getitem__(self, index):
        src_sent = self.src_sents[index]
        tgt_sent = self.tgt_sents[index]
        rate_sent = self.rate_vocab.token2id[self.rate_sents[index]]
        cate_sent = self.cate_vocab.token2id[self.app_cates[index]]
        senti_sent = self.senti_vocab.token2id[self.senti_sents[index]]
        src_seq = self.tokens2ids(src_sent, self.src_vocab.token2id, append_BOS=False, append_EOS=True)
        tgt_seq = self.tokens2ids(tgt_sent, self.tgt_vocab.token2id, append_BOS=False, append_EOS=True)

        return src_sent, tgt_sent, src_seq, tgt_seq, rate_sent, cate_sent, senti_sent

    def tokens2ids(self, tokens, token2id, append_BOS=True, append_EOS=True):
        seq = []
        if append_BOS: seq.append(BOS)
        seq.extend([token2id.get(token, UNK) for token in tokens])
        if append_EOS: seq.append(EOS)
        return seq

    def load_sents(self, fpath):
        src_sents = []
        tgt_sents = []
        rate_sents = []
        app_cates = []
        senti_sents = []
        with open(fpath) as file:
            for sent in tqdm(file.readlines()):
                terms = sent.split('***')
                if len(terms) < 8:    # check term length
                    continue
                # check whether the info complete
                senti_sent = terms[7].strip().split()
                try:
                    if int(senti_sent[1]) * 1.5 + int(senti_sent[0]) < 0:
                        senti_sents.append(senti_sent[1])
                    else:
                        senti_sents.append(senti_sent[0])
                except:
                    continue
                src_sent = terms[4]
                tgt_sent = terms[5]
                src_tokens = [token for token in src_sent.split()]
                tgt_tokens = [token for token in tgt_sent.split()]
                src_sents.append(src_tokens)
                tgt_sents.append(tgt_tokens)
                rate_sent = terms[1]
                rate_sents.append(rate_sent)    # np.eye(5)[np.subtract(rate_sent, 1)]
                app_cates.append(self.app_cate_dict[terms[0]])
        return src_sents, tgt_sents, rate_sents, app_cates, senti_sents

    def load_cates(self, catepath):
        cate_fr = open(catepath)
        app_cate_dict = literal_eval(cate_fr.readlines()[0])
        cate_fr.close()
        return app_cate_dict

    def build_cate_counter(self, sents):
        counter = Counter()
        for sent in tqdm(sents):
            counter[sent] += 1
        return counter

    def build_counter(self, sents):
        counter = Counter()
        for sent in tqdm(sents):
            counter.update(sent)
        return counter

    def build_vocab(self, counter, max_vocab_size):
        vocab = AttrDict()
        vocab.token2id = {'<PAD>': PAD, '<BOS>': BOS, '<EOS>': EOS, '<UNK>': UNK}
        vocab.token2id.update(
            {token: _id + 4 for _id, (token, count) in tqdm(enumerate(counter.most_common(max_vocab_size)))})
        vocab.id2token = {v: k for k, v in tqdm(vocab.token2id.items())}
        return vocab

    def build_rate_vocab(self, counter):
        ### Can be
        vocab = AttrDict()
        vocab.token2id = {token: _id for _id, (token, count) in tqdm(enumerate(counter.most_common()))}
        vocab.id2token = {v: k for k, v in tqdm(vocab.token2id.items())}
        return vocab


class Embedding():
    """
    Load embedding file
    """
    def __init__(self, filename, embedding_size):
        self.filename = filename
        self.embedding_size = embedding_size

    def load_word_vectors(self):
        word_to_index = {}
        word_vectors = []

        with open(self.filename) as fp:
            for line in tqdm(fp.readlines(), leave=False):
                line = line.split(" ")

                word = line[0]
                word_to_index[word] = len(word_to_index)

                vec = np.array([float(x) for x in line[1:]])
                word_vectors.append(vec)

        return word_to_index, word_vectors

    def load_keyword_embedding(self):
        keyword_fr = open(args.keyword_filename)
        keyword_dict = json.load(keyword_fr)
        keyword_fr.close()

        keyword_embedding = {}
        cates = list(keyword_dict.keys())
        ext_cate_embedding = torch.nn.Embedding(len(cates)+1, args.feature_vec_size)
        for cate, keywords in keyword_dict.items():
            for keyword in keywords:
                keyword_embedding[keyword] = ext_cate_embedding(Variable(torch.LongTensor([cates.index(cate)])))[0]

        keyword_embedding["_other_"] = ext_cate_embedding(Variable(torch.LongTensor([len(cates)])))[0]
        return keyword_embedding

    def load_embedding_plus_keyword(self, word_vocab):
        '''
        :param word_vocab: Word to index dictionary
        :param embedding_size: The size of word embedding
        :return:
        '''
        word_to_index, word_vectors = self.load_word_vectors()
        vocab_size = len(word_vocab.token2id)
        embedding = np.zeros((vocab_size, self.embedding_size + args.feature_vec_size))
        keyword_embedding = self.load_keyword_embedding()
        unk_count = 0
        for word, emb_index in tqdm(word_vocab.token2id.items()):
            if word == word_vocab.id2token[PAD]:
                continue
            elif word in [word_vocab.id2token[BOS], word_vocab.id2token[EOS], word_vocab.id2token[UNK]]:
                glove_vec = np.random.rand(self.embedding_size, )
                keyword_tensor = keyword_embedding["_other_"].data.numpy()
                embedding[emb_index, :] = np.concatenate((glove_vec, keyword_tensor), 0)
            elif word in word_to_index:
                glove_index = word_to_index[word.lower()]
                glove_vec = word_vectors[glove_index]  ##cuda.
                if word in keyword_embedding:
                    keyword_tensor = keyword_embedding[word].data.numpy()
                else:
                    keyword_tensor = keyword_embedding["_other_"].data.numpy()
                embedding[emb_index, :] = np.concatenate((glove_vec, keyword_tensor), 0)
            else:
                glove_vec = embedding[UNK]
                # keyword_tensor = keyword_embedding["_other_"].data.numpy()
                unk_count += 1
                embedding[emb_index, :] = glove_vec
        print('- Unknown word count: {}'.format(unk_count))
        print('=' * 100 + '\n')
        return torch.from_numpy(embedding).float()

    def load_embedding(self, word_vocab):
        '''
        :param word_vocab: Word to index dictionary
        :param embedding_size: The size of word embedding
        :return:
        '''
        word_to_index, word_vectors = self.load_word_vectors()
        vocab_size = len(word_vocab.token2id)
        embedding = np.zeros((vocab_size, self.embedding_size))
        unk_count = 0
        for word, emb_index in tqdm(word_vocab.token2id.items()):
            if word == word_vocab.id2token[PAD]:
                continue
            elif word in [word_vocab.id2token[BOS], word_vocab.id2token[EOS], word_vocab.id2token[UNK]]:
                embedding[emb_index, :] = np.random.rand(self.embedding_size, )
            elif word in word_to_index:
                glove_index = word_to_index[word.lower()]
                glove_vec = torch.FloatTensor(word_vectors[glove_index])  ##cuda.
                embedding[emb_index, :] = glove_vec
            else:
                embedding[emb_index, :] = embedding[UNK]
                unk_count += 1

            # embedding[index] = vector
            #
            #
            # if word in word_to_index:
            #     glove_index = word_to_index[word.lower()]
            #     glove_vec = torch.FloatTensor(word_vectors[glove_index])    ##cuda.
            #     embedding[emb_index, :]= glove_vec
            # elif emb_index == PAD:
            #     continue
            # elif emb_index in [BOS, EOS]:
            #     embedding[emb_index, :] = np.random.rand(self.embedding_size, )
            # else:
            #     unk_count += 1
            #     embedding[emb_index, :] = np.random.rand(self.embedding_size, )

        print('- Unknown word count: {}'.format(unk_count))
        print('=' * 100 + '\n')
        return torch.from_numpy(embedding).float()


def collate_fn(data):
    """
        Creates mini-batch tensors from (src_sent, tgt_sent, src_seq, tgt_seq).
        We should build a custom collate_fn rather than using default collate_fn,
        because merging sequences (including padding) is not supported in default.
        Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).

        Args:
            data: list of tuple (src_sents, tgt_sents, src_seqs, tgt_seqs)
            - src_sents, tgt_sents: batch of original tokenized sentences
            - src_seqs, tgt_seqs: batch of original tokenized sentence ids
        Returns:
            - src_sents, tgt_sents (tuple): batch of original tokenized sentences
            - src_seqs, tgt_seqs (variable): (max_src_len, batch_size)
            - src_lens, tgt_lens (tensor): (batch_size)
    """

    def _pad_sequences(seqs):
        lens = [len(seq) for seq in seqs]
        padded_seqs = torch.zeros(len(seqs), max(lens)).long()
        for i, seq in enumerate(seqs):
            end = lens[i]
            padded_seqs[i, :end] = torch.LongTensor(seq[:end])
        return padded_seqs, lens

    # Sort a list by *source* sequence length (descending order) to use `pack_padded_sequence`.
    # The *target* sequence is not sorted <-- It's ok, cause `pack_padded_sequence` only takes
    # *source* sequence, which is in the EncoderRNN
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # Seperate source and target sequences.
    src_sents, tgt_sents, src_seqs, tgt_seqs, rate_sents, cate_sents, senti_sents= zip(*data)

    # Merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lens = _pad_sequences(src_seqs)
    tgt_seqs, tgt_lens = _pad_sequences(tgt_seqs)

    # (batch, seq_len) => (seq_len, batch)
    src_seqs = src_seqs.transpose(0, 1)
    tgt_seqs = tgt_seqs.transpose(0, 1)

    return src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens, rate_sents, cate_sents, senti_sents

# if __name__ == "__main__":
#     max_vocab_size = 50
#     embedding_size = 50
#     train_dataset = Dataset(data_file=args.train_filename, max_vocab_size=max_vocab_size)
#     valid_dataset = Dataset(data_file=args.valid_filename, src_vocab=train_dataset.src_vocab, tgt_vocab=train_dataset.tgt_vocab)
#
#     batch_size = 48
#     train_iter = DataLoader(dataset=train_dataset,
#                             batch_size=batch_size,
#                             shuffle=True,
#                             num_workers=4,
#                             collate_fn=collate_fn)
#     valid_iter = DataLoader(dataset=valid_dataset,
#                             batch_size=batch_size,
#                             shuffle=False,
#                             num_workers=4,
#                             collate_fn=collate_fn)
#
#     src_vocab_size = len(train_dataset.src_vocab.token2id)
#     tgt_vocab_size = len(train_dataset.tgt_vocab.token2id)
#     src_embedding = torch.nn.Embedding(src_vocab_size, embedding_size, padding_idx=PAD)
#     tgt_embedding = torch.nn.Embedding(tgt_vocab_size, embedding_size, padding_idx=PAD)
#     tgt_embedding.weight = src_embedding.weight
#
#     train_embedding = Embedding(filename=args.glove_filename, embedding_size=embedding_size).load_embedding(train_dataset.src_vocab)
#


# class PreTrainedEmbeddings(object):
#     def __init__(self, glove_filename):
#         self.word_to_index, self.word_vectors = load_word_vectors(glove_filename)
#         self.word_vector_size = len(self.word_vectors[0])
#
#         self.index_to_word = {v: k for k, v in self.word_to_index.items()}
#         self.index = AnnoyIndex(self.word_vector_size, metric='euclidean')
#         print('Building Index')
#         for _, i in tqdm_notebook(self.word_to_index.items(), leave=False):
#             self.index.add_item(i, self.word_vectors[i])
#         self.index.build(50)
#         print('Finished!')
#
#     def get_embedding(self, word):
#         return self.word_vectors[self.word_to_index[word]]
#
#     def closest(self, word, n=1):
#         vector = self.get_embedding(word)
#         nn_indices = self.index.get_nns_by_vector(vector, n)
#         return [self.index_to_word[neighbor] for neighbor in nn_indices]
#
#     def closest_v(self, vector, n=1):
#         nn_indices = self.index.get_nns_by_vector(vector, n)
#         return [self.index_to_word[neighbor] for neighbor in nn_indices]
#
#     def sim(self, w1, w2):
#         return np.dot(self.get_embedding(w1), self.get_embedding(w2))
#
# glove = PreTrainedEmbeddings(args.glove_filename)
# print(glove.closest('apple', n=5))

