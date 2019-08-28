import sys
# sys.path.append('/data/d0/gds/cygao/python/site-packages/lib/')

import torch
from embedding import Dataset, args, collate_fn, Embedding, PAD
from torch.utils.data import DataLoader
from encoder import EncoderRNN
from decoder import LuongAttnDecoderRNN
from parameter import opts, LOAD_CHECKPOINT, pred_test
from util import USE_CUDA, get_gpu_memory_usage, write_to_tensorboard
from torch import optim
from tqdm import tqdm
from trainer import train
from checkpoint import save_checkpoint
from evaluator import evaluate
import time
import numpy as np
import pickle
from translate import translate
from tester import _load_test_data, _valid_test

if LOAD_CHECKPOINT:
    from parameter import checkpoint

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


start_time = time.time()
### parameter
max_vocab_size = opts.max_vocab_size #50
embedding_size = opts.word_vec_size
batch_size = opts.batch_size
feature_size = opts.feature_vec_size


train_dataset = Dataset(data_file=args.train_filename, max_vocab_size=max_vocab_size)
valid_dataset = Dataset(data_file=args.valid_filename, src_vocab=train_dataset.src_vocab, tgt_vocab=train_dataset.tgt_vocab, rate_vocab=train_dataset.rate_vocab,       cate_vocab=train_dataset.cate_vocab,
                        senti_vocab=train_dataset.senti_vocab)


### Initialize embedding
src_vocab_size = len(train_dataset.src_vocab.token2id)
tgt_vocab_size = len(train_dataset.tgt_vocab.token2id)
if opts.use_keyword:
    src_embedding = torch.nn.Embedding(src_vocab_size, embedding_size + args.feature_vec_size, padding_idx=PAD)
    tgt_embedding = torch.nn.Embedding(tgt_vocab_size, embedding_size + args.feature_vec_size, padding_idx=PAD)
else:
    src_embedding = torch.nn.Embedding(src_vocab_size, embedding_size, padding_idx=PAD)
    tgt_embedding = torch.nn.Embedding(tgt_vocab_size, embedding_size, padding_idx=PAD)
### Initialize embeddings of external features
if opts.use_sent_rate:
    ext_rate_size = len(train_dataset.rate_vocab.token2id)
    ext_rate_emebdding = torch.nn.Embedding(ext_rate_size, feature_size)
else:
    ext_rate_emebdding = None

if opts.use_app_cate:
    ext_appcate_size = len(train_dataset.cate_vocab.token2id)
    ext_appcate_embedding = torch.nn.Embedding(ext_appcate_size, feature_size)
else:
    ext_appcate_embedding = None

if opts.use_sent_len:
    ext_seqlen_embedding = torch.nn.Embedding(opts.use_sent_len, feature_size)
else:
    ext_seqlen_embedding = None

if opts.use_sent_senti:
    ext_senti_size = len(train_dataset.senti_vocab.token2id)
    ext_senti_embedding = torch.nn.Embedding(ext_senti_size, feature_size)
else:
    ext_senti_embedding = None

# (torch.tensor([train_dataset.rate_vocab.token2id[i] for i in train_dataset.rate_sents], dtype=torch.long))
# tgt_embedding.weight = src_embedding.weight




train_iter = DataLoader(dataset=train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=4,
                        collate_fn=collate_fn)
valid_iter = DataLoader(dataset=valid_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=4,
                        collate_fn=collate_fn)


### Initialize model
encoder = EncoderRNN(embedding=src_embedding,
                     rnn_type=opts.rnn_type,
                     hidden_size=opts.hidden_size,
                     num_layers=opts.num_layers,
                     dropout=opts.dropout,
                     bidirectional=opts.bidirectional)

decoder = LuongAttnDecoderRNN(encoder, embedding=tgt_embedding,
                              attention=opts.attention,
                              tie_embeddings=opts.tie_embeddings,
                              dropout=opts.dropout,
                              tie_ext_feature=opts.tie_ext_feature,
                              ext_rate_embedding=ext_rate_emebdding,
                              ext_appcate_embedding=ext_appcate_embedding, ext_seqlen_embedding=ext_seqlen_embedding, ext_senti_embedding=ext_senti_embedding
                              )

print("emb start")
if opts.pretrained_embeddings:
    if opts.use_keyword:
        train_embedding = Embedding(filename=args.glove_filename, embedding_size=embedding_size).load_embedding_plus_keyword(train_dataset.src_vocab)
        target_embedding = Embedding(filename=args.glove_filename, embedding_size=embedding_size).load_embedding_plus_keyword(
            train_dataset.tgt_vocab)
    else:
        train_embedding = Embedding(filename=args.glove_filename, embedding_size=embedding_size).load_embedding(train_dataset.src_vocab)
        target_embedding = Embedding(filename=args.glove_filename, embedding_size=embedding_size).load_embedding(train_dataset.tgt_vocab)
    encoder.embedding.weight.data.copy_(train_embedding)
    decoder.embedding.weight.data.copy_(target_embedding)
    if opts.fixed_embeddings:
        encoder.embedding.weight.requires_grad = False
        decoder.embedding.weight.requires_grad = False
    else:
        decoder.embedding.weight.requires_grad = True
print("emb end")
if LOAD_CHECKPOINT:
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
if USE_CUDA:
    encoder.cuda()
    decoder.cuda()
FINE_TUNE = True
if FINE_TUNE:
    encoder.embedding.weight.requires_grad = True

print('='*100)
print('Model log:\n')
print(encoder)
print(decoder)
print('- Encoder input embedding requires_grad={}'.format(encoder.embedding.weight.requires_grad))
print('- Decoder input embedding requires_grad={}'.format(decoder.embedding.weight.requires_grad))
print('- Decoder output embedding requires_grad={}'.format(decoder.W_s.weight.requires_grad))
print('='*100 + '\n')

# Initialize optimizers (we can experiment different learning rates)
encoder_optim = optim.Adam([p for p in encoder.parameters() if p.requires_grad], lr=opts.learning_rate, weight_decay=opts.weight_decay)
decoder_optim = optim.Adam([p for p in decoder.parameters() if p.requires_grad], lr=opts.learning_rate, weight_decay=opts.weight_decay)




if not pred_test:
    # Start training
    from datetime import datetime
    # from tensorboardX import SummaryWriter
    # --------------------------
    # Configure tensorboard
    # --------------------------
    print("Start training......")
    model_name = 'seq2seq'
    datetime = ('%s' % datetime.now()).split('.')[0].replace(' ', '_')
    experiment_name = '{}_{}'.format(model_name, datetime)
    tensorboard_log_dir = './tensorboard-logs/{}/'.format(experiment_name)
    # writer = SummaryWriter(tensorboard_log_dir)

    # --------------------------
    # Configure training
    # --------------------------
    num_epochs = opts.num_epochs
    print_every_step = opts.print_every_step
    save_every_step = opts.save_every_step
    # For saving checkpoint and tensorboard
    global_step = 0 if not LOAD_CHECKPOINT else checkpoint['global_step']

    # --------------------------
    # Load test data
    # --------------------------
    test_src_texts, test_tgt_texts, test_rate_sents, test_cate_sents, test_senti_sents = _load_test_data(args.test_filename, train_dataset)

    # --------------------------
    # Start training
    # --------------------------
    total_loss = 0
    total_corrects = 0
    total_words = 0
    prev_gpu_memory_usage = 0
    max_accuracy = 0.0
    max_bleu = 0.0
    save_count = 0  ## Save attention weights to file

    for epoch in range(num_epochs):
        for batch_id, batch_data in tqdm(enumerate(train_iter)):

            # Unpack batch data
            src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens, rate_sents, cate_sents, senti_sents = batch_data

            # Ignore batch if there is a long sequence.
            max_seq_len = max(src_lens + tgt_lens)
            if max_seq_len > opts.max_seq_len:
                print('[!] Ignore batch: sequence length={} > max sequence length={}'.format(max_seq_len, opts.max_seq_len))
                continue

            # Train.
            loss, pred_seqs, attention_weights, num_corrects, num_words, \
            encoder_grad_norm, decoder_grad_norm, clipped_encoder_grad_norm, clipped_decoder_grad_norm \
                = train(src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens, rate_sents, cate_sents, senti_sents, encoder, decoder, encoder_optim, decoder_optim, opts)

            # Statistics.
            global_step += 1
            total_loss += loss
            total_corrects += num_corrects
            total_words += num_words
            total_accuracy = 100 * np.divide(total_corrects, float(total_words))

            # Save checkpoint.
            # if global_step % save_every_step == 0:
            #     if total_accuracy > max_accuracy:
            #         max_accuracy = total_accuracy
            #         checkpoint_path = save_checkpoint(opts, experiment_name, encoder, decoder, encoder_optim, decoder_optim,
            #                                       total_accuracy, total_loss, global_step)
            #
            #         print('=' * 100)
            #         print('Save checkpoint to "{}".'.format(checkpoint_path))
            #         print('=' * 100 + '\n')

            # Print statistics and write to Tensorboard.
            if global_step % print_every_step == 0:
                curr_gpu_memory_usage = get_gpu_memory_usage(device_id=torch.cuda.current_device())
                diff_gpu_memory_usage = curr_gpu_memory_usage - prev_gpu_memory_usage
                prev_gpu_memory_usage = curr_gpu_memory_usage

                print('=' * 100)
                print('Training log:')
                print('- Epoch: {}/{}'.format(epoch, num_epochs))
                print('- Global step: {}'.format(global_step))
                print('- Total loss: {}'.format(total_loss))
                print('- Total corrects: {}'.format(total_corrects))
                print('- Total words: {}'.format(total_words))
                print('- Total accuracy: {}'.format(total_accuracy))
                print('- Current GPU memory usage: {}'.format(curr_gpu_memory_usage))
                print('- Diff GPU memory usage: {}'.format(diff_gpu_memory_usage))
                print('=' * 100 + '\n')

                total_loss = 0
                total_corrects = 0
                total_words = 0


                # write_to_tensorboard(writer, global_step, total_loss, total_corrects, total_words, total_accuracy,
                #                      encoder_grad_norm, decoder_grad_norm, clipped_encoder_grad_norm,
                #                      clipped_decoder_grad_norm,
                #                      encoder, decoder,
                #                      gpu_memory_usage={
                #                          'curr': curr_gpu_memory_usage,
                #                          'diff': diff_gpu_memory_usage
                #                      })

                test_start_time = time.time()
                bleu_score, pls, attention_test_ws, out_texts = _valid_test(test_src_texts, test_tgt_texts, test_rate_sents, test_cate_sents,
                                              test_senti_sents, train_dataset, encoder, decoder, opts.max_seq_len)
                print("Test time cost is ", (time.time()-test_start_time)/3600, "hrs.")
                if bleu_score > max_bleu:
                    max_bleu = bleu_score

                    att_fw = open(os.path.join(opts.attention_res_fp, str(max_bleu) + '_' + str(save_count)), 'w')
                    pickle.dump(attention_test_ws, att_fw)
                    # for id_att, weights in enumerate(attention_test_ws):
                    #     pickle.dump(test_src_texts[id_att]+'**'+','.join([str(w) for w in weights]), att_fw)
                    att_fw.close()
                    text_fw = open(os.path.join(opts.outtext_fp, str(max_bleu) + '_' + str(save_count)), 'w')
                    for id_text, text in enumerate(out_texts):
                        text_fw.write(test_src_texts[id_text] + "**" + text + '\n')
                    text_fw.close()
                    save_count += 1
                print("max blue score is ", max_bleu, "pls is ", pls)
                print("Current period is ", (time.time()-start_time)/3600,"hrs.")
                if total_accuracy > max_accuracy:
                    max_accuracy = total_accuracy
                    print("max accuracy is ", max_accuracy)


                    # checkpoint_path = save_checkpoint(opts, experiment_name, encoder, decoder, encoder_optim, decoder_optim,
                    #                              total_accuracy, total_loss, global_step)

                    # print('=' * 100)
                    # print('Save checkpoint to "{}".'.format(checkpoint_path))
                    # print('=' * 100 + '\n')

                    # Start to evaluate valid set
                    # total_val_loss = 0
                    # total_val_corrects = 0
                    # total_val_words = 0
                    #
                    # for batch_valid_id, batch_valid_data in tqdm(enumerate(valid_iter)):
                    #     src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens, rate_sents, cate_sents, senti_sents = batch_valid_data
                    #
                    #     loss, pred_seqs, attention_weights, num_corrects, num_words \
                    #         = evaluate(src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens, rate_sents, cate_sents, senti_sents, encoder, decoder)
                    #
                    #     total_val_loss += loss
                    #     total_val_corrects += num_corrects
                    #     total_val_words += num_words
                    #     total_val_accuracy = 100 * np.divide(total_val_corrects, float(total_val_words))
                    #
                    # print('=' * 100)
                    # print('Validation log:')
                    # print('- Total loss: {}'.format(total_val_loss))
                    # print('- Total corrects: {}'.format(total_val_corrects))
                    # print('- Total words: {}'.format(total_val_words))
                    # print('- Total accuracy: {}'.format(total_val_accuracy))
                    # print('=' * 100 + '\n')
                    #
                    # total_val_loss = 0
                    # total_val_corrects = 0
                    # total_val_words = 0


            # Free memory
            del src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens, \
                loss, pred_seqs, attention_weights, num_corrects, num_words, \
                encoder_grad_norm, decoder_grad_norm, clipped_encoder_grad_norm, clipped_decoder_grad_norm








else:
    # --------------------------
    ### Start translation
    # --------------------------
    test_fr = open(args.test_filename)
    test_src_texts = []
    test_tgt_texts = []
    # external feature
    test_rates = []
    test_cates = []
    test_sentis = []
    for line in test_fr.readlines():
        terms = line.split('***')
        if len(terms) < 8:  # check term length
            continue
        senti_sent = terms[7].strip().split()
        try:
            if int(senti_sent[1]) * 1.5 + int(senti_sent[0]) < 0:
                test_sentis.append(senti_sent[1])
            else:
                test_sentis.append(senti_sent[0])
        except:
            continue
        test_cates.append(train_dataset.app_cate_dict[terms[0]])
        test_rates.append(terms[1])
        test_src_texts.append(terms[4])
        test_tgt_texts.append(terms[5])

    test_rate_sents = [train_dataset.rate_vocab.token2id[i] for i in test_rates]
    test_cate_sents = [train_dataset.cate_vocab.token2id[i] for i in test_cates]
    test_senti_sents = [train_dataset.senti_vocab.token2id[i] for i in test_sentis]

    # test_src_texts = [line.split('***')[4] for line in test_fr.readlines()]
    # test_tgt_texts = [ for line in test_fr.readlines()]
    print(len(test_src_texts), len(test_tgt_texts))
    out_texts = []
    for idx, src_text in tqdm(enumerate(test_src_texts)):
        _, out_text, _ = translate(src_text.strip(), test_rate_sents[idx], test_cate_sents[idx], test_senti_sents[idx], train_dataset, encoder, decoder, max_seq_len=opts.max_seq_len)
        out_texts.append(src_text.strip() + '***' +test_tgt_texts[idx].strip() +'***' +out_text+'\n')
        # if idx%100 == 0:
        #     print("already translate to %d th sentence." % idx)
        # print("> %s" % src_text)
        # print("= %s" % test_tgt_texts[idx])
        # print("< %s" % out_text)
        # if idx == 10:
        #     break

    from parameter import checkpoint_path
    fw_name = checkpoint_path.split('/')
    dir_name = os.path.join('/research/lyu1/cygao/workspace/data', 'pred', fw_name[-2])
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(os.path.join(dir_name, fw_name[-1].strip('.pt')), 'w') as f:
        f.writelines(out_texts)
    test_fr.close()