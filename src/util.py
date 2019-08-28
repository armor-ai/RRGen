import torch
from torch.autograd import Variable
import torch.nn.functional as F
import subprocess


""" Enable GPU training """
USE_CUDA = torch.cuda.is_available()
print('Use_CUDA={}'.format(USE_CUDA))
if USE_CUDA:
    # You can change device by `torch.cuda.set_device(device_id)`
    torch.cuda.set_device(0)
    print('current_device={}'.format(torch.cuda.current_device()))

def sequence_mask(sequence_length, max_len=None):
    """
    Caution: Input and Return are VARIABLE.
    """
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    mask = seq_range_expand < seq_length_expand

    return mask


def masked_cross_entropy(logits, target, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.

    The code is same as:

    weight = torch.ones(tgt_vocab_size)
    weight[padding_idx] = 0
    criterion = nn.CrossEntropyLoss(weight.cuda(), size_average)
    loss = criterion(logits_flat, losses_flat)
    """
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    # Note: mask need to bed casted to float!
    losses = losses * mask.float()
    loss = losses.sum() / mask.float().sum()

    # (batch_size * max_tgt_len,)
    pred_flat = log_probs_flat.max(1)[1]
    # (batch_size * max_tgt_len,) => (batch_size, max_tgt_len) => (max_tgt_len, batch_size)
    pred_seqs = pred_flat.view(*target.size()).transpose(0, 1).contiguous()
    # (batch_size, max_len) => (batch_size * max_tgt_len,)
    mask_flat = mask.view(-1)

    # `.float()` IS VERY IMPORTANT !!!
    # https://discuss.pytorch.org/t/batch-size-and-validation-accuracy/4066/3
    num_corrects = int(pred_flat.eq(target_flat.squeeze(1)).masked_select(mask_flat).float().data.sum())
    num_words = length.data.sum()

    return loss, pred_seqs, num_corrects, num_words


def variable2numpy(var):
    """ For tensorboard visualization """
    return var.data.cpu().numpy()


def write_to_tensorboard(writer, global_step, total_loss, total_corrects, total_words, total_accuracy,
                         encoder_grad_norm, decoder_grad_norm, clipped_encoder_grad_norm, clipped_decoder_grad_norm,
                         encoder, decoder, gpu_memory_usage=None):
    # scalars
    if gpu_memory_usage is not None:
        writer.add_scalar('curr_gpu_memory_usage', gpu_memory_usage['curr'], global_step)
        writer.add_scalar('diff_gpu_memory_usage', gpu_memory_usage['diff'], global_step)

    writer.add_scalar('total_loss', total_loss, global_step)
    writer.add_scalar('total_accuracy', total_accuracy, global_step)
    writer.add_scalar('total_corrects', total_corrects, global_step)
    writer.add_scalar('total_words', total_words, global_step)
    writer.add_scalar('encoder_grad_norm', encoder_grad_norm, global_step)
    writer.add_scalar('decoder_grad_norm', decoder_grad_norm, global_step)
    writer.add_scalar('clipped_encoder_grad_norm', clipped_encoder_grad_norm, global_step)
    writer.add_scalar('clipped_decoder_grad_norm', clipped_decoder_grad_norm, global_step)

    # histogram
    for name, param in encoder.named_parameters():
        name = name.replace('.', '/')
        writer.add_histogram('encoder/{}'.format(name), variable2numpy(param), global_step, bins='doane')
        if param.grad is not None:
            writer.add_histogram('encoder/{}/grad'.format(name), variable2numpy(param.grad), global_step, bins='doane')

    for name, param in decoder.named_parameters():
        name = name.replace('.', '/')
        writer.add_histogram('decoder/{}'.format(name), variable2numpy(param), global_step, bins='doane')
        if param.grad is not None:
            writer.add_histogram('decoder/{}/grad'.format(name), variable2numpy(param.grad), global_step, bins='doane')


def detach_hidden(hidden):
    """ Wraps hidden states in new Variables, to detach them from their history. Prevent OOM.
        After detach, the hidden's requires_grad=Fasle and grad_fn=None.
    Issues:
    - Memory leak problem in LSTM and RNN: https://github.com/pytorch/pytorch/issues/2198
    - https://github.com/pytorch/examples/blob/master/word_language_model/main.py
    - https://discuss.pytorch.org/t/help-clarifying-repackage-hidden-in-word-language-model/226
    - https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426
    -
    """
    if type(hidden) == Variable:
        hidden.detach_()  # same as creating a new variable.
    else:
        for h in hidden: h.detach()


def get_gpu_memory_usage(device_id):
    """Get the current gpu usage. """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.decode().strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map[device_id]