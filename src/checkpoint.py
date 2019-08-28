import os
import torch
import torch._utils

def load_checkpoint(checkpoint_path):
    # It's weird that if `map_location` is not given, it will be extremely slow.
    # try:
    #     torch._utils._rebuild_tensor_v2
    # except AttributeError:
    #     def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    #         tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
    #         tensor.requires_grad = requires_grad
    #         tensor._backward_hooks = backward_hooks
    #         return tensor
    #
    #     torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    return torch.load(checkpoint_path, map_location=lambda storage, loc: storage)


def save_checkpoint(opts, experiment_name, encoder, decoder, encoder_optim, decoder_optim,
                    total_accuracy, total_loss, global_step):
    checkpoint = {
        'opts': opts,
        'global_step': global_step,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'encoder_optim_state_dict': encoder_optim.state_dict(),
        'decoder_optim_state_dict': decoder_optim.state_dict()
    }
    ext_opts = [int(opts.use_sent_rate), int(opts.use_sent_senti), int(opts.use_sent_len), int(opts.use_app_cate), int(opts.use_keyword), int(opts.tie_ext_feature)]
    ext_opts = ''.join(map(str, ext_opts))
    checkpoint_path = '/research/lyu1/cygao/workspace/data/checkpoints/single_wc%s_hs%s_ln%s_dp%.1f_rslckt%s/%s_acc_%.2f_loss_%.2f_step_%d.pt' % (
    str(opts.word_vec_size), str(opts.hidden_size), str(opts.num_layers), opts.dropout, ext_opts, experiment_name, total_accuracy, total_loss, global_step)

    directory, filename = os.path.split(os.path.abspath(checkpoint_path))

    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        if total_accuracy < 70:  # remove files if accuracy less than 80%
            files = os.listdir(directory)
            for f in files:
                os.remove(os.path.join(directory, f))

    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path