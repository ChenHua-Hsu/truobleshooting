import pandas as pd
import torch
import matplotlib.pyplot as plt

def _get_gpu_mem(synchronize=True, empty_cache=True):
    return torch.cuda.memory_allocated(), torch.cuda.memory_cached()

def _generate_mem_hook(handle_ref, mem, idx, hook_type, exp):
    def hook(self, *args):
        if len(mem) == 0 or mem[-1]["exp"] != exp:
            call_idx = 0
        else:
            call_idx = mem[-1]["call_idx"] + 1

        mem_all, mem_cached = _get_gpu_mem()
        torch.cuda.synchronize()
        mem.append({
            'layer_idx': idx,
            'call_idx': call_idx,
            'layer_type': type(self).__name__,
            'exp': exp,
            'hook_type': hook_type,
            'mem_all': mem_all,
            'mem_cached': mem_cached,
        })

    return hook

def _add_memory_hooks(idx, mod, mem_log, exp, hr):
    h = mod.register_forward_pre_hook(_generate_mem_hook(hr, mem_log, idx, 'pre', exp))
    hr.append(h)

    h = mod.register_forward_hook(_generate_mem_hook(hr, mem_log, idx, 'fwd', exp))
    hr.append(h)

    h = mod.register_backward_hook(_generate_mem_hook(hr, mem_log, idx, 'bwd', exp))
    hr.append(h)

def log_mem(model, input_, device, mem_log=None, exp=None):
    mem_log = mem_log or []
    exp = exp or f'exp_{len(mem_log)}'
    hr = []
    for idx, module in enumerate(model.modules()):
        _add_memory_hooks(idx, module, mem_log, exp, hr)

    try:
        loss,_,_,_ = model(input_)
        loss = loss.sum() # Just meant to check memory usage, so loss function does not necessarily to be consistent with original ones.
        loss.backward()
    finally:
        [h.remove() for h in hr]

        return mem_log


def plot_mem(
        df,
        exps=None,
        normalize_call_idx=True,
        normalize_mem_all=True,
        filter_fwd=False,
        return_df=False,
        output_file=None,
        batch_size=None,
        time=None
):
    if exps is None:
        exps = df.exp.drop_duplicates()

    fig, ax = plt.subplots(figsize=(20, 10))
    for exp in exps:
        df_ = df[df.exp == exp]
        df_.mem_all = df_.mem_all / (1024*1024*1024)

        if normalize_call_idx:
            df_.call_idx = df_.call_idx / df_.call_idx.max()

        if normalize_mem_all:
            df_.mem_all = df_.mem_all - df_[df_.call_idx == df_.call_idx.min()].mem_all.iloc[0]
            df_.mem_all = df_.mem_all // 2 ** 20

        if filter_fwd:
            layer_idx = 0
            callidx_stop = df_[(df_["layer_idx"] == layer_idx) & (df_["hook_type"] == "fwd")]["call_idx"].iloc[0]
            df_ = df_[df_["call_idx"] <= callidx_stop]
            # df_ = df_[df_.call_idx < df_[df_.layer_idx=='bwd'].call_idx.min()]

        categories = df_['layer_type'].unique()
        for color_idx, category in enumerate(categories):
            df_cat_ = df_[df_['layer_type'] == category]
            plt.scatter(df_cat_['call_idx'], df_cat_['mem_all'], label=category, color=plt.cm.tab20(color_idx))
            #plot = df_.plot(ax=ax, x='call_idx', y='mem_all', label=category)
        plt.legend()
        plt.xlabel('Steps')
        plt.ylabel('Memory [GB]')
        plt.title('Memory Performance')
        if not time is None:
            plt.text(df_.call_idx.max() * 0.05, df_.mem_all.max() * 0.8, 'time/batch=%.3f s'%(time), fontsize=12)
        if not batch_size is None:
            plt.text(df_.call_idx.max() * 0.05, df_.mem_all.max() * 0.76, 'batch size: %d'%batch_size, fontsize=12)
        if output_file: 
            plt.savefig(output_file)
            #plot.get_figure().savefig(output_file)

    if return_df:
        return df_
