"""Verify causal-mask alignment for full, chunked and single-token attention.

PyTorch, CPU float64. No model weights, training, GPU or network required.
Optional --figure needs Matplotlib. The illustrated masks are exact boolean arrays.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F


def manual_attention(q, k, v, allowed):
    scores = q @ k.transpose(-2, -1) / q.shape[-1]**.5
    assert allowed.any(dim=-1).all(), 'This reference expects at least one valid key per query.'
    return torch.softmax(scores.masked_fill(~allowed, float('-inf')), dim=-1) @ v


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('cache-mask-results.json'))
    parser.add_argument('--figure', type=Path)
    args = parser.parse_args()
    torch.manual_seed(20260928)
    torch.set_num_threads(1)
    batch, heads, length, dimension = 2, 2, 8, 4
    q, k, v = [torch.randn(batch, heads, length, dimension, dtype=torch.float64) for _ in range(3)]
    full_mask = torch.arange(length)[None, :] <= torch.arange(length)[:, None]
    reference = manual_attention(q, k, v, full_mask)
    torch.testing.assert_close(F.scaled_dot_product_attention(q,k,v,is_causal=True), reference,
                               atol=1e-12,rtol=1e-12)
    for chunks in [[1]*length, [3,2,3], [5,3]]:
        outputs=[];past=0
        for size in chunks:
            stop=past+size
            query_positions=torch.arange(past,stop)
            key_positions=torch.arange(stop)
            allowed=key_positions[None,:] <= query_positions[:,None]
            outputs.append(F.scaled_dot_product_attention(q[:,:,past:stop],k[:,:,:stop],v[:,:,:stop],
                                                          attn_mask=allowed,is_causal=False,dropout_p=0.))
            past=stop
        torch.testing.assert_close(torch.cat(outputs,dim=-2),reference,atol=1e-12,rtol=1e-12)

    past, size = 3, 2
    stop=past+size
    query=q[:,:,past:stop];keys=k[:,:,:stop];values=v[:,:,:stop]
    correct_mask=torch.arange(stop)[None,:] <= torch.arange(past,stop)[:,None]
    upper_left=torch.arange(stop)[None,:] <= torch.arange(size)[:,None]
    correct=F.scaled_dot_product_attention(query,keys,values,attn_mask=correct_mask,dropout_p=0.)
    wrong=F.scaled_dot_product_attention(query,keys,values,is_causal=True,dropout_p=0.)
    torch.testing.assert_close(wrong,manual_attention(query,keys,values,upper_left),atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(correct,reference[:,:,past:stop],atol=1e-12,rtol=1e-12)
    wrong_error=float((wrong-correct).abs().max())
    assert wrong_error > .1
    # For a single appended token, every existing key is visible ONLY when
    # there are no padding entries or unfilled/future cache slots.
    single=F.scaled_dot_product_attention(q[:,:,-1:],k,v,is_causal=False,dropout_p=0.)
    torch.testing.assert_close(single,reference[:,:,-1:],atol=1e-12,rtol=1e-12)
    wrong_single=F.scaled_dot_product_attention(q[:,:,-1:],k,v,is_causal=True,dropout_p=0.)
    torch.testing.assert_close(wrong_single,v[:,:,:1],atol=1e-12,rtol=1e-12)

    # Batched left padding plus preallocated future slots: positions and key
    # validity are explicit per sample. Queries here are all valid tokens.
    query_positions=torch.tensor([[3,4],[1,2]])
    key_positions=torch.tensor([[0,1,2,3,4,5,6,7],[-1,-1,0,1,2,3,4,5]])
    key_valid=torch.tensor([[1,1,1,1,1,0,0,0],[0,0,1,1,1,0,0,0]],dtype=torch.bool)
    batched_mask=(key_positions[:,None,:] <= query_positions[:,:,None]) & key_valid[:,None,:]
    batched_mask=batched_mask[:,None,:,:]
    padded=F.scaled_dot_product_attention(query,k,v,attn_mask=batched_mask,dropout_p=0.)
    padded_reference=manual_attention(query,k,v,batched_mask)
    torch.testing.assert_close(padded,padded_reference,atol=1e-12,rtol=1e-12)
    # Change forbidden K/V dramatically. A valid mask makes outputs invariant.
    invalid=~key_valid[:,None,:,None]
    changed_k=torch.where(invalid,k+1000,k)
    changed_v=torch.where(invalid,v-1000,v)
    invariant=F.scaled_dot_product_attention(query,changed_k,changed_v,
                                            attn_mask=batched_mask,dropout_p=0.)
    torch.testing.assert_close(invariant,padded,atol=1e-12,rtol=1e-12)
    first_mask = batched_mask[:, :, :1]
    first_forbidden = ~first_mask[:, 0, 0, :, None]
    changed_k = torch.where(first_forbidden[:, None], k + 777, k)
    changed_v = torch.where(first_forbidden[:, None], v - 777, v)
    first_invariant = F.scaled_dot_product_attention(query[:, :, :1], changed_k, changed_v,
                                                    attn_mask=first_mask, dropout_p=0.)
    torch.testing.assert_close(first_invariant, padded[:, :, :1], atol=1e-12, rtol=1e-12)
    if args.figure:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        fig,axes=plt.subplots(1,2,figsize=(9.8,3.5),constrained_layout=True)
        for ax,mask,title in zip(axes,[upper_left,correct_mask],
                                ['Upper-left: wrong for these cached queries','Global positions: queries 3 and 4']):
            ax.imshow(mask.numpy(),cmap=ListedColormap(['#f1f4f7','#91b9db']),vmin=0,vmax=1)
            ax.set(xticks=range(stop),yticks=range(size),yticklabels=[3,4],
                   xlabel='Global key position',ylabel='Global query position',title=title)
            for i in range(size):
                for j in range(stop):ax.text(j,i,'allow' if mask[i,j] else 'block',ha='center',va='center',fontsize=10)
        args.figure.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(args.figure,dpi=160);plt.close(fig)
    result={'torch':torch.__version__,'device':'cpu','dtype':'float64','seed':20260928,
            'qkv_shape':[batch,heads,length,dimension],'past_tokens':past,'query_chunk':size,
            'correct_mask':correct_mask.int().tolist(),'incorrect_upper_left_mask':upper_left.int().tolist(),
            'correct_chunk_max_abs_error':float((correct-reference[:,:,past:stop]).abs().max()),
            'incorrect_chunk_max_abs_error':wrong_error,
            'incorrect_single_token_max_abs_error':float((wrong_single-reference[:,:,-1:]).abs().max()),
            'padding_and_unused_slots_max_abs_error':float((padded-padded_reference).abs().max()),
            'checks':['full reference','three chunk partitions','single appended token',
                      'batched key padding','unfilled cache slots','forbidden-entry invariance',
                      'future valid-key invariance for the first query'],
            'scope':'Attention operator only; no positional embedding, transformer layers, generation benchmark or fused GPU-kernel claim.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
