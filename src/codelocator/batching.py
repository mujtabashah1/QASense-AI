"""Deterministic length bucketing with a bounded padded-token budget."""
import random

def token_batches(indices,lengths,token_budget,max_examples=32):
    batch=[]; longest=0
    for index in indices:
        length=lengths[index]
        if length>token_budget: raise ValueError('A sequence exceeds the token budget')
        if batch and (max(longest,length)*(len(batch)+1)>token_budget or len(batch)>=max_examples):
            yield batch; batch=[]; longest=0
        batch.append(index); longest=max(longest,length)
    if batch: yield batch

def training_groups(lengths,seed,effective_batch_size,token_budget):
    rng=random.Random(seed); indices=list(range(len(lengths))); rng.shuffle(indices)
    groups=[]
    for start in range(0,len(indices),256):
        bucket=sorted(indices[start:start+256],key=lambda i:lengths[i])
        groups.extend(bucket[i:i+effective_batch_size] for i in range(0,len(bucket),effective_batch_size))
    rng.shuffle(groups)
    for group in groups:
        yield group,list(token_batches(group,lengths,token_budget,max_examples=effective_batch_size))
