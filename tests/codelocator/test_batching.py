from codelocator.batching import training_groups,token_batches

def test_bounded_batches_and_coverage():
    lengths=[5,20,40,128,256]*100
    groups=list(training_groups(lengths,42,16,512))
    assert sorted(i for group,_ in groups for i in group)==list(range(len(lengths)))
    assert all(len(group)<=16 for group,_ in groups)
    for _,batches in groups:
        for batch in batches: assert len(batch)*max(lengths[i] for i in batch)<=512
    assert groups==list(training_groups(lengths,42,16,512))

def test_prediction_order_indices():
    lengths=[5,256,9,128,40]
    order=sorted(range(len(lengths)),key=lengths.__getitem__)
    batches=list(token_batches(order,lengths,512))
    assert sorted(i for batch in batches for i in batch)==list(range(5))
