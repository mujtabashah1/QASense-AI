import pytest
from codelocator_v2.gate import require_reviewed_gate
from codelocator_v2.runtime import micro_groups

def test_gate_requires_explicit_review_approval():
    with pytest.raises(RuntimeError,match='paused for explicit user review'): require_reviewed_gate(False)
    assert require_reviewed_gate(True)['passed'] is True

def test_microbatch_and_effective_batch_bounds():
    lengths=[256]*33
    groups=list(micro_groups(lengths,0,4))
    assert all(len(group)<=16 for group,_ in groups)
    assert all(len(micro)<=4 for _,micros in groups for micro in micros)
    assert sum(len(group) for group,_ in groups)==33

def test_checkpoint_schema_contains_resume_critical_state():
    import torch
    from codelocator_v2.train import _checkpoint_state
    model=torch.nn.Linear(2,2); optimizer=torch.optim.AdamW(model.parameters()); scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1); scaler=torch.amp.GradScaler('cuda',enabled=False)
    state=_checkpoint_state(model,optimizer,scheduler,scaler,{'batch':4},{'split':'hash'},1,123,'training',456,7.5,1968,[],.4,0)
    required={'model_state_dict','optimizer_state_dict','scheduler_state_dict','scaler_state_dict','epoch','next_group_index','phase','global_optimizer_step','rng_state','split_identity','config'}
    assert required<=state.keys()
    assert state['next_group_index']==123 and state['global_optimizer_step']==456
