def test_upper_layers_receive_gradients_with_frozen_lower_encoder():
    import torch
    from transformers import RobertaConfig,RobertaForSequenceClassification
    from codelocator.train_bug_type import configure_trainable_layers
    config=RobertaConfig(vocab_size=64,hidden_size=16,num_hidden_layers=12,num_attention_heads=2,intermediate_size=32,max_position_embeddings=64,num_labels=2)
    model=RobertaForSequenceClassification(config)
    policy=configure_trainable_layers(model,4)
    assert policy['frozen_encoder_layers']==list(range(8)) and policy['trained_encoder_layers']==[8,9,10,11]
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train()
    loss=model(input_ids=torch.tensor([[0,4,5,6,2]]),attention_mask=torch.ones(1,5,dtype=torch.long),labels=torch.tensor([1])).loss
    loss.backward()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
