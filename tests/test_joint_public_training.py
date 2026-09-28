import torch
from training.transformer.train_joint_public import objective


def test_joint_objective_routes_both_losses_to_shared_features():
    features=torch.tensor([.3,-.8],requires_grad=True)
    recognition=(features-1).square().sum()
    notation=(features+2).square().sum()
    loss=objective(recognition,notation)
    loss.backward()
    expected=2*(features.detach()-1)+2*(features.detach()+2)/12
    assert torch.allclose(features.grad,expected)
    assert not torch.equal(features.grad,2*(features.detach()-1))


def test_flattened_recognition_loss_preserves_mask_smoothing_and_gradients():
    from types import SimpleNamespace
    from training.transformer.train_joint_public import deterministic_cross_entropy
    logits=torch.randn(2,7,5,dtype=torch.float64,requires_grad=True)
    target=torch.randint(0,5,(2,7));target[0,4:]=-100
    old=torch.nn.functional.cross_entropy(logits.transpose(1,2),target,ignore_index=-100,label_smoothing=.1)
    new=deterministic_cross_entropy(SimpleNamespace(ignore_index=-100),logits,target,label_smoothing=.1)
    assert torch.allclose(old,new,atol=1e-12)
    assert torch.allclose(torch.autograd.grad(old,logits,retain_graph=True)[0],torch.autograd.grad(new,logits)[0],atol=1e-12)
