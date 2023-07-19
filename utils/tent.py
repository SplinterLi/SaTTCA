from copy import deepcopy
from loss import TestLoss
import torch
import torch.nn as nn
import torch.jit

# The implementation of TTA baseline is following https://github.com/DequanWang/tent

class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=10, entropy = False, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss = TestLoss()
        self.steps = steps
        self.entropy = entropy
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()
        for i in range(len(x)):
            x[i].requires_grad_(True)

        for _ in range(self.steps):
            if self.entropy:
                outputs = forward_and_adapt(x[0], self.model, self.optimizer)
            else:
                outputs = forward_and_click(x[0], x[1], self.model, self.optimizer, self.loss)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
def sigmoid_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return x.sigmoid() * x


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model(x)
    # adapt
    loss = sigmoid_entropy(outputs).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs


@torch.enable_grad()
def forward_and_click(x, y, model, optimizer, loss_fn):
    """Forward and adapt model on InstanceNorm layer.

    Get entropy loss and scare-awera loss of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model(x)
    # adapt
    loss_dict, center_value = loss_fn(outputs, y)
    loss = loss_dict['total_loss']
    if loss > 1e-7:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return outputs, loss_dict, center_value

def collect_params(model):
    """Collect the affine scale + shift parameters from InstanceNorm.

    Walk the model's modules and collect all InstanceNorm parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.InstanceNorm3d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.InstanceNorm3d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.InstanceNorm3d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
