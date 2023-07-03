import torch
import torch.nn as nn
import torch.nn.functional as F



def _binary_dice_loss(output, target):
    smooth = 1e-8
    intersection = (output * target).sum()
    dice_loss = 1 - (2 * intersection)\
        / (output.sum() + target.sum() + smooth)

    return dice_loss


class BinaryDiceLoss(nn.Module):
    """
    Binary dice loss.
    """

    def forward(self, output, target):
        output = output.sigmoid()
        dice_loss = _binary_dice_loss(output, target)

        return dice_loss

class BinaryDiceTestLoss(nn.Module):
    """
    Binary dice loss.
    """

    def forward(self, output, target):
        center_value = output[target == 1]
        output = output.sigmoid()
        dice_loss = _binary_dice_loss(output, target)

        return dice_loss, center_value.mean().item()

class SegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.seg_losses = nn.ModuleDict()
        self.seg_losses.update([("dice_loss", BinaryDiceLoss())])
        self.seg_losses.update([("bce_loss", nn.BCEWithLogitsLoss())])
        self.weight = [1, 0.5]

    def forward(self, outputs, targets):
        dice_loss = self.seg_losses["dice_loss"](outputs, targets) * self.weight[0]
        bce_loss = self.seg_losses["bce_loss"](outputs,targets) * self.weight[1]
        loss_dict = {"dice_loss": dice_loss,
                     "bce_loss": bce_loss}
        total_loss = sum([loss_dict[k] for k in loss_dict.keys()])
        loss_dict["total_loss"] = total_loss
        return loss_dict


class TestLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.seg_losses = nn.ModuleDict()
        self.seg_losses.update([("dice_loss", BinaryDiceTestLoss())])
        self.seg_losses.update([("bce_loss", nn.BCEWithLogitsLoss())])
        self.weight = [1, 0.5]

    def forward(self, outputs, targets):
        fliter = torch.zeros_like(outputs) - 100000
        fliter[targets==1] = 0
        outputs = outputs + fliter
        dice_loss, center_value = self.seg_losses["dice_loss"](outputs, targets) * self.weight[0]
        bce_loss = self.seg_losses["bce_loss"](outputs, targets) * self.weight[1]
        loss_dict = {"dice_loss": dice_loss,
                     "bce_loss": bce_loss}
        total_loss = sum([loss_dict[k] for k in loss_dict.keys()])
        loss_dict["total_loss"] = total_loss
        return loss_dict, center_value
    
    
class TestPointLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.seg_losses = nn.ModuleDict()
        self.seg_losses.update([("dice_loss", BinaryDiceLoss())])
        self.seg_losses.update([("bce_loss", nn.BCEWithLogitsLoss())])
        self.weight = [1, 0.5]

    def forward(self, outputs, targets, input_shape):
        fliter = torch.zeros_like(outputs) - 100000
        fliter[targets==1] = 0
        fliter[:, :, int(input_shape[2]/2), int(input_shape[3]/2), int(input_shape[4]/2)] = 0
        outputs = outputs + fliter
        dice_loss = self.seg_losses["dice_loss"](outputs, targets) * self.weight[0]
        bce_loss = self.seg_losses["bce_loss"](outputs, targets) * self.weight[1]
        loss_dict = {"dice_loss": dice_loss,
                     "bce_loss": bce_loss}
        total_loss = sum([loss_dict[k] for k in loss_dict.keys()])
        loss_dict["total_loss"] = total_loss
        return loss_dict
    
    
def region_loss(input: torch.Tensor, target: torch.Tensor, exclude_bg: bool = False) -> torch.Tensor:
    """Loss based on region-based incorrect pixel count. Objective function to minimize.
    Based on https://ieeexplore.ieee.org/document/9433775
    Args:
        input: logits tensor with shape :math:`(N, C, H, W)` where C = number of classes.
        label: labels tensor with shape :math:`(N, H, W)` where each value
          is :math:`0 ≤ targets[i] ≤ C−1`.
    Returns:
        torch.Tensor: region-based loss. Value between 0. and 1.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxNxHxW. \
            Got: {input.shape}")

    if not input.shape[-2:] == target.shape[-2:]:
        raise ValueError(f"input and target shapes must be the same. \
            Got: {input.shape} and {target.shape}")

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. \
            Got: {input.device} and {target.device}")

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.sigmoid(input)

    # create the labels one hot tensor

    rl = (target * (1 - input_soft) + (1 - target) * input_soft).sum(dim=(2, 3))
    rl = rl / input[0, 0].numel()

    offset = 1 if exclude_bg else 0
    return rl.mean()


class RegionLoss(nn.Module):

    def __init__(self, exclude_bg: bool = True) -> None:
        super().__init__()
        self.exclude_bg = exclude_bg

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return region_loss(input, target, self.exclude_bg)


def shape_loss(input: torch.Tensor, target: torch.Tensor, distance_maps:torch.Tensor,
               exclude_bg: bool = False) -> torch.Tensor:
    """Loss based on shape. Objective function to minimize.
    Based on https://ieeexplore.ieee.org/document/9433775
    Args:
        input: logits tensor with shape :math:`(N, C, H, W)` where C = number of classes.
        label: labels tensor with shape :math:`(N, H, W)` where each value
          is :math:`0 ≤ targets[i] ≤ C-1`.
    Returns:
        torch.Tensor: shape-based loss. Value between 0. and 1.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxNxHxW. \
            Got: {input.shape}")

    if not input.shape[-2:] == target.shape[-2:]:
        raise ValueError(f"input and target shapes must be the same. \
            Got: {input.shape} and {target.shape}")

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. \
            Got: {input.device} and {target.device}")

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.sigmoid(input)
    # input_soft: torch.Tensor = 1/(1+torch.exp(-input/10))

    # TODO: potentially find a way to torch-ify this... for now switching back and forth to numpy :(
    # highly uncool because I need to do for loops due to scipy function def

    sl = (distance_maps - input_soft).abs().sum(dim=(2, 3))
    sl = sl / input[0, 0].numel()
    return sl.mean()


class ShapeLoss(nn.Module):

    def __init__(self, exclude_bg: bool = True) -> None:
        super().__init__()
        self.exclude_bg = exclude_bg

    def forward(self, input: torch.Tensor, target: torch.Tensor, 
                distance_map: torch.Tensor) -> torch.Tensor:
        return shape_loss(input, target, distance_map, self.exclude_bg)


class SegReShLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.seg_losses = nn.ModuleDict()
        self.seg_losses.update([("dice_loss", BinaryDiceLoss())])
        self.seg_losses.update([("bce_loss", nn.BCEWithLogitsLoss())])
        self.seg_losses.update([("shape_loss", ShapeLoss(exclude_bg=False))])
        self.seg_losses.update([("region_loss", RegionLoss(exclude_bg=False))])
        self.weight = [1, 0.5, 0.5, 0.5]

    def forward(self, outputs, targets, distance_maps):
        dice_loss = self.seg_losses["dice_loss"](outputs, targets) * self.weight[0]
        bce_loss = self.seg_losses["bce_loss"](outputs,targets) * self.weight[1]
        shape_loss = self.seg_losses["shape_loss"](outputs, targets, distance_maps) * self.weight[2]
        region_loss = self.seg_losses["region_loss"](outputs, targets) * self.weight[2]
        loss_dict = {"dice_loss": dice_loss,
                     "bce_loss": bce_loss,
                     "shape_loss": shape_loss,
                     "region_loss": region_loss}
        total_loss = sum([loss_dict[k] for k in loss_dict.keys()])
        loss_dict["total_loss"] = total_loss
        return loss_dict