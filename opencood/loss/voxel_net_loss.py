import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxelNetLoss(nn.Module):
    def __init__(self, args):
        super(VoxelNetLoss, self).__init__()
        self.smoothl1loss = nn.SmoothL1Loss(size_average=False)
        self.alpha = args['alpha']
        self.beta = args['beta']
        self.reg_coe = args['reg']
        self.loss_dict = {}

    def forward(self, output_dict, target_dict):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        rm = output_dict['rm']
        psm = output_dict['psm']

        pos_equal_one = target_dict['pos_equal_one']
        neg_equal_one = target_dict['neg_equal_one']
        targets = target_dict['targets']

        p_pos = F.sigmoid(psm.permute(0, 2, 3, 1))
        rm = rm.permute(0, 2, 3, 1).contiguous()
        rm = rm.view(rm.size(0), rm.size(1), rm.size(2), -1, 7)
        targets = targets.view(targets.size(0), targets.size(1),
                               targets.size(2), -1, 7)
        pos_equal_one_for_reg = pos_equal_one.unsqueeze(
            pos_equal_one.dim()).expand(-1, -1, -1, -1, 7)

        rm_pos = rm * pos_equal_one_for_reg
        targets_pos = targets * pos_equal_one_for_reg

        cls_pos_loss = -pos_equal_one * torch.log(p_pos + 1e-6)
        cls_pos_loss = cls_pos_loss.sum() / (pos_equal_one.sum() + 1e-6)

        cls_neg_loss = -neg_equal_one * torch.log(1 - p_pos + 1e-6)
        cls_neg_loss = cls_neg_loss.sum() / (neg_equal_one.sum() + 1e-6)

        reg_loss = self.smoothl1loss(rm_pos, targets_pos)
        reg_loss = reg_loss / (pos_equal_one.sum() + 1e-6)
        conf_loss = self.alpha * cls_pos_loss + self.beta * cls_neg_loss

        total_loss = self.reg_coe * reg_loss + conf_loss

        self.loss_dict.update({'total_loss': total_loss,
                               'reg_loss': reg_loss,
                               'conf_loss': conf_loss})

        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict['total_loss']
        reg_loss = self.loss_dict['reg_loss']
        conf_loss = self.loss_dict['conf_loss']

        print("[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
              " || Loc Loss: %.4f" % (
                  epoch, batch_id + 1, batch_len,
                  total_loss.item(), conf_loss.item(), reg_loss.item()))

        writer.add_scalar('Regression_loss', reg_loss.item(),
                          epoch*batch_len + batch_id)
        writer.add_scalar('Confidence_loss', conf_loss.item(),
                          epoch*batch_len + batch_id)
