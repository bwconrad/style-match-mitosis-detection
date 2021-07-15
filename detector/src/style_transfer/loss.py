import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, pred, target):
        # Single layer's features
        if not isinstance(pred, list):
            assert pred.size() == target.size()
            return F.mse_loss(pred, target)

        # List of layer features
        assert len(pred) == len(target)
        loss = 0
        for p, t in zip(pred, target):
            loss += F.mse_loss(p, t)

        return loss


class StyleLoss(nn.Module):
    def __init__(self, use_statistics=False):
        super().__init__()
        self.use_statistics = use_statistics

    def get_gram(self, x):
        b, c, h, w = x.size()
        f = x.view(b, c, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * c)
        return G

    def get_mean_std(self, x, eps=1e-6):
        mean = torch.mean(x, axis=(2, 3), keepdim=True)
        std = torch.std(x, dim=(2, 3), keepdim=True) + eps
        return mean, std

    def __call__(self, pred, target):
        assert len(pred) == len(target)

        # Differnce between feature statistics
        if self.use_statistics:
            loss = 0
            for p, t in zip(pred, target):
                assert p.size() == t.size()
                p_mean, p_std = self.get_mean_std(p)
                t_mean, t_std = self.get_mean_std(t)
                loss += F.mse_loss(p_mean, t_mean) + F.mse_loss(p_std, t_std)

        # Difference between feature gram matrices
        else:
            loss = 0
            for p, t in zip(pred, target):
                assert p.size() == t.size()
                p_gram = self.get_gram(p)
                t_gram = self.get_gram(t)
                loss += F.mse_loss(p_gram, t_gram)

        return loss
