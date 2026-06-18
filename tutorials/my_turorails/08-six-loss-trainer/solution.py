"""Solution for Lesson 08: Six-Loss Trainer."""
import torch


def compute_six_losses(
    pred_aa, pred_bb, pred_ab, pred_ba, pred_aba, pred_bab,
    true_a, true_b
):
    l_aa = torch.mean(torch.abs(pred_aa - true_a))
    l_bb = torch.mean(torch.abs(pred_bb - true_b))
    l_ab = torch.mean(torch.abs(pred_ab - true_b))
    l_ba = torch.mean(torch.abs(pred_ba - true_a))
    l_aba = torch.mean(torch.abs(pred_aba - true_a))
    l_bab = torch.mean(torch.abs(pred_bab - true_b))
    total = l_aa + l_bb + l_ab + l_ba + l_aba + l_bab
    return {
        "l_aa": l_aa, "l_bb": l_bb, "l_ab": l_ab,
        "l_ba": l_ba, "l_aba": l_aba, "l_bab": l_bab,
        "total": total,
    }
