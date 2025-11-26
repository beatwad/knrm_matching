from math import log2

import torch
import numpy as np


def num_swapped_pairs(ys_true: torch.torch.Tensor, ys_pred: torch.Tensor) -> int:
    swap_pairs_num = 0
    _, idx = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[idx]

    for i in range(len(ys_true) - 1):
        swap_pairs_num += sum(ys_true_sorted[i:] > ys_true_sorted[i]).item()

    return swap_pairs_num


def compute_gain(y_value: float, gain_scheme: str) -> float:
    if gain_scheme not in ["const", "exp2"]:
        raise ValueError("gain_scheme can have values 'const' and 'exp2' only")
    if gain_scheme == "const":
        return y_value
    return 2**y_value - 1


def dcg(ys_true: torch.Tensor, ys_pred: torch.Tensor, gain_scheme: str, k: int) -> float:
    res = 0
    _, idx = torch.sort(ys_pred, descending=True, dim=0)

    if k is not None:
        idx = idx[:k]

    for i, index in enumerate(idx, start=1):
        res += compute_gain(ys_true[index].item(), gain_scheme) / log2(i + 1)

    return res


def ndcg(
    ys_true: torch.Tensor,
    ys_pred: torch.Tensor,
    gain_scheme: str = "const",
    k: int = None,
) -> float:
    pred_dcg = dcg(ys_true, ys_pred, gain_scheme, k)
    ideal_dcg = compute_ideal_dcg(ys_true, gain_scheme, k)
    return pred_dcg / ideal_dcg


def compute_ideal_dcg(ys_true: torch.Tensor, gain_scheme: str, k: int = None):
    ideal_dcg = dcg(ys_true, ys_true, gain_scheme, k)
    if ideal_dcg == 0:
        return np.nan
    return ideal_dcg


def precision_at_k(ys_true: torch.Tensor, ys_pred: torch.Tensor, k: int) -> float:
    total_num_true = ys_true.sum()
    if total_num_true == 0:
        return -1
    _, idx = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[idx]
    num_true = ys_true_sorted[:k].sum()
    res = num_true / min(k, total_num_true)
    return res


def reciprocal_rank(ys_true: torch.Tensor, ys_pred: torch.Tensor) -> float:
    if ys_true.sum() == 0:
        return 0
    _, idx = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[idx]
    rank = torch.where(ys_true_sorted == 1)[0][0].item()
    return 1 / rank


def p_found(ys_true: torch.Tensor, ys_pred: torch.Tensor, p_break: float = 0.15) -> float:
    p_look = 1
    _p_found = 0
    _, idx = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[idx]

    for cur in ys_true_sorted:
        _p_found += p_look * float(cur)
        p_look = p_look * (1 - float(cur)) * (1 - p_break)

    return _p_found


def average_precision(ys_true: torch.Tensor, ys_pred: torch.Tensor) -> float:
    if ys_true.sum() == 0:
        return -1
    num_correct, rolling_sum = 0, 0
    _, idx = torch.sort(ys_pred, descending=True, dim=0)
    ys_true_sorted = ys_true[idx]

    for idx, val in enumerate(ys_true_sorted, start=1):
        if val == 1:
            num_correct += 1
            rolling_sum += num_correct / idx

    return rolling_sum / num_correct


def listnet_ce_loss(y_i, z_i):
    """
    y_i: (n_i, 1) GT
    z_i: (n_i, 1) preds
    """

    P_y_i = torch.softmax(y_i, dim=0)
    P_z_i = torch.softmax(z_i, dim=0)
    return -torch.sum(P_y_i * torch.log(P_z_i))


def listnet_kl_loss(y_i, z_i):
    """
    y_i: (n_i, 1) GT
    z_i: (n_i, 1) preds
    """
    P_y_i = torch.softmax(y_i, dim=0)
    P_z_i = torch.softmax(z_i, dim=0)
    return -torch.sum(P_y_i * torch.log(P_z_i / P_y_i))


def compute_lambdas(y_true, y_pred, ndcg_scheme="exp2"):
    # рассчитаем нормировку, IdealDCG
    ideal_dcg = compute_ideal_dcg(y_true, gain_scheme=ndcg_scheme)
    N = 1 / ideal_dcg

    # рассчитаем порядок документов согласно оценкам релевантности
    _, rank_order = torch.sort(y_true, descending=True, axis=0)
    rank_order += 1

    with torch.no_grad():
        # получаем все попарные разницы скоров в батче
        pos_pairs_score_diff = 1.0 + torch.exp((y_pred - y_pred.t()))

        # поставим разметку для пар, 1 если первый документ релевантнее
        # -1 если второй документ релевантнее
        Sij = compute_labels_in_batch(y_true)
        # посчитаем изменение gain из-за перестановок
        gain_diff = compute_gain_diff(y_true, ndcg_scheme)

        # посчитаем изменение знаменателей-дискаунтеров
        decay_diff = (1.0 / torch.log2(rank_order + 1.0)) - (1.0 / torch.log2(rank_order.t() + 1.0))
        # посчитаем непосредственное изменение nDCG
        delta_ndcg = torch.abs(N * gain_diff * decay_diff)
        # посчитаем лямбды
        lambda_update = (0.5 * (1 - Sij) - 1 / pos_pairs_score_diff) * delta_ndcg
        lambda_update = torch.sum(lambda_update, dim=1, keepdim=True)
        return lambda_update


def compute_labels_in_batch(y_true):
    # разница релевантностей каждого с каждым объектом
    rel_diff = y_true - y_true.t()

    # 1 в этой матрице - объект более релевантен
    pos_pairs = (rel_diff > 0).type(torch.float32)

    # 1 тут - объект менее релевантен
    neg_pairs = (rel_diff < 0).type(torch.float32)
    Sij = pos_pairs - neg_pairs
    return Sij


def compute_gain_diff(y_true, gain_scheme):
    if gain_scheme == "exp2":
        gain_diff = torch.pow(2.0, y_true) - torch.pow(2.0, y_true.t())
    elif gain_scheme == "diff":
        gain_diff = y_true - y_true.t()
    else:
        raise ValueError(f"{gain_scheme} method not supported")
    return gain_diff


if __name__ == "__main__":
    ys_true = torch.Tensor([3, 1, 3, 2, 3, 1, 2, 2])
    ys_pred = torch.Tensor([1, 2, 3, 3, 2, 2, 1, 1])
