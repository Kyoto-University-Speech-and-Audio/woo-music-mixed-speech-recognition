from itertools import permutations

import sys
import torch
import torch.nn.functional as F

EPS = 1e-8


def cal_loss(source, estimate_source, source_lengths):
    """
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B]
    """
    si_snr = cal_si_snr(source, estimate_source, source_lengths)
    loss = 0 - torch.mean(si_snr)
    return loss, si_snr, estimate_source



def cal_L1PMSE_loss(source, estimate_source, source_lengths):
    l1pmse = cal_L1PMSE(source, estimate_source, source_lengths)
    loss = torch.mean(l1pmse)
    return loss, l1pmse, estimate_source

def cal_L1P_sd_sdr_loss(source, estimate_source, source_lengths):
    """
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B]
    """
    l1p_sd_sdr = cal_L1P_sd_sdr(source, estimate_source, source_lengths)
    loss = 0 - torch.mean(l1p_sd_sdr)
    return loss, l1p_sd_sdr, estimate_source

def cal_sd_sdr_loss(source, estimate_source, source_lengths):
    """
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B]
    """
    sd_sdr = cal_sd_sdr(source, estimate_source, source_lengths)
    loss = 0 - torch.mean(sd_sdr)
    return loss, sd_sdr, estimate_source



def cal_si_snr(source, estimate_source, source_lengths):
    """Calculate SI-SNR with PIT training.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B], each item is between [0, T]
    """


    assert source.size() == estimate_source.size()
    B, C, T = source.size()
    # mask padding position along T
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with PIT
    # change to not use PIT

    # reshape to use broadcast
    s_target = zero_mean_target  # [B, C, T]
    s_estimate = zero_mean_estimate  # [B, C, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=2, keepdim=True)  # [B, C, 1]
    s_target_energy = torch.sum(s_target ** 2, dim=2, keepdim=True) + EPS  # [B, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, T]
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=2) / (torch.sum(e_noise ** 2, dim=2) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C]

    return pair_wise_si_snr


def cal_L1P_sd_sdr(source, estimate_source, source_lengths):
    """Calculate SI-SNR with PIT training.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B], each item is between [0, T]
    """



    assert source.size() == estimate_source.size()
    B, C, T = source.size()
    # mask padding position along T
    mask = get_mask(source, source_lengths)
    estimate_source *= mask



    # reshape to use broadcast
    s_target = source*mask  # [B, C, T]
    s_estimate = estimate_source  # [B, C, T]
    # alpha = <s', s> / ||s||^2 -> alpha * s_target = pair_wise_proj
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=2, keepdim=True)  # [B, C, 1]
    s_target_energy = torch.sum(s_target ** 2, dim=2, keepdim=True) + EPS  # [B, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, T]

    # e_noise = s_target - s'
    e_noise = s_target - s_estimate # [B, C, T]
    # Log 1 Plus SD-SDR = 10 * log_10(||alpha * s_target||^2 / (1 + ||e_noise||^2) ) 
    pair_wise_l1p_sd_sdr = (1 + torch.sum(pair_wise_proj ** 2, dim=2)) / (1 + torch.sum(e_noise ** 2, dim=2)) # [B, C]
    pair_wise_l1p_sd_sdr = 10 * torch.log10(pair_wise_l1p_sd_sdr + EPS)  # [B, C]

    return pair_wise_l1p_sd_sdr


def cal_sd_sdr(source, estimate_source, source_lengths):
    """Calculate SI-SNR with PIT training.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B], each item is between [0, T]
    """



    assert source.size() == estimate_source.size()
    B, C, T = source.size()
    # mask padding position along T
    mask = get_mask(source, source_lengths)
    estimate_source *= mask


    # Step 2. SI-SNR with PIT
    # change to not use PIT

    # reshape to use broadcast
    s_target = source*mask  # [B, C, T]
    s_estimate = estimate_source  # [B, C, T]
    # alpha = <s', s> / ||s||^2 -> alpha * s_target = pair_wise_proj
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=2, keepdim=True)  # [B, C, 1]
    s_target_energy = torch.sum(s_target ** 2, dim=2, keepdim=True) + EPS  # [B, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, T]

    # e_noise = s_target - s'
    e_noise = s_target - s_estimate # [B, C, T]
    # SD-SDR = 10 * log_10(||alpha * s_target||^2 / ||e_noise||^2)
    pair_wise_sd_sdr = torch.sum(pair_wise_proj ** 2, dim=2) / (torch.sum(e_noise ** 2, dim=2) + EPS) # [B, C]
    pair_wise_sd_sdr = 10 * torch.log10(pair_wise_sd_sdr + EPS)  # [B, C]

    return pair_wise_sd_sdr

def cal_L1PMSE(source, estimate_source, source_lengths):
    """Calculate SI-SNR with PIT training.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B], each item is between [0, T]
    """



    assert source.size() == estimate_source.size()
    B, C, T = source.size()
    # mask padding position along T
    mask = get_mask(source, source_lengths)
    estimate_source *= mask


    # 10 log10 (1 + sum( |s(t) - s'(t)| ** 2 ))
    s_target = source*mask  # [B, C, T]
    s_estimate = estimate_source  # [B, C, T]

    s_mse = torch.sum(torch.abs(s_target-s_estimate)**2, dim=2) # [B, C]
    s_l1pmse = 10 * torch.log10(1 + s_mse) # [B, C]

    return s_l1pmse


def reorder_source(source, perms, max_snr_idx):
    """
    Args:
        source: [B, C, T]
        perms: [C!, C], permutations
        max_snr_idx: [B], each item is between [0, C!)
    Returns:
        reorder_source: [B, C, T]
    """
    B, C, *_ = source.size()
    # [B, C], permutation whose SI-SNR is max of each utterance
    # for each utterance, reorder estimate source according this permutation
    max_snr_perm = torch.index_select(perms, dim=0, index=max_snr_idx)
    # print('max_snr_perm', max_snr_perm)
    # maybe use torch.gather()/index_select()/scatter() to impl this?
    reorder_source = torch.zeros_like(source)
    for b in range(B):
        for c in range(C):
            reorder_source[b, c] = source[b, max_snr_perm[b][c]]
    return reorder_source


def get_mask(source, source_lengths):
    """
    Args:
        source: [B, C, T]
        source_lengths: [B]
    Returns:
        mask: [B, 1, T]
    """
    B, _, T = source.size()
    mask = source.new_ones((B, 1, T))
    for i in range(B):
        mask[i, :, source_lengths[i]:] = 0
    return mask


if __name__ == "__main__":
    torch.manual_seed(123)
    B, C, T = 2, 3, 12
    # fake data
    source = torch.randint(4, (B, C, T))
    estimate_source = torch.randint(4, (B, C, T))
    source[1, :, -3:] = 0
    estimate_source[1, :, -3:] = 0
    source_lengths = torch.LongTensor([T, T-3])
    print('source', source)
    print('estimate_source', estimate_source)
    print('source_lengths', source_lengths)
    
    loss, max_snr, estimate_source, reorder_estimate_source = cal_loss(source, estimate_source, source_lengths)
    print('loss', loss)
    print('max_snr', max_snr)
    print('reorder_estimate_source', reorder_estimate_source)
