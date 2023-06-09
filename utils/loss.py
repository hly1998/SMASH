import torch
import torch.nn.functional as F

def compute_reconstr_loss(logprob_word, doc_mat):
    return -torch.mean(torch.sum(logprob_word * doc_mat, dim=1))


def compute_reconstr_noise_loss(logprob_word, doc_mat, epsilon):
    logprob_word = logprob_word.reshape(int(logprob_word.shape[0] / doc_mat.shape[0]), -1, doc_mat.shape[1])
    return -torch.mean(epsilon.t() * torch.sum(logprob_word * doc_mat, dim=2))


def relevance_propagation_v1(z, long_z):
    a = torch.mm(z, z.t()) / z.size()[-1]
    b = torch.mm(long_z, long_z.t()) / long_z.size()[-1]
    c = 1 - torch.eye(z.size()[0]).cuda()
    a = a * c
    b = b * c
    dp_loss = torch.sum(torch.abs(a - b)) / (z.size()[0] * (z.size()[0] - 1))
    return dp_loss


def code_balance_global_v2(num_bits, em_out, batch, alpha=1.0, beta=1.0):
    batch_size = batch.shape[0]
    balance_w = torch.nn.functional.softmax(torch.abs(torch.sum(em_out, dim=0)))
    bit_balance_loss = torch.sum(torch.abs(torch.sum(batch, dim=0)).mul(balance_w)) / (num_bits)
    I_matrix = torch.eye(num_bits).cuda()
    em_size = em_out.shape[0]
    uncorrelation_w = torch.nn.functional.softmax(torch.abs(em_out.t().mm(em_out) / em_size - I_matrix))
    bit_uncorrelation_loss = torch.pow(torch.norm((batch.t().mm(batch) / batch_size - I_matrix).mul(uncorrelation_w)), 2) / (num_bits * num_bits)
    loss = alpha * bit_balance_loss + beta * bit_uncorrelation_loss
    return loss
