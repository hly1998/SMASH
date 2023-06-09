import torch
from torch.autograd import Function
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")

class ExemplarMemory(Function):
    @staticmethod
    def forward(ctx, inputs, idxs, em, em_time_count, time_max, em_alpha, loss_change_term):
        ctx.save_for_backward(inputs, idxs, em, em_time_count)
        for idx in idxs:
            em_time_count[idx] = time_max
        em_out = em[(em_time_count > ((1 - loss_change_term) * (1 - em_alpha) * time_max)).nonzero()].squeeze(dim=1)
        return em_out

    @staticmethod
    def backward(ctx, grad_output):
        inputs, idxs, em, em_time_count = ctx.saved_tensors
        # grad_inputs = grad_output.clone()[:inputs.shape[0], :]
        for idx, z_code in zip(idxs, inputs):
            em[idx] = torch.sign(z_code)
        # return grad_inputs, None, None
        return None, None, None, None, None, None, None


class ExemplarMemory_warm_up(Function):
    @staticmethod
    def forward(ctx, inputs, idxs, em, em_time_count, time_max, em_alpha, loss_change_term):
        ctx.save_for_backward(inputs, idxs, em, em_time_count)
        # em_time_count = em_time_count - 1
        for idx, z_code in zip(idxs, inputs):
            em[idx] = torch.sign(z_code)
            em_time_count[idx] = time_max
            # print(em_time_count[idx], idx)
        em_out = em[(em_time_count > ((1 - loss_change_term) * (1 - em_alpha) * time_max)).nonzero()].squeeze(dim=1)
        return em_out

    @staticmethod
    def backward(ctx, grad_output):
        inputs, idxs, em, em_time_count = ctx.saved_tensors
        return None, None, None, None, None, None, None


class SMASH(nn.Module):
    def __init__(self,
                 dataset,
                 vocabSize,
                 latentDim,
                 device,
                 em_length,
                 time_max,
                 em_alpha,
                 sigma=0.3,
                 dropoutProb=0.):
        super(SMASH, self).__init__()

        self.dataset = dataset
        self.hidden_dim = 1000
        self.long_bit_code = 128
        self.vocabSize = vocabSize
        self.latentDim = latentDim
        self.dropoutProb = dropoutProb
        self.device = device
        self.time_max = time_max
        self.em_alpha = em_alpha
        self.sigma = sigma
        self.em = nn.Parameter(torch.zeros([em_length, latentDim]))
        self.em_time_count = torch.ones(em_length) * time_max
        self.encoder = nn.Sequential(
            nn.Linear(self.vocabSize, self.hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(p=dropoutProb))
        self.encoder_long_bit_code = nn.Sequential(
            nn.Linear(self.hidden_dim, self.long_bit_code), nn.Tanh())

        self.longz_to_z = nn.Linear(self.long_bit_code, self.latentDim)
        self.decoder = nn.Sequential(nn.Linear(self.latentDim, self.vocabSize),
                                     nn.LogSoftmax(dim=1))

    def encode(self, doc_mat):
        h = self.encoder(doc_mat)
        long_z = self.encoder_long_bit_code(h)
        z = self.longz_to_z(long_z)
        z = torch.nn.functional.tanh(z)
        return z

    def forward(self, document_mat, idxs, epoch, loss_change_term, n_sample=3):
        document_mat_noise = self.add_noise(document_mat, n_sample)
        h = self.encoder(document_mat)
        noise_h = self.encoder(document_mat_noise)
        long_z_e = self.encoder_long_bit_code(h)
        noise_long_z_e = self.encoder_long_bit_code(noise_h)
        z = self.longz_to_z(long_z_e)
        z_noise = self.longz_to_z(noise_long_z_e)
        z = torch.nn.functional.tanh(z)
        z_noise = torch.nn.functional.tanh(z_noise)
        self.em_time_count = self.em_time_count - 1
        if epoch < 10:
            em_out = ExemplarMemory_warm_up.apply(z, idxs, self.em, self.em_time_count, self.time_max, self.em_alpha, loss_change_term)
        else:
            em_out = ExemplarMemory.apply(z, idxs, self.em, self.em_time_count, self.time_max, self.em_alpha, loss_change_term)
        prob_w = self.decoder(z)
        noise_prob_w = self.decoder(z_noise)
        return prob_w, noise_prob_w, z, z_noise, long_z_e, noise_long_z_e, em_out

    def add_noise(self, doc_mat, n_sample=3):
        x = torch.abs(torch.normal(torch.zeros([n_sample, doc_mat.shape[0], doc_mat.shape[1]]), 1 - self.sigma))
        x = torch.where(x > 1, torch.tensor(1.0), x)
        noise_matrix = torch.bernoulli(x).to(self.device)
        return torch.reshape(noise_matrix * doc_mat, [-1, doc_mat.shape[1]])

    def get_name(self):
        return "SMASH"

    def get_binary_code(self, train, test):
        train_zy = [(self.encode(xb.to(self.device)), yb)
                    for xb, _, yb in train]
        train_z, train_y = zip(*train_zy)
        train_z = torch.cat(train_z, dim=0)
        train_y = torch.cat(train_y, dim=0)

        test_zy = [(self.encode(xb.to(self.device)), yb) for xb, _, yb in test]
        test_z, test_y = zip(*test_zy)
        test_z = torch.cat(test_z, dim=0)
        test_y = torch.cat(test_y, dim=0)

        mid_val, _ = torch.median(train_z, dim=0)
        train_b = (train_z > mid_val).type(torch.cuda.ByteTensor)
        test_b = (test_z > mid_val).type(torch.cuda.ByteTensor)

        del train_z
        del test_z

        return train_b, test_b, train_y, test_y
