import argparse
import os
import torch
from utils import *
import torch.optim as optim
from model import *
from scipy import sparse
from scipy.sparse import csc_matrix
import numpy as np


def train(args):
    if not args.gpunum:
        parser.error("Need to provide the GPU number.")
    if not args.dataset:
        parser.error("Need to provide the dataset.")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpunum
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset, data_fmt = args.dataset.split('.')
    if dataset in ['reuters', 'tmc', 'rcv1', 'edudata', 'edudata20']:
        single_label_flag = False
    else:
        single_label_flag = True
    if single_label_flag:
        train_set = SingleLabelTextDataset('dataset/{}'.format(dataset),
                                           subset='train',
                                           bow_format=data_fmt,
                                           download=True)
        test_set = SingleLabelTextDataset('dataset/{}'.format(dataset),
                                          subset='test',
                                          bow_format=data_fmt,
                                          download=True)
    else:
        train_set = MultiLabelTextDataset('dataset/{}'.format(dataset),
                                          subset='train',
                                          bow_format=data_fmt,
                                          download=True)
        test_set = MultiLabelTextDataset('dataset/{}'.format(dataset),
                                         subset='test',
                                         bow_format=data_fmt,
                                         download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=args.batch_size,
                                              shuffle=True)
    setup_seed(2023)
    num_bits = args.nbits
    n_sample = args.n_sample
    top_k = args.top_k
    num_features = train_set[0][0].size(0)
    best_precision = 0
    best_precision_epoch = 0
    time_max = int(len(train_set) / args.batch_size)
    model = SMASH(dataset,
                  num_features,
                  num_bits,
                  dropoutProb=0.1,
                  time_max=time_max,
                  em_alpha=args.em_alpha,
                  sigma=args.sigma,
                  device=device,
                  em_length=len(train_set))
    model.to(device)
    num_epochs = args.num_epochs
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if dataset == 'ng20':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=5e3,
                                                    gamma=0.96)
    L_max = 0
    L_t_minus_1 = 0
    transfrom_flag = False
    transfrom_count = 0
    for epoch in range(num_epochs):
        total_loss = []
        reconstr_loss = []
        propagation_loss = []
        balance_loss = []
        model.train()
        for _, (xb, idxs, yb) in enumerate(train_loader):
            # print(L_max, L_t_minus_1)
            xb = xb.to(device)
            yb = yb.to(device)
            loss_change_term = np.abs(L_max - L_t_minus_1) / (L_max + 0.00001)
            logprob_w, logprob_w_noise, z, z_noise, long_z, noise_long_z_e, em_out = model(xb, idxs, epoch, loss_change_term, n_sample)
            # 计算epsilon
            # 这里这个值是手动设置，和长bit code长度一致
            noise_z_reshape = noise_long_z_e.reshape(xb.shape[0], 128, n_sample)
            mult = torch.matmul(long_z.unsqueeze(1), noise_z_reshape).squeeze(1)
            epsilon = torch.nn.functional.softmax(mult, dim=1)
            # print(logprob_w_noise.shape, epsilon.shape)

            rec_loss = compute_reconstr_loss(logprob_w, xb) + compute_reconstr_noise_loss(logprob_w_noise, xb, epsilon)
            pro_loss = relevance_propagation_v1(z, long_z) + relevance_propagation_v1(z_noise, noise_long_z_e)
            loss = rec_loss + args.lsc_weight * pro_loss
            if transfrom_flag:
                ba_loss = code_balance_global_v2(num_bits,
                                                    em_out,
                                                    z,
                                                    alpha=args.bb_weight,
                                                    beta=args.bd_weight)
                loss = loss + ba_loss
            # if loss.item() > L_max:
            #     L_max = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            L_t_minus_1 = loss.item()
            total_loss.append(loss.item())
            reconstr_loss.append(rec_loss.item())
            propagation_loss.append(pro_loss.item())
            if transfrom_flag:
                balance_loss.append(ba_loss.item())
            else:
                balance_loss.append(0)
        model.eval()
        if np.mean(total_loss) > L_max:
            L_max = np.mean(total_loss)
        with torch.no_grad():
            train_b, test_b, train_y, test_y = model.get_binary_code(
                train_loader, test_loader)
            precision = precisionK(train_b, test_b, train_y, test_y, num_bits)
        if precision < best_precision and epoch > 10:
            transfrom_count = transfrom_count + 1
        else:
            transfrom_count = 0
        if transfrom_count == 5:
            transfrom_flag = True
        if precision > best_precision:
            best_precision = precision
            best_precision_epoch = epoch + 1
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_recall': best_precision,
            }
            save_file = os.path.join(
                './checkpoint',
                '{},{},{},{},{}_best.pth'.format(args.dataset, args.nbits,
                                                    args.bb_weight,
                                                    args.bd_weight,
                                                    args.lsc_weight))
            print('saving the best model!')
            torch.save(state, save_file)
        print(
            'total_loss:{:.4f} reconstr_loss:{:.4f} propagation_loss:{:.4f} balance_loss:{:.4f} MAX_Loss: {:.4f} Best Precision:({}){:.4f} Now Precision:({}){:.4f}'
            .format(np.mean(total_loss), np.mean(reconstr_loss), 
                    np.mean(propagation_loss),
                    np.mean(balance_loss), L_max, best_precision_epoch,
                    best_precision, epoch + 1, precision))
    return best_precision, best_precision_epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-g",
                        "--gpunum",
                        help="GPU number to train the model.")
    parser.add_argument("-d", "--dataset", help="Name of the dataset.")
    parser.add_argument("-b",
                        "--nbits",
                        help="Number of bits of the embedded vector.",
                        type=int)
    parser.add_argument("--dropout",
                        help="Dropout probability (0 means no dropout)",
                        default=0.1,
                        type=float)
    parser.add_argument("--num_epochs", default=30, type=int)
    parser.add_argument("--raduis_r", default=1, type=int)
    parser.add_argument("--top_k", default=100, type=int)
    parser.add_argument("--model", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--lsc_weight", default=0.5, type=float)
    parser.add_argument("--bb_weight", default=1, type=float)
    parser.add_argument("--bd_weight", default=0.01, type=float)
    parser.add_argument("--em_alpha", default=0.5, type=float)
    parser.add_argument("--sigma", default=0.6, type=float)
    parser.add_argument("--n_sample", default=3, type=int)
    args = parser.parse_args()
    train(args)
