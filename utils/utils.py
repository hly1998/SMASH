import numpy as np
import torch
from tqdm import tqdm
import time
import pickle
import math
import scipy.io
import faiss
import random


def hash2number(x):
    result = x[0]
    for code in x[1:]:
        result = result * 2
        result = result + code
    return int(result)


def number2hash(x, latentDim):
    str_bit = bin(x)[2:]
    return_bits = np.zeros(latentDim)
    for idx, bit in enumerate(str_bit[::-1]):
        if bit == '1':
            return_bits[-(idx + 1)] = 1
        else:
            return_bits[-(idx + 1)] = 0
    return return_bits


def retrieve_topk(query_b, doc_b, topK, batch_size=100):
    n_bits = doc_b.size(1)
    n_train = doc_b.size(0)
    n_test = query_b.size(0)

    topScores = torch.cuda.ByteTensor(n_test,
                                      topK + batch_size).fill_(n_bits + 1)
    topIndices = torch.cuda.LongTensor(n_test, topK + batch_size).zero_()

    testBinmat = query_b.unsqueeze(2)

    for batchIdx in tqdm(range(0, n_train, batch_size), ncols=0, leave=False):
        s_idx = batchIdx
        e_idx = min(batchIdx + batch_size, n_train)
        numCandidates = e_idx - s_idx

        trainBinmat = doc_b[s_idx:e_idx]
        trainBinmat.unsqueeze_(0)
        trainBinmat = trainBinmat.permute(0, 2, 1)
        trainBinmat = trainBinmat.expand(testBinmat.size(0), n_bits,
                                         trainBinmat.size(2))

        testBinmatExpand = testBinmat.expand_as(trainBinmat)

        scores = (trainBinmat ^ testBinmatExpand).sum(dim=1)
        indices = torch.arange(start=s_idx, end=e_idx, step=1).type(
            torch.cuda.LongTensor).unsqueeze(0).expand(n_test, numCandidates)

        topScores[:, -numCandidates:] = scores
        topIndices[:, -numCandidates:] = indices

        topScores, newIndices = topScores.sort(dim=1)
        topIndices = torch.gather(topIndices, 1, newIndices)

    return topIndices


def compute_precision_at_k(retrieved_indices, query_labels, doc_labels, topK,
                           is_single_label):
    n_test = query_labels.size(0)

    Indices = retrieved_indices[:, :topK]
    if is_single_label:
        test_labels = query_labels.unsqueeze(1).expand(n_test, topK)
        topTrainLabels = [
            torch.index_select(doc_labels, 0, Indices[idx]).unsqueeze_(0)
            for idx in range(0, n_test)
        ]
        topTrainLabels = torch.cat(topTrainLabels, dim=0)
        relevances = (test_labels == topTrainLabels).type(
            torch.cuda.ShortTensor)
    else:
        topTrainLabels = [
            torch.index_select(doc_labels, 0, Indices[idx]).unsqueeze_(0)
            for idx in range(0, n_test)
        ]
        topTrainLabels = torch.cat(topTrainLabels,
                                   dim=0).type(torch.cuda.ShortTensor)
        test_labels = query_labels.unsqueeze(1).expand(
            n_test, topK, topTrainLabels.size(-1)).type(torch.cuda.ShortTensor)
        relevances = (topTrainLabels & test_labels).sum(dim=2)
        relevances = (relevances > 0).type(torch.cuda.ShortTensor)

    true_positive = relevances.sum(dim=1).type(torch.cuda.FloatTensor)
    true_positive = true_positive.div_(topK)
    prec_at_k = torch.mean(true_positive)
    return prec_at_k


def precisionK(train_b, test_b, train_y, test_y, bit_num, k=100):
    def bit2int(c):
        n = 0
        for x in c[:-1]:
            n = (n + x) << 1
        n = n + c[-1]
        return n

    def bit2int8(codes):
        '''
        bit to unit8
        '''
        bits_num = codes.shape[1]
        int8len = int(bits_num / 8)+1
        new_codes = []
        for code in codes:
            new_code = []
            for i in range(int8len):
                if i == int8len - 1:
                    new_code.append(bit2int(code[i*8:]))
                else:
                    new_code.append(bit2int(code[i*8:(i+1)*8]))
            new_codes.append(new_code)
        new_codes = np.array(new_codes).astype('uint8')
        return new_codes
    new_bit_num = math.ceil(bit_num / 8)*8
    train_b = train_b.cpu().numpy()
    test_b = test_b.cpu().numpy()
    train_y = train_y.cpu().numpy()
    test_y = test_y.cpu().numpy()
    train_b = bit2int8(train_b)
    test_b = bit2int8(test_b)
    index = faiss.IndexBinaryHash(new_bit_num, bit_num)
    index.add(train_b)
    stats = faiss.cvar.indexBinaryHash_stats
    index.nflip = bit_num
    stats.reset()
    D, I = index.search(test_b, k)
    acc = 0
    total = 0
    for doc_ids, l in zip(I, test_y):
        for doc_id in doc_ids:
            if train_y[doc_id] == l:
                acc = acc + 1
            total = total + 1
    acc = acc / total
    return acc


def compute_recall_at_k(retrieved_indices, query_labels, doc_labels, topK,
                        is_single_label):
    number_per_topic = {}
    for label in doc_labels:
        if label.item() not in number_per_topic:
            number_per_topic[label.item()] = 1
        else:
            number_per_topic[label.item()] = number_per_topic[label.item()] + 1
    n_test = query_labels.size(0)
    Indices = retrieved_indices[:, :topK]

    if is_single_label:
        test_labels = query_labels.unsqueeze(1).expand(n_test, topK)
        topTrainLabels = [
            torch.index_select(doc_labels, 0, Indices[idx]).unsqueeze_(0)
            for idx in range(0, n_test)
        ]
        topTrainLabels = torch.cat(topTrainLabels, dim=0)
        relevances = (test_labels == topTrainLabels).type(
            torch.cuda.ShortTensor)
    else:
        topTrainLabels = [
            torch.index_select(doc_labels, 0, Indices[idx]).unsqueeze_(0)
            for idx in range(0, n_test)
        ]
        topTrainLabels = torch.cat(topTrainLabels,
                                   dim=0).type(torch.cuda.ShortTensor)
        test_labels = query_labels.unsqueeze(1).expand(
            n_test, topK, topTrainLabels.size(-1)).type(torch.cuda.ShortTensor)
        relevances = (topTrainLabels & test_labels).sum(dim=2)
        relevances = (relevances > 0).type(torch.cuda.ShortTensor)
    true_positive = relevances.sum(dim=1).type(torch.cuda.FloatTensor)
    for i, label in enumerate(query_labels):
        true_positive[i] = true_positive[i] / number_per_topic[label.item()]
    recall_at_k = torch.mean(true_positive)
    return recall_at_k


def get_binary_code(model, train, test):
    train_zy = [(model.encode(xb.to(model.device))[0], yb)
                for xb, _, yb in train]
    train_z, train_y = zip(*train_zy)
    train_z = torch.cat(train_z, dim=0)
    train_y = torch.cat(train_y, dim=0)

    test_zy = [(model.encode(xb.to(model.device))[0], yb)
               for xb, _, yb in test]
    test_z, test_y = zip(*test_zy)
    test_z = torch.cat(test_z, dim=0)
    test_y = torch.cat(test_y, dim=0)

    mid_val, _ = torch.median(train_z, dim=0)
    train_b = (train_z > mid_val).type(torch.cuda.ByteTensor)
    test_b = (test_z > mid_val).type(torch.cuda.ByteTensor)

    del train_z
    del test_z

    return train_b, test_b, train_y, test_y


def get_binary_code_noise(model, train, test, noise_rate):
    def pnoise(doc_mat, noise_rate):
        noise_matrix = torch.bernoulli(
            torch.ones([doc_mat.shape[0], doc_mat.shape[1]]) - noise_rate).to(
                model.device)
        return noise_matrix * doc_mat

    train_zy = [(model.encode(xb.to(model.device))[0], yb) for xb, _, yb in train]
    train_z, train_y = zip(*train_zy)
    train_z = torch.cat(train_z, dim=0)
    train_y = torch.cat(train_y, dim=0)
    test_zy = [(model.encode(pnoise(xb.to(model.device), noise_rate))[0], yb)
               for xb, _, yb in test]
    test_z, test_y = zip(*test_zy)
    test_z = torch.cat(test_z, dim=0)
    test_y = torch.cat(test_y, dim=0)

    mid_val, _ = torch.median(train_z, dim=0)
    train_b = (train_z > mid_val).type(torch.cuda.ByteTensor)
    test_b = (test_z > mid_val).type(torch.cuda.ByteTensor)

    del train_z
    del test_z

    return train_b, test_b, train_y, test_y


def hamming_ball_retrieval(search_type, query_x, query_y, doc_x, doc_y,
                           latentDim, restrict):
    hash_bucket = {}
    for hash_code, label in zip(doc_x, doc_y):
        real = hash2number(hash_code)
        if real not in hash_bucket:
            hash_bucket[real] = []
            hash_bucket[real].append(label.item())
        else:
            hash_bucket[real].append(label.item())
    number_per_topic = {}
    for label in query_y:
        if label.item() not in number_per_topic:
            number_per_topic[label.item()] = 1
        else:
            number_per_topic[label.item()] = number_per_topic[label.item()] + 1
    precision = 0
    recall = 0
    if latentDim <= 12:
        with open('./index/ball_index_' + str(latentDim) + '_5.pickle',
                  'rb') as f:
            ball_search_index = pickle.load(f)
    elif latentDim == 14:
        with open('./index/ball_index_' + str(latentDim) + '_3.pickle',
                  'rb') as f:
            ball_search_index = pickle.load(f)
    else:
        with open('./index/ball_index_' + str(latentDim) + '_2.pickle',
                  'rb') as f:
            ball_search_index = pickle.load(f)
    total_success_knn = 0
    total_error_knn = 0
    if search_type == 'KNN':
        total_search_time = 0
        worst_search_time = 0
        for query, label in zip(query_x, query_y):
            has_found_num = 0
            success = 0
            query_number = hash2number(query)
            start_time = time.time()
            label_result = []
            if latentDim <= 14:
                max_radius = 3
            else:
                max_radius = 2
            for r in range(max_radius + 1):
                num_result = ball_search_index[query_number][r]
                for num in num_result:
                    if num in hash_bucket:
                        label_in_bucket = hash_bucket[num]
                        has_found_num = has_found_num + len(label_in_bucket)
                        if has_found_num >= restrict:
                            label_result = label_result + label_in_bucket[:restrict
                                                                          -
                                                                          len(label_result
                                                                              )]
                            success = 1
                            break
                        else:
                            label_result = label_result + label_in_bucket
                if success == 1:
                    break
            if success == 1:
                total_success_knn = total_success_knn + 1
            else:
                total_error_knn = total_error_knn + 1
            cost_time = time.time() - start_time
            if cost_time > worst_search_time:
                worst_search_time = cost_time
            total_search_time = total_search_time + cost_time
            precision = precision + label_result.count(label.item()) / restrict
            recall = recall + label_result.count(
                label.item()) / number_per_topic[label.item()]
        print("total_success_knn", total_success_knn, "total_error_knn",
              total_error_knn)
        return total_search_time / query_x.shape[
            0], worst_search_time, precision / query_x.shape[
                0], recall / query_x.shape[0]
    elif search_type == 'PLEB':
        total_return_text_num = 0
        worst_return_text_num = 0
        for query, label in zip(query_x, query_y):
            query_number = hash2number(query)
            return_text_num = 0
            label_result = []
            for r in range(restrict + 1):
                num_result = ball_search_index[query_number][r]
                for num in num_result:
                    label_in_bucket = hash_bucket[num]
                    label_result = label_result + label_in_bucket
                    return_text_num = return_text_num + len(label_in_bucket)
            if return_text_num > worst_return_text_num:
                worst_return_text_num = return_text_num
            total_return_text_num = total_return_text_num + return_text_num
            precision = precision + label_result.count(
                label.item()) / total_return_text_num
            recall = recall + label_result.count(
                label.item()) / number_per_topic[label.item()]
        return total_return_text_num / query_x.shape[
            0], worst_return_text_num, precision / query_x.shape[
                0], recall / query_x.shape[0]


def entropy_eval(train_x):
    def entropy(data):
        def Pi(aim, List):
            length = len(list(List))
            aimcount = (list(List)).count(aim)
            pi = (float)(aimcount / length)
            return pi

        data1 = np.unique(data)
        resultEn = 0
        for each in data1:
            pi = Pi(each, data)
            resultEn -= pi * math.log(pi, 2)
        return resultEn
    bucket_num_list = []
    for hash_code in train_x:
        number = hash2number(hash_code)
        bucket_num_list.append(int(number))
    e = entropy(bucket_num_list)
    return e


def entropy_eval_for_kde(train_x, train_y):
    '''
    calculate the entropy in each hash bucket
    '''
    def entropy(data):
        def Pi(aim, List):
            length = len(list(List))
            aimcount = (list(List)).count(aim)
            pi = (float)(aimcount / length)
            return pi

        data1 = np.unique(data)
        resultEn = 0
        for each in data1:
            pi = Pi(each, data)
            resultEn -= pi * math.log(pi, 2)
        return resultEn
    bucket_num_list = {}
    for hash_code, tag in zip(train_x, train_y):
        number = hash2number(hash_code)
        if number not in bucket_num_list.keys():
            bucket_num_list[int(number)] = []
            bucket_num_list[int(number)].append(int(tag))
        else:
            bucket_num_list[int(number)].append(int(tag))
    bucket_entropy_list = []
    for tag_list in bucket_num_list.values():
        bucket_entropy_list.append(entropy(tag_list))
    return bucket_entropy_list


def setup_seed(seed):
    '''
    set the random seed for reproduction
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True