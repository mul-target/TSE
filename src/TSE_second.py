import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils import resample
import numpy as np


def attentionScores(var):
    Q_x, K_t, V_t = var[0], var[1], var[2]
    scores = torch.matmul(Q_x, K_t.transpose(0, 1))
    distribution = F.softmax(scores, dim=-1)
    scores = torch.matmul(distribution, V_t)
    return scores


def attentionScoresIRA(var):
    Q_x, K_t, V_t, Q_e, K_e, V_e = var[0], var[1], var[2], var[3], var[4], var[5]


    score_t = torch.matmul(Q_x, K_e.transpose(0, 1))
    distribution_t = F.softmax(score_t, dim=-1)
    scores_t = torch.matmul(distribution_t, V_t)

    score_e = torch.matmul(Q_e, K_t.transpose(0, 1))
    distribution_e = F.softmax(score_e, dim=-1)
    scores_e = torch.matmul(distribution_e, V_e)

    IRAScores = torch.cat([scores_t, scores_e], dim=-1)

    return IRAScores



def attentionScoresIRA1(var):
    Q_x, K_t, V_t, Q_e, K_e, V_e = var[0], var[1], var[2], var[3], var[4], var[5]

    score_t = torch.matmul(Q_x, K_e.transpose(0, 1))
    distribution_t = F.softmax(score_t, dim=-1)
    scores_e = torch.matmul(distribution_t, V_e)

    score_e = torch.matmul(Q_e, K_t.transpose(0, 1))
    distribution_e = F.softmax(score_e, dim=-1)
    scores_t = torch.matmul(distribution_e, V_t)

    IRAScores = torch.cat([scores_t, scores_e], dim=-1)
    return IRAScores


def create_resample(train_sequence, train_sequence_topic, train_enc, train_enc_senti, batch_size=32):
    train_sequence = np.array(train_sequence)
    train_sequence_topic = np.array(train_sequence_topic)
    train_enc = np.array(train_enc)
    train_enc_senti = np.array(train_enc_senti)

    combined_data = np.column_stack((train_sequence, train_sequence_topic, train_enc, train_enc_senti))

    combined_data_tensor = torch.tensor(combined_data, dtype=torch.float32)

    blv = combined_data_tensor[combined_data_tensor[:, 2] == 0]
    print("len blv", len(blv))
    deny = combined_data_tensor[combined_data_tensor[:, 2] == 1]
    print("len deny", len(deny))

    upsampled1 = resample(deny.numpy(), replace=True, n_samples=len(blv), random_state=27)
    upsampled1 = torch.tensor(upsampled1, dtype=torch.float32)

    upsampled = torch.cat((blv, upsampled1))

    indices = torch.randperm(len(upsampled))
    upsampled = upsampled[indices]

    print("After oversample train data:", len(upsampled))
    print("After oversampling, instances of tweet act classes in oversampled data:", torch.sum(upsampled[:, 2] == 1))

    train_sequence_tensor = upsampled[:, 0]
    train_sequence_topic_tensor = upsampled[:, 1]
    train_enc_tensor = upsampled[:, 2]
    train_enc_senti_tensor = upsampled[:, 3]

    train_sequence = train_sequence_tensor.tolist()
    train_sequence_topic = train_sequence_topic_tensor.tolist()
    train_enc = train_enc_tensor.tolist()
    train_enc_senti = train_enc_senti_tensor.tolist()

    return train_sequence, train_sequence_topic, train_enc, train_enc_senti


