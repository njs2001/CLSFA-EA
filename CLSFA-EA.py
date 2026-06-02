# _*_ coding:utf-8 _*_
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import json
import os
import string
import pickle
import lap
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from zhon.hanzi import punctuation
import torch
import torch.nn.functional as F
from torch.optim import Adam
import torch.nn as nn
import torch.sparse as sparse
from utilstorch import load_triples, load_aligned_pair, test
import warnings

warnings.filterwarnings('ignore')


seed = 12345
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    torch.cuda.set_device(0)
else:
    print("No GPU detected, training will be performed on CPU")

ent_names = json.load(open("", "r",encoding="utf-8"))
file_path = ""
d_2gram = {}
d_3gram = {}
count_2gram = 0
count_3gram = 0
no = 0
depth = 3


word_vecs = {}
glove_path = ""
with open(glove_path, encoding='UTF-8') as f:
    for line in tqdm(f.readlines(), desc="Loading GloVe vectors"):
        line = line.split()
        word_vecs[line[0]] = np.array([float(x) for x in line[1:]])

pickle.dump(word_vecs, open("word_vecs.pkl", "wb"))
word_vecs = pickle.load(open("word_vecs.pkl", "rb"))


# 去标点
def remove_punc(str, punc=None):
    if punc is None:
        punc = PUNC
    if punc == '':
        return str
    return ''.join(['' if i in punc else i for i in str])



def get_punctuations():
    en = string.punctuation
    zh = punctuation
    puncs = set()
    for i in (zh + en):
        puncs.add(i)
    return puncs


all_triples, node_size, rel_size = load_triples(file_path, True)
train_pair, test_pair = load_aligned_pair(file_path, ratio=0.3)
triples_df = pd.DataFrame(all_triples, columns=["head", "rel", "tail"])
train_num = len(train_pair)
test_num = len(test_pair)
PUNC = get_punctuations()

entity_num = len(train_pair) + len(test_pair)

train_kg1_ids = train_pair[:, 0].tolist()
train_kg2_ids = train_pair[:, 1].tolist()


def load_rel_emotion_mapping(path):
    rel_emotion_map = {}
    if not os.path.exists(path):
        print(f"Warning: Emotion mapping file '{path}' not found.")
        return rel_emotion_map
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    rel_id = int(parts[0])
                    emotion_label = int(parts[1])
                    confidence = float(parts[2])
                    rel_emotion_map[rel_id] = (emotion_label, confidence)
                except ValueError:
                    print(f"Warning！ '{line}'")
    print(f"Successfully loaded {len(rel_emotion_map)} relation emotion mappings from '{path}'")
    return rel_emotion_map


def batch_get_entity_emotion_features(node_size, all_triples, rel_emotion_map_kg1, rel_emotion_map_kg2):
    entity_emotion = np.zeros((node_size, 3))  # [neg, pos, neu]
    entity_total_conf = np.zeros(node_size)
    triples_np = np.array(all_triples, dtype=np.int64)
    heads = triples_np[:, 0]
    rels = triples_np[:, 1]
    tails = triples_np[:, 2]

    label_weights = {-1: 1.0, 1: 1.0, 0: 0.8}

    unique_rels, counts = np.unique(rels, return_counts=True)
    rel_freq_map = dict(zip(unique_rels, counts))

    for h, r, t in tqdm(zip(heads, rels, tails), desc="Calculating entity emotion features", total=len(heads)):
        if r in rel_emotion_map_kg1:
            lbl, conf = rel_emotion_map_kg1[r]
        elif r in rel_emotion_map_kg2:
            lbl, conf = rel_emotion_map_kg2[r]
        else:
            continue

        if (lbl == 1 or lbl == -1) and conf < 0.4:
            lbl = 0

        rel_weight = 1 / np.log(rel_freq_map.get(r, 1) + 1)
        weighted_conf = conf * rel_weight

        if lbl in label_weights:
            col_idx = 0 if lbl == -1 else 1 if lbl == 1 else 2
            entity_emotion[h, col_idx] += weighted_conf * label_weights[lbl]
            entity_total_conf[h] += weighted_conf * label_weights[lbl]

        if lbl in label_weights:
            col_idx = 0 if lbl == -1 else 1 if lbl == 1 else 2
            entity_emotion[t, col_idx] += weighted_conf * label_weights[lbl]
            entity_total_conf[t] += weighted_conf * label_weights[lbl]

    mask = entity_total_conf > 0
    entity_emotion[mask] = entity_emotion[mask] / entity_total_conf[mask].reshape(-1, 1)
    rand_vec = np.random.random(((~mask).sum(), 3))
    rand_vec = rand_vec / rand_vec.sum(axis=1, keepdims=True)
    entity_emotion[~mask] = rand_vec

    return entity_emotion


def compute_emotion_loss(kg1_emo_feat, pos_kg2_emo_feat, neg_kg2_emo_feat,K):
    pos_emo_sim = F.cosine_similarity(kg1_emo_feat, pos_kg2_emo_feat, dim=1)
    pos_emo_loss = -torch.log(torch.sigmoid(pos_emo_sim) + 1e-8).mean()
    neg_kg2_emo_feat = neg_kg2_emo_feat.reshape(-1, K, 3)
    neg_emo_sim = F.cosine_similarity(kg1_emo_feat.unsqueeze(1), neg_kg2_emo_feat, dim=2)
    neg_emo_loss = -torch.log(1 - torch.sigmoid(neg_emo_sim) + 1e-8).mean()
    return pos_emo_loss + neg_emo_loss


rel_emotion_map_kg1 = load_rel_emotion_mapping("")
rel_emotion_map_kg2 = load_rel_emotion_mapping("")

corpus = ["" for _ in range(node_size)]  # node_size=38154，下标0~38153
for item in ent_names:
    ent_id = item[0]
    if ent_id < 0 or ent_id >= node_size:
        continue
    corpus[ent_id] = ' '.join(item[1])
    corpus[ent_id] = remove_punc(corpus[ent_id].lower())

for name in corpus:
    if name is None or not isinstance(name, str) or not name.strip():
        continue
    for word in name.split():
        if word not in word_vecs:
            no += 1
        for idx in range(len(word) - 1):
            if word[idx:idx + 2] not in d_2gram:
                d_2gram[word[idx:idx + 2]] = count_2gram
                count_2gram += 1
        for idx in range(len(word) - 2):
            if word[idx:idx + 3] not in d_3gram:
                d_3gram[word[idx:idx + 3]] = count_3gram
                count_3gram += 1


tokenizer = lambda x: x.split() if x and isinstance(x, str) else []
vectorizer = CountVectorizer(tokenizer=tokenizer)
X = vectorizer.fit_transform(corpus)
word = vectorizer.get_feature_names_out()
transform = TfidfTransformer()
Y = transform.fit_transform(X)


def get_tfikgf(doc):
    tfikgf = {}
    feature_index = Y[doc, :].nonzero()[1]
    feature_names = vectorizer.get_feature_names_out()
    tfikgf_scores = zip(feature_index, [Y[doc, x] for x in feature_index])
    for w, s in [(feature_names[i], s) for (i, s) in tfikgf_scores]:
        tfikgf[w] = s
    doc_name = corpus[doc]
    if isinstance(doc_name, str) and doc_name.strip():
        for word in doc_name.split():
            if word not in tfikgf:
                tfikgf[word] = 0.0
    return tfikgf


tfikgf_list = []
for i, _ in tqdm(ent_names):
    tfikgf = get_tfikgf(i)
    tfikgf_list.append(tfikgf)
pickle.dump(tfikgf_list, open("fren_nopunc_tfikgf_list_2+3gram.pkl", "wb"))
tfikgf_list = pickle.load(open("fren_nopunc_tfikgf_list_2+3gram.pkl", "rb"))


ent_vec = np.zeros((node_size, 300))

char_vec = np.zeros((node_size, count_2gram + count_3gram))


for i, name in tqdm(enumerate(corpus), desc="Generating word + 2+3gram features"):

    if name is None or not isinstance(name, str) or not name.strip():
        ent_vec[i] = np.random.random(300) - 0.5
        char_vec[i] = np.random.random(count_2gram + count_3gram) - 0.5
        continue
    k = 0
    tfikgf = tfikgf_list[i]
    ent_vec_list = []
    ent_tfikgf_list = []
    for word in name.split():

        if word in word_vecs:
            ent_vec_list.append(word_vecs[word])
            ent_tfikgf_list.append(tfikgf.get(word, 0.0))
            k += 1

        for idx in range(len(word) - 1):
            gram_2 = word[idx:idx + 2]
            if gram_2 in d_2gram:
                char_vec[i, d_2gram[gram_2]] += 1

        for idx in range(len(word) - 2):
            gram_3 = word[idx:idx + 3]
            if gram_3 in d_3gram:
                char_vec[i, count_2gram + d_3gram[gram_3]] += 1

    if k:
        for j in range(len(ent_vec_list)):
            ent_vec[i] += ent_vec_list[j] * ent_tfikgf_list[j]
    else:
        ent_vec[i] = np.random.random(300) - 0.5

    if np.sum(char_vec[i]) == 0:
        char_vec[i] = np.random.random(count_2gram + count_3gram) - 0.5
    ent_vec[i] = ent_vec[i] / (np.linalg.norm(ent_vec[i]) + 1e-8)
    char_vec[i] = char_vec[i] / (np.linalg.norm(char_vec[i]) + 1e-8)


dh = {}
dr = {}
dt = {}
for x, r, y in all_triples:
    if x not in dh:
        dh[x] = 0
    dh[x] += 1
    if y not in dt:
        dt[y] = 0
    dt[y] += 1
    if r not in dr:
        dr[r] = 0
    dr[r] += 1

sparse_rel_matrix_indices = []
sparse_rel_matrix_values = []
for i in range(node_size):
    sparse_rel_matrix_indices.append([i, i])
    sparse_rel_matrix_values.append(np.log(len(all_triples) / node_size))
for h, r, t in all_triples:
    sparse_rel_matrix_indices.append([h, t])
    sparse_rel_matrix_values.append(
        np.log(len(all_triples) * len(all_triples) * len(all_triples) / (dr[r] * dh[h] * dt[t]))
    )
sparse_rel_matrix_indices = torch.LongTensor(sparse_rel_matrix_indices).t()
sparse_rel_matrix_values = torch.FloatTensor(sparse_rel_matrix_values)
sparse_rel_matrix = torch.sparse_coo_tensor(
    indices=sparse_rel_matrix_indices,
    values=sparse_rel_matrix_values,
    size=(node_size, node_size)
).to(device)

entity_emotion_features = batch_get_entity_emotion_features(node_size, all_triples, rel_emotion_map_kg1,
                                                            rel_emotion_map_kg2)

class EmotionAttention(nn.Module):
    def __init__(self, emotion_dim=3):
        super(EmotionAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(emotion_dim, emotion_dim),
            nn.ReLU(),
            nn.Linear(emotion_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn_weight = self.attention(x)
        return x * attn_weight


emotion_attn = EmotionAttention().to(device)
emotion_feature = torch.FloatTensor(entity_emotion_features).to(device)
emotion_feature.requires_grad = True
emotion_feature_weighted = emotion_attn(emotion_feature)


feature = np.concatenate([ent_vec, char_vec], axis=-1)
feature = np.concatenate([feature, emotion_feature_weighted.detach().cpu().numpy()], axis=-1)
feature = torch.FloatTensor(feature).to(device)
feature = F.normalize(feature, p=2, dim=-1)
feature.requires_grad = True


def sample_hard_negative_pairs(train_pair, kg2_ent_vec, kg2_entity_ids, kg2_id_to_idx, neg_num=10, top_k=30):
    neg_pairs = []
    train_kg2_pos_ids = set(train_pair[:, 1])
    for a, b in tqdm(train_pair, desc="Sampling hard negative pairs"):
        if b not in kg2_id_to_idx:
            valid_neg_candidates = [c for c in kg2_entity_ids if c != b and c in train_kg2_pos_ids]
            if len(valid_neg_candidates) < neg_num:
                selected_neg = valid_neg_candidates
            else:
                selected_neg = np.random.choice(valid_neg_candidates, size=neg_num, replace=False)
        else:
            b_idx = kg2_id_to_idx[b]
            b_vec = kg2_ent_vec[b_idx:b_idx + 1]
            kg2_sims = np.matmul(kg2_ent_vec, b_vec.T).squeeze()
            candidate_indices = []
            for idx, entity_id in enumerate(kg2_entity_ids):
                if entity_id != b and entity_id in train_kg2_pos_ids:
                    candidate_indices.append(idx)
            if not candidate_indices:
                selected_neg = []
            else:
                candidate_sims = kg2_sims[candidate_indices]
                top_k_indices = np.argsort(-candidate_sims)[:top_k]
                top_k_candidate_ids = [kg2_entity_ids[candidate_indices[idx]] for idx in top_k_indices]
                if len(top_k_candidate_ids) < neg_num:
                    selected_neg = top_k_candidate_ids
                else:
                    selected_neg = np.random.choice(top_k_candidate_ids, size=neg_num, replace=False)
        for c in selected_neg:
            neg_pairs.append([a, c])
    return np.array(neg_pairs)


kg2_entity_ids = np.array(train_kg2_ids)
kg2_id_to_idx = {entity_id: idx for idx, entity_id in enumerate(kg2_entity_ids)}
kg2_ent_vec = ent_vec[kg2_entity_ids]
kg2_ent_vec = kg2_ent_vec / (np.linalg.norm(kg2_ent_vec, axis=1, keepdims=True) + 1e-8)

neg_pairs = sample_hard_negative_pairs(
    train_pair=train_pair,
    kg2_ent_vec=kg2_ent_vec,
    kg2_entity_ids=kg2_entity_ids,
    kg2_id_to_idx=kg2_id_to_idx,
    neg_num=10,
    top_k=30
)

optimizer = Adam([
    {'params': feature},
    {'params': emotion_feature},
    {'params': emotion_attn.parameters()}
], lr=2e-3)
epochs = 50
tau = 0.1
EMOTION_LOSS_WEIGHT = 0.1

train_pos_pairs = torch.LongTensor(train_pair).to(device)
train_neg_pairs = torch.LongTensor(neg_pairs).to(device)
pos_kg1_ids = train_pos_pairs[:, 0]
pos_kg2_ids = train_pos_pairs[:, 1]
neg_kg2_ids = train_neg_pairs[:, 1]
K = 10

for epoch in range(epochs):
    kg1_feature = torch.index_select(feature, 0, pos_kg1_ids)
    pos_kg2_feature = torch.index_select(feature, 0, pos_kg2_ids)
    neg_kg2_feature = torch.index_select(feature, 0, neg_kg2_ids)

    kg1_emo_feat = torch.index_select(emotion_feature, 0, pos_kg1_ids)
    pos_kg2_emo_feat = torch.index_select(emotion_feature, 0, pos_kg2_ids)
    neg_kg2_emo_feat = torch.index_select(emotion_feature, 0, neg_kg2_ids)

    neg_kg2_feature = neg_kg2_feature.view(-1, K, kg1_feature.size(1))

    pos_sim = F.cosine_similarity(
        kg1_feature,
        pos_kg2_feature,
        dim=1
    ) / tau

    neg_sim = F.cosine_similarity(
        kg1_feature.unsqueeze(1),  # [N, 1, d]
        neg_kg2_feature,  # [N, K, d]
        dim=2
    ) / tau

    logits = torch.cat(
        [pos_sim.unsqueeze(1), neg_sim],
        dim=1
    )

    labels = torch.zeros(logits.size(0), dtype=torch.long).to(device)
    original_contrastive_loss = F.cross_entropy(logits, labels)

    emotion_loss = compute_emotion_loss(kg1_emo_feat, pos_kg2_emo_feat, neg_kg2_emo_feat,K)
    total_loss = original_contrastive_loss + EMOTION_LOSS_WEIGHT * emotion_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    feature.data = F.normalize(feature.data, p=2, dim=-1)
    emotion_feature.data = F.normalize(emotion_feature.data, p=2, dim=-1)


def cal_sims(test_pair, feature):
    test_pair = torch.LongTensor(test_pair).to(device)
    feature_a = torch.index_select(feature, 0, test_pair[:, 0])
    feature_b = torch.index_select(feature, 0, test_pair[:, 1])
    return torch.matmul(feature_a, feature_b.T)


sims = cal_sims(test_pair, feature)
for i in tqdm(range(depth), leave=False):
    feature = torch.sparse.mm(sparse_rel_matrix, feature)
    feature = F.normalize(feature, p=2, dim=-1)
    sims += cal_sims(test_pair, feature)
sims /= depth + 1
sims = sims.detach().cpu().numpy()

# LAPJV
cost, x, y = lap.lapjv(-sims)
print(f"lapjv Assessment result: {np.sum(x == y) / test_num:.4f}")

# Sinkhorn
sims_tf = torch.FloatTensor(sims).to(device)
sims_tf = torch.exp(sims_tf * 50)
for k in tqdm(range(10)):
    sims_tf = sims_tf / (torch.sum(sims_tf, dim=1, keepdim=True) + 1e-8)
    sims_tf = sims_tf / (torch.sum(sims_tf, dim=0, keepdim=True) + 1e-8)
sims_sinkhorn = sims_tf.cpu().numpy()

def test(sims, method):
    if method == "sinkhorn":
        hits1, hits10, mrr = 0, 0, 0
        total = len(sims)
        for i in range(total):
            rank = np.argsort(-sims[i])
            pos = np.where(rank == i)[0][0]
            if pos < 1:
                hits1 += 1
            if pos < 10:
                hits10 += 1
            mrr += 1 / (pos + 1)
        hits1_rate = (hits1 / total) * 100
        hits10_rate = (hits10 / total) * 100
        mrr_rate = (mrr / total) * 100
        print(f"Sinkhorn Assessment result：")
        print(f"Hits@1: {hits1_rate:.2f}% | Hits@10: {hits10_rate:.2f}% | MRR: {mrr_rate:.2f}%")
    else:
        pass

test(sims_sinkhorn, "sinkhorn")
