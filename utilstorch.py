import numpy as np
import os
import torch
import torch.nn.functional as F


def load_triples(file_path, reverse=True):


    def reverse_triples(triples, rel_size):
        reversed_triples = np.zeros_like(triples)
        for i in range(len(triples)):
            reversed_triples[i, 0] = triples[i, 2]
            reversed_triples[i, 2] = triples[i, 0]
            if reverse:
                reversed_triples[i, 1] = triples[i, 1] + rel_size
            else:
                reversed_triples[i, 1] = triples[i, 1]
        return reversed_triples


    with open(file_path + "triples_1", "r", encoding="utf-8") as f:
        triples1 = f.readlines()
    with open(file_path + "triples_2", "r", encoding="utf-8") as f:
        triples2 = f.readlines()


    all_triple_lines = triples1 + triples2
    triples = np.array([line.strip().split("\t") for line in all_triple_lines]).astype(np.int64)


    node_size = max(np.max(triples[:, 0]), np.max(triples[:, 2])) + 1
    rel_size = np.max(triples[:, 1]) + 1


    if reverse:
        reversed_triples = reverse_triples(triples, rel_size)
        all_triples = np.concatenate([triples, reversed_triples], axis=0)
        all_triples = np.unique(all_triples, axis=0)
        final_rel_size = rel_size * 2
    else:
        all_triples = np.unique(triples, axis=0)
        final_rel_size = rel_size

    return all_triples, node_size, final_rel_size


def load_aligned_pair(file_path, ratio=0.3):

    if "sup_ent_ids" in os.listdir(file_path):
        with open(file_path + "ref_ent_ids", "r", encoding="utf-8") as f:
            ref_pairs = f.readlines()
        with open(file_path + "sup_ent_ids", "r", encoding="utf-8") as f:
            sup_pairs = f.readlines()
        aligned_lines = ref_pairs + sup_pairs
    else:

        with open(file_path + "ref_ent_ids", "r", encoding="utf-8") as f:
            aligned_lines = f.readlines()


    aligned_pairs = np.array([line.strip().split("\t") for line in aligned_lines]).astype(np.int64)


    np.random.shuffle(aligned_pairs)
    split_idx = int(len(aligned_pairs) * ratio)
    train_pair = aligned_pairs[:split_idx]
    test_pair = aligned_pairs[split_idx:]

    return train_pair, test_pair


def test(sims, mode="sinkhorn", batch_size=1024):

    sims_tensor = torch.FloatTensor(sims)

    if mode == "sinkhorn":
        hits1, hits10, mrr = 0, 0, 0
        total = len(sims)


        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_sim = sims_tensor[start:end]


            batch_rank = torch.argsort(-batch_sim, dim=-1)


            true_labels = torch.arange(start, end, dtype=torch.long).unsqueeze(1)


            for idx_in_batch in range(batch_rank.shape[0]):

                true_idx = true_labels[idx_in_batch].item()

                rank = batch_rank[idx_in_batch]

                pos = (rank == true_idx).nonzero(as_tuple=True)[0].item()


                if pos < 1:
                    hits1 += 1
                if pos < 10:
                    hits10 += 1
                mrr += 1 / (pos + 1)


        hits1_rate = (hits1 / total) * 100
        hits10_rate = (hits10 / total) * 100
        mrr_rate = (mrr / total) * 100
        print(f"Sinkhorn Assessment Result：")
        print(f"Hits@1: {hits1_rate:.2f}% | Hits@10: {hits10_rate:.2f}% | MRR: {mrr_rate:.2f}%")

    else:

        pred_matches = sims[1]
        true_matches = np.arange(len(pred_matches))
        correct = np.sum(pred_matches == true_matches)
        hits1_rate = (correct / len(pred_matches)) * 100
        print(f"LAPJV Assessment Result：Hits@1: {hits1_rate:.2f}%")