from os import write
import pandas as pd
import torch
import numpy as np

PROSTATE_PATH = "_database/prostate"
REACTOME_PATH = "_database/pathways"

SORTED_USED_GENES = "genes.txt"
SORTED_USED_PATHWAYS = "pathways.txt"

NUM_SPLITS = 20


def get_split_data(train, split):
    """
    Reads training data from training_set and maps to CNA,
    returns X and y with genes and as NumPy arrays
    """
    if train:
        assert split < NUM_SPLITS
    if train:
        ts = pd.read_csv(f"{PROSTATE_PATH}/splits/training_set_{split}.csv")
    else:
        ts = pd.read_csv(f"{PROSTATE_PATH}/splits/validation_set.csv")
    cna = pd.read_csv(f"{PROSTATE_PATH}/processed/P1000_data_CNA_paper.csv")

    num_samples = ts.shape[0]
    num_features = cna.shape[1] - 1  # exclude ID column

    X = np.zeros((num_samples, num_features), dtype=np.float32)
    y = np.zeros(num_samples, dtype=np.float32)
    genes = cna.columns[1:]

    for i, row in ts.iterrows():
        gene_row = cna[cna.iloc[:, 0] == row["id"]]

        if gene_row.empty:
            raise ValueError(f"Sample ID {row['id']} not found in CNA file")

        # Drop ID column and flatten
        X[i, :] = gene_row.iloc[:, 1:].values.flatten()
        y[i] = row["response"]

    return X, y, genes


def gene_getter_helper(pathways):
    genes = set()
    for path in pathways:
        for g in path["genes"]:
            genes.add(g)
    return list(genes)


def get_reactome_pathways():
    pathways = []

    with open(f"{REACTOME_PATH}/Reactome/ReactomePathways.gmt", "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue

            description = parts[0]
            id = parts[1]
            source = parts[2]
            genes = parts[3:]

            pathways.append(
                {
                    "description": description,
                    "id": id,
                    "source": source,
                    "genes": genes,
                }
            )
    return pathways


def write_used_genes_and_pathways():
    """
    Sorts genes and pathways and writes them in a txt file.
    """
    pathways = get_reactome_pathways()
    genes = gene_getter_helper(pathways)
    pathway_ids = [p["id"] for p in pathways]

    genes.sort()
    pathway_ids.sort()

    with open("genes.txt", "w", encoding="utf-8") as f:
        for gene in genes:
            f.write(gene + "\n")

    with open("pathways.txt", "w", encoding="utf-8") as f:
        for id in pathway_ids:
            f.write(id + "\n")


def read_used_genes():
    genes = dict()

    with open(SORTED_USED_GENES, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            genes[line.strip()] = i

    return genes


def read_used_pathways():
    pathways = dict()

    with open(SORTED_USED_PATHWAYS, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            pathways[line.strip()] = i

    return pathways


def generate_mask_matrix():
    """
    Gets used pathways from reactome,
    Makes adjacency matrix from sorted genes and expressions
    """

    genes_idx, pathways_idx = read_used_genes(), read_used_pathways()

    pathways = get_reactome_pathways()

    n = len(genes_idx)
    m = len(pathways_idx)
    mask = torch.zeros((n, m))

    for path in pathways:
        j = pathways_idx[path["id"]]
        for gene in path["genes"]:
            i = genes_idx[gene]
            mask[i, j] = 1

    return mask


def generate_data(train=True, split=None):
    """
    Generate training data for a given batch number,
    matching tensor shape to sorted expected format without trailing zeros.
    """
    X, y, rows = get_split_data(train, split)

    # features_order: dict mapping gene_name -> target column index
    features_order = read_used_genes()

    num_features = max(features_order.values()) + 1
    num_samples = X.shape[0]

    # Initialize transformed array
    X_transformed = np.zeros((num_samples, num_features), dtype=np.float32)

    for col_idx, gene_name in enumerate(rows):
        if gene_name in features_order:
            target_idx = features_order[gene_name]
            X_transformed[:, target_idx] = X[:, col_idx]

    X_tensor = torch.tensor(X_transformed, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    y_tensor = torch.nn.functional.one_hot(y_tensor.long()).float()

    return X_tensor, y_tensor
