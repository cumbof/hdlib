import os
import copy
import time
import re
import random
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold

from hdlib import Space, Vector
from hdlib.arithmetic import bundle, permute


# This is part of SmilesPE
# SmilesPE is not compatible with aarch64
# https://github.com/XinhaoLi74/SmilesPE/blob/e5f27dfea0778966818ac0a9dd23ac646c62707d/SmilesPE/pretokenizer.py#L6
def atomwise_tokenizer(smi, exclusive_tokens = None):
    """
    Tokenize a SMILES molecule at atom-level:
        (1) 'Br' and 'Cl' are two-character tokens
        (2) Symbols with bracket are considered as tokens

    exclusive_tokens: A list of specifical symbols with bracket you want to keep. e.g., ['[C@@H]', '[nH]'].
    Other symbols with bracket will be replaced by '[UNK]'. default is `None`.
    """
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]

    if exclusive_tokens:
        for i, tok in enumerate(tokens):
            if tok.startswith('['):
                if tok not in exclusive_tokens:
                    tokens[i] = '[UNK]'
    return tokens

# This is also part of SmilesPE
# https://github.com/XinhaoLi74/SmilesPE/blob/e5f27dfea0778966818ac0a9dd23ac646c62707d/SmilesPE/pretokenizer.py#L29
def kmer_tokenizer(smiles, ngram=4, stride=1, remove_last = False, exclusive_tokens = None):
    units = atomwise_tokenizer(smiles, exclusive_tokens = exclusive_tokens) #collect all the atom-wise tokens from the SMILES
    if ngram == 1:
        tokens = units
    else:
        tokens = ["".join(units[i:i+ngram]) for i in range(0, len(units), stride) if len(units[i:i+ngram]) == ngram]

    if remove_last:
        if len(tokens[-1]) < ngram: #truncate last whole k-mer if the length of the last k-mers is less than ngram.
            tokens = tokens[:-1]
    return tokens

def brics_tokenizer(smiles: str):
    mol = Chem.MolFromSmiles(smiles)

    return list(BRICS.BRICSDecompose(mol)) if mol else list()

base_dir = Path.cwd() / "tox21"
log_path = base_dir / "runtime.log"  # ← INSERT #2: define where to write your log

# Setup output directories
def setup_dirs(base_dir: str, columns: list):
    os.makedirs(base_dir, exist_ok=True)

    for col in columns:
        os.makedirs(os.path.join(base_dir, col), exist_ok=True)

# Read the Tox21 dataset
data = pd.read_csv("tox21.csv")

columns = data.columns[:12].tolist()
#columns = ['SR-ARE']
setup_dirs(str(base_dir), columns)

space = Space()

start_time = time.perf_counter()

# Filter & pre-tokenize everything once
smiles_to_tokens: dict[str, list[str]] = dict()
skipped_smiles = list()  # To keep track of skipped SMILES
for sm in data['smiles']:
    if "Al" in sm:
        continue

    # Use different tokenizers
    toks = kmer_tokenizer(sm, ngram=4)
    #toks = atomwise_tokenizer(sm)
    #toks = brics_tokenizer(sm)

    if toks:
        smiles_to_tokens[sm] = toks

    else:
        skipped_smiles.append(sm)
        print(f"Skipping invalid SMILES: {sm}")
    
# Build vocabulary
all_tokens = {tok for toks in smiles_to_tokens.values() for tok in toks}
print(f"{len(all_tokens)} unique tokens")                       

for i, tok in enumerate(all_tokens):
    v = Vector(name=tok, vtype="bipolar", seed=i)
    space.insert(v)

# Split positives & negatives without re-tokenizing
# Dictionary containing all positive samples, grouped by column
positive_samples = {col: data.loc[data[col] == 1, "smiles"].tolist() for col in columns}

# Dictionary where keys are columns, values are dictionarys with smiles as keys and their fragments as values
positive_tokenized = {col: {sm: smiles_to_tokens[sm] for sm in positive_samples[col] if sm in smiles_to_tokens} for col in columns}

# Dictionary containing all negative samples, grouped by column
negative_samples = {col: data.loc[data[col] == 0, "smiles"].tolist() for col in columns}

# Dictionary where keys are columns, values are dictionarys with smiles as keys and their fragments as values
negative_tokenized = {col: {sm: smiles_to_tokens[sm] for sm in negative_samples[col] if sm in smiles_to_tokens} for col in columns}

# Create negative sample dict here instead of in the loop. 
# In the loop we will select 100 random samples from the negative samples
random.seed(42)

print(len(positive_tokenized), "columns with positive samples" )

# Function to convert token lists to hypervectors (creating sample vectors)
# Fucntion takes in a list of lists of tokens, creates hypervectors for each list (smiles string)
def create_sample_hvs(token_dict: dict) -> list:
    vecs = list()
    print(len(token_dict), "smiles strings in the token dictionary")
    #assert "[PbH2+2]" in token_dict, "Token '[PbH2+2]' not found in token dictionary. Check your tokenizer or input data."
    print(len(set(token_dict.keys())))

    for sm in token_dict:
        tokens = token_dict[sm]

        try:
            h = copy.deepcopy(space.get([tokens[0]])[0])

            for i, tok in enumerate(tokens[1:]):
                tok_v = copy.deepcopy(space.get([tok])[0])
                h = bundle(h, permute(tok_v, rotate_by=i+1))
                #h = bundle(h, tok_v)  # For BRICS only

            h.name = sm
            vecs.append(h)

        except Exception as e:
            print(tokens)
            raise Exception(f"Error processing SMILES '{sm}': {e}")
            
    return vecs

# Main loop: per column, 100 negative samples
for col in columns:
    print(len(positive_tokenized[col]), "smiles strings in the token dictionary")
    #assert "[PbH2+2]" in positive_tokenized[col], "Token '[PbH2+2]' not found in token dictionary for column: {}".format(col)

    pos_vecs = create_sample_hvs(positive_tokenized[col]) #create sample hypervectors for positive samples

    print(len(pos_vecs), "positive samples for column", col)
    col_dir = os.path.join(str(base_dir), col)   # point at the same base_dir

    # Before this loop, we count how many negative sample we have at this point
    # Divide that number by the # of positive samples, round it down to the nearest integer
    # Kind of like breaking it into folds
    # Can use new kfold object with shuffle = true
    neg_keys = list(negative_tokenized[col].keys())
    n_pos   = len(pos_vecs)
    n_neg   = len(neg_keys)

    # Compute how many "folds" we can make
    n_folds_neg = n_neg // n_pos

    if n_folds_neg < 1:
        raise ValueError(f"Not enough negatives ({n_neg}) vs positives ({n_pos}) to make at least 1 folds")

    # Create a KFold splitter for the negatives
    kf_neg = KFold(n_splits=n_folds_neg, shuffle=True, random_state=42)
    neg_splits = list(kf_neg.split(neg_keys))

    for neg_fold, (_, neg_test_idx) in enumerate(neg_splits, start=1):
        # Pick out the SMILES for this fold’s "test" negatives
        sampled_neg_keys = [neg_keys[i] for i in neg_test_idx]
        sampled_neg_dict = {sm: negative_tokenized[col][sm] for sm in sampled_neg_keys}

        # Vectorize them
        neg_vecs = create_sample_hvs(sampled_neg_dict)
        print(f"{len(neg_vecs)} negative samples for column {col}, fold {neg_fold}")

        sample_dir = os.path.join(col_dir, f'RandomSample{neg_fold}')
        os.makedirs(sample_dir, exist_ok=True)

        # Now run your 10‐fold CV on (pos_vecs vs. neg_vecs) just like before:
        kf_cv = KFold(n_splits=5, shuffle=False)
        pos_splits = list(kf_cv.split(pos_vecs))
        neg_splits_cv = list(kf_cv.split(neg_vecs))

        # Iterate folds
        for cv_fold, ((pos_train_idx, pos_test_idx), (neg_train_idx, neg_test_idx)) in enumerate(zip(pos_splits, neg_splits_cv), start=1):
            fold_start = time.perf_counter()  
            
            fold_dir = os.path.join(sample_dir, f'Fold{cv_fold}')
            os.makedirs(fold_dir, exist_ok=True)

            train_pos = [pos_vecs[i] for i in pos_train_idx]
            test_pos  = [pos_vecs[i] for i in pos_test_idx]
            train_neg = [neg_vecs[i] for i in neg_train_idx]
            test_neg  = [neg_vecs[i] for i in neg_test_idx]

            # Initialize class vectors
            class_pos = None
            print(f"Training set size for positives: {len(train_pos)}")

            for v in train_pos:
                class_pos = copy.deepcopy(v) if class_pos is None else bundle(class_pos, copy.deepcopy(v))

            class_pos.name = 'positive_class'
            print(class_pos)

            class_neg = None
            print(f"Training set size for negatives: {len(train_neg)}")

            for v in train_neg:
                class_neg = copy.deepcopy(v) if class_neg is None else bundle(class_neg, copy.deepcopy(v))

            class_neg.name = 'negative_class'
            print(class_neg)

            # Error-mitigation (100 iterations)
            em_start = time.perf_counter()

            error_rates = list()
            previous_error = 1.0  # Start with a high error rate

            for it in range(100):
                miss = 0
                pos_copy = copy.deepcopy(class_pos)
                neg_copy = copy.deepcopy(class_neg)

                # Positives
                for v in train_pos:
                    if v.dist(class_pos, method='cosine') >= v.dist(class_neg, method='cosine'):
                        pos_copy += v; neg_copy -= v; miss += 1

                # Negatives
                for v in train_neg:
                    if v.dist(class_pos, method='cosine') <= v.dist(class_neg, method='cosine'):
                        pos_copy -= v; neg_copy += v; miss += 1

                err = miss / (len(train_pos) + len(train_neg))
                error_rates.append(err)
                print(f"Iteration {it}: Error rate {err:.6f} (missed {miss} samples)" )

                if err == 0.0:  # If error rate does not improve, stop loop
                    print(f"Early stopping at iteration {it} with error rate {err:.6f}")
                    break

                class_pos, class_neg = pos_copy, neg_copy
                
                previous_error = err

            em_end = time.perf_counter()

            # Write error rates
            with open(os.path.join(fold_dir, 'error_rates.log'), 'w') as f:
                for e in error_rates:
                    f.write(f"{e:.6f}\n")

            # Evaluate on test sets
            y_true, y_pred = list(), list()

            for v in test_pos:
                pred = 'positive' if v.dist(class_pos, 'cosine') < v.dist(class_neg, 'cosine') else 'negative'
                y_true.append('positive')
                y_pred.append(pred)

            for v in test_neg:
                pred = 'negative' if v.dist(class_neg, 'cosine') < v.dist(class_pos, 'cosine') else 'positive'
                y_true.append('negative')
                y_pred.append(pred)

            report = classification_report(y_true, y_pred)

            with open(os.path.join(fold_dir, 'classification_report.txt'), 'w') as f:
                f.write(report)

            cm = confusion_matrix(y_true, y_pred, labels=['positive', 'negative'])

            with open(os.path.join(fold_dir, 'confusion_matrix.txt'), 'w') as f:
                f.write("Labels: ['positive', 'negative']\n")
                f.write(f"Confusion Matrix:\n{cm}\n")

            fold_end = time.perf_counter()
            fold_elapsed = fold_end - fold_start
            em_time = em_end - em_start

            # write a timing file
            with open(os.path.join(fold_dir, 'runtime.log'), 'w') as f:
                f.write(f"Fold {cv_fold} runtime: {fold_elapsed:.2f} seconds\n")
                f.write(f"Error mitigation time: {em_time:.2f} seconds\n")

    print(f"Column {col}: completed {n_folds_neg} negative folds with 10-fold CV each.")

end_time = time.perf_counter()
elapsed   = end_time - start_time

# compute whole hours and whole minutes
hrs  = int(elapsed // 3600)
mins = int((elapsed % 3600) // 60)

with open(log_path, "w") as log_f:
    log_f.write(f"Total runtime: {hrs}h {mins}m\n")
