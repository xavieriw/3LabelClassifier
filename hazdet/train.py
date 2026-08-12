# Imports
# Basic
# !pip install demoji
# !pip install -U sentence-transformers
# !pip install scikit-learn
# !pip install scikit-optimize

import os
import subprocess


def _select_gpu(min_free_mb=4000, required_name_substring=None):
    """
    Automatically pick the least-loaded visible GPU by setting
    CUDA_VISIBLE_DEVICES *before* torch/tensorflow/sentence-transformers
    are imported below -- this has to run first, since GPU selection is
    effectively locked in once those libraries initialize.

    Useful on shared multi-GPU nodes where GPU 0 may already be full from
    another user's/job's process, even though other GPUs on the same node
    are free.

    On nodes with a mixed GPU fleet (e.g. A100s alongside other
    architectures), only GPUs whose name contains `required_name_substring`
    are considered -- otherwise auto-selection can land on an incompatible
    architecture (e.g. an H100) that this TensorFlow/PyTorch build wasn't
    compiled for, causing cryptic runtime errors like
    "Failed call to cudaGetFuncBySymbol: invalid device function" instead
    of a clear one. Set required_name_substring=None to consider all GPUs
    regardless of model.

    If CUDA_VISIBLE_DEVICES is already set (e.g. by SLURM's --gres), only
    those GPU indices are considered -- this respects whatever the
    scheduler granted, while still picking the least-loaded one among them
    if more than one was granted.
    """
    try:
        existing = os.environ.get('CUDA_VISIBLE_DEVICES')
        candidate_indices = None
        if existing:
            candidate_indices = [int(i) for i in existing.split(',') if i.strip() != '']

        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,name,memory.free',
             '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        gpus = []
        for line in output.strip().split('\n'):
            idx_str, name, free_str = [x.strip() for x in line.split(',')]
            idx, free = int(idx_str), int(free_str)
            if candidate_indices is not None and idx not in candidate_indices:
                continue
            if required_name_substring is not None and required_name_substring not in name:
                continue
            gpus.append((idx, name, free))

        if not gpus:
            print(f'WARNING: no visible GPU matched required_name_substring='
                  f'{required_name_substring!r}. Leaving CUDA_VISIBLE_DEVICES '
                  f'unchanged -- torch/tensorflow will use their own defaults.')
            return

        best_idx, best_name, best_free = max(gpus, key=lambda g: g[2])
        if best_free < min_free_mb:
            print(f'WARNING: no matching GPU has more than {min_free_mb} MiB free '
                  f'(best is GPU {best_idx} [{best_name}] with {best_free} MiB free). '
                  f'Proceeding anyway -- you may hit an out-of-memory error.')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(best_idx)
        print(f'[GPU auto-select] Using GPU {best_idx} [{best_name}] ({best_free} MiB free)')
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        # nvidia-smi not available/parsable (e.g. a CPU-only machine) --
        # leave CUDA_VISIBLE_DEVICES as-is and let torch/tf fall back to CPU
        pass


_select_gpu()

import pandas as pd
import numpy as np
import random
import pickle as pk
import warnings
from scipy.stats import beta
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import skopt
from skopt.space import Real, Integer,Categorical
from skopt import BayesSearchCV, gp_minimize
from skopt.utils import use_named_args
import demoji
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from xgboost import XGBClassifier

# sklearn's SVC(probability=True) -- used throughout this file to get
# predict_proba() for the 3-class AI/Not AI/Other output -- is deprecated
# as of sklearn 1.9 in favor of CalibratedClassifierCV(SVC(), ensemble=False),
# and is slated for REMOVAL in sklearn 1.11. Silencing this specific warning
# only hides the noise; it does not fix the upcoming break. When this
# codebase upgrades past sklearn 1.11, every `SVC(probability=True)` call in
# this file will need to be migrated to CalibratedClassifierCV instead.
warnings.filterwarnings(
    'ignore',
    message=r".*`probability` parameter was deprecated.*",
    category=FutureWarning,
    module=r"sklearn\.svm\._base",
)

# The NN classifier is implemented in PyTorch rather than Keras/TensorFlow.
# Official TensorFlow pip wheels don't ship pre-compiled CUDA kernels for
# Hopper-architecture GPUs (compute capability sm_90 -- H100/H200), only a
# generic PTX fallback that has to be JIT-compiled at runtime (slow, and
# prone to errors like "Failed call to cudaGetFuncBySymbol: invalid device
# function"). PyTorch's Hopper support is far more mature, and this file
# already depends on it anyway via sentence-transformers, so standardizing
# on one GPU backend avoids the issue entirely.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Qwen/Qwen3-Embedding-0.6B produces 1024-dimensional embeddings
# (see https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
EMBEDDING_MODEL = 'Qwen/Qwen3-Embedding-0.6B'
EMBEDDING_DIM = 1024

# The classifier predicts whether a Reddit comment indicates the linked
# image/content is AI-generated, not AI-generated, or something else
# (unclear/off-topic/etc). To make that judgment, it's given the full
# context a human annotator would have had: the post title, the post's
# own text (if any), and the comment body itself -- combined into a
# single string and embedded as one vector (see _combine_context below).
CLASS_COLUMNS = {'AI (Y/N)': 0, 'Not AI (Y/N)': 1, 'Other (Y/N)': 2}
CLASS_LABELS = ['AI', 'Not AI', 'Other']


def _parse_bool(val):
    #Robustly interpret a cell as a boolean. Handles native Python bool
    #(from .xlsx), and string forms like 'TRUE'/'False'/'Y'/'1' (from
    #.csv/.tsv) -- note that a naive bool('False') is WRONG (any non-empty
    #string is truthy in Python), so this must be handled explicitly.
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    s = str(val).strip().upper()
    return s in ('TRUE', '1', 'YES', 'Y')


def _combine_context(title, parent_text, comment_body):
    #Combine a Reddit comment's post title, parent (submission) text, and
    #the comment body itself into a single string for embedding -- giving
    #the model the same context a human annotator would have had when
    #judging whether the comment indicates AI, Not AI, or Other.
    parts = []
    for label, val in [('Title', title), ('Post text', parent_text), ('Comment', comment_body)]:
        if val is not None and str(val).strip() and str(val).strip().lower() != 'nan':
            parts.append(f'{label}: {val}')
    return '\n\n'.join(parts)


def to_class_labels(y):
    #Convert a one-hot encoded (n, 3) label matrix into a single-column
    #vector of integer class labels (0=AI, 1=Not AI, 2=Other).
    #Used for the sklearn-style classifiers (RF, SVC, XGB), which expect
    #integer class labels rather than one-hot vectors.
    return np.argmax(y, axis=1)


def extract_qualtrics_data(file='Tweet annotation.csv'):
    # make sure we only include annotators who completed full survey
    progress_pct = 100
    # find all data where humans completed the task and took > 3 minutes, 20 seconds (200 seconds)
    duration_seconds = 200

    data = pd.read_csv(file,low_memory=False,lineterminator='\n')
    
    # make columns readable
    data.columns = [c.replace('_1','') if not ('_' in c and 'Q' not in c) else 'Q'+c.replace('_1','') for c in data.columns]
    # this question checks if someone is paying attention; everyone chose "4" so it is not needed
    data = data.drop(['Q11119'],axis=1)
    cols = data.columns
    # where are the questions for each text located
    pos_qs = [ii for ii,c in enumerate(cols) if 'Q' in c]
    # number of questions per annotations
    num_qs = 11
    # where do the questions start (starting from the first and ending at the 10th annotation)
    q_start_pos = [22-num_qs]+pos_qs[5::11]
    # here we extract the exact questions
    col_text = {}
    for c in cols:
        col_text[c] = data.iloc[0][c]
    questions = [col_text[c] for c in cols[q_start_pos[1]:q_start_pos[1]+11]]
    # demographics
    demogs = [col_text[c] for c in cols if 'QD' in c]
    # all of the key data: text, questions, demographics per annotator
    annotations = {'text':[]}
    for q in questions:
        annotations[q] = []
    for d in demogs:
        annotations[d] = []

    # data cleaning
    data = data[12:]
    data = data.loc[data['Progress'].astype(float).values==progress_pct,]
    data = data.loc[data['Duration (in seconds)'].astype(float).values>duration_seconds,]
    return data,annotations

def reshape_qualtrics_data(data,annotations):
    
    # for each row of this file, find the exact text humans annotated
    for ii,row in data.iterrows():
        if ii > 0:
            # all the tweets are those whose elements are not null and not questions
            texts = [col_text[c] for null,c in zip(row.isnull()[q_start_pos[0]+num_qs:],cols[q_start_pos[0]+num_qs:]) if not null and 'Q' not in c and c in col_text.keys()][:-4]
            line_annots = {}
            # for each text,
            for q1,text in zip(q_start_pos[1:],texts):
                # positions of all the questions associated with the text
                q_cols = cols[q1:q1+num_qs]
                # add tweet
                line_annots['text'] = [text]
                # add questions
                for q,c in zip(questions,q_cols):
                    line_annots[q]=[row[c]]
                for c in cols:
                    # demographic questions
                    if 'QD' in c:
                        line_annots[col_text[c]]=[row[c]]
            # if we have found all the data
            if set(list(line_annots.keys()))== set(list(annotations.keys())):
                # append text, questions, demographics
                for q1,text in zip(q_start_pos[1:],texts):
                    q_cols = cols[q1:q1+num_qs]
                    annotations['text'].append(text)
                    for q,c in zip(questions,q_cols):
                        annotations[q].append(row[c])
                    for c in cols:
                        if 'QD' in c:
                            annotations[col_text[c]].append(row[c])
                lens= []
                for key in annotations.keys():
                    lens.append(len(annotations[key]))
    return annotations

def clean_qualtrics_data(file='Tweet annotation.csv'):
    data,annotations = extract_qualtrics_data(file)
    annotations = reshape_qualtrics_data(data,annotations)
    annotations = pd.DataFrame(annotations)
    hazard_col = 'Does the tweet describe a hazard (something that could impose harm or other costs on the author of the tweet or on others)?'
    benefit_col = 'Does the tweet describe a benefit (something that provides resources, opportunities, or other good things to the author of the tweet or to others)?'
    return annotations,hazard_col,benefit_col

def create_features(file='Tweet annotation.csv'):
    # extract reshaped data
    
    raw_qualtrics = False
    if raw_qualtrics:
        annotations,hazard_col,benefit_col = clean_qualtrics_data(file)
        text_haz_ben=annotations[['text',hazard_col,benefit_col]]
        text_haz_ben[hazard_col] = text_haz_ben[hazard_col].replace('Yes',1).replace('No',0).dropna()
        text_haz_ben[benefit_col] = text_haz_ben[benefit_col].replace('Yes',1).replace('No',0).dropna()
        unique_text = text_haz_ben['text'].drop_duplicates()
        haz_ben_annots = text_haz_ben.groupby('text')
        GT_labels = {'text':[],'hazard':[],'benefit':[],'old_text':[]}
        for t in unique_text:
            annots = haz_ben_annots.get_group(t)
            # require at least 2 annotations
            if len(annots) <= 2: continue
            ben = annots[benefit_col].values.sum()/len(annots)
            haz = annots[hazard_col].values.sum()/len(annots)
            all_replace={}
            replaced_text = []
            # replace emojis
            replace = demoji.findall(t)
            new_t = t
            for word, initial in replace.items():
                new_t = new_t.replace(word, initial)
            GT_labels['old_text'].append(t)
            GT_labels['text'].append(new_t)
            GT_labels['hazard'].append(haz)
            GT_labels['benefit'].append(ben)
        GT_labels = pd.DataFrame(GT_labels)
    elif file.endswith('.xlsx') or file.endswith('.xls'):
        GT_labels = pd.read_excel(file)
    else:
        GT_labels = pd.read_csv(file)
    #Sentences are encoded by calling model.encode()
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Accept either the friendly lowercase column names ('title',
    # 'parent_text', 'comment_body') or the spreadsheet's exact headers
    # ('Title', 'Parent Text', 'Comment Body', 'AI (Y/N)', ...) -- match
    # case-insensitively so either format works without renaming columns.
    col_map = {str(c).strip().lower(): c for c in GT_labels.columns}

    def _find_col(*names):
        for name in names:
            if name.lower() in col_map:
                return col_map[name.lower()]
        return None

    title_col = _find_col('title')
    parent_col = _find_col('parent_text', 'parent text')
    body_col = _find_col('comment_body', 'comment body')
    if title_col is None or parent_col is None or body_col is None:
        raise Exception(f'ERROR: {file} must contain "title", "parent_text", and '
                         f'"comment_body" columns (or their spreadsheet-header '
                         f'equivalents "Title", "Parent Text", "Comment Body").')

    label_cols = {}
    for label_name, idx in CLASS_COLUMNS.items():
        c = _find_col(label_name)
        if c is None:
            raise Exception(f'ERROR: Could not find expected column "{label_name}" in {file}. '
                             f'Expected columns: {list(CLASS_COLUMNS.keys())}.')
        label_cols[idx] = c

    # Build one training example per row: the combined title/post-text/
    # comment-body context, labeled with whichever single one of
    # AI/Not AI/Other was marked True for that row. Rows where zero or
    # more than one of those three were marked True are skipped rather
    # than guessed at.
    texts = []
    labels = []
    skipped_ambiguous = 0
    skipped_blank = 0
    for _, row in GT_labels.iterrows():
        combined = _combine_context(row.get(title_col), row.get(parent_col), row.get(body_col))
        if not combined.strip():
            skipped_blank += 1
            continue

        flags = [_parse_bool(row.get(label_cols[idx])) for idx in range(3)]
        true_labels = [i for i, v in enumerate(flags) if v]
        if len(true_labels) != 1:
            skipped_ambiguous += 1
            continue

        texts.append(combined)
        labels.append(true_labels[0])

    if skipped_blank:
        print(f'NOTE: skipped {skipped_blank} row(s) with no title/parent_text/comment_body content.')
    if skipped_ambiguous:
        print(f'NOTE: skipped {skipped_ambiguous} row(s) where AI/Not AI/Other '
              f"weren't exactly one boolean set to True.")
    print(f'Building features for {len(texts)} labeled rows.')

    embeddings = model.encode(texts, show_progress_bar=True)
    X = np.array([e.astype('float32') for e in embeddings])
    # one-hot encode the 3 classes: AI / Not AI / Other
    y = np.zeros((len(labels), 3), dtype='float32')
    y[np.arange(len(labels)), labels] = 1
    return X,y

def hyperparameter_tune_model(X,y,search_space,model):
    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.9, random_state=42)
    # y is one-hot encoded for the 3 classes (title/parent_text/comment_body);
    # sklearn-style classifiers expect a single column of integer class labels
    y_train = to_class_labels(y_train)
    y_test = to_class_labels(y_test)
    optimizer = BayesSearchCV(
    estimator=model,
    search_spaces=search_space,
    scoring=None,
    cv=5,
    n_iter=10,
    return_train_score=False,
    n_jobs=-1
    )
    optimizer.fit(X_train, y_train)
    rf_best_hyperparameters = optimizer.best_params_
    best_score = optimizer.best_score_
    return rf_best_hyperparameters,best_score

class _MLP(nn.Module):
    """
    A small feed-forward network, structurally equivalent to what the old
    Keras build_model() produced for layers=[256, 256, 3]:
        Dense(256, relu) applied directly to the input (no dropout before it)
        -> Dropout -> Dense(256, relu)
        -> Dropout -> Dense(3)  [raw logits -- no activation here]

    The final layer intentionally outputs raw logits rather than
    softmax probabilities, since nn.CrossEntropyLoss applies
    log-softmax internally (more numerically stable than the old
    softmax-then-categorical_crossentropy approach). Callers that need
    class probabilities should apply torch.softmax() to the output
    (see TorchMLPClassifier.predict_proba below).
    """
    def __init__(self, nx, layers, keep_prob):
        super().__init__()
        dims = [nx] + list(layers)
        self.first = nn.Linear(dims[0], dims[1])
        self.rest = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(1, len(dims) - 1)])
        self.dropout = nn.Dropout(1 - float(keep_prob))
        self.n_rest = len(self.rest)

    def forward(self, x):
        x = F.relu(self.first(x))
        for i, layer in enumerate(self.rest):
            x = self.dropout(x)
            x = layer(x)
            if i < self.n_rest - 1:
                x = F.relu(x)
            # last layer: no activation -- raw logits for CrossEntropyLoss
        return x


class TorchMLPClassifier:
    """
    A thin sklearn-like wrapper around the PyTorch _MLP network above.
    Exposes .fit()/.predict_proba()/.predict(), so it's a drop-in
    replacement for the old Keras-based NN wherever this file calls it,
    and -- unlike the old Keras model, which had to be saved separately
    via .save() -- it can be pickled with pk.dump()/pk.load() exactly
    like the RF/SVC/XGB classifiers elsewhere in this file, so
    inference.py's loading code works identically for every model type.
    """
    def __init__(self, nx=None, layers=(256, 256, 3), lambtha=0.0, keep_prob=0.5, device=None):
        self.nx = nx
        self.layers = layers
        self.lambtha = lambtha
        self.keep_prob = keep_prob
        self.device = device or DEVICE
        self.model = None
        self.history = {'loss': [], 'val_loss': []}

    def _build(self, nx):
        self.nx = nx
        self.model = _MLP(nx, self.layers, self.keep_prob).to(self.device)

    def fit(self, X_train, y_train, batch_size=32, epochs=100,
            validation_data=None, alpha=0.001, beta1=0.9, beta2=0.999,
            early_stopping=False, patience=0, learning_rate_decay=False,
            decay_rate=1, verbose=False, shuffle=False):
        #Trains the network using mini-batch gradient descent, mirroring
        #the behavior of the old Keras train_model()/return_trained_model():
        #  - y_train/y_val are one-hot (n, classes) arrays; converted to
        #    integer class labels for nn.CrossEntropyLoss
        #  - early_stopping: stop once val_loss hasn't improved for
        #    `patience` consecutive epochs (matches Keras EarlyStopping's
        #    default behavior: no restoring of best weights)
        #  - learning_rate_decay: alpha_0 = alpha / (1 + decay_rate*epoch),
        #    matching the original learning_rate_decay() schedule
        #  - shuffle=False by default, matching the original calls (batches
        #    are drawn in the given row order, not reshuffled each epoch)

        if self.model is None:
            self._build(X_train.shape[1])

        X_train_t = torch.tensor(np.asarray(X_train, dtype='float32'), device=self.device)
        y_train_labels = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long, device=self.device)

        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_t = torch.tensor(np.asarray(X_val, dtype='float32'), device=self.device)
            y_val_labels = torch.tensor(np.argmax(y_val, axis=1), dtype=torch.long, device=self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(alpha),
                                      betas=(float(beta1), float(beta2)),
                                      weight_decay=float(self.lambtha))
        loss_fn = nn.CrossEntropyLoss()

        n = X_train_t.shape[0]
        batch_size = int(batch_size)
        best_val_loss = np.inf
        epochs_since_improvement = 0

        for epoch in range(epochs):
            if learning_rate_decay:
                lr = alpha / (1 + decay_rate * epoch)
                for g in optimizer.param_groups:
                    g['lr'] = lr

            self.model.train()
            if shuffle:
                perm = torch.randperm(n, device=self.device)
                X_epoch, y_epoch = X_train_t[perm], y_train_labels[perm]
            else:
                X_epoch, y_epoch = X_train_t, y_train_labels

            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                end = start + batch_size
                xb, yb = X_epoch[start:end], y_epoch[start:end]
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.shape[0]
            epoch_loss /= n
            self.history['loss'].append(epoch_loss)

            if validation_data is not None:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(X_val_t)
                    val_loss = loss_fn(val_logits, y_val_labels).item()
                self.history['val_loss'].append(val_loss)

                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_since_improvement = 0
                    else:
                        epochs_since_improvement += 1
                        if epochs_since_improvement > patience:
                            break

        # Move to CPU once training is done, so pickling this object
        # (via pk.dump in train_best_model) doesn't tie the saved model
        # file to whichever GPU/CUDA setup happened to train it.
        self.model.to('cpu')
        self.device = torch.device('cpu')
        return self

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(np.asarray(X, dtype='float32'), device=self.device)
            probs = torch.softmax(self.model(X_t), dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        # kept for parity with call sites that used the old Keras
        # clf.predict(X) -- returns the same (n, 3) class probabilities
        # as predict_proba
        return self.predict_proba(X)


def hyperparameter_tune_nn(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.9, random_state=42)
    # y is already one-hot encoded for the 3 classes (title/parent_text/comment_body);
    # the NN uses the full one-hot target with categorical_crossentropy + softmax

    # Setting the bounds of network parameters for Bayesian optimization
    # (mirrors the bounds previously passed to GPyOpt)
    search_space = [
        Real(0.00005, 0.005, name='lambtha'),
        Real(0.05, 0.95, name='keep_prob'),
        Real(0.0001, 0.005, name='alpha'),
        Real(0.9, 0.999, name='beta1'),
        Categorical([32, 128], name='batch_size'),
    ]

    # Creating the objective function for skopt's Bayesian optimization.
    # @use_named_args unpacks each proposed point in search_space into
    # named keyword arguments, so object_function receives plain scalar
    # values (unlike GPyOpt, which passed a batched 2D array).
    @use_named_args(search_space)
    def object_function(lambtha, keep_prob, alpha, beta1, batch_size):
        #Function that sets hyperparameters of the PyTorch network, trains it,
        #and returns the loss to be minimized:
        #    lambtha is the L2 regularization parameter (Adam weight_decay)
        #    keep_prob is the probability that a node will be kept for dropout
        #    alpha is the learning rate in Adam optimizer
        #    beta1 is the first Adam optimization parameter
        #    batch_size is the size of the batch used for mini-batch  gradient
        #    descent

        #Returns the validation loss of the model

        # 3 output units (raw logits, softmax applied at predict time) for
        # the 3 classes: title / parent_text / comment_body
        clf = TorchMLPClassifier(nx=EMBEDDING_DIM, layers=(256, 256, 3),
                                  lambtha=lambtha, keep_prob=keep_prob)

        # Training the model using early stopping
        epochs = 100
        random_indices = list(range(len(X_train)))
        random.shuffle(random_indices)
        train_ind = random_indices[:int(0.8*len(X_train))]
        valid_ind = random_indices[int(0.8*len(X_train)):]
        X_train_i = X_train[train_ind]
        Y_train_i = y_train[train_ind]
        X_valid = X_train[valid_ind]
        Y_valid = y_train[valid_ind]
        clf.fit(X_train_i, Y_train_i, batch_size=batch_size, epochs=epochs,
                validation_data=(X_valid, Y_valid), alpha=alpha, beta1=beta1,
                early_stopping=True, patience=3, learning_rate_decay=True)
        return clf.history['val_loss'][-1]

    #Stop conditions (mirrors the previous GPyOpt run: 30 iterations)
    max_iter = 30

    #Running the method
    result = gp_minimize(
        object_function,
        search_space,
        n_calls=max_iter,
        random_state=42,
    )

    nn_hyperparameters = [(c,v) for c,v in zip(['lambtha','keep_prob','alpha','beta1','batch_size'],result.x)]
    return nn_hyperparameters

def hyperparameter_tune_all_models(file):
    X,y = create_features(file)
    search_space ={
        'n_estimators':Integer( 10, 150),
        'max_depth':Integer( 5, 50),
        'min_samples_split':Integer( 2,20),
        'max_features':Categorical(['sqrt','log2',None]),
        'class_weight':Categorical(['balanced',None]),
        'ccp_alpha':Real(0.0,0.01)#,
    }
    rf_best_hyperparameters,_ = hyperparameter_tune_model(X,y,search_space,RandomForestClassifier())

    search_space ={
    'C':Real(0.01,10),
    'kernel':Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
    'degree':Integer(1,4),
    'gamma':Categorical(['auto','scale']),
    'shrinking':Categorical([False,True]),
    'class_weight':Categorical(['balanced',None]),
    } 
    svc_best_hyperparameters,_ = hyperparameter_tune_model(X,y,search_space,SVC(probability=True))
    
    search_space ={
    'n_estimators':Integer( 10, 100),
    'max_depth':Integer( 5, 50),
    'max_leaves':Integer( 20,200),
    'max_bin':Integer( 2,200),
    'tree_method':Categorical(['auto', 'exact', 'approx', 'hist']),
    'gamma':Real(0.0,0.1),
    'colsample_bytree':Real(0.1,1.0),
    'colsample_bylevel':Real(0.1,1.0),
    'reg_alpha':Real(0.0,10.0), 
    'reg_lambda':Real(0.0,10.0),
    'importance_type':Categorical(['gain','weight','cover','total_gain','total_cover'])
    }
    # device='cpu' is explicit and required here: recent xgboost (2.x+) wheels
    # bundle GPU support by default and will try to initialize CUDA even when
    # GPU acceleration isn't requested, which can fail with
    # "cudaErrorInsufficientDriver" if xgboost's bundled CUDA runtime
    # dependencies (its own nvidia-*-cuXX pip packages) target a newer CUDA
    # version than the node's actual NVIDIA driver supports. RF/SVC/XGB are
    # CPU-only by design in this pipeline anyway -- only the embedding step
    # and the NN training are meant to use the GPU -- so this just makes
    # that explicit instead of relying on xgboost's own auto-detection.
    xgb_best_hyperparameters,_ = hyperparameter_tune_model(X,y,search_space,XGBClassifier(device='cpu'))
    nn_best_hyperparameters = [v[1] for v in hyperparameter_tune_nn(X,y)]
    params = [rf_best_hyperparameters,svc_best_hyperparameters,xgb_best_hyperparameters,nn_best_hyperparameters]
    # save the best model...
    train_best_model(X,y,params)


def train_nn(nn_best_hyperparameters,X_train, y_train,embedding_dim=EMBEDDING_DIM):
    lambtha, keep_prob, alpha, beta1, batch_size = nn_best_hyperparameters
    print(nn_best_hyperparameters)
    print(lambtha)
    # 3 output units (raw logits) for the 3 classes: title / parent_text / comment_body
    clf = TorchMLPClassifier(nx=embedding_dim, layers=(256, 256, 3),
                              lambtha=lambtha, keep_prob=keep_prob)

    # Training the model using early stopping and saving the best model
    epochs = 100
    random_indices = list(range(len(X_train)))
    random.shuffle(random_indices)
    train_ind = random_indices[:int(0.8*len(X_train))]
    valid_ind = random_indices[int(0.8*len(X_train)):]
    X_train_i = X_train[train_ind]
    Y_train_i = y_train[train_ind]
    X_valid = X_train[valid_ind]
    Y_valid = y_train[valid_ind]
    clf.fit(X_train_i, Y_train_i, batch_size=batch_size, epochs=epochs,
            validation_data=(X_valid, Y_valid), alpha=alpha, beta1=beta1,
            early_stopping=True, patience=3, learning_rate_decay=True)
    return clf

def predict_model (model,model_params,X,y,ii):
    random_state = 999
    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.9, random_state=random_state)
    # y is one-hot encoded for the 3 classes (title/parent_text/comment_body).
    # The NN consumes the full one-hot target; the sklearn-style classifiers
    # (RF/SVC/XGB) expect a single column of integer class labels.

    # random seed
    np.random.seed(ii*314159)
    boot_indices = np.random.randint(0,len(X_test),len(X_test))

    X_boot = X_test[boot_indices]
    y_boot = y_test[boot_indices]
    if model == 'NN':
        clf = train_nn(model_params,X_train, y_train)
        y_pred = clf.predict(X_boot)  # shape (n, 3) softmax probabilities
    elif model == 'RF':
        y_train_labels = to_class_labels(y_train)
        kwargs = {key:value for key,value in model_params.items()}
        clf = RandomForestClassifier(**kwargs)
        clf.fit(X_train, y_train_labels)
        y_pred = clf.predict_proba(X_boot)  # shape (n, 3)

    elif model == 'SVC':
        y_train_labels = to_class_labels(y_train)
        kwargs = {key:value for key,value in model_params.items()}
        kwargs['probability'] = True
        clf = SVC(**kwargs)
        clf.fit(X_train, y_train_labels)
        y_pred = clf.predict_proba(X_boot)  # shape (n, 3)
    elif model == 'XGB':
        y_train_labels = to_class_labels(y_train)
        kwargs = {key:value for key,value in model_params.items()}
        kwargs.setdefault('device', 'cpu')  # see note in hyperparameter_tune_all_models
        clf = XGBClassifier(**kwargs)
        clf.fit(X_train, y_train_labels)
        y_pred = clf.predict_proba(X_boot)  # shape (n, 3)
    # return integer class labels for y_boot so the caller can score
    # multiclass predictions consistently across all model types
    return y_pred,to_class_labels(y_boot)


def eval_best_model(X,y,params,num_evals = 50,eval_metric='roc_auc'):
    rf_best_hyperparameters,svc_best_hyperparameters,xgb_best_hyperparameters,nn_best_hyperparameters = params
    # X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.9, random_state=999)#42)
    # y_train = y_train[:,0].round().reshape(-1,1)
    model_params = {'RF':rf_best_hyperparameters,'SVC':svc_best_hyperparameters,'XGB':xgb_best_hyperparameters,'NN':nn_best_hyperparameters}
    #metrics = {'NN_auc':[],'NN_f1':[],'RF_auc':[],'RF_f1':[],'SVM_auc':[],'SVM_f1':[],'XGB_auc':[],'XGB_f1':[],'gpt_auc':[],'gpt_f1':[],'base_f1':[],'gpt_auc':[],'gpt_f1':[],'gpt_soc_auc':[],'gpt_soc_f1':[],'gpt_lib_auc':[],'gpt_lib_f1':[],'gpt4_auc':[],'gpt4_f1':[]}
    best_model_performance = 0
    model_performance = {}
    for model in ['NN','SVC','RF','XGB']:
        print('Calculating performance of ',model)
        performance_metric_boot = []
        for ii in range(num_evals):
            y_pred,y_boot = predict_model (model,model_params[model],X,y,ii) 
            if eval_metric ==  'roc_auc':
                # y_pred is (n, 3) class probabilities, y_boot is (n,) integer class labels.
                # On small datasets, a given bootstrap draw can occasionally miss one of
                # the 3 classes entirely (e.g. no "Not AI" examples in that draw) --
                # roc_auc_score's multiclass mode requires every class present in y_pred's
                # columns to also appear in y_boot, so it raises ValueError in that case.
                # Rather than crashing the whole evaluation run over one unlucky draw,
                # skip that draw and continue; this becomes vanishingly rare as dataset
                # size grows, but is worth guarding against regardless.
                try:
                    performance = roc_auc_score(y_boot, y_pred, multi_class='ovr')
                except ValueError as e:
                    print(f'WARNING: skipping a bootstrap evaluation for {model} '
                          f'(eval {ii}) -- {e}')
                    continue
            elif eval_metric == 'f1':
                performance = f1_score(y_boot, np.argmax(y_pred, axis=1), average='macro')
            else:
                raise Exception('ERROR: Evaluation metric not recognized.')
            performance_metric_boot.append(performance)
        if not performance_metric_boot:
            print(f'WARNING: every bootstrap evaluation for {model} was skipped '
                  f'(dataset likely too small for reliable evaluation). Treating '
                  f'its performance as 0.')
            model_performance[model] = [0.0, 0.0]
            continue
        mean_performance = np.mean(performance_metric_boot)
        std_performance = np.std(performance_metric_boot)
        model_performance[model] = [mean_performance,std_performance]
        if mean_performance > best_model_performance:
            best_model = model
            best_model_performance = mean_performance
    return best_model,model_performance


def train_best_model(X,y,params):
    rf_best_hyperparameters,svc_best_hyperparameters,xgb_best_hyperparameters,nn_best_hyperparameters = params
    model_params = {'RF':rf_best_hyperparameters,'SVC':svc_best_hyperparameters,'XGB':xgb_best_hyperparameters,'NN':nn_best_hyperparameters}
    # y stays one-hot encoded for the 3 classes (title/parent_text/comment_body);
    # each model branch below converts it to the format it needs.
    # find the best model
    best_model,performance = eval_best_model(X,y,params,num_evals = 50,eval_metric='roc_auc')
    print(performance)
    filename = 'finalized_performance.sav'
    pk.dump(performance, open(filename, 'wb'))
    if best_model == 'NN':
        clf = train_nn(model_params[best_model],X, y)

    elif best_model == 'RF':
        y_labels = to_class_labels(y)
        kwargs = {key:value for key,value in model_params[best_model].items()}
        clf = RandomForestClassifier(**kwargs)
        clf.fit(X, y_labels)
    
    elif best_model == 'SVC':
        y_labels = to_class_labels(y)
        kwargs = {key:value for key,value in model_params[best_model].items()}
        kwargs['probability'] = True
        clf = SVC(**kwargs)
        clf.fit(X, y_labels)
    elif best_model == 'XGB':
        y_labels = to_class_labels(y)
        kwargs = {key:value for key,value in model_params[best_model].items()}
        kwargs.setdefault('device', 'cpu')  # see note in hyperparameter_tune_all_models
        clf = XGBClassifier(**kwargs)
        clf.fit(X, y_labels)

    # All 4 model types (including NN, now that it's a picklable
    # TorchMLPClassifier rather than a Keras model) are saved the same way,
    # so inference.py's pk.load()-based loading works identically regardless
    # of which model type ended up being the best.
    filename = 'finalized_model_'+best_model+'.sav'
    pk.dump(clf, open(filename, 'wb'))
    return
