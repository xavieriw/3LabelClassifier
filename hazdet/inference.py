import os
import subprocess


def _select_gpu(min_free_mb=4000, required_name_substring=None):
    """
    Automatically pick the least-loaded visible GPU by setting
    CUDA_VISIBLE_DEVICES *before* torch/sentence-transformers are
    imported below -- this has to run first, since GPU selection is
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
                  f'unchanged -- torch will use its own defaults.')
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
        # leave CUDA_VISIBLE_DEVICES as-is and let torch fall back to CPU
        pass


_select_gpu()

import pandas as pd
import numpy as np
import pickle as pk
from sentence_transformers import SentenceTransformer

# Qwen/Qwen3-Embedding-0.6B produces 1024-dimensional embeddings
# (see https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
EMBEDDING_MODEL = 'Qwen/Qwen3-Embedding-0.6B'

# The model predicts whether a Reddit comment indicates the linked
# image/content is AI-generated, not AI-generated, or something else
# (unclear/off-topic/etc), using the same context it was trained on: the
# post title, the post's own text (if any), and the comment body.
CLASS_LABELS = ['AI', 'Not AI', 'Other']


def _parse_bool(val):
    #Robustly interpret a cell as a boolean -- see train.py for why a
    #naive bool(val) is wrong for string values like 'False'.
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    s = str(val).strip().upper()
    return s in ('TRUE', '1', 'YES', 'Y')


def _combine_context(title, parent_text, comment_body):
    #Combine a Reddit comment's post title, parent (submission) text, and
    #the comment body itself into a single string for embedding -- must
    #match train.py's _combine_context exactly, since the model was
    #trained on strings built this same way.
    parts = []
    for label, val in [('Title', title), ('Post text', parent_text), ('Comment', comment_body)]:
        if val is not None and str(val).strip() and str(val).strip().lower() != 'nan':
            parts.append(f'{label}: {val}')
    return '\n\n'.join(parts)


def read_any(file):
    if file.endswith('.csv') or file.endswith('.tsv') :
        df = pd.read_csv(file)
    elif file.endswith('.json'):
        df = pd.read_json(file)
    elif file.endswith('.xml'):
        df = pd.read_xml(file)
    elif file.endswith('.xls') or file.endswith('.xlsx'):
        df = pd.read_excel(file)
    elif file.endswith('.hdf'):
        df = pd.read_hdf(file)           
    elif file.endswith('.sql'):
        df = pd.read_sql(file)
    else:
        raise ValueError(f'ERROR: Unsupported filetype: {file}')
    return df

def load_data(file):
    data = read_any(file)
    # Accept either the friendly lowercase column names ('title',
    # 'parent_text', 'comment_body') or the spreadsheet's exact headers
    # ('Title', 'Parent Text', 'Comment Body'), matched case-insensitively.
    col_map = {str(c).strip().lower(): c for c in data.columns}

    def _find_col(*names):
        for name in names:
            if name.lower() in col_map:
                return col_map[name.lower()]
        return None

    title_col = _find_col('title')
    parent_col = _find_col('parent_text', 'parent text')
    body_col = _find_col('comment_body', 'comment body')
    if title_col is None or parent_col is None or body_col is None:
        raise Exception('ERROR: Could not find "title", "parent_text", and "comment_body" '
                         'columns (or their spreadsheet-header equivalents "Title", '
                         '"Parent Text", "Comment Body"). Please check the input file.')
    return data, title_col, parent_col, body_col

def inference(file,model_filename,sentence_tf= EMBEDDING_MODEL):
    data,title_col,parent_col,body_col = load_data(file)
    if data is None or len(data) == 0:
        raise Exception('ERROR: data file is blank: ',file)
    if not os.path.exists(model_filename):
        raise Exception('ERROR: model file is missing: ',model_filename)
    try:
        clf = pk.load(open(model_filename,'rb'))
    except:
        raise Exception('ERROR: model file could not be open (perhaps it is corrupted or in the wrong format): ',model_filename)
    try:
        model = SentenceTransformer(sentence_tf)
    except:
        raise Exception('Error loading sentence transformer: ',sentence_tf)
    print('Using ',sentence_tf,': make sure this is the best transformer for your data.')

    # Build the same combined title/post-text/comment context per row that
    # the model was trained on (see train.py's _combine_context)
    combined_texts = [
        _combine_context(row.get(title_col), row.get(parent_col), row.get(body_col))
        for _, row in data.iterrows()
    ]

    all_pred = []  # will hold one (3,) class-probability row per input text
    batch = []
    num_parsed = 1000
    for ii,text in enumerate(combined_texts):
        batch.append(text)
        if ii % num_parsed == 0:
            try:
                embeddings = model.encode(batch)
                # clf.predict_proba returns shape (n, 3): one probability per
                # class (AI / Not AI / Other)
                pred_prob = list(clf.predict_proba(embeddings))
                all_pred+=pred_prob
                batch = []
            except:
                raise Exception('ERROR: one or more text could not be parsed for classification. Look around index ',data.index.values[ii],'-',data.index.values[(ii+1)*num_parsed])
        # catching any leftovers
    try:
        embeddings = model.encode(batch)
        pred_prob = list(clf.predict_proba(embeddings))
        all_pred+=pred_prob
        batch = []
    except:
        raise Exception('ERROR: one or more text could not be parsed for classification. Look around index ',data.index.values[ii],'-',data.index.values[(ii+1)])

    all_pred = np.array(all_pred)  # shape (n, 3)
    data['prob_AI'] = all_pred[:,0]
    data['prob_Not_AI'] = all_pred[:,1]
    data['prob_Other'] = all_pred[:,2]
    # final prediction: whichever of the 3 classes has the highest probability
    data['predicted_class'] = [CLASS_LABELS[i] for i in np.argmax(all_pred, axis=1)]
    return data

            
