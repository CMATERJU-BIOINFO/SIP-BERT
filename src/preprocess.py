import numpy as np
import pandas as pd

ALL_AAS = 'ACDEFGHIKLMNPQRSTUVWXY'
ADDITIONAL_TOKENS = ['<OTHER>', '<START>', '<END>', '<PAD>']
ADDED_TOKENS_PER_SEQ = 2

n_aas = len(ALL_AAS)
aa_to_token_index = {aa: i for i, aa in enumerate(ALL_AAS)}
additional_token_to_index = {token: i + n_aas for i, token in enumerate(ADDITIONAL_TOKENS)}
token_to_index = {**aa_to_token_index, **additional_token_to_index}
index_to_token = {index: token for token, index in token_to_index.items()}
n_tokens = len(token_to_index)

def parse_seq(seq):
    if isinstance(seq, str):
        return seq
    elif isinstance(seq, bytes):
        return seq.decode('utf8')
    else:
        raise TypeError('Unexpected sequence type: %s' % type(seq))

def tokenize_seq(seq):
    other_token_index = additional_token_to_index['<OTHER>']
    return [additional_token_to_index['<START>']] + \
           [aa_to_token_index.get(aa, other_token_index) for aa in parse_seq(seq)] + \
           [additional_token_to_index['<END>']]

def pad_sequences(seqs, fixed_length):
    padded_seqs = []
    for seq in seqs:
        tokenized_seq = tokenize_seq(seq)
        seq_len = len(tokenized_seq)
        if seq_len > fixed_length:
            raise ValueError(f"Sequence length {seq_len} exceeds fixed length {fixed_length}")
        padding_needed = fixed_length - seq_len
        left_padding = padding_needed // 2
        right_padding = padding_needed - left_padding
        padded_seq = ([additional_token_to_index['<PAD>']] * left_padding +
                      tokenized_seq +
                      [additional_token_to_index['<PAD>']] * right_padding)
        padded_seqs.append(padded_seq)
    return np.array(padded_seqs, dtype=np.int32)

def extract_go_annotations(df, uniref_df):
    go_annotations = []
    for protein_id in df['Protein']:
        go_annotation = uniref_df[uniref_df['Protein'] == protein_id]['GO_annotation'].values
        if len(go_annotation) > 0:
            go_annotations.append(go_annotation[0])
        else:
            go_annotations.append(np.zeros(3))
    return np.array(go_annotations, dtype=np.float32)

def preprocess_data(df, uniref_df, fixed_length):
    seqs = df['seq'].tolist()
    padded_seqs = pad_sequences(seqs, fixed_length)
    go_annotations = extract_go_annotations(df, uniref_df)
    labels = df['label'].values
    return df[['Protein', 'seq', 'label']], padded_seqs, go_annotations, labels