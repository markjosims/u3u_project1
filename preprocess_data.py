import pandas as pd
import sys
import numpy as np
from imblearn.over_sampling import ADASYN

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    all_pos = set()
    pos_cols = [col for col in df.columns if col.endswith('pos')]
    for col in pos_cols:
        all_pos.update(df[col].unique())
    all_pos = list(all_pos)+['<s>', '</s>']
    all_pos.sort()
    pos_onehot_col_dict = {}
    for col in pos_cols:
        for pos_tag in all_pos:
            pos_mask = df[col]==pos_tag
            pos_onehot_col_dict[col+'_is_'+pos_tag]=pos_mask.astype(int)
    pos_onehot_df = pd.DataFrame(pos_onehot_col_dict)
    pos_onehot_cols = pos_onehot_df.columns.tolist()
    df=pd.concat([df, pos_onehot_df], axis=1)
    freq_cols = [col for col in df.columns if col.endswith('freq')]
    num_words = len(df)
    num_sent = df['prev_word_pos_is_<s>'].value_counts()[1]
    sent_token_freq=num_sent/num_words
    sent_token_per_bill=sent_token_freq*10**9
    sent_token_zipf_freq = np.log10(sent_token_per_bill)
    for col in freq_cols:
        df.loc[df[col]==-1, col]=sent_token_zipf_freq
    word_cols = [col for col in df.columns if col.endswith('word')]
    wordlen_cols = []
    for col in word_cols:
        wordlen_col=col+'_len'
        df[wordlen_col]=df[col].str.len()
        wordlen_cols.append(wordlen_col)

    df['prev_word_same']=(df['word']==df['prev_word']).astype(int)
    df['next_word_same']=(df['word']==df['next_word']).astype(int)
    word_equal_cols = ['prev_word_same', 'next_word_same']

    y = df.edit_type
    majority_n = sum(y == 0)
    feature_cols = wordlen_cols + word_equal_cols + pos_onehot_cols
    X_ada = df[feature_cols]
    y_ada = df["edit_type"]

    adasyn = ADASYN(sampling_strategy={1: int(majority_n * 0.3), 3: int(majority_n * 0.3)},random_state=42)
    X_re, y_re = adasyn.fit_resample(X_ada, y_ada)

    df_ada = pd.DataFrame(X_re, columns=feature_cols)
    df_ada["edit_type"] = y_re
    return df_ada

if __name__ == '__main__':
    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)
    df_preproc = preprocess_df(df)
    df_preproc.to_csv(csv_path.replace('.csv', '-preprocessed.csv'), index=False)