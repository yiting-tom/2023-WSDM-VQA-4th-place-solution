"""
The information are separated by tabs.
0. uniq-id
1. image-id
2. text
3. region-coord (separated by commas)
4. image base64 string

e.g.)
79_1    237367  A woman in a white blouse holding a glass of wine.  230.79,121.75,423.66,463.06 9j/4AAQ...1pAz/9k=

"""
import re
import logging

import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer

from configs import paths, consts
from wsdm_data import utils


L: logging.Logger = logging.getLogger(logging.basicConfig(level=logging.INFO))

nltk.download('wordnet')
nltk.download('omw-1.4')
wnl = WordNetLemmatizer()


def remove_no_one_yes_nothing(x):
    if re.match(
        r"(^everywhere$)|(^nowhere$)|(^when )|(^how to )|(^no one$)|(^yes$)|(^nothing$)",
        x.lower(),
    ):
        return ""
    return x

def lemmatize(x):
    x = wnl.lemmatize(x, 'n')
    # remove preposition
    result = []
    for i in x.split(' '):
        if i not in ['in', 'on']:
            result.append(i)
    return ' '.join(result)


def process_static_data(
    df: pd.DataFrame,
    df_vqa: pd.DataFrame,
    df_vqa_answer: pd.DataFrame,
) -> pd.DataFrame:
    # pack bounding box
    df_vqa['bbox'] = df[['height', 'width']]\
        .astype(str)\
        .agg(
            lambda x: ','.join(
                ('0', x['width'], '0', x['height'])
            ),
            axis=1,
        )

    # rename question_id as unique_id
    df_vqa.rename(
        columns={'question_id': 'unique_id'},
        inplace=True,
    )

    # merge vqa answer
    df_vg = df_vqa.join(
        df_vqa_answer,
        on='unique_id',
        rsuffix='_text',
    )

    # rename answer as text
    df_vg['text'] = df_vg['answer_text']

    # concat original question
    df_vg['text'] = df['question'].str.cat(df_vg['text'], sep=' ')

    return df_vg

def main():
    # load vqa data
    L.info(f"read VQA tsv file from {paths.VQA_TSV}")
    df_vqa = utils.load_tsv(paths.VQA_TSV)

    # load original csv provided by official
    L.info(f"read formatted test input data from {paths.FORMATED_TEST_PKL}")
    df = pd.read_pickle(paths.FORMATED_TEST_PKL)

    L.info(f"load ofa vqa prediction")
    df_ofa_answer: pd.Series = utils.load_json(paths.OFA_PRED_JSON)['answer']
    # filter out invalid text
    df_ofa_answer = df_ofa_answer.map(remove_no_one_yes_nothing)
    df_ofa_answer = df_ofa_answer.map(lemmatize)
    
    L.info(f"load mplug vqa prediction")
    df_mplug_answer: pd.Series = pd.read_pickle(paths.MPLUG_PRED_PKL)['text']
    # filter out invalid text
    df_mplug_answer = df_mplug_answer.map(remove_no_one_yes_nothing)
    df_mplug_answer = df_mplug_answer.map(lemmatize)

    L.info(f"concat ofa and mplug vqa prediction")
    df_vqa_answer = df_ofa_answer.str.cat(
        df_mplug_answer,
        sep=' , ',
    )

    # process static data
    L.info(f"Processing static data: test")
    df_vg = process_static_data(
        df=df,
        df_vqa=df_vqa,
        df_vqa_answer=df_vqa_answer,
    )

    # only keep necessary columns
    L.info(f"Saving VG data: test to {paths.VG_TSV}")
    utils.save_tsv(paths.VG_TSV, df_vg[consts.VG_TSV_COLUMNS])

if __name__ == "__main__":
    main()
