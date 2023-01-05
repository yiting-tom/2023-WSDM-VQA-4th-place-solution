"""Generate VQA data from the WSDM csv file and images.

1. read csv file which was defined in configs/paths.py::TEST_CSV
2. reformat the question by utils/text_formatter.py::reformat_question
3. extract the image id from the image url
4. convert the image from jpg to base64
5. (ongoing) extract the candidates from the image by VinVL
6. reorder the columns to match the VQA format
    a. question_id
    b. image_id
    c. question
    d. answer
    e. candidate
    f. image
7. save the data to configs/paths.py::VQA_DATASET / "test.tsv"
"""
import logging
from functools import partial

import pandas as pd

from wsdm_data import utils
from configs import paths, consts


L: logging.Logger = logging.getLogger(
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'))

def pipeline(d: pd.Series) -> dict:
    """pipeline

    Args:
        d (pd.Series): row of the dataframe
        image_dir (Path): directory of the images.

    Returns:
        pd.Series: processed row of the dataframe
    """
    image_filepath = paths.TEST_IMG / d['image']

    return dict(
        image=utils.filepath_to_base64(image_filepath),
    )

def generate_candidates():
    import numpy as np
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    df = pd.read_csv(paths.TEST_CSV)
    results = []
    object_detect = pipeline(
        Tasks.image_object_detection,
        model='damo/cv_tinynas_object-detection_damoyolo-m',
    )
    for imgname in df['image']:
        img_path = str(paths.TEST_IMG / imgname)
        result = object_detect(img_path)
        results.append('&&'.join(result['labels']))
    return np.array(results)

def generate_vqa_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    # rename the columns
    df['image_id'] = df['image']
    # apply image_dir into the pipeline
    part_pipeline = partial(pipeline)
    # apply the pipeline
    df.update(pd.DataFrame(
        df.apply(
            part_pipeline,
            axis=1
        ).values.tolist()
    ))
    # add question_id data by using the index
    df['question_id'] = df.index
    # add mock confidence and answer data
    df['answer'] = "0|!+"
    # add mock candidate data
    df['candidate'] = generate_candidates()
    # return the result
    return df[consts.VQA_TSV_COLUMNS]

def main():
    # load the original csv data
    L.info(f"Loading formatted input test pkl data: {paths.FORMATED_TEST_PKL}")
    df = pd.read_pickle(paths.FORMATED_TEST_PKL)

    # generate the vqa input data
    L.info(f"Generating VQA input data")
    df = generate_vqa_data(df=df)

    # save the data
    L.info(f"Saving VQA input data: {paths.VQA_TSV}")
    utils.save_tsv(paths.VQA_TSV, df)

if __name__ == '__main__':
    main()