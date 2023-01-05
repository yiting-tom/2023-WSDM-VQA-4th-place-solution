import logging
import pandas as pd

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from configs import paths


L: logging.Logger = logging.getLogger(logging.basicConfig(level=logging.INFO))
MODEL_ID = 'damo/mplug_visual-question-answering_coco_large_en'

def generate_vqa_input_pairs(df: pd.DataFrame) -> list:
    return [
        {
            'image': str(paths.TEST_IMG / row['image']),
            'question': row['question'],
        } for _, row in df.iterrows()
    ]

def main():
    L.info(f"read formatted input test data: {paths.FORMATED_TEST_PKL}")
    df = pd.read_pickle(paths.FORMATED_TEST_PKL)

    L.info(f"generate vqa input pairs")
    input_pairs = generate_vqa_input_pairs(df[['image', 'question']])

    L.info(f"load pipeline: {MODEL_ID}")
    pipeline_vqa = pipeline(
        task=Tasks.visual_question_answering,
        model=MODEL_ID,
        device='gpu',
    )

    L.info(f"run pipeline: {MODEL_ID}")
    mplug_predict = pipeline_vqa(input_pairs)

    L.info(f"save mplug_predict to {paths.MPLUG_PRED_PKL}")
    pd.to_pickle(pd.DataFrame(mplug_predict), paths.MPLUG_PRED_PKL)

if __name__ == '__main__':
    main()
