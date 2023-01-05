import logging

import pandas as pd
from configs import paths
from wsdm_data import utils


L: logging.Logger = logging.getLogger(
    logging.basicConfig(level=logging.INFO))

def format_answer_and_save_to_mnt(
    test_df: pd.DataFrame,
    predict: pd.DataFrame,
) -> None:
    # merge predict and test_df
    output = pd.merge(
        left=test_df,
        right=predict,
        left_on="uniq_id",
        right_index=True,
    )[['image', 'box']]

    # format bounding box
    output[['left', 'top', 'right', 'bottom']] = pd.DataFrame(
        list(output['box'].values)
    )
    del output['box']

    def correct_ans(box, wh) -> dict:
        """correct_ans 
        
        remove negative value and set right and bottom to image size
        if it is larger than image size

        Args:
            box (np.ndarray): The bounding box tuple
            wh (np.ndarray): The image size tuple

        Returns:
            dict: The corrected bounding box with left, top, right, bottom
        """
        box[box < 0] = 0
        left, top, right, bottom = box
        if left > right:
            right = wh[0]
        if top > bottom:
            bottom = wh[1]
        return dict(left=left, top=top, right=right, bottom=bottom)

    # correct bounding box
    correct = []
    for i, d in output.iterrows():
        correct.append(correct_ans(
            box=d[['left', 'top', 'right', 'bottom']],
            wh=test_df[['width', 'height']].loc[i].values)
        )

    # transform correct bounding boxes to dataframe
    out_ans = pd.DataFrame.from_dict(correct)

    # merge the image id and output answer
    out_file = pd.concat(
        [test_df['image'], out_ans],
        axis=1,
    )

    # save to mnt
    L.info(f"Saving final result to {paths.SUBMIT_FILEPATH}")
    out_file.to_csv(paths.SUBMIT_FILEPATH, index=False)

def main():
    L.info(f"Loading test.csv from {paths.TEST_CSV}")
    test_df = pd.read_csv(paths.TEST_CSV)

    # add uniq_id
    test_df['uniq_id'] = test_df.index

    L.info(f"Loading vg_predict from {paths.VG_PRED_JSON}")
    predict = utils.load_json(paths.VG_PRED_JSON)

    # convert to correct format and export to mnt
    format_answer_and_save_to_mnt(
        test_df=test_df,
        predict=predict,
    )

if __name__ == "__main__":
    main()