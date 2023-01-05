"""The main entry point for reproductions.

1. format_input_test_csv
2. generate_vqa_input
3. vqa_predict_ofa
4. vqa_predict_mplug
5. generate_vg_input
6. vg_predict_ofa
7. format_final_answer
"""
import time
import logging

from configs import paths
from wsdm_data import utils


L: logging.Logger = logging.getLogger(logging.basicConfig(level=logging.INFO))

def run_script(filename: str):
    cmd = (
        f"cd {paths.ROOT}"
        " && "
        f"python3 {filename}"
    )
    utils.run_and_wait(cmd)

def main():
    start = time.time()
    # == format the test csv ==
    run_script("format_input_test_csv.py")
    assert paths.FORMATED_TEST_PKL.exists()

    # == generate the VQA dataset from the formatted input ==
    run_script("generate_vqa_input.py")
    assert paths.VQA_TSV.exists()

    # == predict the VQA answer (ofa) ==
    run_script("vqa_predict_ofa.py")
    assert paths.OFA_PRED_JSON.exists()

    # == predict the VQA answer (mplug) ==
    run_script("vqa_predict_mplug.py")
    assert paths.MPLUG_PRED_PKL.exists()

    # == gnerate the VG dataset from the original dataset ==
    run_script("generate_vg_input.py")
    assert paths.VG_TSV.exists()

    # == generate the VG prediction ==
    run_script("vg_predict_ofa.py")
    assert paths.VG_PRED_JSON.exists()

    # == format the final answer and export to mnt/output/answer.csv ==
    run_script("format_final_answer.py")
    assert paths.SUBMIT_FILEPATH.exists()

    L.info(f"total time: {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    main()