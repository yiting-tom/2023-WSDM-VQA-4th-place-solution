from pathlib import Path

ROOT: Path = Path(__file__).parent.parent

# Change "reproduction" to "/mnt" for reproducing the results
DATASET: Path = ROOT / "dataset"
MNT: Path = Path("/mnt")

# The original data provided by WSDM
TEST_CSV: Path = MNT / "data" / "test.csv"
TEST_IMG: Path = MNT / "data" / "imgs"
OUTPUT_DIR: Path = MNT / "output"

# The generated data
FORMATED_TEST_PKL: Path = DATASET / "formated_test.pkl"

VQA_TSV: Path = DATASET / "vqa.tsv"

OFA_PRED_JSON: Path = DATASET / "vqa_predict.json"
MPLUG_PRED_PKL: Path = DATASET / "mplug_predict.pkl"

VG_TSV: Path = DATASET / "vg.tsv"
VG_PRED_JSON: Path = DATASET / "vg_predict.json"

SUBMIT_FILEPATH: Path = MNT / "output" / "answer.csv"
