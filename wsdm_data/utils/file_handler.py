from typing import List, Union, Any
from pathlib import Path
import pandas as pd

from configs import consts

def load_tsv(filepath: Path, columns: Union[str, List[str]] = "vqa"):
    if columns == "vqa":
        columns = consts.VQA_TSV_COLUMNS
    elif columns == "vg":
        columns = consts.VG_TSV_COLUMNS
    elif isinstance(columns, list):
        ...
    else:
        raise ValueError(f"columns: {columns} is not supported")

    return pd.read_csv(
        filepath,
        sep='\t',
        names=columns,
    )

def load_json(filepath: Path):
    return pd.read_json(filepath)

def save_tsv(filepath: Path, df: pd.DataFrame):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        filepath,
        sep='\t',
        index=False,
        header=False,
    )