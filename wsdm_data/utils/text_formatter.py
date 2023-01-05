from typing import Union
from pathlib import Path

import pandas as pd

from configs import paths


def url_to_img_filename(url: str) -> str:
    return url.split('/')[-1]

def url_to_img_id(url: str) -> str:
    return url_to_img_filename(url).split('.')[0]

def url_to_img_filepath(url: str, parent_dir: Union[str, Path]) -> str:
	return str(parent_dir / url.split('/')[-1])

def wsdmdata_to_url(data: dict) -> str:
    return f"https://toloka-cdn.azureedge.net/wsdmcup2023/{data['img_id']}.jpg"

def id_to_url(id: str) -> str:
    return f"https://toloka-cdn.azureedge.net/wsdmcup2023/{id}.jpg"

def id_to_img_filepath(id: str, stage: str = 'train') -> str:
    return str(paths.WSDM / stage / (id + '.jpg'))

def series_url_to_img_filename(url: pd.Series) -> pd.Series:
	return url.map(url_to_img_filename)

def series_url_to_img_filepath(url: pd.Series) -> pd.Series:
	return url.map(url_to_img_filepath)