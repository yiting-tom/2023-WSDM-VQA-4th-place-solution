from typing import List

VQA_TSV_COLUMNS: List[str] = [
    'question_id', 'image_id', 'question',
    'answer', 'candidate', 'image',
]
VG_TSV_COLUMNS: List[str] = [
    'unique_id', 'image_id', 'text',
    'bbox', 'image',
]
FORMTTED_PKL_COLUMNS: List[str] = [
    'image', 'width', 'height', 'left',
    'top', 'right', 'bottom', 'question',
]
