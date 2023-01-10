import re
import logging
import unicodedata

import numpy as np
import pandas as pd

from configs import paths, consts


L: logging.Logger = logging.getLogger(logging.basicConfig(level=logging.INFO))

def full_width_to_half_width(s: str) -> str:
	"""full-width to half-width and translate to ascii with unicodedata"""
	return unicodedata.normalize('NFKD', s) \
		.encode('ascii', 'ignore') \
		.decode('utf-8', 'ignore')

def translate_head_punctuations(s: str) -> str:
	"""translate head punctuations to specific format"""
	sp = s.split(' ')
	x = sp[0]
	x = re.sub(r'w[wh]h', 'wh', x)
	x = re.sub(r'\'s', ' is', x)
	x = re.sub(r'\'re', ' are', x)
	x = re.sub(r'\'|\"|-|_|\,|/|\.', '', x)
	sp[0] = x
	result = ' '.join(sp).lower()
	return result.lower()

def reformat_question(q: str) -> str:
	"""reformat question to specific format"""
	q = q.lower()
	q = full_width_to_half_width(q)
	q = translate_head_punctuations(q)

	# if q is not end with '?', add '?'
	if not q.endswith('?'):
		q += '?'

	# remove redundant '?'
	while q.endswith('??'):
		q = q[:-1]

	# if q with multiple questions, only keep the first one
	if len(q.split('?')) > 2:
		q = q.split('?')[0] + '?'

	# capitalize first letter
	q = q.capitalize()

	return q

def main():
	L.info(f"read test csv: {paths.TEST_CSV}")
	df = pd.read_csv(paths.TEST_CSV)

	L.info(f"reformat question")
	df['question'] = df['question'].apply(reformat_question)
	
	# adding missing columns
	missing_columns = list(set(consts.FORMTTED_PKL_COLUMNS).difference(set(df.columns)))
	df[missing_columns] = np.nan
	assert set(consts.FORMTTED_PKL_COLUMNS) == set(df.columns)

	L.info(f"save test csv: {paths.FORMATED_TEST_PKL}")
	paths.FORMATED_TEST_PKL.parent.mkdir(parents=True, exist_ok=True)
	df[consts.FORMTTED_PKL_COLUMNS].to_pickle(paths.FORMATED_TEST_PKL)

if __name__ == '__main__':
	main()
