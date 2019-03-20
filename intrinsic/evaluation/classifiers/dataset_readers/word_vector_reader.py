from typing import Dict, List, Iterator
import json
import logging

from overrides import overrides

import tqdm
import os
import sys
import codecs
import numpy as np

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, ArrayField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer


logger = logging.getLogger(__name__)


@DatasetReader.register("word_vectors")
class WordVectorDatasetReader(DatasetReader):
	"""
	Reads a text file for classification task
	Expected format for each input line: word and tag (optional), separated by a tab. 

	The output of ``read`` is a list of ``Instance``s with the following fields:
		token: ``TokenField``
		label: ``LabelField`` 

	Parameters
	----------
	lazy : ``bool`` (optional, default=False)
		Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
		take longer per batch.  This also allows training with datasets that are too large to fit
		in memory.

	"""

	def __init__(self,
				 lazy: bool = False,
				 tokenizer: Tokenizer = None,
				 token_indexers: Dict[str, TokenIndexer] = None) -> None:
		super().__init__(lazy)
		self._tokenizer = tokenizer or WordTokenizer()
		self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}


	@overrides
	def text_to_instance(self, token: List[Token], label: str = None) -> Instance:
		
		token_field = TextField(token, self._token_indexers)
		fields = {'token': token_field}
		if label is not None:
		   fields['label'] = LabelField(label)
		
		return Instance(fields)


	@overrides
	def _read(self, file_path: str) -> Iterator[Instance]:
		"""
		Reads input file.
		
		Args:
			file_path (str): path for file
		"""

		with codecs.open(file_path, encoding='utf-8') as f:
			for line in f:
				items = line.strip().split('\t')
				token = items[0]
				# label is optional
				if len(items) > 1:
					label = items[1]
				else:
					label = None

				yield self.text_to_instance([Token(token)], label)
