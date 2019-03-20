from overrides import overrides

from allennlp.data import Instance
from allennlp.common.util import JsonDict
from allennlp.predictors.predictor import Predictor

import numpy as np
import json

@Predictor.register('word-classifier')
class WordClassifierPredictor(Predictor):
	"""
	Predictor wrapper for the WordClassifier.
	"""
	# @overrides
	def dump_line(self, outputs:JsonDict) -> str:

		return outputs['label'] + '\n'


