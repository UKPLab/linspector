from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides


@Model.register("contextual_classifier")
class ContextualClassifier(Model):
    """
    This is a simple feedforward classifier. Given an embedding of a sentence, we want to predict some linguistic feature (tag) for one token for every sentence.

    Parameters
    ----------
    classifier_feedforward : ``FeedForward``
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:

        super(ContextualClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.classifier_feedforward = classifier_feedforward

        self.metrics = {"accuracy": CategoricalAccuracy()}
        self.loss = torch.nn.CrossEntropyLoss()
        self.vocab = vocab
        initializer(self)

    @overrides
    def forward(self,
                tokens: dict,
                token_indices,
                labels: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Variable, input tokens, required (tokens = list of sentences (sentence = list of word vectors)
        labels : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.
        token_indices : list of indices. For every sentence exists one index which binds a token to the label
        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """

        embedded_token = self.text_field_embedder(tokens)
        embedded_token = torch.squeeze(embedded_token,
                                       dim=1)  # Removes all rows from matrix with dimension = 1. But why?
        word_vectors = []
        # Extract word vectors for which we have a label
        for sentenceNr in range(0, len(embedded_token.tolist())):
            index = token_indices[sentenceNr]
            relevant_word_vector = embedded_token[sentenceNr][index].tolist()
            word_vectors += relevant_word_vector
        device = tokens["tokens"].device
        word_vectors = torch.tensor(word_vectors, device=device)
        logits = self.classifier_feedforward(word_vectors)
        output_dict = {'logits': logits}

        if labels is not None:
            loss = self.loss(logits, labels)
            for metric in self.metrics.values():
                metric(logits, labels)
            output_dict['loss'] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """

        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels") for x in argmax_indices]
        output_dict['label'] = labels

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
