import codecs
from collections import defaultdict

import torch
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder


@TokenEmbedder.register("sentence_embedding")
class SentenceEmbedding(TokenEmbedder):
    """
        Embedder for contextual embeddings. which reads a file of the format 'sentence TAB index TAB vector'.
    """

    def read_file(self, path):
        self.embs = defaultdict(lambda: defaultdict())
        with codecs.open(path, encoding='utf-8') as f:
            for line in f:
                # Read sentence, index and word vector
                sp = line.split("\t")
                vector_str = sp[2]
                vector = []
                for n in vector_str.split(" "):
                    try:
                        vector.append(float(n))
                    except ValueError:
                        break
                index = int(sp[1])
                sentence = sp[0]

                # Save vector in a dict
                self.embs[sentence][index] = vector

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                word_inputs: torch.Tensor = None) -> torch.Tensor:
        """

        :param inputs: list of sentences (sentence = list of token indices)
        :param word_inputs: not used
        :return: tensor which contains a list of embedded sentences (every sentence is a list of word vectors)
        """
        if self.output_dim is None or self.output_dim == 0:
            raise NotImplementedError

        # Get tokens from token indices
        max_sentences_length = len(inputs[0].tolist())
        sentences = []
        for i in inputs:
            token_list = []
            for j in i:
                if j.item() != 0:
                    token = self.vocab.get_token_from_index(j.item())
                    token_list += [token]
            sentences += [token_list]

        sentence_emb = []

        # Read the embeddings from the dict
        for sentence_list in sentences:
            sentence = " ".join(sentence_list[0:-1])
            index = int(sentence_list[-1])

            try:
                word_embedding = self.embs[sentence][index]
            except KeyError:
                print("KEY ERROR " + sentence + " INDEX " + str(index))
                word_embedding = [0] * self.output_dim

            vector_list = []

            # Add zeros to the returning tensor for all tokens without vectors. AllenNLP wants an embedding for every token
            if index != 0:
                for i in range(0, index):
                    vector_list += [[0] * self.output_dim]
            vector_list += [word_embedding]

            for i in range(0, max_sentences_length - index - 1):
                vector_list += [[0] * self.output_dim]

            sentence_emb += [vector_list]

        # Create tensor
        device = inputs.device
        # print(sentence_emb)
        tensor = torch.tensor(sentence_emb, device=device)

        return tensor

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'SentenceEmbedding':
        cls.vocab = vocab
        embedding_dim = params["embedding_dim"]
        pretrained_file = params["pretrained_vector_file"]
        return cls(pretrained_file, embedding_dim)

    def __init__(self, file, vector_size) -> None:
        super().__init__()
        self.embs = {}
        self.output_dim = vector_size
        self.read_file(file)
