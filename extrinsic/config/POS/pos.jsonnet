//https://github.com/allenai/allennlp/blob/53a555c4f4b2e9dcd69c2cd8906bbc4c2ca9da91/tutorials/getting_started/walk_through_allennlp/simple_tagger.json

{
  "dataset_reader": {
    "type": "sequence_tagging",
    "word_tag_delimiter": "##",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
    }
  },
  "train_data_path": "[TRAIN]",
  "validation_data_path": "[DEV]",
  "test_data_path": "[TEST]",
  "model": {
    "type": "simple_tagger",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 300,
            "pretrained_file": "[VEC]",
            "trainable": false,
        },
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": 300,
      "hidden_size": 100,
      "num_layers": 1,
      "dropout": 0.5,
      "bidirectional": true
    }
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size": 32,
  },
  "trainer": {
    "optimizer": "adam",
    "num_epochs": 40,
    "patience": 10,
    "cuda_device": -1,
    "validation_metric": "+accuracy"
  },
  "evaluate_on_test": true
}
