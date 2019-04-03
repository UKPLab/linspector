// source https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz
{
    "dataset_reader": {
        "type": "universal_dependencies",
        "use_language_specific_pos": false,
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            },
        }
    },
    "vocabulary": {
        "non_padded_namespaces": ["pos"],
    },
    "iterator": {
        "type": "bucket",
        "batch_size": 128,
        "sorting_keys": [
            [
                "words",
                "num_tokens"
            ]
        ]
    },
    "model": {
        "type": "biaffine_parser",
        "arc_representation_dim": 200,
        "dropout": 0.3,
        "encoder": {
            "type": "stacked_bidirectional_lstm",
            "hidden_size": 250,
            "input_size": 400,
            "num_layers": 2,
            "recurrent_dropout_probability": 0.3,
            "use_highway": true
        },
        "initializer": [
            [
                ".*feedforward.*weight",
                {
                    "type": "xavier_uniform"
                }
            ],
            [
                ".*feedforward.*bias",
                {
                    "type": "zero"
                }
            ],
            [
                ".*tag_bilinear.*weight",
                {
                    "type": "xavier_uniform"
                }
            ],
            [
                ".*tag_bilinear.*bias",
                {
                    "type": "zero"
                }
            ],
            [
                ".*weight_ih.*",
                {
                    "type": "xavier_uniform"
                }
            ],
            [
                ".*weight_hh.*",
                {
                    "type": "orthogonal"
                }
            ],
            [
                ".*bias_ih.*",
                {
                    "type": "zero"
                }
            ],
            [
                ".*bias_hh.*",
                {
                    "type": "lstm_hidden_bias"
                }
            ]
        ],
        "input_dropout": 0.3,
        "pos_tag_embedding": {
            "embedding_dim": 100,
            "sparse": true,
            "vocab_namespace": "pos"
        },
        "tag_representation_dim": 200,
        "text_field_embedder": {
            "tokens": {
                "type": "embedding",
                "embedding_dim": 300,
                "pretrained_file": "[VEC]",
                "sparse": true,
                "trainable": false
            }
        },
        "use_mst_decoding_for_validation": true
    },
    "train_data_path": "[TRAIN]",
    "validation_data_path": "[DEV]",
    "test_data_path": "[TEST]",
    "trainer": {
        "cuda_device": 3,
        "grad_norm": 5,
        "num_epochs": 30,
        "optimizer": {
            "type": "dense_sparse_adam",
            "betas": [
                0.9,
                0.9
            ]
        },
        "patience": 10,
        "validation_metric": "+LAS"
    },
    "evaluate_on_test": true
}
