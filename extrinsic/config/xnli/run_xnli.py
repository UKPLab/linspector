# -*- coding: utf-8 -*-
# @Author: claravania
# @Date:   2019-03-12 11:49:05
# @Last Modified by:   claravania
# @Last Modified time: 2019-03-15 14:29:13

import json
import shutil
import sys
import os

from allennlp.commands import main

basic_config = "scripts/xnli.jsonnet"

serialization_dir = "models"

train_prefix_by_lang = {"de": "data/XNLI-MT-1.0/XNLI-MT-1.0/multinli/multinli.train.de.jsonl", 
						"es": "data/XNLI-MT-1.0/XNLI-MT-1.0/multinli/multinli.train.es.jsonl",
						"tr": "data/XNLI-MT-1.0/XNLI-MT-1.0/multinli/multinli.train.tr.jsonl",
						"ru": "data/XNLI-MT-1.0/XNLI-MT-1.0/multinli/multinli.train.ru.jsonl"}

dev_prefix_by_lang = {"de": "data/XNLI-1.0/XNLI-1.0/xnli.dev.de.jsonl", 
						"es": "data/XNLI-1.0/XNLI-1.0/xnli.dev.es.jsonl",
						"tr": "data/XNLI-1.0/XNLI-1.0/xnli.dev.tr.jsonl",
						"ru": "data/XNLI-1.0/XNLI-1.0/xnli.dev.ru.jsonl"}


test_prefix_by_lang = {"de": "data/XNLI-1.0/XNLI-1.0/xnli.test.de.jsonl", 
						"es": "data/XNLI-1.0/XNLI-1.0/xnli.test.es.jsonl",
						"tr": "data/XNLI-1.0/XNLI-1.0/xnli.test.tr.jsonl",
						"ru": "data/XNLI-1.0/XNLI-1.0/xnli.test.ru.jsonl"}

lang = sys.argv[1]
embed = sys.argv[2]
base_embed_path = "../../embeddings/xnli"

embed_path = os.path.join(base_embed_path, lang, embed, "final_embeds.vec")
lang_specific_config = {}
lang_specific_config["train_data_path"] = train_prefix_by_lang[lang]
lang_specific_config["validation_data_path"] = dev_prefix_by_lang[lang]
lang_specific_config["test_data_path"] = test_prefix_by_lang[lang]


if embed=='elmo':
	lang_specific_config["model"] = {"text_field_embedder": {
						             "token_embedders": {
						                "tokens": {
						                    "type": "embedding",
						                    "pretrained_file": embed_path,
						                    "embedding_dim": 1024,
						                    "trainable": False
						                	}
						            	}
						        	},
						        	    "encoder": {
							            "type": "lstm",
							            "input_size": 1024,
							            "hidden_size": 300,
							            "num_layers": 1,
							            "bidirectional": True
							        }}
else:
	lang_specific_config["model"] = {"text_field_embedder": {
						             "token_embedders": {
						                "tokens": {
						                    "type": "embedding",
						                    "pretrained_file": embed_path,
						                    "embedding_dim": 300,
						                    "trainable": False
						                }
						            }
						        }}

	
overrides = json.dumps(lang_specific_config)

model_dir = os.path.join(serialization_dir, lang, embed)
shutil.rmtree(model_dir, ignore_errors=True)

sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    basic_config,
    "-s", model_dir,
    "-o", overrides,
    "-f",
]

main()