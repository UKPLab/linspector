import json
import shutil
import sys
import os

from allennlp.commands import main

basic_config = "config/NER/ner.jsonnet"

serialization_dir = "ner_out"

shutil.rmtree(serialization_dir, ignore_errors=True)

path_prefix_by_lang = {"de": "data/ner/de/", 
			"es": "data/ner/es/",
			"fi": "data/ner/fi/",
			"ru": "data/ner/ru/",
			"tr": "data/ner/tr/"}

embeddings = ['w2v', 'bpe', 'fasttext', 'elmo','muse_supervised']
base_embed_path = "../../embeddings" # adjust path
for lang in path_prefix_by_lang:
	for embed in embeddings:
		embed_path = os.path.join(base_embed_path, lang, embed, "final_embeds.vec")
		lang_specific_config = {}
		lang_specific_config["train_data_path"] = path_prefix_by_lang[lang]+"train.txt"
		lang_specific_config["validation_data_path"] = path_prefix_by_lang[lang]+"dev.txt"
		lang_specific_config["test_data_path"] = path_prefix_by_lang[lang]+"test.txt"

		if embed=='elmo':
			lang_specific_config["model"] = {"text_field_embedder": {"token_embedders": {"tokens": {"embedding_dim": 1024, "pretrained_file": embed_path}}}, "encoder": {"input_size": 1024}}
		else:
			lang_specific_config["model"] = {"text_field_embedder": {"token_embedders": {"tokens": {"embedding_dim": 300, "pretrained_file": embed_path}}}, "encoder": {"input_size": 300}}
			

		overrides = json.dumps(lang_specific_config)

		sys.argv = [
		    "allennlp",  # command name, not used by main
		    "train",
		    basic_config,
		    "-s", os.path.join(serialization_dir, lang, embed),
		    "-o", overrides,
		    "-f"
		]

		main()
