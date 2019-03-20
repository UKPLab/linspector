import json
import shutil
import sys
import os

from allennlp.commands import main

basic_config = "ner.jsonnet"

serialization_dir = "ner_out"

shutil.rmtree(serialization_dir, ignore_errors=True)

path_prefix_by_lang = {"de": "../data/ner/de/", 
						"es": "../data/ner/es/",
						"fi": "../data/ner/fi/",
						"ru": "../data/ner/ru/",
						"tr": "../data/ner/tr/"}

embeddings_by_lang = {lang: f"/home/kuznetsov/Projects/subword/w2v_sbw/clean_vec/{lang}-vectors.vec" for lang in path_prefix_by_lang}

for lang in path_prefix_by_lang:
	out = os.path.join(serialization_dir, lang)
	lang_specific_config = {}
	lang_specific_config["train_data_path"] = path_prefix_by_lang[lang]+"train.txt"
	lang_specific_config["validation_data_path"] = path_prefix_by_lang[lang]+"dev.txt"
	lang_specific_config["test_data_path"] = path_prefix_by_lang[lang]+"test.txt"
	lang_specific_config["model"] = {"text_field_embedder": {"token_embedders": {"tokens": {"pretrained_file": embeddings_by_lang[lang]}}}}

	overrides = json.dumps(lang_specific_config)

	sys.argv = [
	    "allennlp",  # command name, not used by main
	    "train",
	    basic_config,
	    "-s", os.path.join(serialization_dir, lang),
	    "-o", overrides,
	]

	main()