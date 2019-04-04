import json
import shutil
import sys
import os

from allennlp.commands import main

basic_config = "config/dep/depparse.jsonnet"

serialization_dir = "dep_out"

shutil.rmtree(serialization_dir, ignore_errors=True)

path_prefix_by_lang = {"de": "data/ud_dep/de/de_gsd-ud-", 
						"es": "data/ud_dep/es/es_ancora-ud-",
						"fi": "data/ud_dep/fi/fi_tdt-ud-",
						"tr": "data/ud_dep/tr/tr_imst-ud-",
						"ru": "data/ud_dep/ru/ru_syntagrus-ud-"}

embeddings = ['bpe', 'fasttext', 'elmo','muse_supervised', 'w2v']
base_embed_path = "../../embeddings"
for lang in path_prefix_by_lang:
	for embed in embeddings:
		embed_path = os.path.join(base_embed_path, lang, embed, "final_embeds.vec")
		lang_specific_config = {}
		lang_specific_config["train_data_path"] = path_prefix_by_lang[lang]+"train.conllu"
		lang_specific_config["validation_data_path"] = path_prefix_by_lang[lang]+"dev.conllu"
		lang_specific_config["test_data_path"] = path_prefix_by_lang[lang]+"test.conllu"
		
		if embed=='elmo':
			lang_specific_config["model"] = {"text_field_embedder": {"tokens": {"embedding_dim": 1024, "pretrained_file": embed_path}}, "encoder": {"input_size": 1124}}
		else:
			lang_specific_config["model"] = {"text_field_embedder": {"tokens": {"embedding_dim": 300, "pretrained_file": embed_path}}, "encoder": {"input_size": 400}}
			
		overrides = json.dumps(lang_specific_config)

		sys.argv = [
		    "allennlp",  # command name, not used by main
		    "train",
		    basic_config,
		    "-s", os.path.join(serialization_dir, lang, embed),
		    "-o", overrides,
		    "-f",
		]

		main()