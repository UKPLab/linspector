import json
import shutil
import sys
import os

from allennlp.commands import main

basic_config = "pos.jsonnet"

serialization_dir = "pos_out"

shutil.rmtree(serialization_dir, ignore_errors=True)

path_prefix_by_lang = {"de": "../data/ud_pos/de/de_gsd-ud-", 
						"es": "../data/ud_pos/es/es_ancora-ud-",
						"fi": "../data/ud_pos/fi/fi_tdt-ud-",
						"ru": "../data/ud_pos/ru/ru_syntagrus-ud-",
						"tr": "../data/ud_pos/tr/tr_imst-ud-"}

#embeddings_by_lang = {lang: f"/home/kuznetsov/Projects/subword/w2v_sbw/clean_vec/{lang}-vectors.vec" for lang in path_prefix_by_lang}
embeddings = ['bpe', 'fasttext', 'elmo','muse_supervised']
base_embed_path = "/home/sahin/Workspace/embeddings"
for lang in path_prefix_by_lang:
	for embed in embeddings:
		#out = os.path.join(serialization_dir, lang, embed)
		embed_path = os.path.join(base_embed_path, lang, embed, "final_embeds.vec")
		lang_specific_config = {}
		lang_specific_config["train_data_path"] = path_prefix_by_lang[lang]+"train.conllu.pos"
		lang_specific_config["validation_data_path"] = path_prefix_by_lang[lang]+"dev.conllu.pos"
		lang_specific_config["test_data_path"] = path_prefix_by_lang[lang]+"test.conllu.pos"
		
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
		    "-f",
		]

		main()