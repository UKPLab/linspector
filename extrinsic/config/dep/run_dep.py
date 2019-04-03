import json
import shutil
import sys
import os

from allennlp.commands import main

basic_config = "depparse.jsonnet"

serialization_dir = "dep_out"

shutil.rmtree(serialization_dir, ignore_errors=True)

path_prefix_by_lang = {"de": "../../downstream_multilingual_data/ud/UD_German-GSD/de_gsd-ud-", 
						"es": "../../downstream_multilingual_data/ud/UD_Spanish-AnCora/es_ancora-ud-",
						"fi": "../../downstream_multilingual_data/ud/UD_Finnish-TDT/fi_tdt-ud-",
						"tr": "../../downstream_multilingual_data/ud/UD_Turkish-IMST/tr_imst-ud-",
						"ru": "../../downstream_multilingual_data/ud/UD_Russian-SynTagRus/ru_syntagrus-ud-"}

#embeddings_by_lang = {lang: f"/home/kuznetsov/Projects/subword/w2v_sbw/clean_vec/{lang}-vectors.vec" for lang in path_prefix_by_lang}
embeddings = ['bpe', 'fasttext', 'elmo','muse_supervised']
base_embed_path = "/home/sahin/Workspace/embeddings"
for lang in path_prefix_by_lang:
	for embed in embeddings:
		#out = os.path.join(serialization_dir, lang, embed)
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