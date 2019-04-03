# -*- coding: utf-8 -*-
# @Author: claravania
# @Date:   2019-01-21 10:43:25
# @Last Modified by:   claravania
# @Last Modified time: 2019-04-01 16:03:48

import os
import argparse

from collections import defaultdict



def main():
	parser = argparse.ArgumentParser()

	# argument options
	parser.add_argument('--data_dir', type=str, required=True,
						help='path to probing test data dir.')
	parser.add_argument('--embedding_dir', type=str, required=True,
						help='path to pretrained embeddings.')
	parser.add_argument('--embedding_types', type=str, required=True,
						help='types of embedding. If more than one, separate by a comma \
						without whitespaces, e.g., word2vec,bpe,fasttext')
	parser.add_argument('--mode', type=str, required=True,
						help='classification or contrastive (for paired test)')
	parser.add_argument('--model_dir', type=str, default='probing_models',
						help='directory to store model files')
	parser.add_argument('--json_config_dir', type=str, default='json_configs',
						help='directory to put json config files')
	parser.add_argument('--bash_dir', type=str, default='bash_scripts',
						help='directory to put bash scripts')
	parser.add_argument('--languages', type=str, default='all',
						help='languages to test. If more than one, separate by a comma. \
						without whitespaces, e.g., russian,finnish,german. Use \'all\' to \
						include all supported languages.')
	args = parser.parse_args()
	generate_scripts(args)


def make_executable(path):
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2    # copy R bits to X
    os.chmod(path, mode)


def generate_scripts(args):

	# supported languages
	langId = {'portuguese': 'pt',
				'french': 'fr',
				'serbo-croatian': 'sh',
				'polish': 'pl',
				'czech':'cs',
				'modern-greek':'el',
				'catalan': 'ca',
				'bulgarian': 'bg',
				'danish': 'da',
				'estonian': 'et',
				'quechua': 'qu',
				'swedish': 'sv',
				'armenian': 'hy',
				'macedonian': 'mk',
				'arabic': 'ar',
				'dutch': 'nl',
				'hungarian': 'hu',
				'italian': 'it',
				'romanian':'ro',
				'ukranian': 'uk',
				'german': 'de',
				'finnish': 'fi',
				'russian': 'ru',
				'turkish': 'tr',
				'spanish': 'es'
		   }
	mode = args.mode
	embeddings = args.embedding_types.split(',')
	data_dir = args.data_dir

	# NOTE:
	# embeddings dir should have the following directory structure
	# embeddings_dir/language_id/embedding_type/final_embeds.vec
	# example: embeddings/de/word2vec/final_embeds.vec
	embedding_dir = args.embedding_dir

	# directory of output files
	json_config_dir = args.json_config_dir
	bash_dir = args.bash_dir
	model_dir = args.model_dir

	languages = {}
	if args.languages != 'all':
		language_set = args.languages.split(',')
	else:
		language_set = langId.keys()
	for lang in language_set:
		languages[lang] = langId[lang]
	
	# load feature template
	json_template = os.path.join("scripts", "subword" + mode + ".json")

	# this file lists number of possible values for each feature
	feature_set = os.path.join("scripts", mode + "_features.txt")
	

	# load feature dicts, currently only supported for five languages
	# German, Turkish, Finnish, Spanish, Russian
	# see feature_set file for examples
	feat_dict = defaultdict(lambda: defaultdict(int))
	with open(feature_set) as f:
		for line in f:
			line = line.strip()
			feature, lang, count, _ = line.split('\t')
			if lang in languages:
				feature = feature.replace(' ', '_')
				feat_dict[feature][lang] = count


	# generate JSON config files
	for emb_type in embeddings:
		for task in feat_dict.keys():
			for lang in feat_dict[task]:

				output_file = os.path.join(json_config_dir, emb_type, task + '-' + lang + '.json')
				os.makedirs(os.path.dirname(output_file), exist_ok=True)
				fout = open(output_file, 'w')

				embedding_file = os.path.join(embedding_dir, langId[lang], emb_type, 'final_embeds.vec')
				if not os.path.isfile(embedding_file):
					sys.stderr("Embedding is not available for ", lang, ". This language will be skipped.")
					continue
				with open(json_template) as f:
					for line in f:
						if '_data_path' in line:
							path = data_dir + '/' + task + '/' + lang
							newline = line.replace('[data_dir]/[feat]/[lang]', path)
							fout.write(newline)
						elif emb_type == 'elmo' and 'embedding_dim' in line:
							newline = line.replace('300', '1024')
							fout.write(newline)
						elif emb_type == 'elmo' and 'input_dim' in line:
							if mode == 'classification':
								newline = line.replace('300', '1024')
							elif mode == 'contrastive':
								newline = line.replace('600', '2048')
							fout.write(newline)
						elif 'hidden_dims' in line:
							newline = line.replace('x', str(feat_dict[task][lang])) 
							fout.write(newline)
						elif 'pretrained_embedding_file' in line:
							newline = line.replace('pretrained_embedding_file', embedding_file)
							fout.write(newline)
						else:
							fout.write(line)
				fout.close()


	# generate bash scripts
	for emb_type in embeddings:

		# train scripts
		bash_file = os.path.join(bash_dir, 'run_' + mode + '_' + emb_type + '.sh')
		os.makedirs(os.path.dirname(bash_file), exist_ok=True)

		with open(bash_file, 'w', encoding='utf-8') as f:

			# write headings
			f.write('#!/bin/bash\n\n')
			f.write('export PYTHON_PATH=$PATH\nexport PYTHONIOENCODING=utf-8\n\n')

			for task in feat_dict.keys():
				for lang in feat_dict[task]:
					json_file = os.path.join(json_config_dir, emb_type, task + '-' + lang + '.json')
					model_path = os.path.join(model_dir, lang, emb_type, task)
					f.write('allennlp train ' + json_file + ' -s ' + model_path + ' --include-package classifiers\n')

		make_executable(bash_file)


if __name__ == '__main__':
	main()
