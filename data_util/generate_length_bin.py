# -*- coding: utf-8 -*-
# @Author: claravania
# @Date:   2019-02-08 15:18:12
# @Last Modified by:   claravania
# @Last Modified time: 2019-02-11 14:16:12


import os


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

bins = {4:1, 8:2, 12:3, 16:4, 20:5, 100: 6}
#languages = ['finnish', 'german', 'turkish', 'spanish', 'russian']
languages = ['arabic', 'armenian', 'bulgarian', 'catalan', 'czech', 'danish', 'dutch','estonian', 'french', 'hungarian', 'italian', 'macedonian', 'modern-greek', 'polish', 'portuguese', 'quechua', 'romanian', 'serbo-croatian', 'swedish']

for language in languages:

	source_dir = os.path.join('../final_tests_for_other_langs/CharacterCount', language)
	target_dir = os.path.join('../final_tests_for_other_langs/CharacterBin', language)

	filenames = os.listdir(source_dir)
	for fname in filenames:
		source_file = os.path.join(source_dir, fname)
		target_file = os.path.join(target_dir, fname)
		ensure_dir(target_file)
		fout = open(target_file, 'w')
		with open(source_file) as f:
			for line in f:
				word, _ = line.strip().split('\t')
				len_bin = 6  # maximum length
				for max_length in sorted(bins.keys()):
					if len(word) <= max_length:
						len_bin = bins[max_length]
						break
				fout.write(word + '\t' + str(len_bin) + '\n')
						
		fout.close()



