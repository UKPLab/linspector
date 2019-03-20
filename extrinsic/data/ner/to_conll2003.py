import glob, shutil, os
import random, math
from sklearn.model_selection import train_test_split

random.seed(1)

# Target format CoNLL-2003: word POS CHUNK NER

def _tuples_to_seq(tuples):
	return " ".join([f"{w}##{tag}" for w, tag in tuples])

def _seq_to_tuples(seq):
	return [t.split("##") for t in seq.split(" ")]

def _split(data, train, dev, test):
	random.shuffle(data)
	chunk = math.ceil(len(data)/100.0)
	ds = {"train": data[:chunk*train], "dev": data[chunk*train:chunk*(train+dev)], "test": data[chunk*(train+dev):]}
	return ds

# Finnish: word \t NER tag \t another NER tag?...
def convert_finnish(srcn, tgtn):
	with open(srcn) as src:
		with open(tgtn, "w") as tgt:
			for line in src:
				line = line.strip()
				if not line.startswith("<"):
					if line:
						word, tag, _ = line.split("\t")
						tgt.write(" ".join([word, "X", "X", tag]))
					tgt.write("\n")

# German: token_id \t word \t NER \t ?
def convert_german(srcn, tgtn):
	with open(srcn) as src:
			with open(tgtn, "w") as tgt:
				for line in src:
					line = line.strip()
					if not line.startswith("#"):
						if line:
							tid, word, tag, _ = line.split("\t")
							tgt.write(" ".join([word, "X", "X", tag]))
						tgt.write("\n")

# Russian: word \t NER \t another_NER
# No train/dev/test split, gotta do everything ourselves...
def convert_russian(srcn, tgt_folder):
	data = []
	sentence = []
	for line in open(srcn):
		line = line.strip()
		if line:
			if not line.startswith("-DOCSTART-"):
				try:
					word, ner, _ = line.split(" ")
					sentence += [(word, ner)]
				except ValueError:
					print(f"[ERR] {line}")
		else:
			if len(sentence)>0:
				data += [_tuples_to_seq(sentence)]
				sentence = []
	
	ds = _split(data, 60, 20, 20)
	for dset, data in ds.items():
		with open(os.path.join(tgt_folder, f"{dset}.txt"), "w") as f:
			for sentence in data:
				for w, t in _seq_to_tuples(sentence):
					f.write(" ".join([w, "X", "X", t])+"\n")
				f.write("\n")

# Spanish CoNLL-2002: word NER
def convert_spanish(srcn, tgtn):
	with open(srcn) as src:
		with open(tgtn, "w") as tgt:
			for line in src:
				line = line.strip()
				if line:
					word, tag = line.split(" ")
					tgt.write(" ".join([word, "X", "X", tag]))
				tgt.write("\n")


# Turkish: in-line format: domain \t tag_seq \t word_seq
def convert_turkish(srcf, tgt_folder):
	data = []
	for line in open(srcf):
		data += [line.strip()]
	ds = _split(data, 60, 20, 20)
	for dset, sentences in ds.items():
		with open(os.path.join(tgt_folder, f"{dset}.txt"), "w") as f:
			for sentence in sentences:
				domain, tags, words = sentence.split("\t")
				for t, w in zip(tags.split(" "), words.split(" ")):
					f.write(" ".join([w, "X", "X", t])+"\n")
				f.write("\n")




if __name__ == "__main__":
	out_dir = "_out"
	shutil.rmtree(out_dir)
	os.mkdir(out_dir)
	out_lang_dir = {lang: os.path.join(out_dir, lang) for lang in ["de", "es", "fi", "ru", "tr"]}
	[os.mkdir(x) for x in out_lang_dir.values()]

	print("fi")
	for fn in glob.glob("finnish/*.csv"):
		name = os.path.split(fn)[-1]
		convert_finnish(fn, os.path.join(out_lang_dir["fi"], name))

	print("de")
	for fn in glob.glob("german/germeval2014/*.tsv"):
		if not fn.endswith("unlabeled.tsv"):
			name = os.path.split(fn)[-1]
			convert_german(fn, os.path.join(out_lang_dir["de"], name))

	print("ru")
	convert_russian("russian/wikiner-ru.conll", out_lang_dir["ru"])

	print("es")
	for fn in glob.glob("spanish/conll2002/*"):
		name = os.path.split(fn)[-1]
		convert_spanish(fn, os.path.join(out_lang_dir["es"], name))

	print("tr")
	convert_turkish("turkish/TWNERTC_TC_Coarse Grained NER_DomainDependent_NoiseReduction.DUMP", out_lang_dir["tr"])







