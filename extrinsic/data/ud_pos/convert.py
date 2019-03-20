import os, glob, shutil
import sys

for src in ["de", "es", "fi", "ru", "tr"]:
	print(src)
	for f in glob.glob(src+"/*.conllu"):
		name = os.path.split(f)[-1]
		with open(os.path.join(src, name+".pos"), "w") as out_file:
			s = []
			for line in open(f, "r"):
				if not line.startswith("#"):	
					line = line.strip()
					if line:
						v = line.strip().split("\t")
						form = v[1].replace(" ", "_")  # bug in syntagrus data, a couple of numbers are given as e.g. 12 000, breaking the AllenNLP reader
						pos = v[3]
						s += [(form, pos)]
					else:
						if len(s)>0:
							out_file.write(" ".join([form+"##"+pos for (form, pos) in s])+"\n")
							s = []

