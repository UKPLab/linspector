import os

# relative paths in case we call them from other modules

LANG_ISO_SRC = os.path.join(os.path.dirname(__file__), "language_iso_codes.txt")
UNIMORPH_SCHEMA_SRC = os.path.join(os.path.dirname(__file__), "unimorph-schema.tsv")

SOURCE = {}
# SIGMORPHON-2016
# Task 1: Inflection (lemma.POS + target tag -> inflected form)
# Task 2: Reinflection (form.POS + source tag + target tag -> inflected form)
# Task 3: Unlabeled Reinflection (form.POS + target tag -> inflected form)
# MSD: UD-like, category=value; with ambiguity
SOURCE["sigmorphon16"] = os.path.join(os.path.dirname(__file__), "../unzipped/sigmorphon2016-master/data")

# CoNLL-2017 SIGMORPHON
# Task 1: Inflection (lemma.POS + target tag -> inflected form)
# Task 2: Paradigm cell filling (lemma.POS + incomplete paradigm -> fill remaining cells)
# MSD: Unimorph
SOURCE["sigmorphon17"] = os.path.join(os.path.dirname(__file__), "../unzipped/conll2017-master/all/task1")  # TODO: test - only answers

# CoNLL-2018 SIGMORPHON
# Task 1: Inflection
# Task 2: Inflection in context (cloze task). Two tracks: with morphological info and without
# MSD: Unimorph
SOURCE["sigmorphon18"] = os.path.join(os.path.dirname(__file__), "../unzipped/conll2018-master/task1/all")

# SIGMORPHON-2019
# Task 1: Inflection, high-resource + some low-resource forms -> low_resource forms
# Task 2: Morphological analysis and lemmatization in context
# MSD: Unimorph
SOURCE["sigmorphon19"] = os.path.join(os.path.dirname(__file__), "../unzipped/sigmorphon19_task1/task1/")

# Unimorph
# Full paradigms with lemmata and forms
# MSD: Unimorph
SOURCE["unimorph"] = os.path.join(os.path.dirname(__file__), "../unzipped/unimorph")