import glob
from collections import defaultdict


from data_util.global_var import *


def load_ds(ds_name, lang=None, tree_bank=False):
    read_fn = {"sigmorphon16": read_ds_sigmorphon16,
               "sigmorphon17": read_ds_sigmorphon17_18,
               "sigmorphon18": read_ds_sigmorphon17_18,
               "sigmorphon19": read_ds_sigmorphon19,
               "sigmorphon19-2": read_ds_sigmorphon19_task2,
               "unimorph": read_ds_unimorph}
    return read_fn[ds_name](SOURCE[ds_name], lang, tree_bank)


######################################


def read_single(src, file_format):
    assert file_format in ["unimorph", "sigmorphon16", "sigmorphon19-2"]
    if file_format == "sigmorphon19-2":
        return read_single_sentence(src)
    out = []
    with open(src) as f:
        for line in f:
            line = line.strip()
            if line:
                v = line.split("\t")
                if file_format == "unimorph":  # most datasets are lemma-form-msd
                    out += [{"lemma": v[0], "form": v[1], "msd": v[2].strip().rstrip()}]  # sometimes there is a space
                elif file_format == "sigmorphon16":  # sigmorphon-2016 is special
                    v = line.split("\t")
                    out += [{"lemma": v[0], "form": v[2], "msd": v[1].strip().rstrip()}]

    return out


def read_single_sentence(src, ud_dict):
    out = []
    to_line = None
    with open(src) as f:
        sentence = []
        for line in f:
            line = line.strip()
            if line:
                v = line.split("\t")
                if v[0].startswith("# sent_id"):
                    sent_id = v[0][v[0].index("= ") + 2:]
                if not v[0].isdigit():  # Ignore commentaries
                    continue
                line = v[0]
                if v[0] == "1" or v[0].startswith("1-"):  # New sentence
                    if len(sentence) != 0:
                        out += [sentence]
                        to_line = None
                    sentence = []
                form = v[1]
                if sent_id in ud_dict:
                    for x in ud_dict[sent_id]:
                        if x[0] == line:
                            to_line = x[1]
                            form = x[2]
                            word = {"lemma": v[2], "form": form, "msd": ""}
                            sentence += [word]
                if to_line is not None and to_line >= line:
                    sentence[-1]["msd"] += v[5]
                    if line == to_line:
                        to_line = None
                else:
                    word = {"lemma": v[2], "form": form, "msd": v[5]}

                    sentence += [word]
        if len(sentence) != 0:
            out += [sentence]
            to_line = None

    return out


# add support for reading individual lang
# TODO: add to other datasets if necessary
def read_ds_unimorph(path, lang=None, iso_map_src=LANG_ISO_SRC):
    iso_map = {}  # unimorph uses language ISO codes
    with open(iso_map_src) as f:
        for line in f:
            iso_code, name = line.strip().split("\t")
            name = name.lower().replace(" ", "-")
            iso_map[iso_code] = name

    out = defaultdict(list)
    for f in glob.glob(os.path.join(path, "*.txt")):
        iso_code = os.path.splitext(os.path.split(f)[-1])[0]
        iso_code = iso_code.split("-")[0]  # for cases like fin-1 + fin-2
        if lang is None:
            out[iso_map[iso_code]] += read_single(f, "unimorph")
        elif iso_map[iso_code] == lang:
            # then only read the file for the language
            out[iso_map[iso_code]] += read_single(f, "unimorph")
    return out


def read_ds_sigmorphon16(path, lang=None, treeBank=False):
    out = defaultdict(list)
    for f in glob.glob(os.path.join(path, "*-task1-train")):  # only training data for task 1
        lang = os.path.split(f)[-1].split("-")[-3]
        out[lang] = read_single(f, "sigmorphon16")
    return out


def read_ds_sigmorphon17_18(path, lang=None, treeBank=False):
    out = defaultdict(list)
    for f in glob.glob(os.path.join(path, "*-train-high")):  # train-high incorporates medium and low
        lang = os.path.split(f)[-1].split("-")[-3]
        out[lang] = read_single(f, "unimorph")
    return out


def read_ds_sigmorphon19(path, lang=None, treeBank=False):
    out = defaultdict(list)
    for fd in os.listdir(path):
        for f in glob.glob(os.path.join(path, fd, "*-train-*")):  # for low-resource, only train-low is given
            lang = os.path.split(f)[-1].split("-")[-3]
            out[lang] = read_single(f, "unimorph")
    return out

def sigmorphon_read_ud(src):
    last_sent_id = -1
    sent_id = 0
    all_splitted_words = dict()
    splitted_words = None
    with open(src) as f:
        sentence = []
        for line in f:
            line = line.strip()
            if line:
                v = line.split("\t")
                if v[0].startswith("# sent_id"):
                    last_sent_id = sent_id
                    sent_id = v[0][v[0].index("= ") + 2:]
                    continue
                elif v[0].startswith("#"):
                    continue

                line = v[0]
                if line == "1" or line.startswith("1-"):  # New sentence
                    if sent_id == 0 or last_sent_id == sent_id:
                        raise IndexError("No valid sent_id " + str(sent_id))
                    if splitted_words is not None:
                        all_splitted_words[last_sent_id] = splitted_words
                    splitted_words = list()

                if "-" in line:
                    from_line = line[:line.index("-")]
                    to_line = line[line.index("-") + 1:]
                    word = v[1]
                    splitted_words.append([from_line, to_line, word])
        if splitted_words is not None:
            all_splitted_words[sent_id] = splitted_words

    return all_splitted_words


def sigmorphon_get_splitted_words(folder_name):
    ud_path = "../unzipped/ud-treebanks-v2.4/"
    ud_dict = dict()
    for f in glob.glob(os.path.join(ud_path, folder_name, "*-train.conllu")):
        ud_dict.update(sigmorphon_read_ud(f))
    for f in glob.glob(os.path.join(ud_path, folder_name, "*-dev.conllu")):
        ud_dict.update(sigmorphon_read_ud(f))
    for f in glob.glob(os.path.join(ud_path, folder_name, "*-test.conllu")):
        ud_dict.update(sigmorphon_read_ud(f))
    print(folder_name)
    return ud_dict

def read_ds_sigmorphon19_task2(path, lang=None, tree_bank=False):
    out = defaultdict(list)

    # Read train and dev file for given language
    for fd in os.listdir(path):
        folder_name = os.path.split(fd)[-1]
        folder_lang = folder_name.split("_")[1].split("-")[0].lower()
        if tree_bank:
            cur_lang = folder_name
        else:
            cur_lang = folder_lang
        if folder_lang == lang or lang is None:
            ud_dict = sigmorphon_get_splitted_words(folder_name)
            for f in glob.glob(os.path.join(path, fd, "*-train.*")):
                out[cur_lang] += read_single_sentence(f, ud_dict)
            for f in glob.glob(os.path.join(path, fd, "*-dev.*")):
                out[cur_lang] += read_single_sentence(f, ud_dict)
    return out
