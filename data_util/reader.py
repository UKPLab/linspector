import glob, os
from collections import defaultdict
from data_util.global_var import *


def load_ds(ds_name, lang=None):
    read_fn = {"sigmorphon16": read_ds_sigmorphon16,
              "sigmorphon17": read_ds_sigmorphon17_18,
              "sigmorphon18": read_ds_sigmorphon17_18,
              "sigmorphon19": read_ds_sigmorphon19,
              "unimorph": read_ds_unimorph}
    return read_fn[ds_name](SOURCE[ds_name], lang)

######################################


def read_single(src, file_format):
    assert file_format in ["unimorph", "sigmorphon16"]
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

# add support for reading individual lang
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
        elif iso_map[iso_code]==lang:
            # then only read the file for the language
            out[iso_map[iso_code]] += read_single(f, "unimorph")
    return out

def read_ds_sigmorphon16(path, lang=None):
    out = defaultdict(list)
    for f in glob.glob(os.path.join(path, "*-task1-train")):  # only training data for task 1
        lang = os.path.split(f)[-1].split("-")[-3]
        out[lang] = read_single(f, "sigmorphon16")
    return out


def read_ds_sigmorphon17_18(path, lang=None):
    out = defaultdict(list)
    for f in glob.glob(os.path.join(path, "*-train-high")):  # train-high incorporates medium and low
        lang = os.path.split(f)[-1].split("-")[-3]
        out[lang] = read_single(f, "unimorph")
    return out


def read_ds_sigmorphon19(path, lang=None):
    out = defaultdict(list)
    for fd in os.listdir(path):
        for f in glob.glob(os.path.join(path, fd, "*-train-*")):  # for low-resource, only train-low is given
            lang = os.path.split(f)[-1].split("-")[-3]
            out[lang] = read_single(f, "unimorph")
    return out

