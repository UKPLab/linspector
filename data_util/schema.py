from collections import defaultdict

from data_util.global_var import *


class UnimorphSchema:
    # schema in unimorph-schema.tsv automatically extracted from https://unimorph.github.io/doc/unimorph-schema.pdf
    # using tabula https://github.com/tabulapdf/tabula
    def __init__(self, src=UNIMORPH_SCHEMA_SRC, ignore_feats=("Deixis")):
        code_ix = defaultdict()
        feature_value = defaultdict(list)
        for line in open(src):
            feat, val, code = line.strip().split("\t")
            if feat not in ignore_feats:
                code = code.upper()
                assert code not in code_ix, f"Duplicate code {code}"  # PROX can be Case or Deixis
                code_ix[code] = (feat, val)
                feature_value[feat] += [(val, code)]

        self.code_ix = code_ix
        self.feature_value = feature_value

    # interpret single code
    def get_feature(self, code):
        code = code.upper().replace("{", "").replace("}", "").replace("/", "+").split(
            "+")  # sometimes several values are given
        out = []
        for c in code:
            assert c in self.code_ix, f"Unknown code {c}, {code}"  # just to be sure
            out += [self.code_ix[c]]

        return out[0][0], "+".join(sorted([x[1] for x in out]))

    # decode full Unimorph MSD
    def decode_msd(self, msd):
        msd = msd.split(";")
        feature_values = []
        residue = []  # whatever we couldn't decypher
        for code in msd:
            try:
                feature_values += [self.get_feature(code)]
            except AssertionError:
                residue += [code]
        out = {}
        for feature, value in feature_values:
            # TODO: actually there are quite a few such errors
            # assert feature not in out, f"Duplicate value for {feature}, {msd}"
            out[feature] = value
        return out, residue
