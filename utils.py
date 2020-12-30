def load_vocabulary(path):

    print(" load start")

    w2i = {}
    i2w = {}
    index = 0

    with open(path,"r",encoding="utf-8") as f:
        for line in f.readlines():
            word = line[:-1]
            i2w[index] = word
            index += 1

    print("vocab from: {}, containing words: {}".format(path, len(i2w)))

    return i2w


# {('ORG', '奥地利队')}
def extract_kvpairs_in_bio(bio_seq, word_seq):
    assert len(bio_seq) == len(word_seq)
    pairs = set()
    pre_bio = "O"
    v = ""
    for i, bio in enumerate(bio_seq):
        if (bio == "O"):
            if v != "": pairs.add((pre_bio[2:], v))
            v = ""
        elif (bio[0] == "B"):
            if v != "": pairs.add((pre_bio[2:], v))
            v = word_seq[i]
        elif (bio[0] == "I"):
            if (pre_bio[0] == "O") or (pre_bio[2:] != bio[2:]):
                if v != "": pairs.add((pre_bio[2:], v))
                v = ""
            else:
                v += word_seq[i]
        pre_bio = bio
    if v != "": pairs.add((pre_bio[2:], v))
    return pairs

############################
####### cal_f1_score #######
############################

def cal_f1_score(preds, golds):
    assert len(preds) == len(golds)
    p_sum = 0
    r_sum = 0
    hits = 0
    for pred, gold in zip(preds, golds):
        p_sum += len(pred)
        r_sum += len(gold)
        for label in pred:
            if label in gold:
                hits += 1
    p = hits / p_sum if p_sum > 0 else 0
    r = hits / r_sum if r_sum > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1