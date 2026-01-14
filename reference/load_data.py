import cyrtranslit


def load_corpus_tokens(filepath_list, model_name="BERTic", conllup=False):
    dataset = []
    wordlist = set()
    s = 0
    for filepath in filepath_list:
        txt = open(filepath, encoding="utf-8").read()
        for sentence in txt.split("\n\n")[:-1]:
            sentence_id = ""
            sentence_list = []
            for line in sentence.split("\n"):
                if not line.startswith("#"):
                    token = line.split("\t")[1]
                    if not conllup:
                        tag = line.split("\t")[3]
                    else:
                        tag = line.split("\t")[4]
                    lemma = line.split("\t")[2]
                    if "SrBERTa" in model_name:
                        token = cyrtranslit.to_cyrillic(token, "sr")
                        lemma = cyrtranslit.to_cyrillic(lemma, "sr")

                    sentence_list.append([s, token, tag, lemma, sentence_id])
                    wordlist.add(token)
                    wordlist.add(token.upper())
                    wordlist.add(token.lower())
                    wordlist.add(token.capitalize())
                elif (
                    line.startswith("#") and ("sent id" in line) or ("sent_id" in line)
                ):
                    sentence_id = line.split("=")[1].strip()
            if len(sentence_list) > 0:
                dataset.extend(sentence_list)
            s += 1

    return dataset, wordlist
