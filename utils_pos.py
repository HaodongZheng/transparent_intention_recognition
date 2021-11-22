import string
from nltk.stem import WordNetLemmatizer

# Punctuation characters
punct = set(string.punctuation)

# Morphology rules used to assign unknown word tokens
noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
verb_suffix = ["ate", "ify", "ise", "ize"]
adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
adv_suffix = ["ward", "wards", "wise"]


def get_word_tag(line, vocab): 
    if not line.split():
        word = "--n--"
        tag = "--s--"
        return word.lower(), tag
    else:
        word, tag = line.split()
        if word.lower() not in vocab: 
            # Handle unknown words
            word = assign_unk(word.lower())
        return word, tag
    return None 


def preprocess(vocab, data_fp, isFileName=True):
    """
    Preprocess data
    """
    orig = []
    prep = []
    
    if (isFileName):
    # Read data
      with open(data_fp, "r") as data_file:

        for cnt, word in enumerate(data_file):

            # End of sentence
            if not word.split():
                orig.append(word.strip().lower())
                word = "--n--"
                prep.append(word.lower())
                continue

            # Handle unknown words
            elif word.strip().lower() not in vocab:
                orig.append(word.strip().lower())
                word = assign_unk(word.lower())
                prep.append(word.lower())
                continue

            else:
                orig.append(word.strip().lower())
                prep.append(word.strip().lower())
    
    else:
        for word in data_fp.split():

            # End of sentence
            if word.strip() in set(['.','!','?']):
                orig.append(word.strip().lower())
                word = "--n--"
                prep.append(word.lower())
                continue

            # Handle unknown words
            elif word.strip().lower()  not in vocab:
                orig.append(word.strip().lower())
                word = assign_unk(word.lower())
                prep.append(word.lower())
                continue

            else:
                orig.append(word.strip().lower())
                prep.append(word.strip().lower())
                
    if isFileName:
        assert(len(orig) == len(open(data_fp, "r").readlines()))
        assert(len(prep) == len(open(data_fp, "r").readlines()))

    return orig, prep


def assign_unk(tok):
    """
    Assign unknown word tokens
    """
    # Digits
    if any(char.isdigit() for char in tok):
        return "--unk_digit--"

    # Punctuation
    elif any(char in punct for char in tok):
        return "--unk_punct--"

    # Upper-case
    elif any(char.isupper() for char in tok):
        return "--unk_upper--"

    # Nouns
    elif any(tok.endswith(suffix) for suffix in noun_suffix):
        return "--unk_noun--"

    # Verbs
    elif any(tok.endswith(suffix) for suffix in verb_suffix):
        return "--unk_verb--"

    # Adjectives
    elif any(tok.endswith(suffix) for suffix in adj_suffix):
        return "--unk_adj--"

    # Adverbs
    elif any(tok.endswith(suffix) for suffix in adv_suffix):
        return "--unk_adv--"

    return "--unk--"
