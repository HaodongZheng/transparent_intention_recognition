from sklearn import metrics
from nltk.corpus import stopwords
import time as timer
import csv
from collections import defaultdict
from nltk.corpus.reader.util import find_corpus_fileids
import numpy as np
import math
from operator import itemgetter
from nltk.stem import WordNetLemmatizer
from utils_pos import get_word_tag, preprocess
import pandas as pd
from collections import defaultdict
import math
import numpy as np
import matplotlib.pyplot as plt
import nltk
from utils_pos import get_word_tag, preprocess
import pickle
from nltk.corpus import wordnet as wn

def write_dict_to_csvfile(file_name, dictionary):
    keys = dictionary.keys()
    with open(file_name, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows([dictionary])

def read_dict_from_csvfile(file_name, type_of_value):
    a_csv_file = open(file_name, "r")
    dict_reader = csv.DictReader(a_csv_file)
    ordered_dict_from_csv = list(dict_reader)[0]
    dict_from_csv = dict(ordered_dict_from_csv)
    if type_of_value == "float":
        for key, value in dict_from_csv.items():
            dict_from_csv[key] = float(value)
        return dict_from_csv
    else:
        if type_of_value == "int":
            for key, value in dict_from_csv.items():
                dict_from_csv[key] = int(value)
            return dict_from_csv
    return dict_from_csv


# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: create_dictionaries
def create_dictionaries(training_corpus, vocab):
    """
    Input:
        training_corpus: a corpus where each line has a word followed by its tag.
        vocab: a dictionary where keys are words in vocabulary and value is an index
    Output:
        emission_counts: a dictionary where the keys are (tag, word) and the values are the counts
        transition_counts: a dictionary where the keys are (prev_tag, tag) and the values are the counts
        tag_counts: a dictionary where the keys are the tags and the values are the counts
    """

    # initialize the dictionaries using defaultdict
    emission_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    tag_counts = defaultdict(int)
    word_counts = defaultdict(int)
    df_counts = defaultdict(int)
    document_count = 0
    # Initialize "prev_tag" (previous tag) with the start state, denoted by '--s--'
    prev_tag = "--s--"

    # use 'i' to track the line number in the corpus
    i = 0

    # Each item in the training corpus contains a word and its POS tag
    # Go through each word and its tag in the training corpus
    tmp_existing_words_list = []

    for word_tag in training_corpus:

        # Increment the word_tag count
        i += 1
        # Increment the document frequency
        if word_tag == "\n":
            document_count += 1
            for existing_word in set(tmp_existing_words_list):
                df_counts[existing_word] += 1

            tmp_existing_words_list = []
        # Every 50,000 words, print the word count
        if i % 50000 == 0:
            print(f"word count = {i}")

        ### START CODE HERE (Replace instances of 'None' with your code) ###
        # get the word and tag using the get_word_tag helper function (imported from utils_pos.py)
        word, tag = get_word_tag(word_tag, vocab)
        word = word.lower()

        tmp_existing_words_list.append(word)
        # Increment the transition count for the previous word and tag
        transition_counts[(prev_tag, tag)] += 1

        # Increment the emission count for the tag and word
        emission_counts[(tag, word)] += 1

        # Increment the tag count
        tag_counts[tag] += 1

        # Increment the word count
        word_counts[word] += 1

        # Set the previous tag to this tag (for the next iteration of the loop)
        prev_tag = tag

    idf_dict = defaultdict(float)
    for key, value in df_counts.items():
        idf_dict[key] = math.log(document_count / df_counts[key])
    sum_idf = sum(idf_dict.values())
    normalized_idf_dict = defaultdict(float)
    for key in idf_dict.keys():
        normalized_idf_dict[key] = idf_dict[key] / sum_idf

    return (
        emission_counts,
        transition_counts,
        tag_counts,
        word_counts,
        normalized_idf_dict,
    )


def create_transition_matrix(alpha, tag_counts, transition_counts):
    """
    Input:
        alpha: number used for smoothing
        tag_counts: a dictionary mapping each tag to its respective count
        transition_counts: transition count for the previous word and tag
    Output:
        A: matrix of dimension (num_tags,num_tags)
    """
    # Get a sorted list of unique POS tags
    all_tags = sorted(tag_counts.keys())

    # Count the number of unique POS tags
    num_tags = len(all_tags)

    # Initialize the transition matrix 'A'
    A = np.zeros((num_tags, num_tags))

    # Get the unique transition tuples (previous POS, current POS)
    trans_keys = set(transition_counts.keys())

    ### START CODE HERE (Replace instances of 'None' with your code) ###

    # Go through each row of the transition matrix A
    for i in range(num_tags):

        # Go through each column of the transition matrix A
        for j in range(num_tags):

            # Initialize the count of the (prev POS, current POS) to zero
            count = 0

            # Define the tuple (prev POS, current POS)
            # Get the tag at position i and tag at position j (from the all_tags list)
            key = tuple([all_tags[i], all_tags[j]])

            # Check if the (prev POS, current POS) tuple
            # exists in the transition counts dictionary
            if key in trans_keys:  # complete this line

                # Get count from the transition_counts dictionary
                # for the (prev POS, current POS) tuple
                count = transition_counts[key]

            # Get the count of the previous tag (index position i) from tag_counts
            count_prev_tag = tag_counts[all_tags[i]]

            # Apply smoothing using count of the tuple, alpha,
            # count of previous tag, alpha, and total number of tags
            A[i, j] = (count + alpha) / (count_prev_tag + alpha * num_tags)

    ### END CODE HERE ###

    return A




def create_emission_matrix(alpha, tag_counts, emission_counts, vocab):
    """
    Input:
        alpha: tuning parameter used in smoothing
        tag_counts: a dictionary mapping each tag to its respective count
        emission_counts: a dictionary where the keys are (tag, word) and the values are the counts
        vocab: a dictionary where keys are words in vocabulary and value is an index.
               within the function it'll be treated as a list
    Output:
        B: a matrix of dimension (num_tags, len(vocab))
    """

    # get the number of POS tag
    num_tags = len(tag_counts)

    # Get a list of all POS tags
    all_tags = sorted(tag_counts.keys())

    # Get the total number of unique words in the vocabulary
    num_words = len(vocab)

    # Initialize the emission matrix B with places for
    # tags in the rows and words in the columns
    B = np.zeros((num_tags, num_words))

    # Get a set of all (POS, word) tuples
    # from the keys of the emission_counts dictionary
    emis_keys = set(list(emission_counts.keys()))

    ### START CODE HERE (Replace instances of 'None' with your code) ###

    # Go through each row (POS tags)
    for i in range(num_tags):  # complete this line

        # Go through each column (words)
        for j in range(num_words):  # complete this line

            # Initialize the emission count for the (POS tag, word) to zero
            count = 0

            # Define the (POS tag, word) tuple for this row and column
            key = tuple([all_tags[i], vocab[j]])

            # check if the (POS tag, word) tuple exists as a key in emission counts
            if key in emission_counts.keys():  # complete this line

                # Get the count of (POS tag, word) from the emission_counts d
                count = emission_counts[key]

            # Get the count of the POS tag
            count_tag = tag_counts[all_tags[i]]

            # Apply smoothing and store the smoothed value
            # into the emission matrix B for this row and column
            B[i, j] = (count + alpha) / (count_tag + alpha * num_words)

    ### END CODE HERE ###
    return B



def initialize(states, A, B, corpus, vocab):
    """
    Input:
        states: a list of all possible parts-of-speech
        tag_counts: a dictionary mapping each tag to its respective count
        A: Transition Matrix of dimension (num_tags, num_tags)
        B: Emission Matrix of dimension (num_tags, len(vocab))
        corpus: a sequence of words whose POS is to be identified in a list
        vocab: a dictionary where keys are words in vocabulary and value is an index
    Output:
        best_probs: matrix of dimension (num_tags, len(corpus)) of floats
        best_paths: matrix of dimension (num_tags, len(corpus)) of integers
    """
    # Get the total number of unique POS tags
    num_tags = len(states)

    # Initialize best_probs matrix
    # POS tags in the rows, number of words in the corpus as the columns
    best_probs = np.zeros((num_tags, len(corpus)))

    # Initialize best_paths matrix
    # POS tags in the rows, number of words in the corpus as columns
    best_paths = np.zeros((num_tags, len(corpus)), dtype=int)

    # Define the start token
    s_idx = states.index("--s--")
    ### START CODE HERE (Replace instances of 'None' with your code) ###

    # Go through each of the POS tags
    for i in range(num_tags):  # complete this line

        # Handle the special case when the transition from start token to POS tag i is zero
        if A[s_idx, i] == 0:  # complete this line

            # Initialize best_probs at POS tag 'i', column 0, to negative infinity
            best_probs[i, 0] = float("-inf")

        # For all other cases when transition from start token to POS tag i is non-zero:
        else:

            # Initialize best_probs at POS tag 'i', column 0
            # Check the formula in the instructions above
            best_probs[i, 0] = math.log(
                A[s_idx, i]) + math.log(B[i, vocab[corpus[0]]])

    ### END CODE HERE ###
    return best_probs, best_paths




def viterbi_forward(A, B, test_corpus, best_probs, best_paths, vocab):
    """
    Input:
        A, B: The transition and emission matrices respectively
        test_corpus: a list containing a preprocessed corpus
        best_probs: an initilized matrix of dimension (num_tags, len(corpus))
        best_paths: an initilized matrix of dimension (num_tags, len(corpus))
        vocab: a dictionary where keys are words in vocabulary and value is an index
    Output:
        best_probs: a completed matrix of dimension (num_tags, len(corpus))
        best_paths: a completed matrix of dimension (num_tags, len(corpus))
    """
    # Get the number of unique POS tags (which is the num of rows in best_probs)
    num_tags = best_probs.shape[0]

    # Go through every word in the corpus starting from word 1
    # Recall that word 0 was initialized in `initialize()`
    for i in range(1, len(test_corpus)):

        # Print number of words processed, every 5000 words
        if i % 5000 == 0:
            print("Words processed: {:>8}".format(i))

        ### START CODE HERE (Replace instances of 'None' with your code EXCEPT the first 'best_path_i = None') ###
        # For each unique POS tag that the current word can be
        for j in range(num_tags):  # complete this line

            # Initialize best_prob for word i to negative infinity
            best_prob_i = float("-inf")

            # Initialize best_path for current word i to None
            best_path_i = None

            # For each POS tag that the previous word can be:
            for k in range(num_tags):  # complete this line

                # Calculate the probability =
                # best probs of POS tag k, previous word i-1 +
                # log(prob of transition from POS k to POS j) +
                # log(prob that emission of POS j is word i)
                prob = (
                    best_probs[k, i - 1]
                    + math.log(A[k, j])
                    + math.log(B[j, vocab[test_corpus[i]]])
                )

                # check if this path's probability is greater than
                # the best probability up to and before this point
                if prob > best_prob_i:  # complete this line

                    # Keep track of the best probability
                    best_prob_i = prob

                    # keep track of the POS tag of the previous word
                    # that is part of the best path.
                    # Save the index (integer) associated with
                    # that previous word's POS tag
                    best_path_i = k

            # Save the best probability for the
            # given current word's POS tag
            # and the position of the current word inside the corpus
            best_probs[j, i] = best_prob_i

            # Save the unique integer ID of the previous POS tag
            # into best_paths matrix, for the POS tag of the current word
            # and the position of the current word inside the corpus.
            best_paths[j, i] = best_path_i

        ### END CODE HERE ###
    return best_probs, best_paths




def viterbi_backward(best_probs, best_paths, corpus, states):
    """
    This function returns the best path.

    """
    # Get the number of words in the corpus
    # which is also the number of columns in best_probs, best_paths
    m = best_paths.shape[1]
    # Initialize array z, same length as the corpus
    z = [None] * m
    # Get the number of unique POS tags
    num_tags = best_probs.shape[0]

    # Initialize the best probability for the last word
    best_prob_for_last_word = float("-inf")

    # Initialize pred array, same length as corpus
    pred = [None] * m

    ### START CODE HERE (Replace instances of 'None' with your code) ###
    ## Step 1 ##

    # Go through each POS tag for the last word (last column of best_probs)
    # in order to find the row (POS tag integer ID)
    # with highest probability for the last word
    for k in range(num_tags):  # complete this line

        # If the probability of POS tag at row k
        # is better than the previously best probability for the last word:
        if best_probs[k, m - 1] > best_prob_for_last_word:  # complete this line

            # Store the new best probability for the last word
            best_prob_for_last_word = best_probs[k, m - 1]

            # Store the unique integer ID of the POS tag
            # which is also the row number in best_probs
            z[m - 1] = k
    # Convert the last word's predicted POS tag
    # from its unique integer ID into the string representation
    # using the 'states' list
    # store this in the 'pred' array for the last word
    pred[m - 1] = states[z[m - 1]]

    ## Step 2 ##
    # Find the best POS tags by walking backward through the best_paths
    # From the last word in the corpus to the 0th word in the corpus
    for i in range(m - 1, 0, -1):  # complete this line

        # Retrieve the unique integer ID of
        # the POS tag for the word at position 'i' in the corpus
        best_tag_idx = best_paths[z[i], i]  # m-2 tag idx col
        pos_tag_for_word_i = states[best_tag_idx]  # m-2 tag

        # In best_paths, go to the row representing the POS tag of word i
        # and the column representing the word's position in the corpus
        # to retrieve the predicted POS for the word at position i-1 in the corpus
        z[i - 1] = best_tag_idx  # z[m-2] is the idx of best prob for word m-1

        # Get the previous word's POS tag in string form
        # Use the 'states' list,
        # where the key is the unique integer ID of the POS tag,
        # and the value is the string representation of that POS tag
        pred[i - 1] = pos_tag_for_word_i

    ### END CODE HERE ###
    return pred



def compute_accuracy(pred, y):
    """
    Input:
        pred: a list of the predicted parts-of-speech
        y: a list of lines where each word is separated by a '\t' (i.e. word \t tag)
    Output:

    """
    num_correct = 0
    total = 0

    # Zip together the prediction and the labels
    for prediction, y in zip(pred, y):
        ### START CODE HERE (Replace instances of 'None' with your code) ###
        # Split the label into the word and the POS tag
        word_tag_tuple = tuple(y.split())

        # Check that there is actually a word and a tag
        # no more and no less than 2 items
        if len(word_tag_tuple) < 2 or len(word_tag_tuple) > 2:  # complete this line
            continue

        # store the word and tag separately
        word, tag = word_tag_tuple[0], word_tag_tuple[1]

        # Check if the POS tag label matches the prediction
        if prediction == tag:  # complete this line

            # count the number of times that the prediction
            # and label match
            num_correct += 1

        # keep track of the total number of examples (that have valid labels)
        total += 1

        ### END CODE HERE ###
    return num_correct / total



def get_pos_for_sentence(states, A, B, sentence, vocab):
    # sentence here is a list.
    best_probs, best_paths = initialize(states, A, B, sentence, vocab)
    best_probs, best_paths = viterbi_forward(
        A, B, sentence, best_probs, best_paths, vocab
    )
    sentence_pos = viterbi_backward(best_probs, best_paths, sentence, states)
    return sentence_pos



def share_common_word(input_word_list, semantic_pattern_word_list):
    if list(set(semantic_pattern_word_list) & set(input_word_list)) == []:
        if (
            "[Number]" in semantic_pattern_word_list
            or "isnumeric" in semantic_pattern_word_list
        ):
            for word in input_word_list:
                if word.isnumeric():
                    return True
                else:
                    return False
    else:
        return True



def within_length_diff_tolerance(
    input_word_list, semantic_pattern_word_list, tolerance
):
    if abs(len(input_word_list) - len(semantic_pattern_word_list)) <= tolerance:
        return True
    else:
        return False


# Label 1
def get_matching_score_general_idf(
    user_input,
    pattern,
    vocab_document,
    word_weighting_dict,
    source_pos,
    pos_dict,
    verboseFlag=False,
    insert_discount_factor=0.9,
    maximum_length_difference=4,
    not_in_doc_factor=0.5,
    relative_matching_threshold=0.3,
):
    # min_edit_distance_for_sentence
    source = user_input.split()
    target = pattern.split()

    # print(source_lower, target_lower, share_common_word(source_lower, target_lower))
    if (
        share_common_word(source, target)
        and within_length_diff_tolerance(source, target, maximum_length_difference)
    ) or (len(source) == 1 and len(target) == 1 and "[" not in target[0]):
        _, shift_penalty, cost = min_edit_distance_for_tags_without_wordnet_general_idf(
            source,
            target,
            source_pos,
            pos_dict,
            vocab_document,
            word_weighting_dict,
            verboseFlag,
            insert_discount_factor,
            not_in_doc_factor,
            relative_matching_threshold,
        )
    else:
        # set cost to high number:
        cost = 100
        sum_of_weights = sum([word_weighting_dict[x] if x in vocab_document else  not_in_doc_factor   for x in source])
        shift_penalty = relative_matching_threshold * sum_of_weights
        # print(within_length_diff_tolerance(source_lower, target_lower, 5))

    return shift_penalty, cost


def min_edit_distance_for_tags_without_wordnet_general_idf(
    source,
    target,
    source_pos,
    pos_dict,
    vocab_document,
    word_weighting_dict,
    verbose=False,
    insert_discount_factor=1.0,
    not_in_doc_factor=0.5,
    relative_matching_threshold=0.3,
):
    """
    features:
    1. removal costs for words not in the document vocabulary is 0.
    2. insert cost for words in the vocabulary is calculated as a softmax function on all target words.

    """
    m = len(source)
    n = len(target)
    D = np.zeros((m + 1, n + 1), dtype=float)

    for row in range(1, m + 1):
        source_word = source[row - 1]
        if source_word not in vocab_document:
            # I want the words in the doc_vocab have different weighting than those aren't if necessary.
            D[row, 0] = D[row - 1, 0] + not_in_doc_factor
        else:
            D[row, 0] = D[row - 1, 0] + word_weighting_dict[source_word]

    # denominator for the normalization.
    sum_of_weights = D[row, 0]

    shift_penalty = relative_matching_threshold * sum_of_weights

    idx = []
    cols = []
    word_weights = []

    # Fill in row 0, for all columns from 1 to n, both inclusive
    for col in range(1, n + 1):
        target_word = target[col - 1]
        D[0, col] = D[0, col - 1] + word_weighting_dict[target_word]

    # Loop through row 1 to row m, both inclusive
    for row in range(1, m + 1):
        source_word = source[row - 1]
        if source_pos != []:
            source_word_pos = source_pos[row - 1]
        else:
            source_word_pos = []
        # Loop through column 1 to column n, both inclusive
        for col in range(1, n + 1):
            target_word = target[col - 1]
            target_word_pos = pos_dict[target_word]
            # Update the cost at row, col based on previous entries in the cost matrix
            # Refer to the equation calculate for D[i,j] (the minimum of three calculated costs)
            # word_weight = word_counts[source[row - 1]]/(total_word_counts/num_words)
            if source_word not in vocab_document:
                delete_word_weight = (
                    not_in_doc_factor * word_weighting_dict[source_word]
                )

            else:
                delete_word_weight = word_weighting_dict[source_word]

            insert_word_weight = word_weighting_dict[target_word]
            del_cost = delete_word_weight
            ins_cost = insert_discount_factor * insert_word_weight

            # rep_cost can be replaced by (1 - similarity score) value between 0 and 1.
            if source_word_pos in target_word_pos and source_pos != []:
                similarity_score = wordnet_similarity_v2(
                            source_word, target_word, [
                                source_word_pos], target_word_pos
                        )
                if similarity_score < 0.8: # or source_word in vocab_document:
                    similarity_score = 0
                if source_word not in vocab_document:
                    rep_cost = (
                        word_weighting_dict[target_word] + not_in_doc_factor
                    ) * (
                        1
                        - similarity_score
                    )
                else:
                    rep_cost = (
                        word_weighting_dict[target_word]
                        + word_weighting_dict[source_word]
                    ) * (
                        1
                        - similarity_score
                    )
            else:
                rep_cost = del_cost + ins_cost

            # Check to see if source character at the previous row
            # matches the target character at the previous column,
            if source_word == target_word:
                rep_cost = 0
            if target_word == "[Number]" or target_word == "isnumeric":
                if source_word.isnumeric():
                    rep_cost = 0

            compare_array = np.array(
                [
                    D[row - 1, col] + del_cost,
                    D[row, col - 1] + ins_cost,
                    D[row - 1, col - 1] + rep_cost,
                ]
            )

            D[row, col] = min(compare_array)

        idx.append(source_word)
        #  print(source)
        #  print(target, "end")
        word_weights.append(del_cost)
    # Set the minimum edit distance with the cost found at row m, column n
    med = D[m, n]
    # relative_med = D[m, n] / sum_of_weights
    # D = D / sum_of_weights
    cols.append("weighting factor")
    if verbose:
        df = pd.DataFrame(np.array(word_weights).reshape(
            m, 1), index=idx, columns=cols)
        print(df)
        Df = pd.DataFrame(D, index=["empty"] + source,
                          columns=["empty"] + target)
        print(Df)
    # shift_penalty = shift_penalty * 0.9999

    ### END CODE HERE ###
    return D, shift_penalty, med


class Semantic_grammar:
    def __init__(self, intent, semantic_grammar_segment_list, grammar_segment_dict):
        self.intent = intent
        #     self.patterns,self.pattern_backtraces = self.get_all_patterns_naive(semantic_grammar_segment_list)
        self.patterns = []
        self.backtraces = []
        self.taglist = []
        self.patterns = semantic_grammar_segment_list
        for pattern in self.patterns:
            self.backtraces.append({})
        self.complete_flag = False
        times = 0
        while not self.complete_flag:
            (
                self.patterns,
                self.backtraces,
                self.complete_flag,
            ) = self.recursive_grammar(grammar_segment_dict)
            times = times + 1
        self.num_patterns = len(self.patterns)
        self.patterns_with_backtraces = []
        for j in range(0, self.num_patterns):
            self.patterns_with_backtraces.append(
                Pattern_BackTrace_Group(self.patterns[j], self.backtraces[j])
            )
        #     self.grammar = semantic_grammar_segment_list

        self.print_intent()
        self.print_all_patterns()

    def recursive_grammar(self, grammar_segment_dict):
        # recursively applying grammar.

        complete_flag = True
        patterns = []
        backtraces = []
        index = 0
        for pattern in self.patterns:
            tmp_list_1 = []
            tmp_back_trace_list_1 = []
            word_list = pattern.split()
            back_trace = self.backtraces[index].copy()
            index = index + 1
            flag = True
            if intersectionIsEmpty(word_list, grammar_segment_dict.keys()):
                for word in word_list:
                    if "/" in word:
                        complete_flag = False
                        flag = False
                if flag:  # if no non terminal nor combinatorial
                    tmp_back_trace_list_1.append(back_trace)
                    patterns.append(pattern)
                    backtraces = backtraces + tmp_back_trace_list_1
                    continue

            word_list_len = len(word_list)
            word = word_list[0]

            if (
                word not in grammar_segment_dict.keys() and "/" not in word
            ):  # not a non terminal or a combinatorial
                tmp_list_1.append(word)
                tmp_back_trace_list_1.append(back_trace.copy())
            #  print(" static dictionary list 1: " + str(tmp_back_trace_list_1))
            else:
                if "/" not in word:  # a non terminal
                    self.taglist.append(word)
                    tmp_list_1 = grammar_segment_dict[word]  # a list of words
                    for branch in grammar_segment_dict[
                        word
                    ]:  # for each pattern in the grammar segment
                        tmp_back_trace = (
                            back_trace.copy()
                        )  # a dictionary containing back trace information: must use copy()
                        # add pattern to the dictionary
                        tmp_back_trace[word] = branch
                        #      print("add " + str(word) + ":" + str(branch))
                        tmp_back_trace_list_1.append(tmp_back_trace)
                    if word not in back_trace.keys():
                        complete_flag = False

                else:  # a combinatorial
                    tmp_list_1 = word.split("/")
                    print(tmp_list_1)
                    for k in range(0, len(tmp_list_1)):
                        tmp_list_1[k] = tmp_list_1[k].replace("-", " ")
                        tmp_back_trace = back_trace.copy()
                        tmp_back_trace[tmp_list_1[k]] = tmp_list_1[k]
                        tmp_back_trace_list_1.append(tmp_back_trace)
                        complete_flag = False

            for i in range(1, word_list_len):
                word = word_list[i]
                tmp_list_2 = []
                tmp_back_trace_list_2 = []
                if word not in grammar_segment_dict.keys() and "/" not in word:
                    tmp_list_2.append(word)
                    tmp_back_trace_list_2 = tmp_back_trace_list_1.copy()
                    # print(" static dictionary list 2: " + str(tmp_back_trace_list_2))
                else:
                    if "/" not in word:
                        self.taglist.append(word)
                        tmp_list_2 = grammar_segment_dict[word]
                        for trace in tmp_back_trace_list_1:
                            for branch in grammar_segment_dict[word]:
                                tmp_back_trace = trace.copy()
                                tmp_back_trace[word] = branch
                                #           print("add " + str(word) + ":" + str(branch))
                                tmp_back_trace_list_2.append(tmp_back_trace)
                        # infinite loop grammar won't flip the complete flag, the grammar will only iterate for
                        # finite tags.
                        if word not in back_trace.keys():
                            complete_flag = False

                    else:
                        tmp_list_2 = word.split("/")
                        for trace in tmp_back_trace_list_1:
                            for j in range(0, len(tmp_list_2)):
                                tmp_list_2[j] = tmp_list_2[j].replace("-", " ")
                                tmp_back_trace = trace.copy()
                                tmp_back_trace[tmp_list_2[j]] = tmp_list_2[j]
                                tmp_back_trace_list_2.append(tmp_back_trace)
                        complete_flag = False
                # print("1: " + str(tmp_back_trace_list_1))
                # print("2: " + str(tmp_back_trace_list_2))
                tmp_list_1 = combine_list(tmp_list_1, tmp_list_2, " ")
                tmp_back_trace_list_1 = tmp_back_trace_list_2.copy()
            patterns = patterns + tmp_list_1
            backtraces = backtraces + tmp_back_trace_list_1

        return patterns, backtraces, complete_flag

    def print_grammar(self):
        print("Grammar: {0}".format(self.grammar))

    def print_intent(self):
        print("Intent:{0} \n".format(self.intent))

    def print_all_patterns(self):
        i = 0
        for pattern in self.patterns_with_backtraces:
            i = i + 1
            print(
                "Pattern {0}: {1} BackTrace: {2}".format(
                    i, pattern.string, pattern.back_trace_dict
                )
            )
        # print("Pattern {0}: {1}".format(i,pattern))


def combine_list(list_1, list_2, insert_symbol):
    new_list = []
    if list_1 == [] or list_2 == []:
        return list_1 + list_2
    for element_1 in list_1:
        for element_2 in list_2:
            new_list.append(element_1 + insert_symbol + element_2)
    return new_list


def intersectionIsEmpty(lst1, lst2):
    if len(list(set(lst1) & set(lst2))) == 0:
        return True
    else:
        return False


def get_weighted_overlap_sum(list1, list2, weighting_dict):
    overlap_list = list(set(list1) & set(list2))
    weighted_sum = 0
    for element in overlap_list:
        weighted_sum = weighted_sum + weighting_dict[element]
    return weighted_sum


class Pattern_BackTrace_Group:
    def __init__(self, patternString, back_trace_dict):
        self.string = str(patternString)
        self.back_trace_dict = back_trace_dict

    def print_string(self):
        print(self.string)



def get_idf_from_grammar_dict(
    grammar_segment_dict, intent_grammar_dict, smoothing_value
):
    vocab = []
    tf_dict = defaultdict(int)
    df_dict = defaultdict(int)
    idf_dict = defaultdict(float)
    grammar_tag_list = []
    for grammar_tag, grammar_patterns in intent_grammar_dict.items():
        semantic_grammar = Semantic_grammar(
            grammar_tag, grammar_segment_dict[grammar_tag], grammar_segment_dict
        )
        pattern_list = semantic_grammar.patterns
        if grammar_tag not in grammar_tag_list:
            grammar_tag_list.append(grammar_tag)
        #      for segment in semantic_grammar.semantic_grammar_segment_list:
        for pattern in pattern_list:
            for word in pattern.split():
                if word.lower() not in vocab:
                    vocab.append(word.lower())
                tf_dict[(word.lower(), grammar_tag)] = (
                    tf_dict[(word.lower(), grammar_tag)] + 1
                )

    for word in vocab:
        for grammar_tag in grammar_tag_list:
            if tf_dict[(word, grammar_tag)] != 0:
                df_dict[word] += 1
        idf_dict[word] = math.log(len(grammar_tag_list) / df_dict[word])
    # deal with the semantic tags by adding its own idf and the sum of its descendence's idf

    idf_sum = sum(idf_dict.values()) + len(vocab) * smoothing_value
    for word in vocab:
        idf_dict[word] = (idf_dict[word] + smoothing_value) / idf_sum

    return idf_dict, vocab


def combine_list(list_1, list_2, insert_symbol):
    new_list = []
    if list_1 == [] or list_2 == []:
        return list_1 + list_2
    for element_1 in list_1:
        for element_2 in list_2:
            new_list.append(element_1 + insert_symbol + element_2)
    return new_list


def combine_list_for_sum_score(list_1, list_2):
    new_list = []
    if list_1 == [] or list_2 == []:
        return list_1 + list_2
    for element_1 in list_1:
        for element_2 in list_2:
            new_list.append(element_1 + element_2)
    return new_list


def combine_list_of_dictionary(list_1, list_2):
    new_list = []
    if list_1 == [] or list_2 == []:
        return list_1 + list_2
    for dict_1 in list_1:
        for dict_2 in list_2:
            new_dict = dict_2.copy()
            new_dict.update(dict_1.copy())
            new_list.append(new_dict)
    return new_list


def penn_pos_to_wordnet_pos(pos):
    wordnet_pos = []
    for tag in pos:
        if "VB" in tag:
            wordnet_pos.append(wn.VERB)
        elif "NN" in tag:
            wordnet_pos.append(wn.NOUN)
        elif "JJ" in tag:
            wordnet_pos.append(wn.ADJ)
        elif "RB" in tag:
            wordnet_pos.append(wn.ADV)
        else:
            wordnet_pos.append(tag)
    return wordnet_pos


def get_synonym(word, pos):
    synsets = wn.synsets(word, pos=pos)
    syn_lemmas = []
    for synset in synsets:
        for lemma in synset.lemmas():
            syn_lemmas.append(str(lemma.name()))
    syn_lemmas = set(syn_lemmas)
    return syn_lemmas


def wordnet_similarity(word_1, word_2, pos):
    # For code optimization, lemmatizer can be done for all stored patterns in advance.
    lemmatizer = WordNetLemmatizer()
    t = timer.time()
    word_1_stem = lemmatizer.lemmatize(word_1, pos=pos)
    word_2_stem = lemmatizer.lemmatize(word_2, pos=pos)

    synsets_1 = wn.synsets(word_1_stem, pos=pos)
    synsets_2 = wn.synsets(word_2_stem, pos=pos)
    # print(word_1_stem)
    # print(word_2_stem)
    similarity_score_list = []
    if pos == wn.NOUN or pos == wn.VERB:
        for synset_1 in synsets_1:
            for synset_2 in synsets_2:
                similarity_score_list.append(
                    0.0 * synset_1.path_similarity(synset_2)
                    + 1.0 * synset_1.wup_similarity(synset_2)
                )
    else:
        similar_word_list = []
        for synset_1 in synsets_1:
            similar_synsets = synset_1.similar_tos()
            for synset in similar_synsets:
                for lemma in synset.lemmas():
                    similar_word_list.append(str(lemma.name()))
        synonyms_list = get_synonym(word_1, pos)
        if word_2 in synonyms_list:
            similarity_score_list.append(1)
        else:
            if word_2 in similar_word_list:
                similarity_score_list.append(0.8)
            else:
                similarity_score_list.append(0)
    if len(similarity_score_list) == 0:
        similarity_score_list.append(0)
    # print("total time cost: ",time.time()-t)
    return max(similarity_score_list)


def wordnet_similarity_v2(word_1, word_2, pos_1_list, pos_2_list):
    synsets_1 = []
    synsets_2 = []
    shared_pos = list(set(pos_1_list) & set(pos_2_list))[0]
    for pos_1 in pos_1_list:
        synsets = wn.synsets(word_1, pos=shared_pos)
        synsets_1 = synsets_1 + synsets
    for pos_2 in pos_2_list:
        synsets = wn.synsets(word_2, pos=shared_pos)
        synsets_2 = synsets_2 + synsets

    similarity_score_list = []
    if shared_pos == wn.NOUN or shared_pos == wn.VERB:
        for synset_1 in synsets_1:
            for synset_2 in synsets_2:
                similarity_score_list.append(
                    0.0 * synset_1.path_similarity(synset_2)
                    + 1.0 * synset_1.wup_similarity(synset_2)
                )
    elif shared_pos == wn.ADJ or shared_pos == wn.ADV:
        similar_word_list = []
        for synset_1 in synsets_1:
            similar_synsets = synset_1.similar_tos()
            for synset in similar_synsets:
                for lemma in synset.lemmas():
                    similar_word_list.append(str(lemma.name()))
        synonyms_list = get_synonym(word_1, shared_pos)
        if word_2 in synonyms_list:
            similarity_score_list.append(1)
        else:
            if word_2 in similar_word_list:
                similarity_score_list.append(0.8)
            else:
                similarity_score_list.append(0)
    if len(similarity_score_list) == 0:
        similarity_score_list.append(0)
    # print("total time cost: ",time.time()-t)
    return max(similarity_score_list)


def wordnet_similarity_WSD(word_1, word_2, source, target, pos, weighting_dict):
    # For code optimization, lemmatizer can be done for all stored patterns in advance.
    lemmatizer = WordNetLemmatizer()
    word_1_stem = lemmatizer.lemmatize(word_1, pos=pos)
    word_2_stem = lemmatizer.lemmatize(word_2, pos=pos)
    # print("lemmatize time cost: ", time.time() - t)

    context_word_list_1 = source.replace(word_1, "").split()
    context_word_list_2 = target.replace(word_1, "").split()
    synset_1 = WSD_lesk_algorithm(
        context_word_list_1, word_1_stem, pos, weighting_dict)

    synset_2 = WSD_lesk_algorithm(
        context_word_list_2, word_1_stem, pos, weighting_dict)
    # print(word_1_stem)
    # print(word_2_stem)
    similarity_score_list = []
    if pos == wn.NOUN or pos == wn.VERB:
        similarity_score_list.append(
            0.0 * synset_1.path_similarity(synset_2)
            + 1.0 * synset_1.wup_similarity(synset_2)
        )

    else:
        similar_word_list = []
        synsets_1 = [synset_1]
        for synset_1 in synsets_1:
            similar_synsets = synset_1.similar_tos()
            for synset in similar_synsets:
                for lemma in synset.lemmas():
                    similar_word_list.append(str(lemma.name()))
        synonyms_list = get_synonym(word_1, pos)
        if word_2 in synonyms_list:
            similarity_score_list.append(1)
        else:
            if word_2 in similar_word_list:
                similarity_score_list.append(0.9)
            else:
                similarity_score_list.append(0)
    if len(similarity_score_list) == 0:
        similarity_score_list.append(0)
    # print("total time cost: ",time.time()-t)
    return max(similarity_score_list)


def WSD_lesk_algorithm(context, center_word, center_word_pos, weighting_dict):
    # context is a list of words that have been lemmatized.
    if center_word_pos != "":
        synsets = wn.synsets(center_word, pos=center_word_pos)
    else:
        synsets = wn.synsets(center_word)

    best_score = 0
    if synsets != []:
        best_synset = synsets[0]
    else:
        return None
    for synset in synsets:
        # definition = synset.definition()
        # for example in synset.examples():
        #    definition = definition + " " + example

        # definition_word_set = set(definition.split())
        # stopword_set = set(stopwords.words("english"))
        # definition_word_set = definition_word_set - stopword_set - set(center_word)
        score = get_overlap_context_score(
            synset, context, stopwords.words("english"), weighting_dict
        )
        for hyponyms in synset.hyponyms():
            score += get_overlap_context_score(
                synset, context, stopwords.words("english"), weighting_dict
            )

    # score = get_weighted_overlap_sum(
    #     list(definition_word_set), context_word_list, weighting_dict
    # )

    if score > best_score:
        best_synset = synset

    return best_synset


def get_overlap_context_score(synset, context_sentence, stopwords, weighting_dict):
    definition = synset.definition().split()
    definition_pos = get_wordnet_pos_from_nltk(definition)
    gloss = []
    lemmatizer = WordNetLemmatizer()
    # print(definition)
    for i in range(0, len(definition)):

        if definition_pos[i] in ["v", "a", "n", "r"]:
            gloss.append(lemmatizer.lemmatize(
                definition[i].lower(), definition_pos[i]))
            # print(
            #    definition[i].lower(),
            #    lemmatizer.lemmatize(definition[i].lower(), definition_pos[i]),
            #    definition_pos[i],
            # )
        else:
            gloss.append(lemmatizer.lemmatize(definition[i].lower()))

    for example in synset.examples():
        example_sentence = example.split()
        example_pos = get_wordnet_pos_from_nltk(example_sentence)
        for j in range(0, len(example_sentence)):
            if example_pos[j] in ["v", "a", "n", "r"]:
                gloss.append(
                    lemmatizer.lemmatize(
                        example_sentence[j].lower(), example_pos[j])
                )
            else:
                gloss.append(lemmatizer.lemmatize(example_sentence[j].lower()))

    gloss = list(set(gloss).difference(stopwords))

    score = get_weighted_overlap_sum(gloss, context_sentence, weighting_dict)
    # print(definition, score)
    return score


def get_wordnet_pos_from_nltk(tokens):
    tokens_with_pos = nltk.pos_tag(tokens)
    pos_list = []
    for token, pos in tokens_with_pos:
        pos_list.append(pos)

    return penn_pos_to_wordnet_pos(pos_list)


def get_weighted_overlap_sum(list1, list2, weighting_dict):
    overlap_list = list(set(list1) & set(list2))
    weighted_sum = 0
    for element in overlap_list:
        weighted_sum = weighted_sum + weighting_dict[element]
    return weighted_sum

def extract_intent_and_semantic_tags_from_result_for_matching_threshold(result):
        item_index = 0
        top_match_intent = []
        match_intent = []
        tag_value_pair = {}
        best_parse = result[0]
        best_parse_pattern = best_parse[0]
        best_parse_backtrace = best_parse[1]
        for tag in best_parse_pattern.split(" "):
            if "{" in tag:
                match_intent.append(tag)
        for key, value in best_parse_backtrace.items():
            if "[" in key or "<" in key:
                semantic_tag_value = value.split("<-")[0].strip()
                if semantic_tag_value == "isnumeric":
                    semantic_tag_value = value.split("<-")[1].strip()
                semantic_tag_value_processed = ""
                word_index = 0
                for word in semantic_tag_value.split():
                    word_processed = word
                    if "[" in word and word[1].islower():
                        word_processed = word.replace("[", "").replace("]", "")
                    if word_index == 0:
                        semantic_tag_value_processed = word_processed
                    else:
                        semantic_tag_value_processed = semantic_tag_value_processed + " " + word_processed
                    word_index = word_index + 1
                tag_value_pair[key] = semantic_tag_value_processed

        return match_intent, tag_value_pair


def semantic_grammar_parsing_general_idf_for_evaluation(
    input_sentence_raw,
    lowest_idf,
    vocab_document,
    word_weighting_dict,
    grammar_segment_dict,
    vocab_for_pos,
    states, 
    A,
    B,
    intent_taglist_dict,
    pos_dict,
    verbose=False,
    single_word_beam_size=2,
    beam_size=2,
    insert_discount_factor=0.9,
    maximum_length_difference=4,
    not_in_doc_factor=0.5,
    stopword_list=["the", "a", "some", "any", "my"],
    relative_matching_threshold=0.3,
    force_intent_matching=True,
    real_threshold_list=[0.5],
):
    start_time = timer.time()
    table = {}
    # single_word_beam_size = 2  # find a exact match or shift the word
    # beam_size = 2

    original_sentence, input_processed = preprocess(
        vocab_for_pos, input_sentence_raw, False)
    input_sentence_pos = penn_pos_to_wordnet_pos(
        get_pos_for_sentence(states, A, B, input_processed, vocab_for_pos))
    input_sentence_pos_for_lemmatizer = [
        x if (x in ["a", "v", "n", "r"]) else "" for x in input_sentence_pos
    ]

    num_words = len(input_processed)
    lemmatizer = WordNetLemmatizer()
    input_sentence = [
        lemmatizer.lemmatize(
            original_sentence[x], input_sentence_pos_for_lemmatizer[x])
        if (input_sentence_pos_for_lemmatizer[x] != "")
        else original_sentence[x]
        for x in range(0, num_words)
    ]
    new_input_sentence = []
    new_input_sentence_pos_for_lemmatizer = []
    for tmp_index in range(0, num_words):
        word = input_sentence[tmp_index]
        pos = input_sentence_pos_for_lemmatizer[tmp_index]
        if word not in stopword_list:
            new_input_sentence.append(word)
            new_input_sentence_pos_for_lemmatizer.append(pos)

    input_sentence = new_input_sentence
    input_sentence_pos_for_lemmatizer = new_input_sentence_pos_for_lemmatizer
    num_words = len(input_sentence)
    # initialize table

    # parsing process
    for j in range(1, num_words + 1):
        parent = "none"
        tmp_list = []
        original_word = input_sentence[j - 1]
        source_pos = list(input_sentence_pos_for_lemmatizer[j - 1])
        lowest_cost = 100
        for tag, tag_patterns in grammar_segment_dict.items():
            for tag_pattern in tag_patterns:
                tmp_shift_penalty, score = get_matching_score_general_idf(
                    original_word,
                    tag_pattern,
                    vocab_document,
                    word_weighting_dict,
                    source_pos,
                    pos_dict,
                    verboseFlag=False,
                    insert_discount_factor=insert_discount_factor,
                    maximum_length_difference=maximum_length_difference,
                    not_in_doc_factor=lowest_idf,
                    relative_matching_threshold=relative_matching_threshold,
                )
                tmp_list.append(
                    [tag, {tag: tag_pattern + "<-" +
                           original_word}, score, parent]
                )
        if score < lowest_cost:
            lowest_cost = score
        shift_penalty = tmp_shift_penalty
        if lowest_cost > 0:
            tmp_list.append(
                [original_word, {original_word: original_word},
                    shift_penalty, parent]
            )
        else:
            # exact_match would dominate others
            tmp_list_copy = tmp_list.copy()
            for element in tmp_list_copy:
                if element[2] > 0:
                    tmp_list.remove(element)

        tmp_list = sorted(tmp_list, key=itemgetter(2))
        tmp_list = tmp_list[:single_word_beam_size]

        upper_level_finished = False
        loop_list = tmp_list
        while not upper_level_finished:
            next_loop_list = []
            for _pattern, _backtrace, _score, _parent in loop_list:
                for _tag, _tag_patterns in grammar_segment_dict.items():
                    # if upper level exists, then
                    for _tag_pattern in _tag_patterns:
                        if _pattern.strip() == _tag_pattern.strip():
                            new_dict = _backtrace.copy()
                            new_dict.update({_tag: _tag_pattern})
                            new_element = [
                                _tag,
                                new_dict,
                                _score,
                                _parent,
                            ]
                            next_loop_list.append(new_element)
                            tmp_list = [new_element] + tmp_list
            loop_list = next_loop_list

            if next_loop_list == []:
                upper_level_finished = True
        if verbose:
            print("({0},{1}) : {2} \n".format(j - 1, j, tmp_list))

        tmp_list = sorted(tmp_list, key=itemgetter(2))
        table[(j - 1, j)] = tmp_list

        for i in range(j - 2, -1, -1):
            tmp_list = []
            pattern_dict = defaultdict(int)
            # length of the sequence * the relative edit distance threshold
            for k in range(i + 1, j):
                left_element = table[(i, k)]
                down_element = table[(k, j)]
                pattern_list_1 = []
                backtrace_list_1 = []
                score_list_1 = []
                pattern_list_2 = []
                backtrace_list_2 = []
                score_list_2 = []
                # print(left_element)
                for m in range(0, len(left_element)):
                    pattern_list_1.append(left_element[m][0])
                    backtrace_list_1.append(left_element[m][1])
                    score_list_1.append(left_element[m][2])
                for n in range(0, len(down_element)):
                    pattern_list_2.append(down_element[n][0])
                    backtrace_list_2.append(down_element[n][1])
                    score_list_2.append(down_element[n][2])

                tmp_pattern_list = combine_list(
                    pattern_list_1, pattern_list_2, " ")
                tmp_backtrace_list = combine_list_of_dictionary(
                    backtrace_list_1, backtrace_list_2
                )
                tmp_score_list = combine_list_for_sum_score(
                    score_list_1, score_list_2)
                num_patterns = len(tmp_pattern_list)

                parent = "{0} + {1}".format([i, k], [k, j])
                # preserve only one of all the same patterns with the lowest distance
                for l in range(0, num_patterns):
                    source = tmp_pattern_list[l].split()
                    shift_penalty = relative_matching_threshold * sum(
                        [word_weighting_dict[x] for x in source]
                    )
                    original_list = [
                        tmp_pattern_list[l],
                        tmp_backtrace_list[l],
                        shift_penalty + tmp_score_list[l],
                        parent,
                    ]
                    if (
                        pattern_dict[tmp_pattern_list[l]] == 0
                        or pattern_dict[tmp_pattern_list[l]][2] > original_list[2]
                    ):
                        pattern_dict[tmp_pattern_list[l]] = original_list

            for value in pattern_dict.values():
                tmp_list.append(value)
            num_patterns = len(tmp_list)
            tmp_list_copy = tmp_list

            for l in range(0, num_patterns):
                #               if tmp_pattern_list[l] not in tabuList:
                source = tmp_list_copy[l][0].split()
                shift_penalty = relative_matching_threshold * sum(
                    [word_weighting_dict[x] for x in source]
                )
                original_pattern_word_list = tmp_list_copy[l][0].split()
                original_pattern_tag_list = []
                for word in original_pattern_word_list:
                    if "[" in word:
                        original_pattern_tag_list.append(word)

                for tag, tag_patterns in grammar_segment_dict.items():
                    for tag_pattern in tag_patterns:
                        tag_pattern_word_list = tag_pattern.split()
                        tag_pattern_tag_list = []
                        for word in tag_pattern_word_list:
                            if "[" in word:
                                tag_pattern_tag_list.append(word)
                        if (
                            (
                                (
                                    set(tag_pattern_tag_list)
                                    & set(original_pattern_tag_list)
                                )
                                == set()
                            )
                            and set(tag_pattern_tag_list) != set()
                            and set(original_pattern_tag_list) != set()
                        ):
                            continue
                        (tmp_shift_penalty, score,) = get_matching_score_general_idf(
                            tmp_list_copy[l][0],
                            tag_pattern,
                            vocab_document,
                            word_weighting_dict,
                            [],
                            pos_dict,
                            verboseFlag=False,
                            insert_discount_factor=insert_discount_factor,
                            maximum_length_difference=maximum_length_difference,
                            not_in_doc_factor=lowest_idf,
                            relative_matching_threshold=relative_matching_threshold,
                        )

                        # maybe reward a match by returning the shift cost?

                        tmp_dict = {tag: tag_pattern +
                                    "<-" + tmp_list_copy[l][0]}
                        new_tmp_dict = tmp_list_copy[l][1].copy()
                        new_tmp_dict.update(tmp_dict)

                        tmp_list.append(
                            [
                                tag,
                                new_tmp_dict.copy(),
                                score + tmp_list_copy[l][2] - shift_penalty,
                                tmp_list_copy[l][3],
                            ]
                        )

            tmp_list = sorted(tmp_list, key=itemgetter(2))
            tmp_list = tmp_list[:beam_size]
            upper_level_finished = False
            loop_list = tmp_list
            while not upper_level_finished:
                next_loop_list = []
                for _pattern, _backtrace, _score, _parent in loop_list:
                    for _tag, _tag_patterns in grammar_segment_dict.items():
                        for _tag_pattern in _tag_patterns:
                            if _pattern.split() == _tag_pattern.split():
                                new_dict = _backtrace.copy()
                                new_dict.update({_tag: _tag_pattern})
                                new_element = [
                                    _tag,
                                    new_dict,
                                    _score,
                                    _parent,
                                ]
                                next_loop_list.append(new_element)
                                tmp_list = [new_element] + tmp_list
                loop_list = next_loop_list
                if next_loop_list == []:
                    upper_level_finished = True

            tmp_list = sorted(tmp_list, key=itemgetter(2))
            table[(i, j)] = tmp_list
            if verbose:
                print("({0},{1}) : {2} \n".format(i, j, tmp_list))

    if force_intent_matching:  # force intent in the last step
        final_list = []
        index = 0
        for _pattern, _backtrace, _score, _parent in tmp_list:
            if "{" in _pattern:
                final_list.append(tmp_list[index])
                continue
            index = index + 1
            original_pattern_word_list = _pattern.split()
            original_pattern_tag_list = []
            for word in original_pattern_word_list:
                if "[" in word:
                    original_pattern_tag_list.append(word)
            for _tag, _tag_patterns in grammar_segment_dict.items():
                if "{" in _tag:
                    for _tag_pattern in _tag_patterns:
                        _tag_pattern_word_list = _tag_pattern.split()
                        _tag_pattern_tag_list = []
                        for word in _tag_pattern_word_list:
                            if "[" in word:
                                _tag_pattern_tag_list.append(word)
                        if (
                            (
                                (
                                    set(_tag_pattern_tag_list)
                                    & set(original_pattern_tag_list)
                                )
                                == set()
                            )
                            and set(_tag_pattern_tag_list) != set()
                            and set(original_pattern_tag_list) != set()
                        ):
                            continue
                        parent = "final"
                        (tmp_shift_penalty, score,) = get_matching_score_general_idf(
                            _pattern,
                            _tag_pattern,
                            vocab_document,
                            word_weighting_dict,
                            [],
                            pos_dict,
                            verboseFlag=False,
                            insert_discount_factor=insert_discount_factor,
                            maximum_length_difference=100,
                            not_in_doc_factor=lowest_idf,
                            relative_matching_threshold=relative_matching_threshold,
                        )

                        tmp_dict = {_tag: _tag_pattern + "<-" + _pattern}
                        new_tmp_dict = _backtrace.copy()
                        new_tmp_dict.update(tmp_dict)

                        final_list.append(
                            [
                                _tag,
                                new_tmp_dict.copy(),
                                0.1 * score + _score,
                                parent,
                            ]
                        )

        tmp_list = sorted(final_list, key=itemgetter(2))
        print("({0},{1}) : {2} \n".format(0, num_words, tmp_list[:beam_size]))
        result = tmp_list
    if not force_intent_matching:
        match_intent, backtrace = extract_intent_and_semantic_tags_from_result(
            result)
        best_score = 0
        best_result = [["{none}", tmp_list[0][1], 100, "final"]]
        for _tag, _tag_patterns in grammar_segment_dict.items():
            if "{" in _tag:
                taglist = intent_taglist_dict[_tag]
                score = get_weighted_overlap_sum(
                    backtrace.keys(), taglist, word_weighting_dict
                )
                if score > best_score:
                    best_score = score
                    best_result = tmp_list[0]
                    best_result[0] = _tag
        tmp_list = [best_result]

    sum_of_weights = sum(
        [
            word_weighting_dict[x] if x in vocab_document else lowest_idf
            for x in input_sentence
        ]
    )
    end_time = timer.time()
    runtime = start_time - end_time
    if sum_of_weights < word_weighting_dict["{Greetings}"]:
        return [
            [["{none}", tmp_list[0][1], 100, "final: sum_of_weights == 0"]]
        ], runtime

    if len(tmp_list) == 0:
        return [[["{none}", tmp_list[0][1], 100, "final: empty tmp list"]]], runtime
    else:
        final_result = tmp_list[0]
    matching_score = final_result[2] / sum_of_weights
    result = tmp_list

    matching_score = final_result[2] / sum_of_weights

    if verbose:
        print("run time: {0}".format(end_time - start_time))

    result_list = []
    for real_threshold in real_threshold_list:
        if matching_score > real_threshold or "{" not in final_result[0]:
            result_list.append(
                [["{none}", tmp_list[0][1], 100, "final: below threshold"]]
            )
        else:
            result_list.append(tmp_list[:beam_size])

    return result_list, runtime

def semantic_grammar_parsing_general_idf(
    input_sentence_raw,
    lowest_idf,
    vocab_document,
    word_weighting_dict,
    grammar_segment_dict,
    vocab_for_pos,
    states, 
    A,
    B,
    intent_taglist_dict,
    pos_dict,
    verbose=False,
    single_word_beam_size=2,
    beam_size=2,
    insert_discount_factor=0.9,
    maximum_length_difference=4,
    not_in_doc_factor=0.5,
    stopword_list=["the", "a", "some", "any", "my"],
    relative_matching_threshold=0.3,
    force_intent_matching=False,
    real_threshold=0.5,
):
    start_time = timer.time()
    table = {}
    # single_word_beam_size = 2  # find a exact match or shift the word
    # beam_size = 2

    original_sentence, input_processed = preprocess(
        vocab_for_pos, input_sentence_raw, False)
    input_sentence_pos = penn_pos_to_wordnet_pos(
        get_pos_for_sentence(states, A, B, input_processed, vocab_for_pos))
    input_sentence_pos_for_lemmatizer = [
        x if (x in ["a", "v", "n", "r"]) else "" for x in input_sentence_pos
    ]
    num_words = len(input_processed)
    
    lemmatizer = WordNetLemmatizer()

    input_sentence = [
        lemmatizer.lemmatize(
            original_sentence[x], input_sentence_pos_for_lemmatizer[x])
        if (input_sentence_pos_for_lemmatizer[x] != "")
        else original_sentence[x]
        for x in range(0, num_words)
    ]
    new_input_sentence = []
    new_input_sentence_pos_for_lemmatizer = []
    for tmp_index in range(0, num_words):
        word = input_sentence[tmp_index]
        pos = input_sentence_pos_for_lemmatizer[tmp_index]
        if word not in stopword_list:
            new_input_sentence.append(word)
            new_input_sentence_pos_for_lemmatizer.append(pos)

    input_sentence = new_input_sentence
    input_sentence_pos_for_lemmatizer = new_input_sentence_pos_for_lemmatizer
    num_words = len(input_sentence)

    # initialize table

    # parsing process
    for j in range(1, num_words + 1):
        parent = "none"
        tmp_list = []
        original_word = input_sentence[j - 1]
        source_pos = list(input_sentence_pos_for_lemmatizer[j - 1])
        lowest_cost = 100
        for tag, tag_patterns in grammar_segment_dict.items():
            for tag_pattern in tag_patterns:
                tmp_shift_penalty, score = get_matching_score_general_idf(
                    original_word,
                    tag_pattern,
                    vocab_document,
                    word_weighting_dict,
                    source_pos,
                    pos_dict,
                    verboseFlag=False,
                    insert_discount_factor=insert_discount_factor,
                    maximum_length_difference=maximum_length_difference,
                    not_in_doc_factor=lowest_idf,
                    relative_matching_threshold=relative_matching_threshold,
                )
                tmp_list.append(
                    [tag, {tag: tag_pattern + "<-" +
                           original_word}, score, parent]
                )
                if score < lowest_cost:
                    lowest_cost = score
        shift_penalty = tmp_shift_penalty
        if lowest_cost > 0:
            tmp_list.append(
                [original_word, {original_word: original_word},
                    shift_penalty, parent]
            )
        else:
            # exact_match would dominate others
            tmp_list_copy = tmp_list.copy()
            for element in tmp_list_copy:
                if element[2] > 0:
                    tmp_list.remove(element)

        tmp_list = sorted(tmp_list, key=itemgetter(2))
        tmp_list = tmp_list[:single_word_beam_size]

        upper_level_finished = False
        loop_list = tmp_list
        while not upper_level_finished:
            next_loop_list = []
            for _pattern, _backtrace, _score, _parent in loop_list:
                for _tag, _tag_patterns in grammar_segment_dict.items():
                    # if upper level exists, then
                    for _tag_pattern in _tag_patterns:
                        if _pattern.strip() == _tag_pattern.strip():
                            new_dict = _backtrace.copy()
                            new_dict.update({_tag: _tag_pattern})
                            new_element = [
                                _tag,
                                new_dict,
                                _score,
                                _parent,
                            ]
                            next_loop_list.append(new_element)
                            tmp_list = [new_element] + tmp_list
            loop_list = next_loop_list

            if next_loop_list == []:
                upper_level_finished = True
        if verbose:
            print("({0},{1}) : {2} \n".format(j - 1, j, tmp_list))

        tmp_list = sorted(tmp_list, key=itemgetter(2))
        table[(j - 1, j)] = tmp_list

        for i in range(j - 2, -1, -1):
            tmp_list = []
            pattern_dict = defaultdict(int)
            # length of the sequence * the relative edit distance threshold
            for k in range(i + 1, j):
                left_element = table[(i, k)]
                down_element = table[(k, j)]
                pattern_list_1 = []
                backtrace_list_1 = []
                score_list_1 = []
                pattern_list_2 = []
                backtrace_list_2 = []
                score_list_2 = []
                # print(left_element)
                for m in range(0, len(left_element)):
                    pattern_list_1.append(left_element[m][0])
                    backtrace_list_1.append(left_element[m][1])
                    score_list_1.append(left_element[m][2])
                for n in range(0, len(down_element)):
                    pattern_list_2.append(down_element[n][0])
                    backtrace_list_2.append(down_element[n][1])
                    score_list_2.append(down_element[n][2])

                tmp_pattern_list = combine_list(
                    pattern_list_1, pattern_list_2, " ")
                tmp_backtrace_list = combine_list_of_dictionary(
                    backtrace_list_1, backtrace_list_2
                )
                tmp_score_list = combine_list_for_sum_score(
                    score_list_1, score_list_2)
                num_patterns = len(tmp_pattern_list)

                parent = "{0} + {1}".format([i, k], [k, j])
                # preserve only one of all the same patterns with the lowest distance
                for l in range(0, num_patterns):
                    source = tmp_pattern_list[l].split()
                    shift_penalty = relative_matching_threshold * sum(
                        [word_weighting_dict[x] for x in source]
                    )
                    original_list = [
                        tmp_pattern_list[l],
                        tmp_backtrace_list[l],
                        shift_penalty + tmp_score_list[l],
                        parent,
                    ]
                    if (
                        pattern_dict[tmp_pattern_list[l]] == 0
                        or pattern_dict[tmp_pattern_list[l]][2] > original_list[2]
                    ):
                        pattern_dict[tmp_pattern_list[l]] = original_list

            for value in pattern_dict.values():
                tmp_list.append(value)
            num_patterns = len(tmp_list)
            tmp_list_copy = tmp_list

            for l in range(0, num_patterns):
                #               if tmp_pattern_list[l] not in tabuList:
                source = tmp_list_copy[l][0].split()
                shift_penalty = relative_matching_threshold * sum(
                    [word_weighting_dict[x] for x in source]
                )
                original_pattern_word_list = tmp_list_copy[l][0].split()
                original_pattern_tag_list = []
                for word in original_pattern_word_list:
                    if "[" in word:
                        original_pattern_tag_list.append(word)

                for tag, tag_patterns in grammar_segment_dict.items():
                    for tag_pattern in tag_patterns:
                        tag_pattern_word_list = tag_pattern.split()
                        tag_pattern_tag_list = []
                        for word in tag_pattern_word_list:
                            if "[" in word:
                                tag_pattern_tag_list.append(word)
                        if (
                            (
                                (
                                    set(tag_pattern_tag_list)
                                    & set(original_pattern_tag_list)
                                )
                                == set()
                            )
                            and set(tag_pattern_tag_list) != set()
                            and set(original_pattern_tag_list) != set()
                        ):
                            continue

                        (tmp_shift_penalty, score,) = get_matching_score_general_idf(
                            tmp_list_copy[l][0],
                            tag_pattern,
                            vocab_document,
                            word_weighting_dict,
                            [],
                            pos_dict,
                            verboseFlag=False,
                            insert_discount_factor=insert_discount_factor,
                            maximum_length_difference=maximum_length_difference,
                            not_in_doc_factor=lowest_idf,
                            relative_matching_threshold=relative_matching_threshold,
                        )
                        # maybe reward a match by returning the shift cost?

                        tmp_dict = {tag: tag_pattern +
                                    "<-" + tmp_list_copy[l][0]}
                        new_tmp_dict = tmp_list_copy[l][1].copy()
                        new_tmp_dict.update(tmp_dict)

                        tmp_list.append(
                            [
                                tag,
                                new_tmp_dict.copy(),
                                score + tmp_list_copy[l][2] - shift_penalty,
                                tmp_list_copy[l][3],
                            ]
                        )

            tmp_list = sorted(tmp_list, key=itemgetter(2))
            tmp_list = tmp_list[:beam_size]
            upper_level_finished = False
            loop_list = tmp_list
            while not upper_level_finished:
                next_loop_list = []
                for _pattern, _backtrace, _score, _parent in loop_list:
                    for _tag, _tag_patterns in grammar_segment_dict.items():
                        for _tag_pattern in _tag_patterns:
                            if _pattern.split() == _tag_pattern.split():
                                new_dict = _backtrace.copy()
                                new_dict.update({_tag: _tag_pattern})
                                new_element = [
                                    _tag,
                                    new_dict,
                                    _score,
                                    _parent,
                                ]
                                next_loop_list.append(new_element)
                                tmp_list = [new_element] + tmp_list
                loop_list = next_loop_list
                if next_loop_list == []:
                    upper_level_finished = True

            tmp_list = sorted(tmp_list, key=itemgetter(2))
            table[(i, j)] = tmp_list
            if verbose:
                print("({0},{1}) : {2} \n".format(i, j, tmp_list))

    if force_intent_matching:  # force intent in the last step
        final_list = []
        index = 0
        for _pattern, _backtrace, _score, _parent in tmp_list:
            if "{" in _pattern:
                final_list.append(tmp_list[index])
                continue
            index = index + 1
            original_pattern_word_list = _pattern.split()
            original_pattern_tag_list = []
            for word in original_pattern_word_list:
                if "[" in word:
                    original_pattern_tag_list.append(word)
            for _tag, _tag_patterns in grammar_segment_dict.items():
                if "{" in _tag:
                    for _tag_pattern in _tag_patterns:
                        _tag_pattern_word_list = _tag_pattern.split()
                        _tag_pattern_tag_list = []
                        for word in _tag_pattern_word_list:
                            if "[" in word:
                                _tag_pattern_tag_list.append(word)

                        if (
                            (
                                (
                                    set(_tag_pattern_tag_list)
                                    & set(original_pattern_tag_list)
                                )
                                == set()
                            )
                            and set(_tag_pattern_tag_list) != set()
                            and set(original_pattern_tag_list) != set()
                        ):
                            continue
                        parent = "final"

                        (tmp_shift_penalty, score,) = get_matching_score_general_idf(
                            _pattern,
                            _tag_pattern,
                            vocab_document,
                            word_weighting_dict,
                            [],
                            pos_dict,
                            verboseFlag=False,
                            insert_discount_factor=insert_discount_factor,
                            maximum_length_difference=100,
                            not_in_doc_factor=lowest_idf,
                            relative_matching_threshold=relative_matching_threshold,
                        )
                        tmp_dict = {_tag: _tag_pattern + "<-" + _pattern}
                        new_tmp_dict = _backtrace.copy()
                        new_tmp_dict.update(tmp_dict)

                        final_list.append(
                            [
                                _tag,
                                new_tmp_dict.copy(),
                                0.1 * score + _score,
                                parent,
                            ]
                        )

        tmp_list = sorted(final_list, key=itemgetter(2))
        print("({0},{1}) : {2} \n".format(0, num_words, tmp_list[:beam_size]))

    final_result = tmp_list[0]

    sum_of_weights = sum(
        [
            word_weighting_dict[x] if x in vocab_document else lowest_idf
            for x in input_sentence
        ]
    )

    if sum_of_weights == 0:
        sum_of_weights = 0.000000001
    matching_score = final_result[2] / sum_of_weights
    result = tmp_list

    if not force_intent_matching:
        match_intent, backtrace = extract_intent_and_semantic_tags_from_result(
            result)
        best_score = 0
        best_result = [["{none}", tmp_list[0][1], 100, "final"]]
        for _tag, _tag_patterns in grammar_segment_dict.items():
            if "{" in _tag:
                taglist = intent_taglist_dict[_tag]
                score = get_weighted_overlap_sum(
                    backtrace.keys(), taglist, word_weighting_dict
                )
                if score > best_score:
                    best_score = score
                    best_result = tmp_list[0]
                    best_result[0] = _tag
        tmp_list = [best_result]

    match_intent, backtrace = extract_intent_and_semantic_tags_from_result_for_matching_threshold(
         result)
    _, _backtrace = extract_intent_and_semantic_tags_from_result(
        result)
    print(_backtrace)
    print(result[0])
    print(lowest_idf,word_weighting_dict["i"])
    taglist = intent_taglist_dict[match_intent[0]]
    score = get_weighted_overlap_sum(
        backtrace.keys(), taglist, word_weighting_dict
    )
    num_of_real_tag = 0
    for tag in taglist:
        if tag[1].isupper():
             num_of_real_tag += 1

    for tag in backtrace:
        print(tag, word_weighting_dict[tag])

#    matching_score = score/sum_of_weights
    print(score, sum_of_weights, matching_score)
#    if len(_backtrace) == 0 and num_of_real_tag > 0:
#        return [["{none}", tmp_list[0][1], 100, "final"]]

    end_time = timer.time()
    if verbose:
        print("run time: {0}".format(end_time - start_time))
    if matching_score > real_threshold or "{" not in final_result[0] or sum_of_weights < 0.3 * word_weighting_dict["{Greetings}"]:
        return [["{none}", tmp_list[0][1], 100, "final"]]

    return tmp_list[:beam_size]


def initialize_through_recalculating(cnf_option=False):

    # load in the training corpus
    with open("WSJ_02-21.pos", "r") as f:
        training_corpus = f.readlines()

    # vocab: dictionary that has the index of the corresponding words
    with open("hmm_vocab.txt", "r") as f:
        voc_l = f.read().split("\n")

    vocab = {}
    vocab_list = []
    # Get the index of the corresponding words.
    for i, word in enumerate(sorted(voc_l)):
        vocab_list.append(word.lower())

    vocab_list = list(set(vocab_list))

    for i in range(0, len(vocab_list)):
        vocab[vocab_list[i]] = i

    (
        emission_counts,
        transition_counts,
        tag_counts,
        word_counts,
        normalized_idf_dict,
    ) = create_dictionaries(training_corpus, vocab)

    write_dict_to_csvfile("normalized_idf_dict.csv", normalized_idf_dict)
    # get all the POS states
    states = sorted(tag_counts.keys())
    with open("states.data", "wb") as filehandle:
        # store the data as binary data stream
        pickle.dump(states, filehandle)
    
    alpha = 0.001
    A = create_transition_matrix(alpha, tag_counts, transition_counts)
    np.savetxt("A.txt", A)
    B = create_emission_matrix(alpha, tag_counts, emission_counts, list(vocab))
    np.savetxt("B.txt", B)

    # label grammar definition
    """ Grammar Pieces Dictionary """
    grammar_pieces_dict = defaultdict(str)
    entry_point_dict = defaultdict(str)

    grammar_pieces_dict["[NeedQuery]"] = [
        "Do I need to", "Should I", "Am I required to", "Does [Chalmers] require me"]
    grammar_pieces_dict["[Inform]"] = ["give me", "tell me", "show me", "show"]
    grammar_pieces_dict["[RequestInfo]"] = [
        "may-i/can-i/could-i know",
        "can-you/could-you/would-you/will-you [Inform]/find ",
        "I [Desire] to know/find", 
        "what are",
        "Where/how can/could/do I get/find information [RelatedTo]"]
    grammar_pieces_dict["[RelatedTo]"] = ["related-to/regarding/about/on"]
    grammar_pieces_dict["[Help]"] = ["help me", "help"]
    grammar_pieces_dict["[Desire]"] = ["want/need/desire/would-like"]
    grammar_pieces_dict["[DegreeType]"] = ["bachelor", "master", "doctoral"]
    grammar_pieces_dict["[Chalmers]"] = [
        "Chalmers",
        "Chalmers University of Technology",
        "CTH",
    ]
    grammar_pieces_dict["[ProgramType]"] = [
        "a [DegreeType] program",
        "[DegreeType] programs",
    ]
    grammar_pieces_dict["[EntryRequirements]"] = [
        "the entry requirements",
        "the eligibility criteria",
        "the admission criteria",
        "the admission requirements",
    ]


    grammar_pieces_dict["[HaveTrouble]"] = [
        "I have difficulty/trouble",

    ]


    grammar_pieces_dict["[Exist]"] = [
        "Is there",
        "Are there",
        "Do you have/know",
    ]
    grammar_pieces_dict["[UserAskForMethod]"] = [
        "How do/can/could I",
        "[Exist] a way to",
        "Do you know a way to",
        "Do you know how to",
        "what do I need-to/ do to",
        "what should/can/could I do to",
    ]

    grammar_pieces_dict["[AskAmount]"] = ["How much", "How many"]

    """
    grammar_pieces_dict["[DT]"] = [
        "the",
        "my",
        "his",
        "her",
        "a",
        "our",
        "their",
        "your",
        "any",
        "some"
    ]

    """


    grammar_pieces_dict["[Operand1]"] = ["isnumeric"]
    grammar_pieces_dict["[Operand2]"] = ["isnumeric"]
  #  grammar_pieces_dict["[TimeSuffice]"] = ["a_m", "p_m"]

    entry_point_dict["{Duration}"] = [
        "from [Operand1] to [Operand2]",
   ]

    entry_point_dict["{DivisionOperation}"] = [
       "[Operand1] divide [Operand2]",
       "[Operand2] divided by [Operand1]"

   ]


    """ Entry Points Dictionary """
    entry_point_dict["{Greetings}"] = [
        "Hello",
        "Hi",
        "Greetings",
        "Good morning",
        "Good afternoon",
        "Good evening",
    ]

    entry_point_dict["{GreetingsResponse}"] = [
        "Nice/glad to see/meet you",
    ]
    entry_point_dict["{Goodbye}"] = ["Goodbye", "Bye", "Ciao","see you"]

    entry_point_dict["{GetEntryRequirement}"] = [
        # including program type is important for assigning weights to the Tag and associated words
        "I  [Desire] to become a [DegreeType] student",
        "[UserAskForMethod] become a [DegreeType] student",
        "[RequestInfo] [EntryRequirements]",
        "[HaveTrouble] finding [EntryRequirements]",
    ]

    entry_point_dict["{GetAvailablePrograms}"] = [
        "[RequestInfo] available/ programs/[ProgramType]/programs",  # u can replace this by introducing CFG rules (ADJ NP == NP ADJ) 
        "[Exist] programs",
        "[HaveTrouble] finding available-[ProgramType] information",
    ]

    entry_point_dict["{CountrySpecificQuery}"] = [
        "[RequestInfo]  country",
        "[UserAskForMethod]  country",
     ]
    
    entry_point_dict["{RequestHelpOnApplication}"] = [
        "[UserAskForMethod] [Help] with application system/",
        "I [Desire] [Help] on application system/",
        "[HaveTrouble] with application system/",
    ]

    entry_point_dict["{NeedEnglishScore}"] = [
        "[NeedQuery] prove  English skills",
        "[NeedQuery] provide/submit  English test score",
    ]

    entry_point_dict["{ConvertCredict}"] = [
        "[UserAskForMethod] convert the credicts to-[Chalmrs]-system/",
        "[RequestInfo] how to convert the credicts to-[Chalmrs]-system/",
    ]

    entry_point_dict["{NeedGreScore}"] = [
        "[NeedQuery] provide/submit Gre test/score",
    ]

    entry_point_dict["{AdmissionChanceQuery}"] = [
        "[RequesInfo] chance of admission",
        "[UserAskForMethod] chance of admission",
    ]

    entry_point_dict["{TransferMajor}"] = [
        "can I apply for a transfer to [NewMajor]",
        "[UserAskForMethod] transfer program/major",
    ] 

    entry_point_dict["{HowFindCourse}"] = [
        "[UserAskForMethod] find courses of-program/",
    ]

    entry_point_dict["{WhereFindSchedule}"] = [
        "[UserAskForMethod] find schedule for courses",
        "[UserAskForMethod] find course schedule",
    ] 

    entry_point_dict["{SocholarshipQuery}"] = [
        "Does [Chalmers] offer/provide scholarship",
        "[Exist] scholarship",
        "[UserAskForMethod] apply for scholarship",
    ]

    entry_point_dict["{TransferScholarship}"] = [
        "Can I keep scholarship if I apply for [NewMajor]",
    ]


    entry_point_dict["{NeedTuitionFeeQuery}"] = [
        "[NeedQuery] pay/ [TuitionFee]",
    ]

    grammar_pieces_dict["[TuitionFee]"] = ["tuition fee"]

    entry_point_dict["{TuitionFeeAmountQuery}"] = [
        "[RequestInfo] [AskAmount] the [TuitionFee] is/take/cost",
        "[AskAmount] is the [TuitionFee]",
        "[AskAmount] do I spend on/for the [TuitionFee]"
        "[AskAmount] does the [TuitionFee] take/cost",
    ]

    entry_point_dict["{CredictCardQuery}"] = [
        "[NeedQuery] credit card",
    ]

    grammar_pieces_dict["[LivingCost]"] = ["living cost/expense"]
    entry_point_dict["{LivingCostAmountQuery}"] = [
        "[RequestInfo][AskAmount] the [LivingCost] is/take/",
        "[AskAmount] is the [LivingCost]",
        "[AskAmount] do i spend on/for the [LivingCost]",
        "[AskAmount] does the [LivingCost] take/cost",
    ]

    grammar_pieces_dict["[Accommodation]"] = ["accommodation", "housing", "houses"]

    entry_point_dict["{NeedFindAccommodationQuery}"] = [
        "[NeedQuery] find/seek [Accommodation]",
        "Does [Chalmers] guarantee [Accommodation] for [NewStudent]"
    ]

    entry_point_dict["{WhereFindAccommodationQuery}"] = [
        "Where may-i/can-i/could-i/do-i find/get [Accommodation]",
        "[RequestInfo] [Accommodation]"
        "[UserAskForMethod] find/seek/get [Accommodation]",
    ]
    grammar_pieces_dict["[NewMajor]"] = ["another/new major/program"]
    grammar_pieces_dict["[NewStudent]"] = ["incoming/new students"]
    grammar_pieces_dict["[ResidentPermit]"] = ["resident permit/permission"]

    entry_point_dict["{NeedApplyResidentPermitQuery}"] = [
        "[NeedQuery] have/apply-for/ a [ResidentPermit]",  
    ]

    entry_point_dict["{HowApplyResidentPermitQuery}"] = [
        "Where may-i/can-i/could-i/do-i apply-for/have/find a [ResidentPermit]",
        "[UserAskForMethod]/[HaveTrouble] apply-for/have/find a [ResidentPermit]",
    ]

    entry_point_dict["{WhenApplyResidentPermit}"] = [
        "When may-i/can-i/could-i/do-i apply-for a [ResidentPermit]",
        "Tips on applying for a [ResidentPermit]",
    ]

    entry_point_dict["{NeedLearnSwedish}"] = [
        "[NeedQuery] learn swedish",
    ]

    entry_point_dict["{HowLearnSwedish}"] = [
        "[UserAskForMethod] learn-swedish/take-swedish-courses",
        "[RequestInfo] learn swedish/take-swedish-courses",
    ]
    grammar_pieces_dict["[PersonalNumber]"] = ["personal number/identity"]
    entry_point_dict["{HowGetPersonalNumber}"] = [
        "[UserAskForMethod] apply-for/get/have [PersonalNumber]",
        "[RequestInfo] [PersonalNumber]",
    ]

    
    entry_point_dict["{HowStartBankAccount}"] = [
        "[UserAskForMethod] apply-for/get/have/start a bank account",
        "[RequestInfo] bank account",
    ]

    entry_point_dict["{PublicTransportQuery}"] = [
        "How does public/ transportation work ",
        "[RequestInfo] transportation",
    ]

    entry_point_dict["{PublicTransportQuery}"] = [
        "How does public/ transportation work ",
        "[RequestInfo] transportation",
    ]
    grammar_pieces_dict["[HealthCare]"] = ["health care"]
    entry_point_dict["{HealthCareQuery}"] = [
        "How does public/ [HealthCare]work ",
        "[RequestInfo] [HealthCare]",
    ]

    entry_point_dict["{HowApplyForStudyPlace}"] = [
        "[UserAskForMethod] apply for [ProgramType]/study place",
    ]

    entry_point_dict["{InformStudentType}"] = [
        "I am [StudentType]",
        "[StudentType]",
        "I am [NewStudent]"
    ]
    

    grammar_pieces_dict["[StudentType]"] = [
        "non EU/EEA student",
        "international student",
        "EU/EEA citizen/student",
        "Switzerland student/citizen",
    ]


    entry_point_dict["{Appreciation}"] = ["I appreciate it","Thank you","Thanks"]
# you can consider removing some of the stop words.
    entry_point_dict["{Positive1}"] = ["Yes", "Sure","I agree/think so","ok","good"]

    entry_point_dict["{Negative}"] = [
        "No",
        "I do not agree/think-so",
        "not good",
        "bad"
    ]




    entry_point_dict["{RequestStudyRoomBooking}"] = [
        "[Exist] study rooms",
        "can-you/could-you/would-you/will-you [Help] book a study room",
        "I  [Desire] to book a study room",
    ]


    """
    entry_point_dict["{ResponseToBookingTime}"] = [
        "[Exist] study room",
        "can-you/could-you/would-you/will-you [Help] book a study room",
        "I  [Desire] to book a study room",
    ]
    """


    entry_point_dict["{RequestAgentName}"] = [
        "[RequestInfo] your name",
        "What should I call you",
    ]

    entry_point_dict["{RequestAgentFunction}"] = [
         "[RequestInfo] what you can do",
         "What can you [Help] me with",
         "what can you do"
    ]


    stopword_list = ["the","a","some","my","any"]

    grammar_pieces_dict.update(entry_point_dict.copy())
    lemmatizer = WordNetLemmatizer()
    pos_dict_for_grammar_terminal = defaultdict(list)
    for tag, patterns in grammar_pieces_dict.items():
        tmp_patterns = []
        for pattern in patterns:
            tmp_pattern_list = [""]
            word_list = pattern.split()
            _, pattern_processed = preprocess(vocab, pattern, False)
            pattern_pos = penn_pos_to_wordnet_pos(
                get_pos_for_sentence(states,A,B,pattern_processed,vocab))
            word_index = 0
            for word in word_list:
                if word.lower() in stopword_list:
                    tmp_word = ""
                elif "[" not in word and "{" not in word and "<" not in word:
                    pos = pattern_pos[word_index]
                    if pos in ["v", "n", "a", "r"]:
                        tmp_word = lemmatizer.lemmatize(word.lower(), pos)
                        if pos not in pos_dict_for_grammar_terminal[tmp_word] and "/" not in tmp_word:
                            pos_dict_for_grammar_terminal[tmp_word].append(pos)
                    else:
                        tmp_word = word.lower()
                else:
                    tmp_word = word
                word_index = word_index + 1
                if "/" not in tmp_word:
                    for i in range(0, len(tmp_pattern_list)):
                        tmp_pattern_list[i] = tmp_pattern_list[i] + " " + tmp_word
                else:
                    branch_list = tmp_word.split("/")
                    for j in range(0, len(branch_list)):
                        branch_list[j] = branch_list[j].replace("-", " ")

                        original_words = branch_list[j].split()

                        new_original_words = []
                        for orig_word in original_words:
                            if "[" in orig_word or "<" in orig_word:
                                new_original_words.append(orig_word)
                            else:                               
                                new_original_words.append(orig_word.lower())
                        original_words = new_original_words

                        _, word_list_for_branch = preprocess(
                            vocab, branch_list[j], False
                        )                    
                        if original_words != []:
                            pos_list = penn_pos_to_wordnet_pos(
                                get_pos_for_sentence(
                                    states, A, B, word_list_for_branch, vocab)
                            )
                        tmp_branch = ""
                        for index in range(0, len(word_list_for_branch)):
                            if original_words[index].lower() in stopword_list:
                                lemma_word = ""
                            elif pos_list[index] in ["v", "n", "a", "r"] :
                                lemma_word = lemmatizer.lemmatize(
                                    original_words[index], pos_list[index]
                                )
                                if "[" not in lemma_word:
                                    if (
                                        pos_list[index]
                                        not in pos_dict_for_grammar_terminal[lemma_word]
                                    ):
                                        pos_dict_for_grammar_terminal[lemma_word].append(
                                            pos_list[index]
                                        )
                            else:
                                lemma_word = original_words[index]

                            tmp_branch = tmp_branch + " " + lemma_word
                        branch_list[j] = tmp_branch
                    new_tmp_pattern_list = combine_list(
                        tmp_pattern_list, branch_list, " ")
                    tmp_pattern_list = new_tmp_pattern_list
            tmp_patterns = tmp_patterns + tmp_pattern_list
        grammar_pieces_dict[tag] = tmp_patterns
    print(pos_dict_for_grammar_terminal)

    for tag, patterns in grammar_pieces_dict.items():
        patterns_padding = []
        for pattern in patterns:
            patterns_padding.append(pattern + " ")
        grammar_pieces_dict[tag] = patterns_padding

    # Chomsky Form
    # convert all terminal into dummy non-terminal if not in the original grammar
    dummy_dict = defaultdict()
    for tag, patterns in grammar_pieces_dict.items():
        pattern_index = 0
        for pattern in patterns:
            word_list = pattern.split()
            for word in word_list:
                if "[" not in word and "{" not in word and "<" not in word and len(word_list)!=1:

                    original = " " + word + " "
                    new = " [" + word + "] "
                    dummy_dict["[" + word + "]"] = [word]
                    grammar_pieces_dict[tag][pattern_index] = grammar_pieces_dict[tag][
                        pattern_index
                    ].replace(original, new)
            pattern_index = pattern_index + 1
        print(tag, grammar_pieces_dict[tag])
    grammar_pieces_dict.update(dummy_dict.copy())

    if cnf_option == True:
        # convert all grammars into binary form.
        additional_dict = defaultdict()
        for tag, patterns in grammar_pieces_dict.items():
            pattern_index = 0
            for pattern in patterns:
                word_list = pattern.split()
                len_pattern = len(word_list)
                previous_pattern = " " + word_list[0]
                previous_tag = word_list[0]

                for i in range(1, len_pattern):
                    word = word_list[i]
                    tmp_pattern = previous_pattern + " " + word
                    tmp_tag = previous_tag + "+" + word

                    if i == len_pattern - 1:
                        grammar_pieces_dict[tag][pattern_index] = tmp_pattern
                    else:
                        additional_dict[tmp_tag] = [tmp_pattern]
                        previous_pattern = " " + tmp_tag
                        previous_tag = tmp_tag
                pattern_index = pattern_index + 1
        grammar_pieces_dict.update(additional_dict.copy())
        
        idf_calculated_from_grammar, vocab_document_2 = get_idf_from_grammar_dict(
            grammar_pieces_dict, entry_point_dict, 0.001
        )


    # modify the code to show only a few words with the lowest idf

    idf_calculated_from_grammar, vocab_document_2 = get_idf_from_grammar_dict(
        grammar_pieces_dict, entry_point_dict, 0.001
    )
    lowest_grammar_idf = min(idf_calculated_from_grammar.values())

    # calculate weighting for semantic tags
    keys = []
    values = []
    for key, pattern_list in grammar_pieces_dict.items():
        semantic_grammar = Semantic_grammar(key, pattern_list, grammar_pieces_dict)
        patterns = semantic_grammar.patterns
        keys.append(key)
        vocab_document_2.append(key)
        tmp_idf_sum = 0

        if key[0] == "<":  # deal with positoin flexible components
            values.append(0)
            continue
        for pattern in patterns:
            words = pattern.split()
            for word in words:
                tmp_idf_sum = tmp_idf_sum + idf_calculated_from_grammar[word.lower()]
                if key == "[DesireAction]":
                    print(word, tmp_idf_sum)
        value = tmp_idf_sum / len(patterns)
        values.append(value)
        idf_calculated_from_grammar[key] = tmp_idf_sum / len(patterns)


    # calculate weighting for semantic tags using the grammar idf but the lowest pattern
    error_list = []
    keys = []
    values = []
    intent_taglist_dict = defaultdict(float)
    for key, pattern_list in grammar_pieces_dict.items():

        semantic_grammar = Semantic_grammar(key, pattern_list, grammar_pieces_dict)
        if "{" in key:
            intent_taglist_dict[key] = list(set(semantic_grammar.taglist))
        patterns = semantic_grammar.patterns
        keys.append(key)
        vocab_document_2.append(key)

        tmp_lowest_idf_sum = 100

        if key[0] == "<":  # deal with positoin flexible components
            values.append(0)
            continue
        for pattern in patterns:
            tmp_idf_sum = 0
            words = pattern.lower().split()
            for word in words:
                tmp_idf_sum = tmp_idf_sum + idf_calculated_from_grammar[word]
            if tmp_idf_sum < tmp_lowest_idf_sum:
                tmp_lowest_idf_sum = tmp_idf_sum
            if key == "{Greetings}":
                print(pattern, tmp_idf_sum, tmp_lowest_idf_sum)

        value = tmp_lowest_idf_sum
        values.append(value)
        # take mean

        idf_calculated_from_grammar[key] = tmp_lowest_idf_sum
        if value == 0:
            error_list.append(key)
            error_list = list(set(error_list))


    # calculate weighting for semantic tags using general idf instead of the grammar idf
    keys = []
    values = []
    for key, pattern_list in grammar_pieces_dict.items():

        semantic_grammar = Semantic_grammar(key, pattern_list, grammar_pieces_dict)
        patterns = semantic_grammar.patterns
        keys.append(key)
        vocab_document_2.append(key)

        tmp_lowest_idf_sum = 100

        if key[0] == "<":  # deal with positoin flexible components
            values.append(0)
            continue
        for pattern in patterns:
            tmp_idf_sum = 0
            words = pattern.lower().split()
            for word in words:
                if normalized_idf_dict[word] == 0:
                    _, tmp_tag_list = preprocess(vocab, word, False)
                    normalized_idf_dict[word] = normalized_idf_dict[tmp_tag_list[0]]
                    print(word, tmp_tag_list[0])

                tmp_idf_sum = tmp_idf_sum + normalized_idf_dict[word]
                if key in error_list:
                    print(word, normalized_idf_dict[word])
            if tmp_idf_sum < tmp_lowest_idf_sum:
                tmp_lowest_idf_sum = tmp_idf_sum
            if key == "{Greetings}":
                print(pattern, tmp_idf_sum, tmp_lowest_idf_sum)

        value = tmp_lowest_idf_sum
        values.append(value)
        # take mean

        normalized_idf_dict[key] = tmp_lowest_idf_sum
        if value == 0:
            error_list.append(key)
            error_list = list(set(error_list))

    return grammar_pieces_dict,idf_calculated_from_grammar,normalized_idf_dict,states,A,B,vocab,vocab_document_2,intent_taglist_dict,pos_dict_for_grammar_terminal,lowest_grammar_idf


def initialize_through_reading_files():
    # Read in all the data in advance
    normalized_idf_dict = defaultdict(
        float, read_dict_from_csvfile("normalized_idf_dict.csv", "float"))
    A = np.loadtxt("A.txt")
    B = np.loadtxt("B.txt")
    with open('states.data', 'rb') as filehandle:
        # store the data as binary data stream
        states = pickle.load(filehandle)
    print(states)

    return A,B,states

    

def initialize_A_B_matrix():
    return True

def initialize_semantic_grammars_and_grammars_idf():
    return True


def initialize_semantic_grammars_and_grammars_idf():    
    return True

def extract_intent_and_semantic_tags_from_result(result):
    item_index = 0
    top_match_intent = []
    match_intent = []
    tag_value_pair = {}
    best_parse = result[0]
    best_parse_pattern = best_parse[0]
    best_parse_backtrace = best_parse[1]
    for tag in best_parse_pattern.split(" "):
        if "{" in tag or "<" in tag:
            match_intent.append(tag)
    for key, value in best_parse_backtrace.items():
        if "[" in key and key[1].isupper() :
            semantic_tag_value = value.split("<-")[0].strip()                        
            if semantic_tag_value == "isnumeric":
                semantic_tag_value = value.split("<-")[1].strip()
            semantic_tag_value_processed = ""
            word_index = 0
            for word in semantic_tag_value.split():
                word_processed = word
                if "[" in word and word[1].islower():
                    word_processed = word.replace("[","").replace("]","")
                if word_index == 0:
                    semantic_tag_value_processed = word_processed
                else:
                    semantic_tag_value_processed = semantic_tag_value_processed + " " + word_processed
                word_index = word_index + 1 
            tag_value_pair[key] = semantic_tag_value_processed
           
    return match_intent,tag_value_pair

    def extract_intent_and_semantic_tags_from_result_for_matching_threshold(result):
        item_index = 0
        top_match_intent = []
        match_intent = []
        tag_value_pair = {}
        best_parse = result[0]
        best_parse_pattern = best_parse[0]
        best_parse_backtrace = best_parse[1]
        for tag in best_parse_pattern.split(" "):
            if "{" in tag :
                match_intent.append(tag)
        for key, value in best_parse_backtrace.items():
            if "[" in key or "<" in key:
                semantic_tag_value = value.split("<-")[0].strip()
                print("semantic_tag_value", semantic_tag_value)
                if semantic_tag_value == "isnumeric":
                    semantic_tag_value = value.split("<-")[1].strip()
                print("semantic_tag_value", semantic_tag_value)
                semantic_tag_value_processed = ""
                word_index = 0
                for word in semantic_tag_value.split():
                    word_processed = word
                    if "[" in word and word[1].islower():
                        word_processed = word.replace("[", "").replace("]", "")
                    if word_index == 0:
                        semantic_tag_value_processed = word_processed
                    else:
                        semantic_tag_value_processed = semantic_tag_value_processed + " " + word_processed
                    word_index = word_index + 1
                tag_value_pair[key] = semantic_tag_value_processed

        return match_intent, tag_value_pair


def method_evaluation_parsing_based_CKY_MED(
    vocab_document,
    idf_calculated_from_grammar,
    test_sentences,
    test_intent_labels,
    grammar_pieces_dict,
    vocab_for_pos,
    states,
    A,
    B,
    lowest_idf,
    intent_taglist_dict,
    pos_dict_for_grammar_terminal,
    insert_discount_factor=0.9,
    maximum_length_difference=4,
    relative_matching_threshold=0.3,
    beam_size=2,
    force_intent_matching=True,
    real_threshold_list=[1],
    ):
    wrong_classified_sentences = []
    wrong_labels = []
    num_threshold = len(real_threshold_list)
    num_correct_classification = num_threshold * [0]
    num_wrong_classification = num_threshold * [0]
    num_empty_entry = num_threshold * [0]
    num_correct_top = num_threshold * [0]
    num_wrong_top = num_threshold * [0]
    num_empty_top = num_threshold * [0]
    report_list = []
    run_time_list = []
    pred_labels_list = [[] for x in range(0, num_threshold)]
    num_sentences = len(test_sentences)

    for i in range(0, num_sentences):
        sentence = test_sentences[i]
        # print(sentence)
        result_list, runtime = semantic_grammar_parsing_general_idf_for_evaluation(
            sentence,
            lowest_idf,
            vocab_document,
            idf_calculated_from_grammar,
            grammar_pieces_dict,
            vocab_for_pos,
            states,
            A,
            B,
            intent_taglist_dict,
            pos_dict_for_grammar_terminal,
            single_word_beam_size=beam_size,
            beam_size=beam_size,
            insert_discount_factor=insert_discount_factor,
            maximum_length_difference=maximum_length_difference,
            stopword_list=[],
            relative_matching_threshold=relative_matching_threshold,
            force_intent_matching=force_intent_matching,
            real_threshold_list=real_threshold_list,
        )
        print(result_list)
        # tmp_list as result
        len_sentence = len(sentence)
        run_time_list.append(
            [
                "beam size",
                beam_size,
                "sentence_length",
                len_sentence,
                "runtime",
                runtime,
            ]
        )

        result_index = 0
        print("num: ", i, " result length:", len(result_list))
        for result in result_list:
            if len(result_list) == 1 and num_threshold != 1:
                for j in range(0, num_threshold):
                    num_empty_top[j] += 1
                    num_empty_entry[j] += 1
                for index in range(0, num_threshold):
                    pred_labels_list[index].append("{none}")
                break
            match_intent = []
            top_match_intent = []
            item_index = 0

            for item in result:
                for tag in item[0].split(" "):
                    if "{" in tag or "<" in tag:
                        match_intent.append(tag)
                        if item_index == 0:
                            top_match_intent.append(tag)
                item_index = item_index + 1
            previous_length = len(pred_labels_list[result_index])
            pred_labels_list[result_index].append(top_match_intent[0])
            after_length = len(pred_labels_list[result_index])
            # print(previous_length == after_length-1)

            if test_intent_labels[i] in top_match_intent:
                num_correct_top[result_index] += 1
            else:
                if top_match_intent != ["{none}"]:
                    num_wrong_top[result_index] += 1
                else:
                    num_empty_top[result_index] += 1

            if test_intent_labels[i] in match_intent:
                num_correct_classification[result_index] += 1

            else:
                #  print(
                #      "Sentence: {0}\n score: {1}\n best match: {2}\n match_intent: {3}\n true_intent: {4}\n back_trace:{5}\n".format(
                #          test_sentences[i],
                #          result[0][2],
                #          result[0][2],
                #          result[0][0],
                #          test_intent_labels[i],
                #          result[0][1],
                #      )
                #  )
                wrong_classified_sentences.append(test_sentences[i])
                wrong_labels.append(test_intent_labels[i])

                if match_intent != ["{none}"]:
                    num_wrong_classification[result_index] += 1
                else:
                    num_empty_entry[result_index] += 1
            result_index = result_index + 1
        print(num_correct_classification,
              num_wrong_classification, num_empty_entry)
        print(num_correct_top, num_wrong_top, num_empty_top)
        print(len(pred_labels_list[result_index - 1]))
    precision_top_list = []
    recall_top_list = []
    F1_measure_top_list = []
    precision_list = []
    recall_list = []
    F1_measure_list = []
    report_list = []
    result_index = 0

    print("pred_labels: {0}".format(pred_labels_list))

    for real_threshold in real_threshold_list:
        if (num_correct_top[result_index] + num_wrong_top[result_index]) == 0:
            precision_top = 0
        else:
            precision_top = num_correct_top[result_index] / (
                num_correct_top[result_index] + num_wrong_top[result_index]
            )
        recall_top = num_correct_top[result_index] / num_sentences
        if precision_top + recall_top == 0:
            F1_measure_top = 0
        else:
            F1_measure_top = (
                2 * (precision_top * recall_top) / (precision_top + recall_top)
            )

        if (
            num_correct_classification[result_index]
            + num_wrong_classification[result_index]
        ) == 0:
            precision = 0
        else:
            precision = num_correct_classification[result_index] / (
                num_correct_classification[result_index]
                + num_wrong_classification[result_index]
            )
        recall = num_correct_classification[result_index] / num_sentences
        if precision + recall == 0:
            F1_measure = 0
        else:
            F1_measure = 2 * (precision * recall) / (precision + recall)
        precision_top_list.append(precision_top)
        recall_top_list.append(recall_top)
        F1_measure_top_list.append(F1_measure_top)
        precision_list.append(precision)
        recall_list.append(recall)
        F1_measure_list.append(F1_measure)
        run_time_list
        print("precision_top_3:: {0}".format(precision))
        print("recall_top_3: {0}".format(recall))
        print("F1_measure_top_3: {0}".format(F1_measure))
        print("precision {0}".format(precision_top))
        print("recall: {0}".format(recall_top))
        print("F1_measure: {0}".format(F1_measure_top))
        print(
            metrics.confusion_matrix(
                test_intent_labels, pred_labels_list[result_index])
        )
        report = metrics.classification_report(
            test_intent_labels,
            pred_labels_list[result_index],
            digits=3,
            labels=list(set(test_intent_labels)),
            output_dict=True,
        )
        report_list.append(report)
        print(report)
        result_index = result_index + 1
    return (
        precision_top_list,
        recall_top_list,
        F1_measure_top_list,
        precision_list,
        recall_list,
        F1_measure_list,
        wrong_classified_sentences,
        wrong_labels,
        report_list,
        run_time_list,
    )


def get_single_result(vocab_document,
                      idf_calculated_from_grammar,
                      test_sentences,
                      test_intent_labels,
                      grammar_pieces_dict,
                      vocab,
                      states,
                      A,
                      B,
                      lowest_grammar_idf,
                      intent_taglist_dict,
                      pos_dict_for_grammar_terminal,
                      relative_matching_threshold,
                      beam_size):
    #real_threshold_list=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7],):
    (
        precision_top_list,
        recall_top_list,
        F1_measure_top_list,
        precision_list,
        recall_list,
        f1_list,
        wrong_classified_sentences,
        wrong_labels,
        report_list,
        runtime_list,
    ) = method_evaluation_parsing_based_CKY_MED(
        vocab_document,
        idf_calculated_from_grammar,
        test_sentences,
        test_intent_labels,
        grammar_pieces_dict,
        vocab,
        states,
        A,
        B,
        lowest_grammar_idf,
        intent_taglist_dict,
        pos_dict_for_grammar_terminal,
        insert_discount_factor=1.0,
        maximum_length_difference=100,
        relative_matching_threshold=relative_matching_threshold,
        beam_size=beam_size,
        force_intent_matching=True,
        real_threshold_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    )

    real_threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    #real_threshold_list=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7],
    precision_list_record_1 = precision_top_list
    recall_list_record_1 = recall_top_list
    index = 0
    record = []
    macro_avg_precision = []
    macro_avg_recall = []
    micro_avg_precision = []
    micro_avg_recall = []
    weighted_avg_precision = []
    weighted_avg_recall = []
    for report in report_list:
        print(real_threshold_list[index])
        print("macro avg", report["macro avg"])
        print("weighted avg", report["weighted avg"])
        record.append(
            [real_threshold_list[index], report["macro avg"], report["weighted avg"]]
        )
        macro_avg_precision.append(report["macro avg"]["precision"])
        macro_avg_recall.append(report["macro avg"]["recall"])
        micro_avg_precision.append(precision_list_record_1[index])
        micro_avg_recall.append(recall_list_record_1[index])
        weighted_avg_precision.append(report["weighted avg"]["precision"])
        weighted_avg_recall.append(report["weighted avg"]["recall"])
        index += 1

    print(precision_list_record_1)
    print(recall_top_list)
    method_name = "CKY-MED"
    relative_threshold = relative_matching_threshold
    with open(
        "record_data_for_"
        + method_name
        + "_"
        + relative_threshold
        + "_"
        + beam_size
        + ".txt",
        "w",
    ) as f:
        f.write(
            "threshold_list ="
            + str(real_threshold_list)
            + "\n"
            + "macro_avg_precision_"
            + method_name
            + "_"
            + relative_threshold
            + "_"
            + beam_size
            + "="
            + str(macro_avg_precision)
            + "\n"
            + "macro_avg_recall_"
            + method_name
            + "_"
            + relative_threshold
            + "_"
            + beam_size
            + "="
            + str(macro_avg_recall)
            + "\n"
            + "micro_avg_precision_"
            + method_name
            + "_"
            + relative_threshold
            + "_"
            + beam_size
            + "="
            + str(micro_avg_precision)
            + "\n"
            + "micro_avg_recall_"
            + method_name
            + "_"
            + relative_threshold
            + "_"
            + beam_size
            + "="
            + str(micro_avg_recall)
            + "\n"
            + "weighted_avg_precision_"
            + method_name
            + "_"
            + relative_threshold
            + "_"
            + beam_size
            + "="
            + str(weighted_avg_precision)
            + "\n"
            + "weighted_avg_recall_"
            + method_name
            + "_"
            + relative_threshold
            + "_"
            + beam_size
            + "="
            + str(weighted_avg_recall)
            + "\n"
            + "runtime_list_"
            + method_name
            + "_"
            + relative_threshold
            + "_"
            + beam_size
            + "="
            + str(runtime_list)
            + "\n"
            + str(record)
        )

    return True
