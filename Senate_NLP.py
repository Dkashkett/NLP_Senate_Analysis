import json
import re
import sys
import gzip
import codecs
import string
from math import inf, log2
from collections import Counter
from spacy.lang.en import English
import re

# The following module is optional but useful for debugging
from traceback_with_variables import activate_by_import


from tqdm import tqdm

def read_and_clean_lines(infile):
    print("\nReading and cleaning text from {}".format(infile))
    lines = []
    with gzip.open(infile, "rt") as f:
        for line in f:
            json_data = json.loads(line)
            if json_data["chamber"] == "Senate":
                # parse json
                text = json_data["text"]
                # filter the text element
                filtered_text = text.replace("\t", "").replace("\n", "").replace("\r", "")
                # append filtered lines to the lines list
                lines.append(json_data["party"] + "\t" + filtered_text)

    return lines


# Input: lines containing <party> TAB <text>
# Writes just the text to outfile
def write_party_speeches(lines, outfile, party_to_write):
    print("{} speeches being written to {}".format(party_to_write, outfile))
    with open(outfile, "w") as f:
        for line in tqdm(lines):
            party, text = line.split("\t")
            if party == party_to_write:
                f.write(text + "\n")


# Read a set of stoplist words from filename, assuming it contains one word per line
# Return a python Set data structure (https://www.w3schools.com/python/python_sets.asp)
def load_stopwords(filename):
    stopwords = []
    with open(filename, "r") as f:
        for line in f:
            # split on the new line
            word = line.split("\n")[0]
            stopwords.append(word)
    return set(stopwords)


# Take a list of string tokens and return all ngrams of length n,
# representing each ngram as a list of  tokens.
# E.g. ngrams(['the','quick','brown','fox'], 2)
# returns [['the','quick'], ['quick','brown'], ['brown','fox']]
# Note that this should work for any n, not just unigrams and bigrams
def ngrams(tokens, n):
    # Returns all ngrams of size n in sentence, where an ngram is itself a list of tokens
    ngrams = []
    # im going to use a sliding window function, left and right are the edges of the window
    left = n
    right = 0
    # keep sliding the window until it reaches the end of the tokens list
    while right < len(tokens) - 1:
        # check that the length of the window is equal to n
        if len(tokens[right:left]) == n:
            ngrams.append(tokens[right:left])
        # move the window forward by 1 place
        right += 1
        left += 1
    return ngrams


def filter_punctuation_bigrams(ngrams):
    # Input: assume ngrams is a list of ['token1','token2'] bigrams
    # Removes ngrams like ['today','.'] where either token is a punctuation character
    # Returns list with the items that were not removed
    punct = string.punctuation
    return [
        ngram for ngram in ngrams if ngram[0] not in punct and ngram[1] not in punct
    ]


def filter_stopword_bigrams(ngrams, stopwords):
    # Input: assume ngrams is a list of ['token1','token2'] bigrams, stopwords is a set of words like 'the'
    # Removes ngrams like ['in','the'] and ['senator','from'] where either word is a stopword
    # Returns list with the items that were not removed

    # using a filter to test that there is no intersection between the stopwords and the ngram
    filtered_ngrams = list(filter(lambda x: stopwords.intersection(x) == set(), ngrams))

    return filtered_ngrams


def normalize_tokens(tokenlist):
    # Input: list of tokens as strings,  e.g. ['I', ' ', 'saw', ' ', '@psresnik', ' ', 'on', ' ','Twitter']
    # Output: list of tokens where
    #   - All tokens are lowercased
    #   - All tokens starting with a whitespace character have been filtered out
    #   - All handles (tokens starting with @) have been filtered out
    #   - Any underscores have been replaced with + (since we use _ as a special character in bigrams)

    # defined a function to clean each individual token
    cleaned_tokens = []
    for token in tokenlist:
        token = str(token)
        token = token.lower().replace("_", "+")
        if (token.startswith(" ") == False) and (token.startswith("@") == False):
            cleaned_tokens.append(token)
    return cleaned_tokens


# Given a counter containing underscore-separated bigrams (e.g. "united_states") and their counts,
# returns a counter with marginal counts for the unigram tokens for either the 1st or 2nd position,
# passed in as param position (0 or 1).
# For position 0:  freq(a,.) = sum over b freq(a_b)
# For position 1:  freq(.,b) = sum over a freq(a_b)
def get_unigram_counts(bigram_counter, position):
    unigram_counter = Counter()
    for bigramstring in bigram_counter:
        tokens = bigramstring.split("_")
        if tokens[position] in unigram_counter.keys():
            unigram_counter[tokens[position]] += bigram_counter[bigramstring]
        else:
            unigram_counter[tokens[position]] = bigram_counter[bigramstring]
    return unigram_counter


def collect_bigram_counts(lines, stopwords, remove_stopword_bigrams=False):
    # Input lines is a list of raw text strings, stopwords is a set of stopwords
    #
    # Create a bigram counter
    # For each line:
    #   Extract all the bigrams from the line
    #   If remove_stopword_bigrams is True:
    #     Filter out any bigram where both words are stopwords
    #   Increment the count for each bigram
    # Return the counter
    #
    # In the returned counter, the bigrams should be represented as string tokens containing underscores.
    #
    if remove_stopword_bigrams:
        print("Collecting bigram counts with stopword-filtered bigrams")
    else:
        print("Collecting bigram counts with all bigrams")

    # Initialize spacy and an empty counter
    print("Initializing spacy")
    nlp = English(
        parser=False
    )  # faster init with parse=False, if only using for tokenization
    counter = Counter()

    # Iterate through raw text lines
    for line in tqdm(lines):

        # Call spacy and get tokens
        tokens = nlp(line)
        # Normalize
        normalized_tokens = normalize_tokens(tokens)
        # Get bigrams
        bigrams = ngrams(normalized_tokens, 2)
        # Filter out bigrams where either token is punctuation
        filtered_bigrams = filter_punctuation_bigrams(bigrams)
        # Optionally filter bigrams that are both stopwords
        if remove_stopword_bigrams:
            filtered_bigrams = filter_stopword_bigrams(filtered_bigrams, stopwords)
        # Increment bigram counts
        for item in filtered_bigrams:
            bigram = item[0] + "_" + item[1]
            if bigram in counter.keys():
                counter[bigram] += 1
            else:
                counter[bigram] = 1

    return counter

def print_sorted_items(dict, n=10, order="ascending"):
    if order == "descending":
        multiplier = -1
    else:
        multiplier = 1
    ranked = sorted(dict.items(), key=lambda x: x[1] * multiplier)
    for key, value in ranked[:n]:
        print(key, value)


################################################################
# Main
################################################################

# Hard-wired variables
# input_speechfile   = "./speeches2020.jsonl.gz"
input_speechfile = "./speeches2020_jan_to_jun.jsonl.gz"
text_dems = "./speeches_dem.txt"
text_reps = "./speeches_rep.txt"
stopwords_file = "./mallet_en_stoplist.txt"
min_freq_for_pmi = 5
topN_to_show = 50


def main():

    # Read in the stopword list
    stopwords = load_stopwords(stopwords_file)

    # Read input speeches as json, and create one file for each party containing raw text one speech per line.
    # Effectively this is creating a corpus with two subcorpora, one each for Democratic and Republican speeches.
    print("\nProcessing text from input file {}".format(input_speechfile))
    lines = read_and_clean_lines(input_speechfile)

    print("\nWriting Democrats' speeches to {}".format(text_dems))
    write_party_speeches(lines, text_dems, "Democrat")

    print("\nWriting Republicans' speeches to {}".format(text_reps))
    write_party_speeches(lines, text_reps, "Republican")

    
    print("\nGetting Dem unigram and bigram counts")
    with open(text_dems) as f:
        dem_speeches = f.readlines()
    dem_bigram_counts = collect_bigram_counts(dem_speeches, stopwords, True)
    dem_unigram_w1_counts = get_unigram_counts(dem_bigram_counts, 0)
    dem_unigram_w2_counts = get_unigram_counts(dem_bigram_counts, 1)
    print("\nTop Dem bigrams by frequency")
    print_sorted_items(dem_bigram_counts, topN_to_show, "descending")

    print("\nGetting Rep unigram and bigram counts")
    with open(text_reps) as f:
        rep_speeches = f.readlines()
    rep_bigram_counts = collect_bigram_counts(rep_speeches, stopwords, True)
    rep_unigram_w1_counts = get_unigram_counts(rep_bigram_counts, 0)
    rep_unigram_w2_counts = get_unigram_counts(rep_bigram_counts, 1)
    print("\nTop Rep bigrams by frequency")
    print_sorted_items(rep_bigram_counts, topN_to_show, "descending")


if __name__ == "__main__":
    main()
