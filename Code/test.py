
import os
from urllib import request
import string
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.lm import MLE, Laplace
from nltk.lm.preprocessing import pad_both_ends, flatten, padded_everygram_pipeline
from nltk.util import ngrams




nltk.download('punkt')
printStats = True


def loadBookResource(url):
    # Get Frankenstein eBook
    book = request.urlopen(url).read().decode('utf8', errors='ignore')

    # Get first chapter, Beautiful Soup would have been useful to remove all the tags.
    text = book.split("Chapter 1\r\n\r\n")[1].split("Chapter 2\r\n\r\n")[0]
    
    return text

def createLanguageModel(n_parameter, data):        

    text_tokenized = [word_tokenize(sentence) for sentence in data]
    text_padded = [list(pad_both_ends(sentence_tokenized, n=n_parameter))
                for sentence_tokenized in text_tokenized]

    if printStats:
        print(f"Number of sentences: {len(data)}")
        n_tokens = 0
        for sentence in text_tokenized:
            n_tokens += len(sentence)
        print(f"Number of tokens: {n_tokens}")
        print(f"Average number of tokens per sentence: {n_tokens/len(data)}")

        flat_text_padded = list(flatten(text_padded))
        unigrams = flat_text_padded
        bigrams = list(ngrams(flat_text_padded, 2))
        trigrams = list(ngrams(flat_text_padded, 3))

        print(f"Most frequent unigrams:\n{nltk.FreqDist(unigrams).most_common(10)}")
        print(f"Most frequent bigrams:\n{nltk.FreqDist(bigrams).most_common(10)}")
        print(f"Most frequent trigrams:\n{nltk.FreqDist(trigrams).most_common(10)}")


    # Create language model:
    # Language model
    corpus, vocab = padded_everygram_pipeline(n_parameter, text_padded)
    # lm = MLE(N_PARAM)  # non-smoothed
    languageModel = Laplace(n_parameter)  # smoothed
    languageModel.fit(corpus, vocab)
    print(list(languageModel.vocab))
    return languageModel
    
def tokenizeText(text):
    text_string = text.lower()

    # Remove punctuation
    string.punctuation = string.punctuation + '“' + '”' + '-' + '’' + '‘' + '—'
    string.punctuation = string.punctuation.replace('.', '')  # keep "." so that can split sentences with NLTK
    text_filtered = "".join([char for char in text_string if char not in string.punctuation])
    text_sentences = sent_tokenize(text_filtered)
    return text_sentences
    
    

def calculatePerplexity(text, languageModel, N_PARAM):
    # **Lower perplexity = more chance of plagiarism**
    # **Higher perplexity = more chance of having generated text**
    # We can do this whole process again with the MLE language model (`lm = MLE(N_PARAM)` instead of `lm = Laplace(N_PARAM)`) to see if a non-smoothed language model produces different perplexity results.
    text_tokenized = nltk.word_tokenize(text)
    text_padded = list(pad_both_ends(text_tokenized, n=N_PARAM))
    text_ngrams = list(ngrams(text_padded, N_PARAM))
    print(text_ngrams)
    print(f"Perplexity: {languageModel.perplexity(text_ngrams)}")

#sentiment analysis:
def processReviews():
    paths_positive = []
    for file_path in os.listdir("./data/positive"):
        if file_path.endswith(".txt"):
            paths_positive.append(f"./data/positive/{file_path}")

    paths_negative = []
    for file_path in os.listdir("./data/negative"):
        if file_path.endswith(".txt"):
            paths_negative.append(f"./data/negative/{file_path}")

    positive_text = ''
    for file_path in paths_positive:
        file_text = open(file_path, encoding="utf8").read()
        positive_text += ' ' + file_text
    print(positive_text)

    negative_text = ''
    for file_path in paths_negative:
        file_text = open(file_path, encoding="utf8").read()
        negative_text += ' ' + file_text
    print(negative_text)



def main():
    N_PARAM = 2 

    data = loadBookResource("http://www.gutenberg.org/files/84/84-0.txt")
    tokenized_text = tokenizeText(data)
    languageModel = createLanguageModel(N_PARAM, tokenized_text)


    print(languageModel.generate(10, random_seed=1))



    # max ngram length of the language model
    calculatePerplexity("several months passed in this manner", languageModel, N_PARAM)


if __name__ == "__main__":
    main()