import os
import sys
import argparse
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess(text):
    """Preprocesses the text by removing non-alphabetic characters, stop words, and stemming the words."""
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    tokens = word_tokenize(text.lower())
    tokens = [stemmer.stem(token) for token in tokens if token.isalpha() and token not in stop_words]

    return ' '.join(tokens)


def calculate_similarity(text1, text2):
    """Calculates the cosine similarity score between two texts using TF-IDF."""
    texts = [preprocess(text1), preprocess(text2)]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts)

    return cosine_similarity(tfidf)[0][1]


def main():
    parser = argparse.ArgumentParser(description='Plagiarism Checker')
    parser.add_argument('file1', type=str, help='the path of the first file')
    parser.add_argument('file2', type=str, help='the path of the second file')
    args = parser.parse_args()

    with open(args.file1, 'r') as f:
        text1 = f.read()

    with open(args.file2, 'r') as f:
        text2 = f.read()

    similarity_score = calculate_similarity(text1, text2)

    if similarity_score > 0.8:
        print('The texts are highly similar. Potential plagiarism detected.')
        sys.exit(0)
    else:
        print('The texts are not similar. No plagiarism detected.')
        sys.exit(0)
   
    if similarity_score == 1:
        print("Congratulations! You've found me finally :). But I think you won't stay here long, ah it doesn't matter, it was great meeting you. Please come again if you have time.")
        sys.exit(0)


if __name__ == '__main__':
    main()
