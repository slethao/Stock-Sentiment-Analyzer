import spacy as sp
import re
import string
import contractions as con
from tensorflow.keras.preprocessing.text import Tokenizer

def rid_of_stop_words(dataset):
    stop_list = ["\n", "the", "to", "a", "", "had", "says", "of", "be", 
                 "\'", "\"", "with", "its", "an", "this", "that", "my", 
                 "mine", "myself", "you", "your", "yours", "yourself", 
                 "he", "him", "his", "himself", "she", "her", "hers", 
                 "herself", "it", "its", "itself", "we", "us", "our", 
                 "ours", "ourselves", "they", "them", "their", "theirs",
                 "themselves"]
    all_words_arry = []
    for record in dataset:
            record_parts = record.split(",")
            for parts in record_parts:
                all_words_arry.append(parts.split(" "))
    
    for word_arry in all_words_arry:
        for word in word_arry:
            if word in stop_list:
                word_arry.remove(word)
        if word_arry == []:
            all_words_arry.remove(word_arry)
    return all_words_arry

def tokenizer_for_words(line):
    # puntucation
    words = []
    words_made = []
    for word in line:
        word = con.fix(word)
        words.append(word) # contractions created here

    for word in words:
        for character in string.punctuation:
            word = word.replace(character, f" {character} ")
        words_made.extend(word.split())
    word_arry = [word for word in words_made if word != " "]
    word_arry = re.findall(r"[\w\-']+|[.,!?;]|\d+(?:\.\d+)?", line)
    return word_arry
    
    # Specialized Financial Terms
    # Handling Company Names and Tickers

def vectorize_for_words():
    pass