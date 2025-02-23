import re
import string
import contractions as con
import numpy
import pandas
from tensorflow.keras.preprocessing.text import Tokenizer

def rid_of_stop_words(dataset):
    stop_list = ["\n", "the", "to", "a", "", "had", "says", "of", "be", 
                 "\'", "\"", "with", "its", "an", "this", "that", "my", 
                 "mine", "myself", "you", "your", "yours", "yourself", 
                 "he", "him", "his", "himself", "she", "her", "hers", 
                 "herself", "it", "its", "itself", "we", "us", "our", 
                 "ours", "ourselves", "they", "them", "their", "theirs",
                 "themselves", "so", "is"]
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
    #@TODO: make a csv here
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
    #@TODO: make a csv here
    return word_arry    

def vectorize_for_token(overall_data):
    # NOTE tow list compressions
    all_tok = []
    for data in overall_data: # tokens
        all_token_v = [token.vector for token in data]
        all_tok += all_token_v
    #@TODO: make a csv here
    # print(all_tok)
    return all_tok

def vectorize_for_sentence(overall_data):
    all_sentence_v = []
    sentence_v = []
    for data in overall_data: 
        for token in data: 
            if token.has_vector:
                all_sentence_v.append(token.vector)
        if all_sentence_v:
            a_sen_vec = sum(all_sentence_v)/len(all_sentence_v)
            sentence_v.append(a_sen_vec)
    #@TODO: make a csv here
    # print(sentence_v)        
    return sentence_v

def group_encoding(sentence_vector, ):
    AI_words = ["artificial intelligence", "ai", "machine learning", "deep learning",
                "neural network", "algorithm", "model", "training", "prediction",
                "classification", "regression", "clustering", "natural language processing",
                "nlp", "computer vision","robotics","ml", "deep learning model", 
                "artificial neural network", "convolutional neural network", 
                "cnn", "recurrent neural network", "rnn", "transformer", "generative ai",
                "large language model", "llm", "reinforcement learning", "supervised learning",
                "unsupervised learning", "data science", "analytics", "cognitive computing",
                "intelligent systems", "expert systems", "chatbot", "ai ethics", "bias detection",
                "explainable ai", "axi", "system"]
    Hardware = ["cpu", "gpu", "ram", "memory", "storage", "disk", "ssd", "hard drive",
                "motherboard", "chip", "processor", "server", "computer", "device",
                "component", "circuit", "transistor", "semiconductor", "hardware",
                "electronics", "computing", "system", "architecture", "peripheral",
                "network", "router", "switch", "cable", "device driver", "firmware",
                "hardware acceleration", "hardware acceleration", "graphics card",
                "sound card", "network card", "i/o", "input/output"]
    Enterprise = ["business", "company", "organization", "corporation", "enterprise",
                  "management", "strategy", "marketing", "strategy", "marketing", "sales",
                  "finance", "accounting", "human resources", "hr", "information technolgy",
                  "it", "software", "cloud computing", "data center", "networking", "security",
                  "solutions", "platform", "service", "product", "client", "customer", "market",
                  "industry", "innovation", "growth", "profit", "revenue", "investment", "consulting",
                  "supply chain", "logistics", "e commerce", "e", "commerce", "b2b", "b2c", "saas",
                  "crm", "erp", "analytics", "data", "business intelligence", "automation", 
                  "digital transformation"]
    groups = {"AI": [1,0,0,0], "Hardware": [0,1,0,0], "Enterprise": [0,0,1,0], "General": [0,0,0,1]}
    hot_ones = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,1,0], [0,0,0,1]]
    all_encode = []

    for sentence_vector, hot_ones in zip(sentence_vector, hot_ones):
        combine_vectors = numpy.concatenate((sentence_vector, hot_ones))
        all_encode.append(combine_vectors)
    #@TODO: make a csv here

    print(numpy.array(all_encode))
    return numpy.array(all_encode)

def formt_to_csv(encoded, file_path):
    groups = [f"feature_{i}" for i in range(encoded.shape[1]-4)]+ [f"category_{i}" for i in range(4)]
    encoded = pandas.DataFrame(encoded, columns=groups)
    encoded.to_csv(file_path, index=False)
    return groups