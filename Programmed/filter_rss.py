import csv
import re
import word_filter as word
import spacy as sp
import Isolation_Model as iso
import pandas
import Tensor_Model as ten

def re_expression(line):
    edited = line
    if "<p>" in line:
        edited = edited.replace("<p>", "")
        if "</p>" in edited:
            edited = edited.replace("</p>", "")
            if "[&" in edited:
                start = edited.index("[")
                edited = edited.replace(edited[start:start+9], "")
                if "&#" in edited:
                    edited = re.sub(r"&#\d+;", "", edited)
    return edited

def filter_bronze():
    file_path_one = "Programmed/RSS Feeder/Bronze/Event.csv"
    file_path_two = "Programmed/RSS Feeder/Silver/Event.csv"

    with open(file_path_one, "r") as raw_data:
        extra_line = 'All rights reserved. For personal use only'
        collected = []
        header = raw_data.readline()
        for line in raw_data.readlines():
            if extra_line not in line:
                collected.append(re_expression(line))
        
        with open(file_path_two, "w") as new_data:
            scribe = csv.writer(new_data)
            scribe.writerow(header.split("\n"))
            for line in collected:
                scribe.writerow(re_expression(line).lower().split("\n"))

def filter_silver():
    file_path_two = "Programmed/RSS Feeder/Silver/Event.csv"
    file_path_three = "Programmed/RSS Feeder/Gold/Event_Final.csv"
    file_path_four = "Programmed/RSS Feeder/Gold/Event_Final_02.csv"
    edited_tokens = []
    all_pre_reocrds = []
    model = sp.load("en_core_web_sm")

    with open(file_path_two, "r") as data:
        data = word.rid_of_stop_words(data.readlines())
        for line in data:
            token = word.tokenizer_for_words(" ".join(line)) # each sentence will be process one at a time
            edited_tokens.append(token)
            #@TODO upload tokens to csv.
        # step two spaCy (language model)
        for token in edited_tokens:
            txt = " ".join(token)
            pre_records = model(txt)
            all_pre_reocrds.append(pre_records)

        token_vector = word.vectorize_for_token(all_pre_reocrds) 
        sentence_vector = word.vectorize_for_sentence(all_pre_reocrds)  
        encoded_01 = word.group_encoding(sentence_vector)
        encoded_02 = word.group_encoding(token_vector)
        group_one = word.formt_to_csv(encoded_01, file_path_three)
        group_two = word.formt_to_csv(encoded_02, file_path_four)
        # isolation forest
        spliting_tool = iso.IsolationModel("feature_1", file_path_three)
        learning_model = ten.TensorModel("group", file_path_three)
        for group in group_one:
            given_df = pandas.DataFrame(group_one)
            spliting_tool.set_x_value(group, given_df.columns)
            anomaly_found = spliting_tool.anomaly_results()
            print(anomaly_found)
        print("end of one csv")
        spliting_tool.set_data_used(file_path_four)
        for group in group_two:
            given_df = pandas.DataFrame(group_two)
            spliting_tool.set_x_value(group, given_df.columns)
            anomaly_found = spliting_tool.anomaly_results()
        print("end of the second csv")


def main():
    filter_bronze()
    filter_silver()

main()
