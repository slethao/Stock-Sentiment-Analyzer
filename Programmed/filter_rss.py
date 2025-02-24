import csv
import re
import word_filter as word
import spacy as sp
import Isolation_Model as iso
import pandas
import Tensor_Model as ten
import numpy
import tensorflow

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
    file_path_one = "Stock-Sentiment-Analyzer/Programmed/RSS Feeder/Bronze/Event.csv"
    file_path_two = "Stock-Sentiment-Analyzer/Programmed/RSS Feeder/Silver/Event.csv"

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
    file_path_two = "Stock-Sentiment-Analyzer/Programmed/RSS Feeder/Silver/Event.csv"
    file_path_three = "Stock-Sentiment-Analyzer/Programmed/RSS Feeder/Gold/Event_Final.csv"
    file_path_four = "Stock-Sentiment-Analyzer/Programmed/RSS Feeder/Gold/Event_Final_02.csv"
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
        sentence_anmol = []
        token_anmol = []
        for group in group_one:
            given_df = pandas.DataFrame(group_one)
            spliting_tool.set_x_value(group, given_df.columns)
            anomaly_found = spliting_tool.anomaly_results()
            sentence_anmol.append(anomaly_found)
            print(anomaly_found)
        final_sen = numpy.array(sentence_anmol)
        sen_anm_df = pandas.DataFrame(final_sen, columns=["feature_0", "feature_1", "feature_2", "feature_4",
                                      "feature_5"])
        sen_anm_df.to_csv("Stock-Sentiment-Analyzer/Programmed/Predicted Data/SENTENCE_TOKEN_AMONOLIES.csv", index=False)
        spliting_tool.set_data_used(file_path_four)
        for group in group_two:
            given_df = pandas.DataFrame(group_two)
            spliting_tool.set_x_value(group, given_df.columns)
            anomaly_found = spliting_tool.anomaly_results()
            token_anmol.append(anomaly_found)
        final_tok = numpy.array(token_anmol)
        tok_anm_df = pandas.DataFrame(final_tok, columns=["feature_0", "feature_1", "feature_2", "feature_4",
                                                          "feature_5"])
        
        summary_writer = tensorflow.summary.create_file_writer("Programmed/logs")
        with summary_writer.as_default():
            for group in tok_anm_df.columns:
                tok_anm_df_series = tok_anm_df[group]
                for time_point, is_event in tok_anm_df_series.items():
                    step = time_point
                    tensorflow.summary.scalar(f"external_events/{group}", is_event, step=step) 

                tensorflow.summary.histogram(f"external_events_histogram/{group}", tok_anm_df_series.values, step=0) 


        tok_anm_df.to_csv("Stock-Sentiment-Analyzer/Programmed/Predicted Data/Token_TOKEN_AMONOLIES.csv", index=False)
        learning_model_01 = ten.TensorModel(sen_anm_df.columns, "Stock-Sentiment-Analyzer/Programmed/Predicted Data/SENTENCE_TOKEN_AMONOLIES.csv")
        learning_model_02 = ten.TensorModel(tok_anm_df.columns, "Stock-Sentiment-Analyzer/Programmed/Predicted Data/Token_TOKEN_AMONOLIES.csv")
        # @TODO: build, compile and train then you are done!!
        # build
        learning_model_01.build_model_other()
        learning_model_02.build_model_other()


def main():
    filter_bronze()
    filter_silver()