import pandas
import file_help_func as file
import Tensor_Model as tensor
import Conclusion as conclude


def main():
    # create six file variables to hold each prediction
    model_obj = tensor.TensorModel("Adj Close")
    file_one = "Programmed/NVIDIA_STOCK_PREDICT_AdjClose.csv"
    file_two = "Programmed/NVIDIA_STOCK_PREDICT_Close.csv"
    file_three = "Programmed/NVIDIA_STOCK_PREDICT_High.csv"
    file_four = "Programmed/NVIDIA_STOCK_PREDICT_Low.csv"
    file_five = "Programmed/NVIDIA_STOCK_PREDICT_Open.csv"
    file_six = "Programmed/NVIDIA_STOCK_PREDICT_Volume.csv"
    all_files = [file_one, file_two, file_three, file_four, file_five, file_six]
    all_attributes = ["Adj Close","Close","High","Low","Open","Volume"]
    for attribute in all_attributes:
        # then use the setter
        model_obj.set_group(attribute)
        # build
        # train
        # predict
        # write out all the tensor flow models
    
    # use the file library to merge the all files
    file.combining_files()

    # then calculate the results in the Conclusion class

if __name__ == "__main__":
    main()