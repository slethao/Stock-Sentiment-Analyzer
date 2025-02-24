import file_help_func as file
import Tensor_Model as tensor
import Conclusion as conclude
import Polarity as polar
import FindingStrategy as plan
import cleaning_data as clean
import Database as db
import filter_rss as rss
import tensorflow

def main():
    # create six file variables to hold each prediction
    # display it on tensor board
    clean.main()
    model_obj = tensor.TensorModel("Adj Close", "Stock-Sentiment-Analyzer/Programmed/Standard Filter/Gold/NVIDIA_STOCK_03.csv")
    file_one = "Stock-Sentiment-Analyzer/Programmed/Predicted Data/NVIDIA_STOCK_PREDICT_AdjClose.csv"
    file_two = "Stock-Sentiment-Analyzer/Programmed/Predicted Data/NVIDIA_STOCK_PREDICT_Close.csv"
    file_three = "Stock-Sentiment-Analyzer/Programmed/Predicted Data/NVIDIA_STOCK_PREDICT_High.csv"
    file_four = "Stock-Sentiment-Analyzer/Programmed/Predicted Data/NVIDIA_STOCK_PREDICT_Low.csv"
    file_five = "Stock-Sentiment-Analyzer/Programmed/Predicted Data/NVIDIA_STOCK_PREDICT_Open.csv"
    file_six = "Stock-Sentiment-Analyzer/Programmed/Predicted Data/NVIDIA_STOCK_PREDICT_Volume.csv"
    all_files = [file_one, file_two, file_three, file_four, file_five, file_six]
    all_attributes = ["Adj Close","Close","High","Low","Open","Volume"]
    file_counter = 0

    """
    NOTE to inspect: tensorboard --inspect --logdir Programmed/logs/my_model0

    rm -rf ./logs

    try:
        tensorboard --logdir Programmed/logs/

    All current graphs:
        -> epoch_learning_rate
        -> epoch_loss
        -> evaluation_loss_vs_iterations 
        -> matplotlib plot   
    """

    for attribute in all_attributes:
        # then use the setter
        model_obj.set_group(attribute)
        # build
        model_obj.build_model()
        # train
        tb_cb = tensorflow.keras.callbacks.TensorBoard(log_dir=f"Stock-Sentiment-Analyzer/Programmed/logs/my_model{file_counter}")
        trained = model_obj.train_model("Stock-Sentiment-Analyzer/Programmed/Standard Filter/Gold/NVIDIA_STOCK_03.csv")
        # evaluate
        model_obj.evaluate_model(trained, tb_cb) # train model parameter needed
        # predict
        # model_obj.predict_model(trained, all_files[file_counter], tb_cb) # train model and fiel path parameter needed
        model_obj.predict_model(trained, all_files[file_counter]) # train model and fiel path parameter needed
        file_counter += 1
    # use the file library to merge the all files
    file.combining_files()

    # then calculate the results in the Conclusion class
    data_stored = "Stock-Sentiment-Analyzer/Programmed/Predicted Data/OVERALL_PREDICTION.csv"
    outcome = conclude.Conclusion(data_stored)
    
    outcome.price_change() # uncomment later (create a chart to put on tensor board)
    outcome.price_move()
    
    # connect the prediction data to the conclusions made 
    outcome.price_cal_with_predict(data_stored)
    outcome.range_cal_with_predict(data_stored)
    outcome.daily_return_with_predict(data_stored)
    outcome.daily_mov_with_predict("Stock-Sentiment-Analyzer/Programmed/Predicted Data/PREDICTED_PRICE_CHANGE.csv")
    outcome.cal_mse(data_stored)
    outcome.cal_rmse("Stock-Sentiment-Analyzer/Programmed/Calculations/MEAN_SQUARE_ERROR.csv")
    outcome.cal_r_square(data_stored, "Close")
    outcome.cal_r_square(data_stored, "Adj Close")
    plan_obj = plan.Strategies(data_stored)
    plan_obj.recomendation()

    # polarity on the results 
    polar_obj = polar.Polarity("Stock-Sentiment-Analyzer/Programmed/Predicted Data/PREDICTED_PRICE_CHANGE.csv",
                               "Stock-Sentiment-Analyzer/Programmed/Predicted Data/PREDICTED_DAILY_RETURN.csv", 
                               "Stock-Sentiment-Analyzer/Programmed/Predicted Data/PREDICTED_PRICE_RANGE.csv") # price change, daily return, mov dir
    polar_obj.sentiment_results() # graph!!!!
    polar_obj.predict_news_events() # works
    # polar_obj.predict_volume()

    # store into a database (pending)
    db.main()
    print("the end, yay")
    rss.main()
    #@TODO tensrobaord here
    print("the end, yay two")

    # # include unit test pytest


if __name__ == "__main__":
    main()
#NOTE put this repo on Kaggle