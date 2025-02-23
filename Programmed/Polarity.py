"""
Sentiment Analysis*opinion mining: 
*good bad neutral* 
"""
import matplotlib as mat
import pandas
import request_rss as request
import plot_transfer as image
import tensorflow

class Polarity:
    def __init__(self, price_change, daily_return, mov_dir):
        self._price_change = pandas.read_csv(price_change)["predict price change"].values
        self._daily_return = pandas.read_csv(daily_return)["Predicted daily return"].values
        self._mov_dir = pandas.read_csv(mov_dir)["predicted range"]

    def sentiment_results(self):
        """
        *use an and operator*
        positive sentiment
            price change > 0 *price increased*
            daily return > 0
            movement direction = "Up"
        negative sentiment
            price change < 0 (price decreased)
            daily return < 0
            movement direction = "Down"
        neutral sentiment
            price change = 0
            daily return = 0
            movement direction = "no change"
        create a bar char on what is happening
        """
        all_groups = ["Positive", "Negative", "Neutral"]
        sentiment_result = []
        for index in range(len(self._price_change)):
            if self._price_change[index] > 0 and self._daily_return[index] > 0: # price movement increase
                sentiment_result.append(1)
            elif self._price_change[index] < 0 and self._daily_return[index] < 0: # price movement decreased
                sentiment_result.append(-1)
            else: # no change movement direction
                sentiment_result.append(0)
        # create a bar graph here
        mat.pyplot.plot(range(len(self._price_change)), sentiment_result)
        mat.pyplot.yticks([-1, 0, 1], ["Negative", "Neutral", "Positive"])
        fig = mat.pyplot.gcf() # Get the current figure

        writer = tensorflow.summary.create_file_writer("Programmed/logs") # Create a writer 
        with writer.as_default():
            tensorflow.summary.image("matplotlib_plot_09", image.plot_to_image(fig), step=0)
    
    def predict_news_events(self):
        """
        external events to see if they 
        correlate with the obsered sentiment
        (api called here)
        """
        request_obj = request.InfoGathering()
        grab_data = request_obj.pulls_data() # works
        request_obj.load_to_csv(grab_data)
        

    # def predict_volume(self):
    #     """
    #     predict volume duing periods of positive 
    #     sentiment to find strong buying pressure
    #     bar graph
    #     """
    #     pass