"""
Sentiment Analysis*opinion mining,: 
good bad neutral* 
"""
class Polarity:
    def __init__(self, price_change, daily_return, mov_dir):
        self._price_change = price_change
        self._daily_return = daily_return
        self._mov_dir = mov_dir

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
        pass
    
    def predict_news_events(self):
        """
        predict volume duing periods of positive sentiment to find strong buying pressure
        bar graph
        """
        pass

    def predict_volume(self):
        """
        external events to see if they correlate with the obsered sentiment
        (api called here)
        """
        pass

