import requests
import csv
import os

""""
look into DigitalRank api (similarweb)
and create an enivormental variable that will be gitirgonred

1. Create a free Similarweb account.
2. Generate a personalized API key.
3. Use the API key to access the data.
"""

class CallAPI():
    def __init__(self, info):
        self.__api_key = os.getenv("STOCK_API")
        self.__url = f"https://api.similarweb.com/v1/similar-rank/top-sites?api_key={self.__api_key}&limit={50}"
        #self.__query = info
        
    def load_to_csv(self):
        current_url = self.__url
        request = requests.get(current_url)
        collected_data = csv.reader(request.iter_lines(decode_unicode= True))
        with open("Programmed/Event.csv", 'w') as events:
            loader = csv.writer(events)
            for line in collected_data:
                loader.writerow(line)
