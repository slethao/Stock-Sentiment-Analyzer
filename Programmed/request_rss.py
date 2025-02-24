import feedparser
import pandas

class InfoGathering():
    def __init__(self):
        self._urls_used = ["https://investor.nvidia.com/investor-resources/rss/default.aspx", # NVIDIA's Investor Relations RSS
                           "https://nvidianews.nvidia.com/rss", # NVIDIA's Newsroom RSS
                           "https://techcrunch.com/feed" # TechCrunch RSS
                           ]
    
    def pulls_data(self):
        """
        AI
        Hardware
        Enterprise
        """
        all_catagories = ["AI", "Hardware", "Enterprise"]
        all_content = {"Title": [], "Link": [], "Summary":[], "Published":[]}
        for url in self._urls_used:
            feed_found = feedparser.parse(url)
            #print(feed_found.status)
            for part in feed_found.entries:
                if f"{part.category}" in all_catagories:
                    all_content["Title"].append(part.title)
                    all_content["Link"].append(part.link)
                    all_content["Summary"].append(part.summary)
                    all_content["Published"].append(part.published)
                    # print(f"title: {part.title}")
                    # print(f"link: {part.link}")
                    # print(f"brief: {part.summary}")
                    # print(f"published: {part.published}")
        return all_content

    def load_to_csv(self, given_content):
        file_path = "Stock-Sentiment-Analyzer/Programmed/RSS Feeder/Bronze/Event.csv"
        content_dataframe = pandas.DataFrame(given_content)
        content_dataframe.to_csv(file_path, index = False)
        
