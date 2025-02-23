import json
import os

class Request():
    def __init__(self):
        self._path = os.path.expanduser("~/.kaggle/kaggle.json")
        self._username = "KAGGLE_USERNAME"
        self._key = "KAGGLE_KEY"
    
    def get_request(self):
        if os.path.exists(self._path):
            try:
                with open(self._path, 'r') as json_file:
                    kaggle_cred = json.load(json_file)
                return kaggle_cred
            except FileNotFoundError:
                return f".json file is not found at {self._path}"
                
            except json.JSONDecodeError:
                return f"Invalid .json file in {self._path}"
                
        elif self._username in os.environ and self._key in os.environ: # if file don't exist
            # check if enviroment variables are set
            kaggle_cred = {
                "user": os.environ[self._username],
                "key": os.environ[self._key]
            }

        # if enviroment variables are set not set and file not there
        else:
            return "Please create a Kaggle API token and store in ~/.kaggle\nReference the Kaggle API documentation for information."

# obj = Request()
# print(obj.get_request())
# print("this works")
#TODO commmit things into the repo

