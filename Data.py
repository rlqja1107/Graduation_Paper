import pandas as pd
import pickle


class EpinionData(object):
    def __init__(self):
        self.n_user = config['n_user']
        self.n_item = config['n_item']
        self.path = config['path']

    def load_data(self, pickle = True):
        if pickle:
            with open(self.path+'/epinions_2.pickle', 'rb') as f:
                self.ratings = pickle.load(f)
        else:
            preprocess_json()
        self.trust = pd.read_table('.\\데이터\\Epinion\\network_trust.txt',
                              sep='\t', header=None, names=['믿는 자', 'trust', '믿음을 당하는자'])
            
    
    def preprocess_json(self):
        data = []
        with open(self.path+'/epinions_2.json', 'r') as f:
            for line in f:
                line = json.dumps(line)
                data.append(json.loads(line.replace("'","\"")))
        item = pd.DataFrame(data)
        
    def preprocess_data(self):
