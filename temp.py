import json
from datetime import datetime


with open('./lstm_nn/model_metadata.json', 'r+') as file:
    models = json.load(file)
    
    models.append({
        'model' : datetime.now().strftime("%d %b - %H %M"),
        'lookback' : 60,
        'train_test_split' : 0.8,
        'ticker' : 'AAPL',
        'start_date' : '2015-01-01',
        'end_date' : '2023-01-01',
        'initial_balance' : 10000,
        'epochs' : 100,
        'batch_size' : 32,
    })
    
    file.seek(0)
    
    json.dump(models, file, indent=4)
    