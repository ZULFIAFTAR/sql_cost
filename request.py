##REF : https://towardsdatascience.com/deploy-a-machine-learning-model-using-flask-da580f84e60c

import requests
url = 'http://localhost:5000/predict2'
r = requests.post(url,json={'sql_statement':'kalau ada sumur diladang'})
print(r.json())