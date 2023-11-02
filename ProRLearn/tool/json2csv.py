import pandas as pd
import json

# JSON data
non_data = pd.read_json('non-vulnerables.json')

vul_data = pd.read_json('vulnerables.json')

non_data = non_data['code']
vul_data = vul_data['code']
non_data = pd.DataFrame(non_data)
non_data['label'] = 0
vul_data = pd.DataFrame(vul_data)
vul_data['label'] = 1
# print DataFrame
data = pd.concat([non_data, vul_data])

data = data.sample(frac=1, random_state=42)
data.to_csv('./reveal.csv',index = False)