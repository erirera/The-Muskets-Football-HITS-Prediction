import json
d = json.load(open('dashboard/top_predictions.json'))
for p in d[:10]:
    print(f"{p['rank']:2}. {p['nationality']:20} {p['name']}")
