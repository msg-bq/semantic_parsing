import json

# f = open("output.jsonl","r",encoding="utf-8")

json_data = []
with open("output.jsonl","r",encoding="utf-8") as f:
    for data in f:
        json_data.append(json.loads(data))

f1 = open("our_weather1.tsv","w",encoding="utf-8")
f1.write(f"domain\tutterance\tsemantic_parse\n")
for data1 in json_data:
    domain = "weather"
    question = data1["sentence"]
    answer = data1["result"]
    f1.write(f"{domain}\t{question}\t{answer}\n")