import csv

with open('train.txt', encoding='utf-8') as txtfile:
    all_text = txtfile.read()
with open('train.csv', mode='w', encoding='utf-8') as csv_file:
    fieldnames = ['text']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'text': all_text})


with open('validation.txt', encoding='utf-8') as txtfile:
    all_text = txtfile.read()
with open('validation.csv', mode='w', encoding='utf-8') as csv_file:
    fieldnames = ['text']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'text': all_text})

print("created train.csv and validation.csv files")
