from collections import Counter
from transformers import BertModel, BertTokenizer

def word_tokenize(text):
    return text.split()


with open("./Dscan_result/data/sim_hash_result.csv", encoding='utf-8', mode='r') as fin:
    text = fin.read()

text = [w for w in word_tokenize(text.lower())]

# print(text)

vocab = dict(Counter(text).most_common(5000))
print(vocab)

pretrain_path = './bert_pretrain'

model = BertModel.from_pretrained(pretrain_path)
tokenizer = BertTokenizer.from_pretrained(pretrain_path)

print(tokenizer.tokenize('400 The plain HTTP request was sent to HTTPS port 400 Bad Request The plain HTTP request was sent to HTTPS port nginx HTTP/1.1 505 HTTP Version Not Supported Server: nginx Data: DATE Content-Type: text/html Content-Length: CONTENT-LENGTH Connection: close 505 HTTP Version Not Supported 505 HTTP Version Not Supported nginx HTTP/1.1 400 Bad Request Server: nginx Data: DATE Content-Type: text/html Content-Length: CONTENT-LENGTH Connection: close 400 The plain HTTP request was sent to HTTPS port 400 Bad Request The plain HTTP request was sent to HTTPS port nginx HTTP/1.1 400 Bad Request Server: nginx Data: DATE Content-Type: text/html Content-Length: CONTENT-LENGTH Connection: close 400 The plain HTTP request was sent to HTTPS port 400 Bad Request The plain HTTP request was sent to HTTPS port nginx HTTP/1.1 400 Bad Request Server: nginx Data: DATE Content-Type: text/html Content-Length: CONTENT-LENGTH Connection: close ETag: ETAG  HTTP/1.1 400 Bad Request Server: nginx Data: DATE Content-Type: text/html Content-Length: CONTENT-LENGTH Connection: close  HTTP/1.1 400 Bad Request Server: nginx Data: DATE Content-Type: text/html Content-Length: CONTENT-LENGTH Connection: close 400 The plain HTTP request was sent to HTTPS port 400 Bad Request The plain HTTP request was sent to HTTPS port nginx HTTP/1.1 405 Not Allowed Server: nginx Data: DATE Content-Type: text/html Content-Length: CONTENT-LENGTH Connection: close 405 Not Allowed 405 Not Allowed nginx HTTP/1.1 405 Not Allowed Server: nginx Data: DATE Content-Type: text/html Content-Length: CONTENT-LENGTH Connection: close 405 Not Allowed 405 Not Allowed nginx'))

for _ in vocab:
    if _ not in tokenizer.vocab.keys():
        # print(_)
        num_added_toks = tokenizer.add_tokens(_)


model.resize_token_embeddings(len(tokenizer))

tokenizer.save_pretrained("./bert_pretrain")

print(tokenizer.tokenize('400 The plain HTTP request was sent to HTTPS port 400 Bad Request The plain HTTP request was sent to HTTPS port nginx HTTP/1.1 505 HTTP Version Not Supported Server: nginx Data: DATE Content-Type: text/html Content-Length: CONTENT-LENGTH Connection: close 505 HTTP Version Not Supported 505 HTTP Version Not Supported nginx HTTP/1.1 400 Bad Request Server: nginx Data: DATE Content-Type: text/html Content-Length: CONTENT-LENGTH Connection: close 400 The plain HTTP request was sent to HTTPS port 400 Bad Request The plain HTTP request was sent to HTTPS port nginx HTTP/1.1 400 Bad Request Server: nginx Data: DATE Content-Type: text/html Content-Length: CONTENT-LENGTH Connection: close 400 The plain HTTP request was sent to HTTPS port 400 Bad Request The plain HTTP request was sent to HTTPS port nginx HTTP/1.1 400 Bad Request Server: nginx Data: DATE Content-Type: text/html Content-Length: CONTENT-LENGTH Connection: close ETag: ETAG  HTTP/1.1 400 Bad Request Server: nginx Data: DATE Content-Type: text/html Content-Length: CONTENT-LENGTH Connection: close  HTTP/1.1 400 Bad Request Server: nginx Data: DATE Content-Type: text/html Content-Length: CONTENT-LENGTH Connection: close 400 The plain HTTP request was sent to HTTPS port 400 Bad Request The plain HTTP request was sent to HTTPS port nginx HTTP/1.1 405 Not Allowed Server: nginx Data: DATE Content-Type: text/html Content-Length: CONTENT-LENGTH Connection: close 405 Not Allowed 405 Not Allowed nginx HTTP/1.1 405 Not Allowed Server: nginx Data: DATE Content-Type: text/html Content-Length: CONTENT-LENGTH Connection: close 405 Not Allowed 405 Not Allowed nginx'))
