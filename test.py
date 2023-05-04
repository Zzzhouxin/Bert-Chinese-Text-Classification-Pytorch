# coding=utf-8
from transformers import BertModel, BertTokenizer

pretrain_path = './bert_pretrain'

model = BertModel.from_pretrained(pretrain_path)
tokenizer = BertTokenizer.from_pretrained(pretrain_path)

print(f'vocab size:{tokenizer.vocab_size}')
print(tokenizer.encode('[PAD]'))
print(tokenizer.encode('[SEP]'))

print(tokenizer.tokenize('lighttpd'))
print(tokenizer.encode('lighttpd'))

print(tokenizer.tokenize('lighttpd'))
print(tokenizer.tokenize('HTTP/1.1 505 HTTP Version Not Supported Server: nginx Data: DATE Content-Type: text/html Content-Length: CONTENT-LENGTH Connection: close 505 HTTP Version Not Supported 505 HTTP Version Not Supported nginx HTTP/1.1 404 Not Found Server: nginx Data: DATE Content-Type: text/html; charset=utf-8 Connection: close Vary: Accept-Encoding 对不起，未找到页面 对不起，未找到页面 > 对不起，未找到页面 对不起，pc端接口已关闭，請使用移动手机端访问～ 对不起   您要访问的页面无法访问或还未面世 ! 我们将尽快修复 ! 对您浏览造成的不便，请谅解 ! HTTP/1.1 404 Not Found Server: nginx Data: DATE Content-Type: text/html; charset=utf-8 Connection: close Vary: Accept-Encoding 对不起，未找到页面 对不起，未找到页面 > 对不起，未找到页面 对不起，pc端接口已关闭，請使用移动手机端访问～ 对不起   您要访问的页面无法访问或还未面世 ! 我们将尽快修复 ! 对您浏览造成的不便，请谅解 ! HTTP/1.1 400 Bad Request Server: nginx Data: DATE Content-Type: text/html Content-Length: CONTENT-LENGTH Connection: close  HTTP/1.1 404 Not Found Server: nginx Data: DATE Content-Type: text/html; charset=UTF-8 Connection: close Vary: Accept-Encoding  HTTP/1.1 404 Not Found Server: nginx Data: DATE Content-Type: text/html; charset=utf-8 Connection: close Vary: Accept-Encoding 对不起，未找到页面 对不起，未找到页面 > 对不起，未找到页面 对不起，pc端接口已关闭，請使用移动手机端访问～ 对不起   您要访问的页面无法访问或还未面世 ! 我们将尽快修复 ! 对您浏览造成的不便，请谅解 ! HTTP/1.1 405 Not Allowed Server: nginx Data: DATE Content-Type: text/html Content-Length: CONTENT-LENGTH Connection: close 405 Not Allowed 405 Not Allowed nginx HTTP/1.1 405 Not Allowed Server: nginx Data: DATE Content-Type: text/html Content-Length: CONTENT-LENGTH Connection: close 405 Not Allowed 405 Not Allowed nginx,nginx'))

# new_tokens = ['lighttpd', 'hospitalization']
# num_added_toks = tokenizer.add_tokens(new_tokens)  # 返回一个数，表示加入的新词数量，在这里是2
#
# # 关键步骤，resize_token_embeddings输入的参数是tokenizer的新长度
# model.resize_token_embeddings(len(tokenizer))
#
# print(tokenizer.tokenize('lighttpd'))
# tokenizer.save_pretrained("./bert_pretrain")