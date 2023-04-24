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
print(tokenizer.tokenize('A  �A  .�A  P�A  u�A     \�A   �          �   ��� BaseException'))

# new_tokens = ['lighttpd', 'hospitalization']
# num_added_toks = tokenizer.add_tokens(new_tokens)  # 返回一个数，表示加入的新词数量，在这里是2
#
# # 关键步骤，resize_token_embeddings输入的参数是tokenizer的新长度
# model.resize_token_embeddings(len(tokenizer))
#
# print(tokenizer.tokenize('lighttpd'))
# tokenizer.save_pretrained("./bert_pretrain")