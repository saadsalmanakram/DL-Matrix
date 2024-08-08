from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model

# BERT Example
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

text = "Hello, world!"
inputs = bert_tokenizer(text, return_tensors='pt')
outputs = bert_model(**inputs)
print(outputs.last_hidden_state)

# GPT Example
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2Model.from_pretrained('gpt2')

text = "Hello, world!"
inputs = gpt2_tokenizer(text, return_tensors='pt')
outputs = gpt2_model(**inputs)
print(outputs.last_hidden_state)
