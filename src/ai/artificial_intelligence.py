from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
import torch


class IntentClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=4):
        config = BertConfig.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)

    def classify(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        _, predicted = torch.max(outputs.logits, 1)
        return predicted.item()


class ResponseGenerator:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    # def generate_response(self, intent):
    #     input_ids = self.tokenizer.encode(intent, return_tensors='pt')
    #     outputs = self.model.generate(input_ids, max_length=100, num_return_sequences=1)
    #     return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    

    def generate_response(self, intent):
        input_ids = self.tokenizer.encode(intent, return_tensors='pt')
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        outputs = self.model.generate(input_ids, max_length=100, num_return_sequences=1, attention_mask=attention_mask)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)