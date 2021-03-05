import re
class Preprocess():
    def __init__(self, tokenizer):
        self.tk = tokenizer

    def work(self, sentence, data, maxlen):
        sentence = re.sub('\,|~|\"|=|<|>|\*|\'', '', sentence)
        sentence = re.sub('\(|\)', ',', sentence)
        sentence = re.sub('[0-9]+', 'num', sentence)
        sentence = re.sub(";+", ';', sentence)
        sentence = re.sub("[?]{2,}", '??', sentence)
        sentence = re.sub("[.]{2,}", '..', sentence)
        sentence = re.sub("[!]{2,}", '!!', sentence)
        ' '.join(('[CLS]', sentence, '[SEP]'))
        #sentence = re.sub('[a-zA-Z]', '', sentence)

        temp_X = self.tk.encode_plus(sentence,
                        add_special_tokens = True, # add [CLS], [SEP]
                        max_length = maxlen, # max length of the text that can go to BERT
                        pad_to_max_length = True, # add [PAD] tokens
                        return_attention_mask = True,)

        return temp_X

    def labeling(self, data, train):
        for i in range(len(data['Emotion'])):
            if data['Emotion'].iloc[i] == '슬픔':
                train.append([0])
            elif data['Emotion'].iloc[i] == '중립':
                train.append([1])
            elif data['Emotion'].iloc[i] == '행복':
                train.append([2])
            elif data['Emotion'].iloc[i] == '공포':
                train.append([3])
            elif data['Emotion'].iloc[i] == '분노':
                train.append([4])
        return train
