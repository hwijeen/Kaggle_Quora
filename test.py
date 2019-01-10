import spacy 
from itertools import chain

def get_char_tokenizer():
    spacy_en = spacy.load('en')
    def char_tokenizer(text):
        sent = []
        words = [tok.text for tok in spacy_en.tokenizer(text)]
        sent = chain.from_iterable(words)
        return list(sent)
    return char_tokenizer


tokenizer = get_char_tokenizer()
print(tokenizer('hi nice to mee you?'))
