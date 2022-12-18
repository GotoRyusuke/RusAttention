# -*- coding: utf-8 -*-
'''
AUTHOR
------
    Goto Ryusuke (yuhang1012long@link.cuhk.edu.hk)
    Find me at:
        https://github.com/GotoRyusuke

DESCRIPTION
-----------
Summariser using spaCy module.

References:
    For the spaCy module:
        https://spacy.io/models
    Statistical extraction:
        Luhn, H. P. (1958). The automatic creation of literature abstracts. IBM Journal of research and development, 2(2), 159-165.

STRUCTURE
---------
-<class> SpacySummariser
| -<method> _summarise
-<END>

'''
import re
import spacy 
import string
from heapq import nlargest
from spacy.lang.en.stop_words import STOP_WORDS

class SpacySummariser:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.punctuation = string.punctuation +  '\n'
    
    def _summarise(self, text:str):
        doc = self.nlp(text)
        tokens = [token.text for token in doc]
        word_freq = {}
        stop_words = list(STOP_WORDS )
        for word in doc:
            if word.text.lower() not in stop_words:
                if word.text.lower() not in self.punctuation:
                    if word.text.lower() not in word_freq.keys():
                        word_freq[word.text.lower()] = 1
                    else:
                        word_freq[word.text.lower()] += 1
        for word in word_freq.keys():
            word_freq[word] = word_freq[word] / max(word_freq.values())  
        sent_tokens = [sent for sent in doc.sents]          
        sent_score = {}
        for sent in sent_tokens:
            for word in sent:
                if word.text.lower() in word_freq.keys():
                    if sent not in sent_score.keys():
                        sent_score[sent] = word_freq[word.text.lower()]
                    else:
                        sent_score[sent] += word_freq[word.text.lower()]
        summary = nlargest(n = 2 , iterable = sent_score , key = sent_score.get)
        return ' '.join([str(sent) for sent in summary])

if __name__ == '__main__':
    test_file_path = './test_file.txt'
    with open(test_file_path, 'r', encoding = 'utf-8') as f:
        text = f.read()
    text = re.sub(r'\n+', '. ', text)
    
    obj = SpacySummariser()
    test = obj._summarise(text)
    print(test)