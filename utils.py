# -*- coding: utf-8 -*-
'''
AUTHOR
------
    Goto Ryusuke (yuhang1012long@link.cuhk.edu.hk)
    Find me at:
        https://github.com/GotoRyusuke

DESCRIPTION
-----------
Utils for AttentionToSummary class and some of the summarisers.

STRUCTURE
---------
-<func> load_dicts
-<func> preprocess_text
-<func> phrase_in_text
-<func> cut_sentence
-<func> cut_text_per_2000

'''
from collections import Counter
import re

def load_dicts(dict_path_list:str)->list:
    '''
    Import dicts from paths as a word/phrase list.

    Parametre
    ---------
    dict_path_list: list
        A list of dict paths.
    
    Return
    ------
        A list of word/phrase, with each element a list
        of word(s) in a word/phrase.
    '''

    dict_phrases = []
    for path in dict_path_list:
        with open(path, 'r', encoding='utf-8') as f:
            dict_phrases += [
                phrase.strip().lower()
                for phrase in f.readlines()
                ]

    return [
        phrase.split()
        for phrase in dict_phrases
        ]

def preprocess_text(text:str)->list:
    '''
    Func to do the following:
    - remove all line breaks
    - remove all spaces larger than 2 char-widths
    - remove all puncts
    - transform to lower case

    Parametre
    ---------
    text: str
        The text to be processed.
    
    Return
    ------
    A list with each element a str of word in the text
    '''

    text = re.sub(r'\n+', '. ', text)
    text = re.sub(r'\s{2,}', '', text)
    text = text.lower()
    for punc in ['.',',','?','!',':',';']:
        text = text.replace(punc, ' ')

    return text.split()

def phrase_in_text(dict_phrases:list, text:str)->bool:
    '''
    Detect dict words/phrases in a given text.

    Parametres
    ----------
    dict_phrases: list
        A list of dict words/phrases. Ideally generated by
        load_dicts func.
    text: str
        A string of text.

    Return
    ------
    False if no dict word/phrase is detected, else True
    '''

    # pre-process the text and get a list of words in it
    text_words = preprocess_text(text)
    # get the set of words in the text
    words_counter = dict(Counter(text_words))
    counter_keys = list(words_counter.keys())
    
    for phrase in dict_phrases:
        # whether the dict phrase is a single word
        if len(phrase) == 1:
            kw = phrase[0]
            # whether it is a lemma
            if '*' not in kw:
                if phrase[0] in words_counter.keys():
                    return True
            else:
                kw = kw.split("*")[0]
                for key in counter_keys:
                    if key[:len(kw)] == kw:
                        return True
        else:
            for w_i in range(len(text_words)):
                flag = True
                for p_w_i, p_w in enumerate(phrase):
                    if '*' not in p_w:
                        if w_i + p_w_i >= len(text_words) or p_w != text_words[w_i + p_w_i]:
                            flag = False
                            break
                    else:
                        p_w = p_w.split('*')[0]
                        if w_i + p_w_i >= len(text_words) or p_w != text_words[w_i + p_w_i][:len(p_w)]:
                            flag = False
                            break
                if flag:
                    return flag
    return False

def cut_sentence(talk_content:str):
    talk_sentences = []
    talk_words = talk_content.split()
    last_sentence_idx = 0
    exception_rule = ["Mr", "Mrs", "Miss", "Ms", "Sir", "Madam", "Dr", "Cllr", "Lady", "Lord", "Professor", "Prof",
                      "Chancellor", "Principal", "President", "Master", "Governer", "Gov", "Attorney", "Atty"]
    for w_i in range(len(talk_words)):
        talk_word = talk_words[w_i]
        if w_i == len(talk_words) - 1:
            talk_sentences.append(" ".join(talk_words[last_sentence_idx: w_i + 1]))
        else:
            if talk_word[-1] in [".", "?", "!"]:
                if talk_word[:-1] not in exception_rule:
                    if talk_words[w_i + 1][0].isupper():
                        talk_sentences.append(" ".join(talk_words[last_sentence_idx: w_i + 1]))
                        last_sentence_idx = w_i + 1
    return talk_sentences

def cut_text_per_2000(text:str):
    '''
    Cut the text into parts if it is longer than 2000 chars.

    Parametre
    ---------
    text: str
        The text to be cut.
    
    Return
    ------
    text_cut: list
        A list with each element a str shorter than 2000 chars
    
    Note
    ----
    This func is used by FinanceSummariser
    
    '''
    
    text_sents = cut_sentence(text)
    text_cut = []
    sent_i = 0
    while sent_i < len(text_sents)-1:
        sub_para = ''
        while len(' '.join([sub_para, text_sents[sent_i]])) < 2000:
            sub_para = ' '.join([sub_para, text_sents[sent_i]])
            sent_i += 1
            if sent_i == len(text_sents): break
        text_cut.append(sub_para)
    
    return text_cut

# dict_path_list = [
#     'rus_dict_lemma.txt',
#     'rus_names.txt',
#     ]
# dict_phrases = load_dicts(dict_path_list)

# test_file_path = './test_file.txt'
# with open(test_file_path, 'r', encoding = 'utf-8') as f:
#     text = f.read()
# text = '''
#     BBC news. Several ballistic missile submarines are approaching the coast
#     of Greece. Hello world!
#     '''
# test = phrase_in_text(dict_phrases, text)
