# -*- coding: utf-8 -*-
'''
AUTHOR
------
    Goto Ryusuke (yuhang1012long@link.cuhk.edu.hk)
    Find me at:
        https://github.com/GotoRyusuke

DESCRIPTION
-----------
Summariser using SBERT

References:
    For the Ssentence-transformers module:
        https://github.com/UKPLab/sentence-transformers
        https://www.sbert.net/
    Papers:
        [1] Reimers, N., & Gurevych, I. (2019). Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084.
        [2] Erkan, G., & Radev, D. R. (2004). Lexrank: Graph-based lexical centrality as salience in text summarization. Journal of artificial intelligence research, 22, 457-479.
        
STRUCTURE
---------
-<class> LexRankSummariser
| -<method> _summarise
-<END>

NOTE
----
Thank Nils Reimers (https://github.com/nreimers) for fixing the
LexRank module and providing the degree_centrality_socres func.
You can find his comments in the issues of the sentence-transformers
github page.

'''

import re
import nltk
import numpy as np
from LexRank import degree_centrality_scores
from sentence_transformers import SentenceTransformer, util

class LexRankSummariser:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def _summarise(self, text:str):
        sentences = nltk.sent_tokenize(text)
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        cos_scores = util.cos_sim(embeddings, embeddings).numpy()
        centrality_scores = degree_centrality_scores(cos_scores, threshold=None)
        most_central_sentence_indices = np.argsort(-centrality_scores)
        
        idx_list = [int(idx)
                    for idx in most_central_sentence_indices[:5]
                    ]
        
        return ' '.join(
            [sentences[idx].strip()
             for idx in idx_list]
            )
    

if __name__ == '__main__':
    test_file_path = './test_file.txt'
    with open(test_file_path, 'r', encoding = 'utf-8') as f:
        text = f.read()
        
    # text = """
    #     New York City (NYC), often called simply New York, is the most populous city in the United States. With an estimated 2019 population of 8,336,817 distributed over about 302.6 square miles (784 km2), New York City is also the most densely populated major city in the United States. Located at the southern tip of the U.S. state of New York, the city is the center of the New York metropolitan area, the largest metropolitan area in the world by urban landmass. With almost 20 million people in its metropolitan statistical area and approximately 23 million in its combined statistical area, it is one of the world's most populous megacities. New York City has been described as the cultural, financial, and media capital of the world, significantly influencing commerce, entertainment, research, technology, education, politics, tourism, art, fashion, and sports. Home to the headquarters of the United Nations, New York is an important center for international diplomacy.
    #     Situated on one of the world's largest natural harbors, New York City is composed of five boroughs, each of which is a county of the State of New York. The five boroughs—Brooklyn, Queens, Manhattan, the Bronx, and Staten Island—were consolidated into a single city in 1898. The city and its metropolitan area constitute the premier gateway for legal immigration to the United States. As many as 800 languages are spoken in New York, making it the most linguistically diverse city in the world. New York is home to more than 3.2 million residents born outside the United States, the largest foreign-born population of any city in the world as of 2016. As of 2019, the New York metropolitan area is estimated to produce a gross metropolitan product (GMP) of $2.0 trillion. If the New York metropolitan area were a sovereign state, it would have the eighth-largest economy in the world. New York is home to the highest number of billionaires of any city in the world.
    #     New York City traces its origins to a trading post founded by colonists from the Dutch Republic in 1624 on Lower Manhattan; the post was named New Amsterdam in 1626. The city and its surroundings came under English control in 1664 and were renamed New York after King Charles II of England granted the lands to his brother, the Duke of York. The city was regained by the Dutch in July 1673 and was subsequently renamed New Orange for one year and three months; the city has been continuously named New York since November 1674. New York City was the capital of the United States from 1785 until 1790, and has been the largest U.S. city since 1790. The Statue of Liberty greeted millions of immigrants as they came to the U.S. by ship in the late 19th and early 20th centuries, and is a symbol of the U.S. and its ideals of liberty and peace. In the 21st century, New York has emerged as a global node of creativity, entrepreneurship, and environmental sustainability, and as a symbol of freedom and cultural diversity. In 2019, New York was voted the greatest city in the world per a survey of over 30,000 people from 48 cities worldwide, citing its cultural diversity.
    #     Many districts and landmarks in New York City are well known, including three of the world's ten most visited tourist attractions in 2013. A record 62.8 million tourists visited New York City in 2017. Times Square is the brightly illuminated hub of the Broadway Theater District, one of the world's busiest pedestrian intersections, and a major center of the world's entertainment industry. Many of the city's landmarks, skyscrapers, and parks are known around the world. Manhattan's real estate market is among the most expensive in the world. Providing continuous 24/7 service and contributing to the nickname The City that Never Sleeps, the New York City Subway is the largest single-operator rapid transit system worldwide, with 472 rail stations. The city has over 120 colleges and universities, including Columbia University, New York University, Rockefeller University, and the City University of New York system, which is the largest urban public university system in the United States. Anchored by Wall Street in the Financial District of Lower Manhattan, New York City has been called both the world's leading financial center and the most financially powerful city in the world, and is home to the world's two largest stock exchanges by total market capitalization, the New York Stock Exchange and NASDAQ.
    #     """
    text = re.sub(r'\n+', '. ', text)
    text = re.sub(r'\s{2,}', '', text)
    
    summariser = LexRankSummariser()
    test = summariser._summarise(text)
        
        