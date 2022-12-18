# -*- coding: utf-8 -*-
'''
AUTHOR
------
    Goto Ryusuke (yuhang1012long@link.cuhk.edu.hk)
    Find me at:
        https://github.com/GotoRyusuke

DESCRIPTION
-----------
Summariser using financial-summarization-pegasus.

References:
    For the FSP module:
        https://huggingface.co/human-centered-summarization/financial-summarization-pegasus
    FSP:
        Passali, T., Gidiotis, A., Chatzikyriakidis, E., & Tsoumakas, G. (2021, April). Towards human-centered summarization: A case study on financial news. In Proceedings of the First Workshop on Bridging Human–Computer Interaction and Natural Language Processing (pp. 21-27).

STRUCTURE
---------
-<class> SpacySummariser
| -<method> _summarise
-<END>

NOTE
----
* This summariser is highly resource-expensive. I tried an RTX 3090, but it quickly
went out of memory after processing less than 10 forms and the speed was around
30sec/file. Apparently, cpu-only calculation led to an even lower speed. Therefore,
this summariser is the least recommended for large amount of data. Nevertheless,
one of its merits lies in that it has been pre-trained on financial texts, and thus 
may lead to a more meaningful summary, considering that our data are also finance-related.
Use it when the workload is around 10 files, each with less than 512 sentences.

** Here I employ an iteration strategy to curtail texts longer than 512 sentences. The logic
is simple: whenever the summariser encounters a text longer than 2000 chars, it cuts the text
into segments, each shorter than 2000 chars; then, it summerises each segment and concatenates
the summaries. If the new text, which is a combination of summaries of all segments of texts,
is still longer than 2000 chars, the iteration goes on.

'''


from transformers import PegasusTokenizer, PegasusForConditionalGeneration, TFPegasusForConditionalGeneration
from utils import cut_text_per_2000

class FinanceSummariser:
    def __init__(self):
        self.model_name = "human-centered-summarization/financial-summarization-pegasus"
        self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name)
    
    def _normal_summariser(self, text:str):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids

        output = self.model.generate(
            input_ids, 
            max_length=32, 
            num_beams=5, 
            early_stopping=True
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
        
    def _summarise(self,text:str):
        while len(text) > 2000:
            text_paras = cut_text_per_2000(text)
            text = ' '.join([
                self._normal_summariser(para)
                for para in text_paras]
                )
        return self._normal_summariser(text)
           

            
            
            
        