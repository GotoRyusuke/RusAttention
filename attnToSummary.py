# -*- coding: utf-8 -*-
'''
AUTHOR
------
    Goto Ryusuke (yuhang1012long@link.cuhk.edu.hk)
    Find me at:
        https://github.com/GotoRyusuke

DESCRIPTION
-----------
Module to conduct the following tasks:
    - read the texts of all items in a form from dirs saved in the summary
    table and concat;
    - conduct summarisation using one of the summariser modules
    - check if rus words in the summary;
    - create a dummy and add it to the summary table

STRUCTURE
---------
-<class> AttentionToSummary
| -<method> _assign_dummy2single_form
| -<method> assign_in_batch
| -<method> threading
-<END>

'''
import re
import pandas as pd
import spacy
import string
import datetime
from heapq import nlargest
from spacySum import SpacySummariser
from lexrankSum import LexRankSummariser
from finSum import FinanceSummariser
from joblib import Parallel, delayed
from spacy.lang.en.stop_words import STOP_WORDS
from utils import load_dicts, phrase_in_text

class AttentionToSummary:
    def __init__(
            self,
            summary_path:str,
            dict_path_list:list,
            store_path:str,
            form_type: str,
            ):
        '''
        Parametres
        ----------
        summary_path: str
            The path to summary excel file.
        dict_path_list: list
            A list of dicts to be used to detect if the text
            if Russian-related.
        store_path: str
            The path to the folder where all the extracted files
            are saved.
        form_type: str
            Should be one of the following:
                - 8-K
                - 10-Q
                - 10-K_Item1A
                - 10-K_Item7
        '''

        # read the summary excel file as a pandas df
        df = pd.read_excel(summary_path)
        # get the list of adrs columns
        adrs_names = [
            adrs_name
            for adrs_name in df.columns
            if '_adrs' in adrs_name
            ]
        # get the item names in this type of Form
        items = [
            adrs.split('_')[0]
            for adrs in adrs_names
            ]
        basic_info = [
            'CIK',
            'co_name',
            'f_type',
            'f_date',
            ]
        
        # initialise summariser
        self.summariser = SpacySummariser()
        # import the words of all dicts as a list
        self.dict_phrases = load_dicts(dict_path_list)
        # drop all other columns except basic info and adrs
        self.df = df.loc[:, basic_info + adrs_names]
        # add attributes for future use
        self.adrs_names = adrs_names
        self.items = items
        self.form_type = form_type
        self.store_path = store_path
    
    def _assign_dummy2single_form(self, idx:int):
        '''
        Get the value of the dummy for a single form

        Parametres
        ----------
        idx: int
            The index of the form in the summary df
        
        Return
        ------
        0 if no Russian-related words/phrases are detected,
        else 1.
        '''
        # get the paths of items that are extracted successfully
        file_paths = [
            self.store_path + path
            for path in list(self.df.loc[idx, self.adrs_names])
            if not isinstance(path, float) and len(path) >= 10
            ]
        
        # read the texts from paths
        text = ''
        for path in file_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    text += f.read() + ' '
            except:
                # some early texts are saved in gbk encoding,
                # which can lead to an error if read through u8 
                with open(path, 'r', encoding='gbk') as f:
                    text += f.read() + ' '
        # simple pre-processing
        text = re.sub(r'\n+', '. ', text)
        text = re.sub(r'\s{2,}', '', text)

        # ensure the text has meaningful contents
        if len(text) > 0:
            if len(text) > 1000000:
                text = text[:1000000]
            # use summariser to get the summary  
            text = self.summariser._summarise(text)
            if phrase_in_text(self.dict_phrases, text):
                return 1
            else: return 0
        else:
            return 0
    
    def assign_in_batch(self,_range:list):
        '''
        Obtain the values of the dummy in a large batch of forms.

        Parametre
        ---------
        _range: list
            A list of indeces of forms in a summary df.
        
        Return
        ------
        df: pandas df
            The fraction of summary df with the column "rus_attention"
        '''
        df = self.df.loc[_range,:]
        for idx in _range:
            df.loc[idx,'rus_attn'] = self._assign_dummy2single_form(idx)
        return df
    
    def threading(self, jobs:int):
        '''
        Employ threading.

        Parametre
        ---------
        jobs: int
            Num of jobs
        
        Return
        ------
        output: pandas df
            Complete summary df with the column "rus_attn"
        '''
        # idx list segmentation
        idx_list = list(self.df.index)
        num_per_job = int(len(idx_list) / jobs)
        idx_list_cut = []
        for i in range(jobs):
            if i != jobs - 1:
                idx_list_cut.append(idx_list[i * num_per_job: (i + 1) * num_per_job])
            else:
                idx_list_cut.append(idx_list[i * num_per_job:])
        
        def multi_run(sub_idx_list):
            sub_df = self.assign_in_batch(sub_idx_list)
            return sub_df

        # deploy threading
        output_dfs = Parallel(n_jobs=jobs, verbose=1)(delayed(multi_run)(sub_list) for sub_list in idx_list_cut)
        output = pd.DataFrame()
        for sub_df in output_dfs:
            output = pd.concat([output, sub_df])
        output.reset_index(drop=True, inplace=True)
        
        # drop empty records
        save = []
        for idx in output.index:
           value = 0
           for item in self.items:
               item_adrs = output.loc[idx, item+'_adrs']
               if item_adrs != ' '  and not isinstance(item_adrs,float):
                   value += 1
           if value != 0: save.append(idx)
        output = output.loc[save, :]
        output.sort_values(by = ['CIK', 'f_date'], inplace=True)
        output.reset_index(drop = True, inplace = True)
        
        # refine f_date
        if self.form_type == '10-Q':
            output['f_date'] = output['f_date'].dt.strftime('%Y-%m-%d')
        elif self.form_type in ['10-K_Item1A', '10-K_Item7']:
            output['f_date'] = [re.sub('/', '-', date) for date in output['f_date']]
     
        return output

    
        
