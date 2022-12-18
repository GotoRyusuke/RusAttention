# -*- coding: utf-8 -*-
from attnToSummary import AttentionToSummary
import time

form_type = '10-Q'
summary_path = f'./new_summary_2022Q2_{form_type}(exact word).xlsx'
dict_path_list = [
    'rus_dict_lemma.txt',
    'rus_names.txt',
    ]
store_path = 'F:/EDGAR/2022Q2_extracted/'

obj = AttentionToSummary(
    summary_path, 
    dict_path_list, 
    store_path,
    form_type,
    )

start = time.time()
result = obj.assign_in_batch(range(50,100))
print('cost {:.2f} seconds'.format(time.time()-start))
# result = obj.threading(16)
# result.to_excel(f'./attn_lexrankSum_{form_type}.xlsx', index=False)


# test_file_path = './test_file.txt'
# with open(test_file_path, 'r', encoding = 'utf-8') as f:
#     text = f.read()
# text = re.sub(r'\n+', '. ', text)
# text = re.sub(r'\s{2,}', '', text)

#start = time.time()
#print('cost {:.2f} seconds'.format(time.time()-start))