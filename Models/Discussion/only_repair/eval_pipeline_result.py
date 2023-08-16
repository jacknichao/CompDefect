import os
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import prettytable
import sys

sys.path.append('../../')
from bleu import _bleu

output_dir = './saved_models'
type2label = pd.read_json('../../data/type2label_all.json', orient='index')
label2type = dict(zip(type2label.loc[:, 0], type2label.index))
dataset2paper = pd.read_json('../../data/dataset2paper.json', orient='index')
dataset2paper = dict(zip(dataset2paper.index, dataset2paper.loc[:, 0]))
table = prettytable.PrettyTable()
table.field_names = ['bugType', '# cnt (TEST)', '# cls correct', '# repair correct', '# cls&repair correct']
table_summary = ['ALL', 0, 0, 0, 0]
df_data = []

all_cnt, cls_cnt, repair_cnt, cls_repair_cnt = 0, 0, 0, 0
with open(os.path.join(output_dir, f'test_data.output')) as f, open(
        os.path.join(output_dir, f'test_data.gold')) as f1, open(
    os.path.join(output_dir, f'test_data.y_trues')) as f2, open(
    os.path.join(output_dir, f'test_data.bug')) as f3, open(
        os.path.join(output_dir, f'test_data_buggy.output'), 'w') as f4, open(
        os.path.join(output_dir, f'test_data_buggy.gold'), 'w') as f5:
    outputs = f.readlines()
    golds = f1.readlines()
    trues = f2.readlines()
    bugs = f3.readlines()
    count_data = []
    accs = []
    accs_buggy = []
    golds_buggy = []
    outputs_buggy = []
    for idx, (output, gold, bug, true) in enumerate(zip(outputs, golds, bugs, trues)):
        if label2type[int(true)] != 'CLEAN':
            accs_buggy.append(output == gold)
        else:
            f4.write(gold + '\n')
            f5.write(output + '\n')
        accs.append(output == gold)


    print('xmatch:', round(np.mean(accs) * 100, 4), f'{sum(accs)} / {len(accs)}')
    dev_bleu = round(
        _bleu(os.path.join(output_dir, f'test_data.gold'),
              os.path.join(output_dir, f'test_data.output')), 4)
    print('bleu:', dev_bleu)

    print('xmatch:', round(np.mean(accs_buggy) * 100, 4), f'{sum(accs_buggy)} / {len(accs_buggy)}')
    dev_bleu = round(
        _bleu(os.path.join(output_dir, f'test_data_buggy.gold'),
              os.path.join(output_dir, f'test_data_buggy.output')), 4)
    print('bleu:', dev_bleu)
