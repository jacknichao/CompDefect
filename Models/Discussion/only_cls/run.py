import argparse
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from tqdm import tqdm
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from tree_sitter import Language, Parser
import sys
sys.path.append('../../')
from bleu import _bleu
from model import Model
from parser import DFG_java
from parser import remove_comments_and_docstrings, tree_to_token_index, index_to_code_token
from utils.myLogger import logging

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

# load parser
dfg_function = {'java': DFG_java}
parsers = {}
for lang in dfg_function:
    Language = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(Language)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


# remove comments, tokenize and extract dataflow
def extract_dataflow(code, parser, lang):
    code = remove_comments_and_docstrings(code, lang)

    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node

        # get token's (start_point, end_point)
        tokens_index = tree_to_token_index(root_node)

        # get token's source_code
        code = code.split('\n')
        code_tokens = [index_to_code_token(idx, code) for idx in tokens_index]

        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)

        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
            logging.WARNING(f'dfg is empty')

        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)

        dfg = new_DFG
    except:
        dfg = []
        logging.WARNING(f'dfg is empty')
    return code_tokens, dfg


class Example(object):
    """a single data instance"""

    def __init__(self, bug_type, fix_func, bug_func, clean_func):
        self.fix_func = fix_func
        self.bug_func = bug_func
        self.clean_func = clean_func
        self.bug_type = bug_type


def read_examples(filename):
    examples = []
    data = pd.read_csv(filename, lineterminator='\n', index_col=0)
    data = data[['bugType', 'func_code_fix', 'func_code_bug', 'func_code_clean']]
    # data = data[data['bugType'] != 'CLEAN'].reset_index(drop=True)
    for item in data.values.tolist():
        examples.append(
            Example(
                *item
            )
        )
    return examples


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 bug_ids,
                 bug_mask,
                 bug_position_ids,
                 bug_dfg_to_code,
                 bug_dfg_to_dfg,
                 clean_ids,
                 clean_mask,
                 clean_position_ids,
                 clean_dfg_to_code,
                 clean_dfg_to_dfg,
                 fix_ids,
                 fix_mask,
                 label,
                 ):
        self.example_id = example_id
        # bug_func
        self.bug_ids = bug_ids
        self.bug_mask = bug_mask
        self.bug_position_ids = bug_position_ids
        self.bug_dfg_to_code = bug_dfg_to_code
        self.bug_dfg_to_dfg = bug_dfg_to_dfg
        self.label = label

        # clean_func
        self.clean_ids = clean_ids
        self.clean_mask = clean_mask
        self.clean_position_ids = clean_position_ids
        self.clean_dfg_to_code = clean_dfg_to_code
        self.clean_dfg_to_dfg = clean_dfg_to_dfg

        # fix_func
        self.fix_ids = fix_ids
        self.fix_mask = fix_mask


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []
    logging.info(args.type2label)
    type2label = pd.read_json(args.type2label, orient='index')
    type2label = dict(zip(type2label.index, type2label.loc[:, 0]))
    for example_index, example in enumerate(tqdm(examples, total=len(examples))):
        cache = []
        for code in [example.bug_func, example.clean_func]:
            # extract data flow
            code_tokens, dfg = extract_dataflow(code, parsers['java'], 'java')
            code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x)
                           for idx, x in enumerate(code_tokens)]

            ori2cur_pos = {}  # token be tokenized to more than one tokens
            ori2cur_pos[-1] = (0, 0)
            for i in range(len(code_tokens)):
                ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
            code_tokens = [y for x in code_tokens for y in x]

            # truncating
            code_tokens = code_tokens[
                          :args.max_source_length + args.data_flow_length - 2 - min(len(dfg), args.data_flow_length)]
            bug_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
            bug_ids = tokenizer.convert_tokens_to_ids(bug_tokens)
            bug_position_ids = [i + tokenizer.pad_token_id + 1 for i in range(len(bug_ids))]
            # variables
            dfg = dfg[:args.max_source_length + args.data_flow_length - len(bug_ids)]
            bug_tokens += [x[0] for x in dfg]
            bug_ids += [tokenizer.unk_token_id for _ in dfg]
            bug_position_ids += [0 for _ in dfg]

            # padding
            padding_length = args.max_source_length + args.data_flow_length - len(bug_ids)
            bug_position_ids += [tokenizer.pad_token_id] * padding_length  # tokenizer.pad_token_id = 0
            bug_ids += [tokenizer.pad_token_id] * padding_length
            bug_mask = [1] * len(bug_tokens)
            bug_mask += [0] * padding_length

            # reindex
            reverse_index = {}
            for idx, x in enumerate(dfg):
                reverse_index[x[1]] = idx
            for idx, x in enumerate(dfg):
                dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
            dfg_to_dfg = [x[-1] for x in dfg]
            dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
            length = len([tokenizer.cls_token])  # = 1
            dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]
            assert len(bug_ids) == len(bug_position_ids)
            cache.append([
                bug_tokens, bug_ids, bug_mask, bug_position_ids, dfg_to_code, dfg_to_dfg
            ])
        assert len(cache) == 2
        # target
        # if stage == 'test':
        #     fix_tokens = tokenizer.tokenize("None")
        # else:
        code_tokens, _ = extract_dataflow(example.fix_func, parsers['java'], 'java')
        code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x)
                       for idx, x in enumerate(code_tokens)]
        code_tokens = [y for x in code_tokens for y in x]

        fix_tokens = code_tokens[:args.max_target_length - 2]
        fix_tokens = [tokenizer.cls_token] + fix_tokens + [tokenizer.sep_token]
        fix_ids = tokenizer.convert_tokens_to_ids(fix_tokens)
        fix_mask = [1] * len(fix_ids)
        padding_length = args.max_target_length - len(fix_ids)
        fix_ids += [tokenizer.pad_token_id] * padding_length
        fix_mask += [0] * padding_length

        bug_item = cache[0]
        clean_item = cache[1]

        features.append(
            InputFeatures(
                example_index,
                *bug_item[1:],
                *clean_item[1:],
                fix_ids,
                fix_mask,
                type2label[example.bug_type]
            )
        )
        if example_index < 0:
            if stage == 'test':
                logging.info('** Example **')
                logging.info("bug_tokens: {}".format([x.replace('\u0120', '_') for x in cache[0][0]]))
                logging.info("bug_ids: {}".format(' '.join(map(str, cache[0][1]))))
                logging.info("bug_mask: {}".format(' '.join(map(str, cache[0][2]))))
                logging.info("bug_position_idx: {}".format(cache[0][3]))
                logging.info("dfg_to_code: {}".format(' '.join(map(str, cache[0][4]))))
                logging.info("dfg_to_dfg: {}".format(' '.join(map(str, cache[0][5]))))

                logging.info("clean_tokens: {}".format([x.replace('\u0120', '_') for x in cache[1][0]]))
                logging.info("clean_ids: {}".format(' '.join(map(str, cache[1][1]))))
                logging.info("clean_mask: {}".format(' '.join(map(str, cache[1][2]))))
                logging.info("clean_position_idx: {}".format(cache[1][3]))
                logging.info("dfg_to_code: {}".format(' '.join(map(str, cache[1][4]))))
                logging.info("dfg_to_dfg: {}".format(' '.join(map(str, cache[1][5]))))

                logging.info("fix_tokens: {}".format([x.replace('\u0120', '_') for x in fix_tokens]))
                logging.info("fix_ids: {}".format(' '.join(map(str, fix_ids))))
                logging.info("fix_mask: {}".format(' '.join(map(str, fix_mask))))
                logging.info("label: {}-{}".format(example.bug_type, type2label[example.bug_type]))

    return features


class TextDataset(Dataset):
    def __init__(self, examples, args):
        self.examples = examples
        self.args = args

    def __len__(self):
        return len(self.examples)

    def get_attn_mask(self, bug_ids, bug_position_ids, bug_dfg_to_code, bug_dfg_to_dfg):
        # calculate graph-guided masked
        bug_attn_mask = np.zeros((self.args.max_source_length + self.args.data_flow_length,
                                  self.args.max_source_length + self.args.data_flow_length), dtype=bool)

        # calculate begin index of node and max length of code
        bug_node_index = sum([i > 1 for i in bug_position_ids])
        bug_max_length = sum([i != 1 for i in bug_position_ids])

        # sequence can attend to seqence
        bug_attn_mask[:bug_node_index, :bug_node_index] = True

        # special tokens attend to all tokens
        for idx, i in enumerate(bug_ids):
            if i in [0, 2]:
                bug_attn_mask[idx, :bug_max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(bug_dfg_to_code):
            if a < bug_node_index and b < bug_node_index:
                bug_attn_mask[idx + bug_node_index, a:b] = True
                bug_attn_mask[a:b, idx + bug_node_index] = True

        # nodes attends to adjacent nodes
        for idx, nodes in enumerate(bug_dfg_to_dfg):
            for a in nodes:
                if a + bug_node_index < len(bug_position_ids):
                    bug_attn_mask[idx + bug_node_index, a + bug_node_index] = True

        return bug_attn_mask

    def __getitem__(self, item):
        bug_attn_mask = self.get_attn_mask(
            self.examples[item].bug_ids,
            self.examples[item].bug_position_ids,
            self.examples[item].bug_dfg_to_code,
            self.examples[item].bug_dfg_to_dfg
        )
        clean_attn_mask = self.get_attn_mask(
            self.examples[item].clean_ids,
            self.examples[item].clean_position_ids,
            self.examples[item].clean_dfg_to_code,
            self.examples[item].clean_dfg_to_dfg
        )

        return (torch.tensor(self.examples[item].bug_ids),
                torch.tensor(self.examples[item].bug_mask),
                torch.tensor(self.examples[item].bug_position_ids),
                torch.tensor(bug_attn_mask),
                torch.tensor(self.examples[item].clean_ids),
                torch.tensor(self.examples[item].clean_mask),
                torch.tensor(self.examples[item].clean_position_ids),
                torch.tensor(clean_attn_mask),
                torch.tensor(self.examples[item].fix_ids),
                torch.tensor(self.examples[item].fix_mask),
                torch.tensor(self.examples[item].label)
                )


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_yaml_data(yaml_file):
    # 打开yaml文件
    with open(yaml_file, 'r', encoding="utf-8") as f:
        file_data = f.read()

    config = yaml.load(file_data, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=False,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=False,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filename.")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename.")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_source_length", default=448, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=448, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--num_labels', type=int, default=16,
                        help="num of multiclass")
    parser.add_argument('--config', type=str, default="", required=True,
                        help="config file")
    parser.add_argument('--use_awl', action='store_true')
    # print arguments
    args = parser.parse_args()
    configs = get_yaml_data(args.config)
    args = vars(args)
    for key in configs.keys():
        args[key] = configs[key]
    args = argparse.Namespace(**args)
    logging.info(args)
    # Setup CUDA, GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.n_gpu = torch.cuda.device_count()
    logging.info(f'n_gpu: {args.n_gpu}')
    args.device = device

    # Set seed
    set_seed(args.seed)

    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name)
    config.num_labels = args.num_labels
    config.cls_dropout_prob = args.cls_dropout_prob
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)

    # build_model
    encoder = model_class.from_pretrained(args.model_name_or_path, config=config)
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model = Model(encoder=encoder, decoder=decoder,
                  config=config, beam_size=args.beam_size, max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id, eos_id=tokenizer_class.sep_token_id)

    if args.load_model_path is not None:
        logging.info(f'reload model from {args.load_model_path}')
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.do_train:
        logging.info('initialize train_dataloader')
        if args.num_labels == 2:
            pkl_filename = os.path.dirname(args.train_filename) + '/train_features_2.pkl'
        else:
            pkl_filename = os.path.dirname(args.train_filename) + '/train_features.pkl'

        if not os.path.exists(pkl_filename):
            train_examples = read_examples(args.train_filename)
            train_features = convert_examples_to_features(train_examples, tokenizer, args, stage='train')
            with open(pkl_filename, 'wb') as f:
                pickle.dump(train_features, f)
        else:
            with open(pkl_filename, 'rb') as f:
                train_features = pickle.load(f)
        train_data = TextDataset(train_features, args)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=args.train_batch_size // args.gradient_accumulation_steps,
                                      num_workers=4)
        # Prepare optimizer and schedule
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouoed_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouoed_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=len(
                                                        train_dataloader) * args.num_train_epochs * 0.1,
                                                    num_training_steps=len(train_dataloader) * args.num_train_epochs)

        # start training
        logging.info('**** Running training ****')
        logging.info(f'  Num examples = {len(train_features)}')
        logging.info(f'  Batch size = {args.train_batch_size}')
        logging.info(f'  Num epochs = {args.num_train_epochs}')

        model.train()
        dev_dataset = {}
        nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss = 0, 0, 0, 0, 0, 1e6
        tr_gen_loss, tr_cls_loss = 0, 0
        best_auc_ovo = 0
        best_auc_ovr = 0
        best_gen_loss = 1e6
        best_cls_loss = 1e6
        best_f1 = 0
        best_ppl = 1e6
        best_xmatch = 0
        for epoch in range(args.num_train_epochs):
            bar = tqdm(train_dataloader, total=len(train_dataloader))
            for batch in bar:
                batch = tuple(t.to(device) for t in batch)
                bug_ids, bug_mask, bug_position_ids, bug_attn_mask, clean_ids, clean_mask, clean_position_ids, clean_attn_mask, fix_ids, fix_mask, labels = batch
                loss = model(bug_ids, bug_mask, bug_position_ids, bug_attn_mask,
                                                                   clean_ids,
                                                                   clean_mask,
                                                                   clean_position_ids, clean_attn_mask, fix_ids,
                                                                   fix_mask, labels, use_awl=args.use_awl)

                if args.n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                tr_loss += loss.item()
                train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)

                bar.set_description(
                    f'epoch: {epoch} loss: {train_loss}')
                nb_tr_examples += bug_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
            if args.do_eval and epoch in [int(args.num_train_epochs) * (i + 1) // 20 for i in range(20)] + [
                args.num_train_epochs - 1]:
                tr_loss, tr_cls_loss, tr_gen_loss, nb_tr_steps, nb_tr_examples = 0, 0, 0, 0, 0
                if 'dev_loss' in dev_dataset:
                    eval_data = dev_dataset['dev_loss']
                else:
                    if args.num_labels == 2:
                        pkl_filename = os.path.dirname(args.dev_filename) + '/eval_loss_features_2.pkl'
                    else:
                        pkl_filename = os.path.dirname(args.dev_filename) + '/eval_loss_features.pkl'

                    if not os.path.exists(pkl_filename):
                        eval_examples = read_examples(args.dev_filename)
                        eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='dev')
                        with open(pkl_filename, 'wb') as f:
                            pickle.dump(eval_features, f)
                    else:
                        with open(pkl_filename, 'rb') as f:
                            eval_features = pickle.load(f)
                    eval_data = TextDataset(eval_features, args)
                    dev_dataset['dev_loss'] = eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                                             batch_size=args.eval_batch_size,
                                             num_workers=4)

                logging.info('  ***** Running evaluation *****')
                logging.info(f'  Num examples = {len(eval_features)}')
                logging.info(f'  Batch size = {args.eval_batch_size}')

                # start evaling model
                model.eval()
                eval_loss, eval_cls_loss, eval_gen_loss = 0, 0, 0
                eval_loss_a, tokens_num = 0, 0
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    bug_ids, bug_mask, bug_position_ids, bug_attn_mask, clean_ids, clean_mask, clean_position_ids, clean_attn_mask, fix_ids, fix_mask, labels = batch

                    with torch.no_grad():
                        loss = model(bug_ids, bug_mask, bug_position_ids,
                                                                         bug_attn_mask, clean_ids,
                                                                         clean_mask,
                                                                         clean_position_ids, clean_attn_mask, fix_ids,
                                                                         fix_mask, labels)

                    if args.n_gpu > 1:
                        loss = loss.mean()
                    eval_loss += loss.item()
                # pricing loss of dev dataset
                result = {
                          'global_step': global_step + 1,
                          'train_loss': round(train_loss, 5),
                          'eval_loss': round(eval_loss, 5),
                          }
                logging.info("  ***** Eval results *****")
                for key in sorted(result.keys()):
                    logging.info(f'  {key} = {str(result[key])}')
                logging.info('  ' + '*' * 20)

                # save last checkpoint
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model
                output_model_file = os.path.join(last_output_dir, 'pytorch_model.bin')
                torch.save(model_to_save.state_dict(), output_model_file)

                if eval_loss < best_loss:
                    logging.info("  Best Loss:%s", eval_loss)
                    logging.info("  " + "*" * 20)
                    best_loss = eval_loss

                    best_loss_output_dir = os.path.join(args.output_dir, 'checkpoint-best-loss')
                    if not os.path.exists(best_loss_output_dir):
                        os.makedirs(best_loss_output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(best_loss_output_dir, 'pytorch_model.bin')
                    torch.save(model_to_save.state_dict(), output_model_file)
                if 'dev_loss' in dev_dataset:
                    eval_data = dev_dataset['dev_loss']
                else:
                    if args.num_labels == 2:
                        pkl_filename = os.path.dirname(args.dev_filename) + '/eval_bleu_features_2.pkl'
                    else:
                        pkl_filename = os.path.dirname(args.dev_filename) + '/eval_bleu_features.pkl'

                    if not os.path.exists(pkl_filename):
                        eval_examples = read_examples(args.dev_filename)
                        # eval_examples = random.sample(eval_examples, min(64, len(eval_examples)))
                        eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='dev')
                        with open(pkl_filename, 'wb') as f:
                            pickle.dump(eval_features, f)
                    else:
                        with open(pkl_filename, 'rb') as f:
                            eval_features = pickle.load(f)
                    eval_data = TextDataset(eval_features, args)
                    dev_dataset['dev_bleu'] = eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                                             batch_size=args.eval_batch_size,
                                             num_workers=4)
                logging.info('\n ***** Running evaluate preds *****')
                logging.info(f'  Num examples = {len(eval_features)}')
                logging.info(f'  Batch size = {args.eval_batch_size}')
                model.eval()

                y_preds = []
                y_trues = []
                y_scores = []
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    bug_ids, bug_mask, bug_position_ids, bug_attn_mask, clean_ids, clean_mask, clean_position_ids, clean_attn_mask, fix_ids, fix_mask, labels = batch
                    with torch.no_grad():
                        cls_prob, preds = model(bug_ids, bug_mask, bug_position_ids, bug_attn_mask, clean_ids,
                                                clean_mask,
                                                clean_position_ids, clean_attn_mask, no_inf=False
                                                )
                    values, index = cls_prob.topk(1)
                    y_preds.append(index.squeeze().cpu().numpy())
                    y_trues.append(labels.squeeze().cpu().numpy())
                    y_scores.append(cls_prob.squeeze().cpu().numpy())

                # eval classification task
                # [all, num_label]
                y_preds = np.concatenate(y_preds, 0)
                y_trues = np.concatenate(y_trues, 0)
                y_scores = np.concatenate(y_scores, 0)
                # o = np.concatenate(o, 0)
                logging.info(np.unique(y_preds))
                logging.info(np.unique(y_trues))
                recall = recall_score(y_trues, y_preds, average='macro')
                precision = precision_score(y_trues, y_preds, average='macro')
                f1 = f1_score(y_trues, y_preds, average='macro')
                # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.

                if len(np.unique(y_trues)) == args.num_labels:
                    try:
                        auc_ovo = roc_auc_score(y_trues, y_scores, multi_class='ovo')
                        auc_ovr = roc_auc_score(y_trues, y_scores, multi_class='ovr')
                    except:
                        logging.info('compute auc error')
                        auc_ovr = 0
                        auc_ovo = 0
                else:
                    auc_ovr = 0
                    auc_ovo = 0

                logging.info("  ***** Eval results *****")

                result = {
                    "eval_recall": float(recall),
                    "eval_precision": float(precision),
                    "eval_f1": float(f1),
                    "eval_auc_ovo": float(auc_ovo),
                    "eval_auc_ovr": float(auc_ovr)
                }

                for key in sorted(result.keys()):
                    logging.info("  %s = %s", key, str(round(result[key], 4)))
                logging.info("  " + "*" * 20)

                if auc_ovo > best_auc_ovo:
                    logging.info("  Best AUC_ovo:%s", auc_ovo)
                    logging.info("  " + "*" * 20)
                    best_auc_ovo = auc_ovo
                    # save best checkpoint for best auc
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-auc-ovo')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(output_dir, 'pytorch_model.bin')
                    torch.save(model_to_save.state_dict(), output_model_file)
                if auc_ovr > best_auc_ovr:
                    logging.info("  Best AUC_ovr:%s", auc_ovr)
                    logging.info("  " + "*" * 20)
                    best_auc_ovr = auc_ovr
                    # save best checkpoint for best auc
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-auc-ovr')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(output_dir, 'pytorch_model.bin')
                    torch.save(model_to_save.state_dict(), output_model_file)
                if f1 > best_f1:
                    logging.info("  Best F1:%s", f1)
                    logging.info("  " + "*" * 20)
                    best_f1 = f1
                    # save best checkpoint for best auc
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-f1')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(output_dir, 'pytorch_model.bin')
                    torch.save(model_to_save.state_dict(), output_model_file)

    if args.do_test:
        files = []
        if args.test_filename is not None:
            files.append(args.test_filename)
        if args.dev_filename is not None:
            files.append(args.dev_filename)
        for idx, file in enumerate(files):
            logging.info(f'Test file: {file}')
            eval_examples = read_examples(file)
            # eval_examples = random.sample(eval_examples, min(16, len(eval_examples)))
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
            eval_data = TextDataset(eval_features, args)

            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                                         batch_size=args.eval_batch_size,
                                         num_workers=4)

            model.eval()
            y_preds = []
            y_trues = []
            y_scores = []
            o_bug = []
            o = []
            p = []
            logging.info('  ***** Running test *****')
            logging.info(f'  Num examples = {len(eval_examples)}')
            logging.info(f'  Batch size = {args.eval_batch_size}')
            for batch in eval_dataloader:
                batch = tuple(t.to(device) for t in batch)
                bug_ids, bug_mask, bug_position_ids, bug_attn_mask, clean_ids, clean_mask, clean_position_ids, clean_attn_mask, fix_ids, fix_mask, labels = batch
                with torch.no_grad():
                    cls_prob, preds = model(bug_ids, bug_mask, bug_position_ids, bug_attn_mask, clean_ids, clean_mask,
                                            clean_position_ids, clean_attn_mask, no_inf=False)
                values, index = cls_prob.topk(1)
                y_scores.append(cls_prob.cpu().numpy())
                y_preds.append(index.squeeze().cpu().numpy())
                y_trues.append(labels.cpu().numpy())
                o.append(fix_ids)
                o_bug.append(bug_ids)
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if tokenizer.sep_token_id in t:
                        t = t[:t.index(tokenizer.sep_token_id)]
                    text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                    # logging.info(t)
                    # logging.info(text)
                    p.append(text)

            model.train()

            y_preds = np.concatenate(y_preds, 0)
            y_trues = np.concatenate(y_trues, 0)
            y_scores = np.concatenate(y_scores, 0)

            logging.info(np.unique(y_preds))
            logging.info(np.unique(y_trues))
            recall = recall_score(y_trues, y_preds, average='macro')
            precision = precision_score(y_trues, y_preds, average='macro')
            f1 = f1_score(y_trues, y_preds, average='macro')
            recall_w = recall_score(y_trues, y_preds, average='weighted')
            precision_w = precision_score(y_trues, y_preds, average='weighted')
            f1_w = f1_score(y_trues, y_preds, average='weighted')
            # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.

            # binary task
            b_y_preds = [0 if i == 0 else 1 for i in y_preds]
            b_y_trues = [0 if i == 0 else 1 for i in y_trues]
            b_y_scores = [1 - i[0] for i in y_scores]
            if len(np.unique(b_y_trues)) == 2:
                b_recall = recall_score(b_y_trues, b_y_preds)
                b_precision = precision_score(b_y_trues, b_y_preds)
                b_f1 = f1_score(b_y_trues, b_y_preds)
                b_auc = roc_auc_score(b_y_trues, b_y_scores)
            else:
                b_recall = 0
                b_precision = 0
                b_f1 = 0
                b_auc = 0
            if len(np.unique(y_trues)) == args.num_labels:
                try:
                    auc_ovo = roc_auc_score(y_trues, y_scores, multi_class='ovo')
                    auc_ovr = roc_auc_score(y_trues, y_scores, multi_class='ovr')
                    auc_ovo_w = roc_auc_score(y_trues, y_scores, multi_class='ovo', average='weighted')
                    auc_ovr_w = roc_auc_score(y_trues, y_scores, multi_class='ovr', average='weighted')
                except Exception as e:
                    logging.info('compute auc error')
                    logging.error(e)
                    logging.info((y_trues.shape, y_scores.shape))
                    auc_ovr = 0
                    auc_ovo = 0
                    auc_ovo_w = 0
                    auc_ovr_w = 0
            else:
                auc_ovr = 0
                auc_ovo = 0
                auc_ovo_w = 0
                auc_ovr_w = 0

            logging.info("  ***** Test results *****")

            result = {
                "eval_recall": float(recall),
                "eval_precision": float(precision),
                "eval_f1": float(f1),
                "eval_auc_ovo": float(auc_ovo),
                "eval_auc_ovr": float(auc_ovr),
                "eval_auc_ovo_w": float(auc_ovo_w),
                "eval_auc_ovr_w": float(auc_ovr_w),
                "eval_binary_recall": float(b_recall),
                "eval_binary_precision": float(b_precision),
                "eval_binary_f1": float(b_f1),
                "eval_binary_auc": float(b_auc),
                "eval_recall_w": float(recall_w),
                "eval_precision_w": float(precision_w),
                "eval_f1_w": float(f1_w),
            }

            for key in sorted(result.keys()):
                logging.info("  %s = %s", key, str(round(result[key], 4)))
            logging.info("  " + "*" * 20)
            return
            # detailed classification
            type2label = pd.read_json(args.type2label, orient='index')
            label2type = dict(zip(type2label.loc[:, 0], type2label.index))
            df_data = []
            for i, j in zip(y_preds, y_trues):
                if i == j:
                    # print(label2type[i])
                    df_data.append(label2type[i])
            from collections import Counter
            counter = Counter(df_data)
            print(counter)
            # return
            o = torch.cat(o, 0)
            o_bug = torch.cat(o_bug, 0)
            predictions = []
            accs = []

            with open(os.path.join(args.output_dir, f'{os.path.basename(file).split(".")[0]}.output'), 'w') as f, open(
                    os.path.join(args.output_dir, f'{os.path.basename(file).split(".")[0]}.gold'), 'w') as f1, open(
                os.path.join(args.output_dir, f'{os.path.basename(file).split(".")[0]}.y_true'), 'w') as f2, open(
                os.path.join(args.output_dir, f'{os.path.basename(file).split(".")[0]}.y_pred'), 'w') as f3, open(
                os.path.join(args.output_dir, f'{os.path.basename(file).split(".")[0]}.y_score'), 'w') as f4:
                print("i'm here")
                for i, (ref, t, t_bug) in enumerate(zip(p, o, o_bug)):
                    print("REF", ref)
                    predictions.append(ref)
                    t = list(t)
                    if tokenizer.sep_token_id in t:
                        t = t[1:t.index(tokenizer.sep_token_id)]
                    t = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                    t_bug = list(t_bug)
                    if tokenizer.sep_token_id in t_bug:
                        t_bug = t_bug[1:t_bug.index(tokenizer.sep_token_id)]
                    t_bug = tokenizer.decode(t_bug, clean_up_tokenization_spaces=False)
                    f.write(ref + '\n')
                    f1.write(t + '\n')
                    f2.write(str(y_trues[i]) + '\n')
                    f3.write(str(y_preds[i]) + '\n')
                    f4.write(str(y_scores[i]) + '\n')
                    accs.append(ref == t)
            dev_bleu = round(
                _bleu(os.path.join(args.output_dir, f'{os.path.basename(file).split(".")[0]}.gold'),
                      os.path.join(args.output_dir, f'{os.path.basename(file).split(".")[0]}.output')), 2)
            xmatch = round(np.mean(accs) * 100, 4)
            logging.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
            logging.info("  %s = %s " % ("xMatch", str(xmatch)))
            logging.info("  " + "*" * 20)


if __name__ == '__main__':
    main()
