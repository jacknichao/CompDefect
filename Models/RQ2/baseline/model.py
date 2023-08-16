import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from utils.myLogger import logging


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = features[:, 0, :]  # [bsz*2, hidden]
        x = x.reshape(-1, x.size(-1) * 2)  # [bsz, 2*hidden]

        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Model(nn.Module):
    """
        Build Seqence-to-Sequence.

        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model.
        * `beam_size`- beam size for beam search.
        * `max_length`- max length of target for beam search.
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search.
    """

    def __init__(self, encoder, config):
        super(Model, self).__init__()
        self.encoder = encoder
        self.classifier = RobertaClassificationHead(config)
        self.config = config
        self.sm = nn.Softmax(dim=-1)

    def forward(self, bug_ids, bug_mask,
                clean_ids, clean_mask,
                labels):
        bs, l = bug_ids.size()
        inputs_ids = torch.cat((bug_ids.unsqueeze(1), clean_ids.unsqueeze(1)), 1).view(bs * 2, l)
        attn_mask = torch.cat((bug_mask.unsqueeze(1), clean_mask.unsqueeze(1)), 1).view(bs * 2,  l)

        # [bsz*2, l, hidden]
        outputs = \
            self.encoder(input_ids=inputs_ids, attention_mask=attn_mask)[
                0]

        # classification task
        # [bsz, h], [bsz, h]
        cls_logits = self.classifier(outputs)
        cls_prob = self.sm(cls_logits)

        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        cls_loss = loss_fct(cls_logits, labels)

        return cls_loss, cls_prob
