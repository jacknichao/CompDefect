import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from utils.myLogger import logging


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size * 2 + 2, config.hidden_size)
        self.dropout = nn.Dropout(config.cls_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        # neural network tensor
        self.W_nn_tensor_one = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_nn_tensor_two = nn.Linear(config.hidden_size, config.hidden_size)
        self.V_nn_tensor = nn.Linear(config.hidden_size * 2, 2)

    def neural_network_tensor_layer(self, inputs1, inputs2):
        b = inputs1.size(0)
        output_one = self.W_nn_tensor_one(inputs1)
        output_one = torch.mul(output_one, inputs2)
        output_one = torch.sum(output_one, dim=1).view(b, 1)

        output_two = self.W_nn_tensor_two(inputs1)
        output_two = torch.mul(output_two, inputs2)
        output_two = torch.sum(output_two, dim=1).view(b, 1)

        W_output = torch.cat((output_one, output_two), dim=1)
        code = torch.cat((inputs1, inputs2), dim=1)
        V_output = self.V_nn_tensor(code)
        return F.relu(W_output + V_output)

    def forward(self, features):
        x = features[:, 0, :]  # [bsz*2, hidden]
        hidden_size = x.size(-1)
        x = x.reshape(-1, x.size(-1) * 2)  # [bsz, 2*hidden]

        inputs1, inputs2 = x[:, :hidden_size], x[:, hidden_size:]
        ntn = self.neural_network_tensor_layer(inputs1, inputs2)

        x = torch.cat((x, ntn), dim=1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        y = x.detach()
        x = self.dropout(x)
        x = self.out_proj(x)
        return x, y

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

    def __init__(self, encoder, decoder, config, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = RobertaClassificationHead(config)
        self.config = config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.sm = nn.Softmax(dim=-1)
        self.tie_weights()

        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)

    def encoder_inputs_embedding(self, source_ids, position_ids, attn_mask):
        nodes_mask = position_ids.eq(0)  # [bsz, seq]
        token_mask = position_ids.ge(2)

        inputs_embedding = self.encoder.embeddings.word_embeddings(source_ids)  # [bsz, seq, hidden]
        nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
        # [bsz, sql, hidden]
        avg_embeddings = torch.einsum('abc, acd->abd', nodes_to_token_mask, inputs_embedding)
        inputs_embedding = inputs_embedding * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]

        return inputs_embedding

    def forward(self, bug_ids, bug_mask, bug_position_ids, bug_attn_mask,
                clean_ids, clean_mask, clean_position_ids, clean_attn_mask,
                fix_ids=None, fix_mask=None, labels=None, no_inf=True, use_awl=False):
        bs, l = bug_ids.size()
        inputs_ids = torch.cat((bug_ids.unsqueeze(1), clean_ids.unsqueeze(1)), 1).view(bs * 2, l)
        position_ids = torch.cat((bug_position_ids.unsqueeze(1), clean_position_ids.unsqueeze(1)), 1).view(bs * 2, l)
        attn_mask = torch.cat((bug_attn_mask.unsqueeze(1), clean_attn_mask.unsqueeze(1)), 1).view(bs * 2, l, l)

        inputs_embeddings = self.encoder_inputs_embedding(inputs_ids, position_ids, attn_mask)
        # [bsz*2, l, hidden]
        outputs = \
            self.encoder(inputs_embeds=inputs_embeddings, attention_mask=attn_mask, position_ids=position_ids)[
                0]

        # classification task
        # [bsz, h], [bsz, h]
        cls_logits, cls_embedding = self.classifier(outputs)
        if fix_ids is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            cls_loss = loss_fct(cls_logits, labels)
            return cls_loss
        else:
            # Predict
            # classification
            cls_prob = self.sm(cls_logits)
            return cls_prob, []


class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                           .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)  # [beam, vocab_n]

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]  # [1, 1, vocab_n]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)  # 来自哪个beam
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size - len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
