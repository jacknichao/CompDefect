import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from utils.myLogger import logging


class ClassificationHead(nn.Module):
    """Head for CompDefect classification task."""

    def __init__(self, config):
        super(ClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size * 2 + 2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
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
        x = features[:, 0, :]
        hidden_size = x.size(-1)
        x = x.reshape(-1, x.size(-1) * 2)

        inputs1, inputs2 = x[:, :hidden_size], x[:, hidden_size:]
        ntn = self.neural_network_tensor_layer(inputs1, inputs2)

        x = torch.cat((x, ntn), dim=1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        y = x
        x = self.dropout(x)
        x = self.out_proj(x)
        return x, y


class CompDefect(nn.Module):
    """
        Build CompDefect model (encoder, classifier, decoder).

        Parameters:

        * `encoder`- encoder of CompDefect model.
        * `decoder`- decoder of CompDefect model.
        * `config`- configuration of encoder model.
        * `beam_size`- beam size for beam search.
        * `max_length`- max length of target for beam search.
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search.
    """

    def __init__(self, encoder, decoder, config, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(CompDefect, self).__init__()
        self.encoder = encoder
        # generator
        self.decoder = decoder
        self.classifier = ClassificationHead(config)
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
        nodes_mask = position_ids.eq(0)
        token_mask = position_ids.ge(2)

        inputs_embedding = self.encoder.embeddings.word_embeddings(source_ids)
        nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]

        avg_embeddings = torch.einsum('abc, acd->abd', nodes_to_token_mask, inputs_embedding)
        inputs_embedding = inputs_embedding * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]

        return inputs_embedding

    def forward(self, bug_ids, bug_mask, bug_position_ids, bug_attn_mask,
                clean_ids, clean_mask, clean_position_ids, clean_attn_mask,
                fix_ids=None, fix_mask=None, labels=None):
        bs, l = bug_ids.size()
        inputs_ids = torch.cat((bug_ids.unsqueeze(1), clean_ids.unsqueeze(1)), 1).view(bs * 2, l)
        position_ids = torch.cat((bug_position_ids.unsqueeze(1), clean_position_ids.unsqueeze(1)), 1).view(bs * 2, l)
        attn_mask = torch.cat((bug_attn_mask.unsqueeze(1), clean_attn_mask.unsqueeze(1)), 1).view(bs * 2, l, l)

        inputs_embeddings = self.encoder_inputs_embedding(inputs_ids, position_ids, attn_mask)
        outputs = \
            self.encoder(inputs_embeds=inputs_embeddings, attention_mask=attn_mask, position_ids=position_ids)[
                0]

        # classification task
        cls_logits, cls_embedding = self.classifier(outputs)

        # generation task
        indices = [i for i in range(0, bs * 2, 2)]
        indices = torch.tensor(indices).cuda()
        bug_outputs = outputs.index_select(0, indices)

        outputs = bug_outputs
        outputs[:, 0, :] = cls_embedding
        encoder_output = outputs.permute([1, 0, 2]).contiguous()
        if fix_ids is not None:
            attn_mask = -1e4 * (1 - self.bias[:fix_ids.shape[1], :fix_ids.shape[1]])
            tgt_embeddings = self.encoder.embeddings(fix_ids).permute([1, 0, 2]).contiguous()
            out = self.decoder(tgt_embeddings, encoder_output, tgt_mask=attn_mask,
                               memory_key_padding_mask=(1 - bug_mask).bool())
            hidden_states = torch.tanh(self.dense(out)).permute([1, 0, 2]).contiguous()
            lm_logits = self.lm_head(hidden_states)
            # shift, so than tokens < n predict n
            active_loss = fix_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()

            shift_labels = fix_ids[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            gen_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                                shift_labels.view(-1)[active_loss])
            cls_loss = loss_fct(cls_logits, labels)
            loss = self.awl(cls_loss, gen_loss)
            return loss, cls_loss, gen_loss, loss * active_loss.sum(), active_loss.sum()

        else:
            # Predict
            # classification
            cls_prob = self.sm(cls_logits)
            # generation
            preds = []
            zero = torch.cuda.LongTensor(1).fill_(0)
            beams = [Beam(self.beam_size, self.sos_id, self.eos_id) for _ in range(bug_ids.size(0))]
            context = encoder_output
            context_mask = bug_mask
            context = torch.cat(
                [context[:, i:i + 1, :].repeat(1, self.beam_size, 1) for i in range(bug_ids.size(0))], dim=1)
            context_mask = torch.cat(
                [context_mask[i:i + 1, :].repeat(self.beam_size, 1) for i in range(bug_ids.size(0))], dim=0)
            # Construct batch x beam_size nxt words.
            input_ids = torch.cat([b.getCurrentState() for b in beams], dim=0)
            input_ids = input_ids.view(-1, 1)

            for _ in range(self.max_length):
                if all((b.done() for b in beams)):
                    break
                attn_mask = -1e4 * (1 - self.bias[:input_ids.shape[1], :input_ids.shape[1]])
                tgt_embeddings = self.encoder.embeddings(input_ids).permute([1, 0, 2]).contiguous()
                out = self.decoder(tgt_embeddings, context, tgt_mask=attn_mask,
                                   memory_key_padding_mask=(1 - context_mask).bool())
                out = torch.tanh(self.dense(out))
                hidden_states = out.permute([1, 0, 2]).contiguous()[:, -1, :]
                out = self.lsm(self.lm_head(hidden_states)).data
                out = out.view(bug_ids.size(0), self.beam_size, -1)

                input_ids = input_ids.view(bug_ids.size(0), self.beam_size, -1)
                input_ids_array = []
                for j, b in enumerate(beams):
                    b.advance(out[j, :])
                    input_ids[j].data.copy_(input_ids[j].data.index_select(0, b.getCurrentOrigin()))
                    tmp = torch.cat((input_ids[j], b.getCurrentState()), -1)
                    input_ids_array.append(tmp)
                input_ids = torch.cat(input_ids_array, dim=0)
            # Extract sentences from beam.
            for b in beams:
                hyp = b.getHyp(b.getFinal())
                pred = b.buildTargetTokens(hyp)[:self.beam_size]

                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p in
                        pred]
                preds.append(torch.cat(pred, 0).unsqueeze(0))

            preds = torch.cat(preds, 0)
            return cls_prob, preds


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
        self.prevKs.append(prevK)  # 
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
