import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from bert import BERTencoder, BERTclassifier, Tagger
import constant

from transformers import AdamW

class Trainer(object):
    def __init__(self, cuda):
        self.encoder = BERTencoder()
        self.classifier = BERTclassifier(opt)
        self.tagger = Tagger()
        self.criterion = nn.CrossEntropyLoss()
        self.criterion2 = nn.BCELoss()
        self.parameters = [p for p in self.classifier.parameters() if p.requires_grad] + [p for p in self.encoder.parameters() if p.requires_grad]+ [p for p in self.tagger.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            self.parameters,
            lr=1e-5,
        )

    def update(self, inputs, epoch):
        self.encoder.train()
        self.classifier.train()
        self.tagger.train()
        self.optimizer.zero_grad()
        words, subj_pos, obj_pos, relation, tagging, has_tag = inputs
        h, b_out = self.encoder(words)
        tagging_output = self.tagger(h)
        # binary relation loss
        loss = self.criterion2(b_out, (~labels).eq(0).to(torch.float32).unsqueeze(1))
        
        if has_tag:
            # tagging loss
            loss += self.criterion2(tagging_output, tagging.unsqueeze(1).to(torch.float32))
            logits = self.classifier(h, tagging, subj_pos, obj_pos)
            # relation extraction loss
            loss += self.criterion(logits, labels)
        elif relation != -1:
            if epoch <= 20:
                ??
            else:
                ???
        loss_val = loss.item()
        # backward
        loss.backward()
        self.optimizer.step()

        return loss_val

    def predict(self, batch):
        self.encoder.eval()
        self.classifier.eval()
        self.tagger.eval()
