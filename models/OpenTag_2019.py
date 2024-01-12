from .basic_module import BasicModule
from transformers import BertModel
import torch
from torchcrf import CRF
from .squeeze_embedding import SqueezeEmbedding

class OpenTag2019(BasicModule):
    def __init__(self, opt, *args, **kwargs):
        super(OpenTag2019, self).__init__(*args, **kwargs)

        self.model_name = 'opentag2019'
        self.opt = opt
        self.embedding_dim = opt.embedding_dim
        self.hidden_dim = opt.hidden_dim
        self.tagset_size = opt.tagset_size

        self.bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.word_embeds = torch.nn.Embedding(30000, self.opt.embedding_dim)

        self.dropout = torch.nn.Dropout(opt.dropout).to(opt.device)

        self.squeeze_embedding = SqueezeEmbedding()

        #CRF
        self.lstm = torch.nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True).to(opt.device)

        self.hidden2tag = torch.nn.Linear(self.hidden_dim*2, self.tagset_size).to(opt.device)
        self.crf = CRF(self.tagset_size, batch_first=True).to(opt.device)

    def calculate_cosin(self, context_output, att_hidden):
        '''
        context_output (batchsize, seqlen, hidden_dim)
        att_hidden (batchsize, hidden_dim)
        '''
        batchsize, seqlen, hidden_dim = context_output.size()
        att_hidden = att_hidden.unsqueeze(1).repeat(1,seqlen,1)

        context_output = context_output.float()
        att_hidden = att_hidden.float()

        cos = torch.sum(context_output*att_hidden, dim=-1)/(torch.norm(context_output, dim=-1)*torch.norm(att_hidden, dim=-1))
        cos = cos.unsqueeze(-1)
        cos_output = context_output*cos
        outputs = torch.cat([context_output, cos_output], dim=-1)

        return outputs

    def forward(self, inputs):
        context, att, target = inputs[0], inputs[1], inputs[2]
        context_len = torch.sum(context != 0, dim=-1)
        att_len = torch.sum(att != 0, dim=-1)

        attention_mask_context = torch.zeros_like(context)
        attention_mask_att = torch.zeros_like(att)
        # If the value in context is not 0, then the value in attention_mask_context is 1
        attention_mask_context[context != 0] = 1
        attention_mask_att[att != 0] = 1

        context = self.squeeze_embedding(context, context_len)
        attention_mask_context = self.squeeze_embedding(attention_mask_context, context_len)
        context = self.bert(context, attention_mask=attention_mask_context).last_hidden_state
        # context = self.word_embeds(context)
        context_output, _ = self.lstm(context)

        att = self.squeeze_embedding(att, att_len)
        attention_mask_att = self.squeeze_embedding(attention_mask_att, att_len)
        att = self.bert(att, attention_mask=attention_mask_att).last_hidden_state
        # att = self.word_embeds(att)
        _, att_hidden = self.lstm(att)
        att_hidden = torch.cat([att_hidden[0][-2],att_hidden[0][-1]], dim=-1)

        outputs = self.calculate_cosin(context_output, att_hidden)
        outputs = self.dropout(outputs)

        outputs = self.hidden2tag(outputs)
        #CRF
        # outputs = outputs.transpose(0,1).contiguous()
        outputs = self.crf.decode(outputs)
        return outputs



    def log_likelihood(self, inputs):
        context, att, target = inputs[0], inputs[1], inputs[2]
        context_len = torch.sum(context != 0, dim=-1)
        att_len = torch.sum(att != 0, dim=-1)
        target_len = torch.sum(target != 0, dim=-1)

        target = self.squeeze_embedding(target, target_len)
        # target = target.transpose(0,1).contiguous()

        attention_mask_context = torch.zeros_like(context)
        attention_mask_att = torch.zeros_like(att)
        # If the value in context is not 0, then the value in attention_mask_context is 1
        attention_mask_context[context != 0] = 1
        attention_mask_att[att != 0] = 1

        context = self.squeeze_embedding(context, context_len)
        attention_mask_context = self.squeeze_embedding(attention_mask_context, context_len)
        context = self.bert(context, attention_mask=attention_mask_context).last_hidden_state
        # context = self.word_embeds(context)
        context_output, _ = self.lstm(context)

        att = self.squeeze_embedding(att, att_len)
        attention_mask_att = self.squeeze_embedding(attention_mask_att, att_len)
        att = self.bert(att, attention_mask=attention_mask_att).last_hidden_state
        # att = self.word_embeds(att)
        _, att_hidden = self.lstm(att)
        att_hidden = torch.cat([att_hidden[0][-2],att_hidden[0][-1]], dim=-1)

        outputs = self.calculate_cosin(context_output, att_hidden)
        outputs = self.dropout(outputs)

        outputs = self.hidden2tag(outputs)
        #CRF
        # outputs = outputs.transpose(0,1).contiguous()

        return - self.crf(outputs, target)

