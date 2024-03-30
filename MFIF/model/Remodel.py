"""
@Time    : 2023/6/14 16:26
@Author  : kilig
@FileName: model.py
@IDE     : PyCharm
"""
import torch
import torch.nn as nn
from transformers import BertModel
from modules.GPLinkerlayer import GlobalPointer
from modules.DGCNN import IDCNN
from modules.Muti_DGCNN import IDCNN_1
from modules.attention import MultiHeadedAttention,LocalMultiHeadedAttention,AtrousMultiHeadedAttention
class REmodel(nn.Module):
    def __init__(self, config):
        super(REmodel, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.bert_path)
        self.entity_model = GlobalPointer(self.config.bert_dim, 2,self.config.hidden_size)
        self.head_model = GlobalPointer(self.config.bert_dim, self.config.num_rel,self.config.hidden_size, RoPE=False,tril_mask=False)
        self.tail_model = GlobalPointer(self.config.bert_dim, self.config.num_rel,self.config.hidden_size,RoPE=False,tril_mask=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(self.config.bert_dim,self.config.bert_dim)
        self.dropout = SpatialDropout(0.1)
        self.layer_norm = nn.LayerNorm(self.config.bert_dim)
        self.idcnn = IDCNN(self.config.bert_dim, self.config.hidden_size)
        self.idcnn_1 = IDCNN_1(self.config.bert_dim, self.config.hidden_size)
        self.linear1 = nn.Linear(self.config.hidden_size, self.config.bert_dim)
        self.linear2 = nn.Linear(self.config.bert_dim*2, self.config.bert_dim)
        self.linear2.weight = nn.Parameter(torch.ones(self.config.bert_dim,self.config.bert_dim*2))
        self.attention =AtrousMultiHeadedAttention(self.config.bert_dim,8)
        # self.attention_1 = MultiHeadedAttention(self.config.bert_dim, 8)
        self.rel_emd = nn.Embedding(self.config.num_rel,self.config.bert_dim)
        self.rel_judgement = MultiNonLinearClassifier(self.config.bert_dim, self.config.num_rel, 0.3)
        self.linear3 = nn.Linear(self.config.bert_dim,self.config.bert_dim)
        self.lstm = nn.LSTM(self.config.bert_dim,self.config.bert_dim//2,num_layers=1,bidirectional=True,batch_first=True)


    def forward(self,data):
        input_ids = data["input_ids"].to(self.device)
        attention_mask = data["attention_mask"].to(self.device)
        rel_mask = data["rel_list"].to(self.device)
        hidden_state = self.bert(input_ids, attention_mask=attention_mask)[0]

        # lstm_state = self.linear3(hidden_state)
        # lstm,_ = self.lstm(lstm_state)
        # lstm = self.dropout(lstm)

        rel_emd = self.rel_emd(rel_mask)
        rel_emd= self.attention(rel_emd)
        h_k_avg = self.masked_avgpool(hidden_state, attention_mask)
        rel_pred = self.rel_judgement(h_k_avg,rel_emd)



        ent_state = rel_pred+ hidden_state


        entity_logits = self.entity_model(ent_state, attention_mask)

        # #
        entity = torch.where(entity_logits==-1.2500e+11,torch.tensor(0.0).cuda(),entity_logits)
        entity = torch.where(entity== -2.5000e+11, torch.tensor(0.0).cuda(), entity)

        rel = torch.chunk(entity,2,dim=1)

        h = rel[0].permute(0,2,3,1)
        t = rel[1].permute(0,2,3,1)

        ent = h+t
        ent = ent.squeeze(-1)

        h = torch.sum(ent,dim=-2)
        t = torch.sum(ent,dim=-1)

        h,t = h.unsqueeze(-1),t.unsqueeze(-1)

        h = self.sigmoid(h)
        t = self.sigmoid(t)



        h,t = torch.where(h==5e-01,torch.tensor(0.0).cuda(),h),torch.where(t==5e-01,torch.tensor(0.0).cuda(),t)
        h_state = h*hidden_state
        h_state = self.linear(h_state)
        h_state = self.dropout(h_state)
        # h_state,_ = self.lstm(h_state)
        # h_state = self.dropout(h_state)
        t_state = t*hidden_state
        t_state = self.linear(t_state)
        t_state = self.dropout(t_state)
        # t_state,_ = self.lstm(t_state)
        # t_state = self.dropout(t_state)
        h_state = h_state+hidden_state
        t_state = t_state+hidden_state

        h_state = self.layer_norm(h_state)
        t_state = self.layer_norm(t_state)

        head_logits = self.head_model(h_state, attention_mask)
        tail_logits = self.tail_model(t_state ,attention_mask)

        return entity_logits,head_logits,tail_logits

    @staticmethod
    def masked_avgpool(sent, mask):
        mask_ = mask.masked_fill(mask == 0, -1e9).float()
        score = torch.softmax(mask_, -1)
        return torch.matmul(score.unsqueeze(1), sent)

class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, tag_size, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
        self.hidden2tag = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.dropout = SpatialDropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features,rel_mask):

        features_tmp = self.linear(input_features)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = self.dropout(features_tmp)
        features_output = self.hidden2tag(features_tmp)

        rel_weight = torch.matmul(features_output,rel_mask)

        return rel_weight

class SpatialDropout(nn.Dropout2d):
    def __init__(self, p=0.6):
        super(SpatialDropout, self).__init__(p=p)

    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x