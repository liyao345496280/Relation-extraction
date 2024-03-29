
class Config(object):
    def __init__(self):
        self.dataset = "webv"
        self.num_rel = 171
        self.batch_size = 4
        self.hidden_size = 64
        self.learning_rate = 1e-5
        # self.bert_path = "./RoBERTa_zh_Large_PyTorch"
        self.bert_path = "./pre_model/bert-small"
        self.train_fn = "./dataset/{}/train_triples.json".format(self.dataset)
        self.dev_fn = "./dataset/{}/dev_triples.json".format(self.dataset)
        self.test_fn = "./dataset/{}/test_triples.json".format(self.dataset)
        self.schema_fn = "./dataset/{}/rel2id.json".format(self.dataset)
        self.checkpoint = "E:\pythonProject\guanxichouqu\TPlinker\cp"
        self.dev_result = "dev_result/dev.json"
        self.epochs = 100
        self.RoPE = True
        self.bert_dim = 512
        self.max_len = 128
