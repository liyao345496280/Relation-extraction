import torch
from model.Remodel import REmodel
from torch.utils.data import DataLoader
from dataloader_E import MyDataset, collate_fn
from tqdm import tqdm
import json
import numpy as np
import os

# from loss.loss import multilabel_categorical_crossentropy
from bert4torch.callbacks import AdversarialTraining
# class FGM():
#     def __init__(self, model):
#         self.model = model
#         self.backup = {}
#
#     def attack(self, epsilon=1., emb_name='word_embeddings'):
#         # emb_name这个参数要换成你模型中embedding的参数名
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and emb_name in name:
#                 self.backup[name] = param.data.clone()
#                 norm = torch.norm(param.grad)
#                 if norm != 0 and not torch.isnan(norm):
#                     r_at = epsilon * param.grad / norm
#                     param.data.add_(r_at)
#
#     def restore(self, emb_name='word_embeddings'):
#         # emb_name这个参数要换成你模型中embedding的参数名
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and emb_name in name:
#                 assert name in self.backup
#                 param.data = self.backup[name]
#         self.backup = {}

class Framework(object):
    def __init__(self, config):
        self.config = config
        self.adversarial_train = AdversarialTraining('vat')
        with open(self.config.schema_fn, "r", encoding="utf-8") as f:
            self.id2label = json.load(f)[1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        def sparse_multilabel_categorical_crossentropy(y_true=None, y_pred=None, mask_zero=False):
            '''
            稀疏多标签交叉熵损失的torch实现
            '''
            shape = y_pred.shape
            y_true = y_true[..., 0] * shape[2] + y_true[..., 1]
            y_pred = y_pred.reshape(shape[0], -1, np.prod(shape[2:]))
            zeros = torch.zeros_like(y_pred[..., :1])
            y_pred = torch.cat([y_pred, zeros], dim=-1)
            if mask_zero:
                infs = zeros + 1e12
                y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)
            y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
            y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
            if mask_zero:
                y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
                y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
            pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
            all_loss = torch.logsumexp(y_pred, dim=-1)
            aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss
            aux_loss = torch.clip(1 - torch.exp(aux_loss), 1e-10, 1)
            neg_loss = all_loss + torch.log(aux_loss)
            loss = torch.mean(torch.sum(pos_loss + neg_loss))
            return loss

        dataset = MyDataset(self.config, self.config.train_fn,is_test=False)
        dev_dataset = MyDataset(self.config, self.config.dev_fn,is_test=False)

        dataloader = DataLoader(dataset, shuffle=True, batch_size=self.config.batch_size,
                                collate_fn=collate_fn, pin_memory=True)
        dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=1,
                                    collate_fn=collate_fn, pin_memory=True)

        model = REmodel(self.config).to(self.device)
        torch.backends.cudnn.enabled = False
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        best_epoch = 0
        best_f1_score = 0
        global_step = 0
        global_loss = 0
        p, r = 0, 0
        # fgm = FGM(model)
        for epoch in range(self.config.epochs):

            for data in tqdm(dataloader):

                model.train()
                optimizer.zero_grad()
                rel_logtis, head_logits, tail_logits = model(data)

                rel_loss = sparse_multilabel_categorical_crossentropy(data["entity_list"].to(self.device), rel_logtis, True)
                head_loss = sparse_multilabel_categorical_crossentropy(data["head_list"].to(self.device), head_logits, True)
                tail_loss = sparse_multilabel_categorical_crossentropy(data["tail_list"].to(self.device), tail_logits, True)

                loss = sum([rel_loss + head_loss +tail_loss]) / 3
                loss.backward()
                # fgm.attack()
                #
                # adv_rel_logtis, adv_head_logits, adv_tail_logits = model(data)
                #
                # adv_rel_loss = sparse_multilabel_categorical_crossentropy(data["entity_list"].to(self.device), adv_rel_logtis,
                #                                                       True)
                # adv_head_loss = sparse_multilabel_categor
                # ical_crossentropy(data["head_list"].to(self.device), adv_head_logits,
                #                                                        True)
                # adv_tail_loss = sparse_multilabel_categorical_crossentropy(data["tail_list"].to(self.device), adv_tail_logits,
                #                                                        True)
                # adv_loss = sum([adv_rel_loss + adv_head_loss + adv_tail_loss]) / 3
                # adv_loss.backward()
                # fgm.restore()
                optimizer.step()

                global_loss += loss.item()
                global_step += 1

            print("epoch {} global_step: {} global_loss: {:5.4f}".format(epoch, global_step, global_loss))

            global_loss = 0
            if (epoch + 1) % 1==0:
                precision, recall, f1_score, predict = self.evaluate(model, dev_dataloader)
                if best_f1_score < f1_score:
                    best_f1_score = f1_score
                    p, r = precision, recall
                    best_epoch = epoch
                    json.dump(predict, open(self.config.dev_result, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
                    print(
                        "epoch {} precision: {:5.4f} recall: {:5.4f} f1_score: {:5.4f} best_epoch: {}".format(epoch, p,
                                                                                                              r,
                                                                                                              f1_score,
                                                                                                              best_epoch))
                    print("save model......")
                    # torch.save(model.state_dict(), self.config.checkpoint)
        print("best_epoch: {} precision: {:5.4f} recall: {:5.4f} f1_score: {:5.4f}".format(best_epoch, p, r, best_f1_score))
    #
    # def evaluate(self, model, dataloader, threshold=-0.5):
    #
    #     model.eval()
    #     predict_num, gold_num, correct_num = 0, 0, 0
    #     predict = []
    #
    #     def to_tuple(data):
    #         tuple_data = []
    #         for i in data:
    #             tuple_data.append(tuple(i))
    #         return tuple(tuple_data)
    #
    #     with torch.no_grad():
    #         for data in tqdm(dataloader):
    #             text = data["text"][0]
    #             token = data["token"][0]
    #             logits = model(data)
    #             outputs = [o.cpu()[0] for o in logits]
    #             outputs[0][:, [0, -1]] -= np.inf
    #             outputs[0][:, :, [0, -1]] -= np.inf
    #
    #             subjects, objects = [], []
    #             for l, h, t in zip(*np.where(outputs[0] > threshold)):
    #                 if l == 0:
    #                     subjects.append((h, t))
    #                 else:
    #                     objects.append((h, t))
    #
    #             spoes = []
    #             for sh, st in subjects:
    #                 for oh, ot in objects:
    #                     sp = np.where(outputs[1][:, sh, oh] >threshold)[0]
    #                     op = np.where(outputs[2][:, st, ot] > threshold)[0]
    #                     rs = set(sp) & set(op)
    #                     for r in rs:
    #                         sub = "".join(token[sh: st + 1])
    #                         sub = sub.replace('[unused1]','')
    #                         sub = sub.replace('##', '')
    #                         relation = self.id2label[str(r)]
    #                         obj = "".join(token[oh: ot + 1])
    #                         obj = obj.replace('[unused1]', '')
    #                         obj = obj.replace('##', '')
    #                         spoes.append((sub, relation, obj))
    #             triple = data["spo_list"][0]
    #             triple = set(to_tuple(triple))
    #             pred = set(spoes)
    #             correct_num += len(triple & pred)
    #             predict_num += len(pred)
    #             gold_num += len(triple)
    #             lack = triple - pred
    #             new = pred - triple
    #             predict.append({"text": text, "gold": list(triple), "predict": list(pred),
    #                             "lack": list(lack), "new": list(new)})
    #         print("correct_num:{} predict_num: {} gold_num: {}".format(correct_num, predict_num, gold_num))
    #         recall = correct_num / (gold_num + 1e-10)
    #         precision = correct_num / (predict_num + 1e-10)
    #         f1_score = 2 * recall * precision / (recall + precision + 1e-10)
    #         print("precision: {:5.4f} recall: {:5.4f} f1_score: {:5.4f}".format(precision, recall, f1_score))
    #     return precision, recall, f1_score, predict

    def evaluate(self, model, dataloader, threshold=-0.5):
        model.eval()
        type_stats = {}  # 初始化统计每种关系类型的字典
        predict_num, gold_num, correct_num = 0, 0, 0  # 总体统计量
        predict = []

        with torch.no_grad():
            for data in tqdm(dataloader):
                # 提取数据
                text = data["text"][0]
                token = data["token"][0]
                triple = data["spo_list"][0]

                # 模型预测
                logits = model(data)
                outputs = [o.cpu()[0] for o in logits]
                outputs[0][:, [0, -1]] -= np.inf
                outputs[0][:, :, [0, -1]] -= np.inf

                # 提取主体和客体
                subjects, objects = [], []
                for l, h, t in zip(*np.where(outputs[0] > threshold)):
                    if l == 0:
                        subjects.append((h, t))
                    else:
                        objects.append((h, t))

                # 构建关系
                spoes = []
                for sh, st in subjects:
                    for oh, ot in objects:
                        sp = np.where(outputs[1][:, sh, oh] > threshold)[0]
                        op = np.where(outputs[2][:, st, ot] > threshold)[0]
                        rs = set(sp) & set(op)
                        for r in rs:
                            sub = "".join(token[sh: st + 1])
                            sub = sub.replace('[unused1]', '').replace('##', '')
                            relation = self.id2label[str(r)]
                            obj = "".join(token[oh: ot + 1])
                            obj = obj.replace('[unused1]', '').replace('##', '')
                            spoes.append((sub, relation, obj))

                # 转换为元组格式
                def to_tuple(data):
                    return tuple(tuple(i) for i in data)

                triple = set(to_tuple(triple))
                pred = set(spoes)

                # 更新关系类型统计数据和总体统计量
                for sub, rel, obj in triple:
                    if rel not in type_stats:
                        type_stats[rel] = {'correct_num': 0, 'predict_num': 0, 'gold_num': 0}
                    type_stats[rel]['gold_num'] += 1
                    gold_num += 1

                for sub, rel, obj in pred:
                    if rel not in type_stats:
                        type_stats[rel] = {'correct_num': 0, 'predict_num': 0, 'gold_num': 0}
                    type_stats[rel]['predict_num'] += 1
                    predict_num += 1
                    if (sub, rel, obj) in triple:
                        type_stats[rel]['correct_num'] += 1
                        correct_num += 1

                # 生成预测结果
                pred_s = {(tup[0], tup[1], tup[2]) for tup in pred}
                triple_s = {(tup[0], tup[1], tup[2]) for tup in triple}
                lack = triple_s - pred_s
                new = pred_s - triple_s
                predict.append({"text": text, "gold": list(triple_s), "predict": list(pred_s),
                                "lack": list(lack), "new": list(new)})

        # 计算并打印每种关系类型的性能指标
        type_performance = {}
        for rel, stats in type_stats.items():
            recall = stats['correct_num'] / (stats['gold_num'] + 1e-10)
            precision = stats['correct_num'] / (stats['predict_num'] + 1e-10)
            f1_score = 2 * recall * precision / (recall + precision + 1e-10)
            type_performance[rel] = {'precision': precision, 'recall': recall, 'f1_score': f1_score}
            print(f"Relation Type: {rel}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
        # 计算并打印总体性能指标
        total_recall = correct_num / (gold_num + 1e-10)
        total_precision = correct_num / (predict_num + 1e-10)
        total_f1_score = 2 * total_recall * total_precision / (total_recall + total_precision + 1e-10)
        print(
            f"Total Precision: {total_precision:.4f}, Total Recall: {total_recall:.4f}, Total F1 Score: {total_f1_score:.4f}")

        return total_precision, total_recall, total_f1_score, predict

    def testall(self, model_name):
        model = REmodel(self.config).to(self.device)
        path = os.path.join(self.config.checkpoint, model_name)
        model.load_state_dict(torch.load(path))

        model.cuda()
        model.eval()
        test_dataset = MyDataset(self.config, self.config.test_fn,is_test=False)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1,
                                    collate_fn=collate_fn, pin_memory=True)
        precision, recall, f1_score, predict = self.evaluate(model, test_dataloader)
        print("f1: {:4.4f}, precision: {:4.4f}, recall: {:4.4f}".format(f1_score, precision, recall))