from torch.utils.data import Dataset
import json
import os
# from pytorch_pretrained_bert import BertTokenizer
import torch
from utils.tokenize import get_tokenizer
import numpy as np


tokenizer = get_tokenizer('E:\pythonProject\guanxichouqu\TPlinker\pre_model\\bert_base_cased\\vocab.txt')

BERT_MAX_LEN = 512

def padding(l, lenth, t):
    """
    :param l:{{(0, 0),(0 ,0),...},{...},{...}}
    :param lenth: num
    :return:
    """
    to_list = [list(i) for i in l]
    array = []
    for i in to_list:
        temp = i
        if len(i) < lenth:
            for j in range(lenth - len(i)):
                temp.append((0, 0))
        else:
            if len(i) > 1:
                if len(i) != lenth:
                    print(lenth)
                    print(len(i))
                    print(i)
                    print(t)
                # assert len(i) == lenth
        array.append(temp)
    return np.array(array)


def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


class MyDataset(Dataset):
    def __init__(self, config, fn,is_test):
        self.config = config
        with open(self.config.schema_fn, "r",encoding='utf-8') as f:
            self.label2id = json.load(f)[0]
        # self.tokenizer = tokenizer
        self.is_test = is_test
        self.tokenizer = tokenizer

        self.data= json.load(open(fn, "r", encoding="utf-8"))

        #N=1,2,3,4,5
        # self.data = [
        #     {"text": item["text"], "triple_list": item["triple_list"]}
        #     for item in self.filtered_data
        #     if len(item["triple_list"]) == 2
        # ]
        def has_overlapping_entities(triples): ##Normal
            entities = set()
            for triple in triples:
                entities.update(triple)
            return len(entities) != len(triples) * 3

        def has_multiple_relations(triples): ##EPO
            entity_relations = {}
            for triple in triples:
                entity_pair = (triple[0], triple[2])
                relation = triple[1]
                if entity_pair in entity_relations:
                    if entity_relations[entity_pair] != relation:
                        return True
                else:
                    entity_relations[entity_pair] = relation
            return False

        def contains_entity_with_multiple_relations(triples):  ##seo
            entity_relations = {}
            for triple in triples:
                entity = triple[0]
                relation = triple[1]
                related_entity = triple[2]

                if entity in entity_relations:
                    if relation != entity_relations[entity][0] or related_entity != entity_relations[entity][1]:
                        return True
                else:
                    entity_relations[entity] = (relation, related_entity)
            return False

        # self.data = [
        #     {"text": item["text"], "triple_list": item["triple_list"]}
        #     for item in self.filtered_data
        #     if contains_entity_with_multiple_relations(item["triple_list"])
        # ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ins_json_data = self.data[idx]
        text = ins_json_data['text']
        text = ' '.join(text.split()[:self.config.max_len])
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > BERT_MAX_LEN:
            tokens = tokens[: BERT_MAX_LEN]
        text_len = len(tokens)
        rels_list = [set() for _ in range(self.config.num_rel)]
        entity_list = [set() for _ in range(2)]
        head_list = [set() for _ in range(self.config.num_rel)]
        tail_list = [set() for _ in range(self.config.num_rel)]
        if not self.is_test:
            s2ro_map = {}

            for triple in ins_json_data['triple_list']:
                triple = (self.tokenizer.tokenize(triple[0])[1:-1], triple[1], self.tokenizer.tokenize(triple[2])[1:-1])
                sub_head_idx = find_head_idx(tokens, triple[0])
                obj_head_idx = find_head_idx(tokens, triple[2])
                if sub_head_idx != -1 and obj_head_idx != -1:
                    sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
                    if sub not in s2ro_map:
                        s2ro_map[sub] = []
                    s2ro_map[sub].append((obj_head_idx, obj_head_idx + len(triple[2]) - 1, self.label2id[triple[1]]))

            if s2ro_map:
                token_ids, segment_ids = self.tokenizer.encode(first=text)
                masks = segment_ids
                if len(token_ids) > text_len:
                    token_ids = token_ids[:text_len]
                    masks = masks[:text_len]
                token_ids = np.array(token_ids)
                masks = np.array(masks) + 1

                for s in s2ro_map:
                    subject_id = (s[0],s[1])
                    entity_list[0].add(subject_id)
                    for ro in s2ro_map.get((s[0],s[1]),[]):
                        relid = ro[2]
                        object_id = (ro[0],ro[1])
                        entity_list[1].add((object_id))
                        rels_list[relid].add(relid)
                        head_list[relid].add((subject_id[0], object_id[0]))
                        tail_list[relid].add((subject_id[1], object_id[1]))

                for label in entity_list + head_list + tail_list:
                    if not label:
                        label.add((0, 0))
                for label in rels_list:
                    if not label:
                        label.add(0)
                entity_list = [list(i) for i in entity_list]
                en1 = len(entity_list[0])
                en2 = len(entity_list[1])
                if en1 < en2:
                    for i in range(en2 - en1):
                        entity_list[0].append([0, 0])
                else:
                    for i in range(en1 - en2):
                        entity_list[1].append([0, 0])
                entity_list = np.array(entity_list)
                entity_list_length = entity_list.shape[1]

                head_length = [len(i) for i in head_list]
                max_hl = max(head_length)
                tail_length = [len(i) for i in tail_list]
                max_tl = max(tail_length)
                head_list = padding(head_list, max_hl, text)
                tail_list = padding(tail_list, max_tl, text)



                return text, tokens, ins_json_data["triple_list"], token_ids, masks, text_len, entity_list_length, \
                       entity_list, head_list, tail_list, max_hl, max_tl,rels_list



def collate_fn(batch):
    text, token, spo_list, input_ids, attention_mask, token_len, length, entity_list, head_list, tail_list, head_len, tail_len,rels_list = zip(*batch)

    cur_batch = len(batch)
    max_text_length = max(token_len)
    max_length = max(length)
    max_head_len = max(head_len)
    max_tail_len = max(tail_len)
    rels_list = np.array(rels_list)
    rel_list = [[item.pop() for item in sublist] for sublist in rels_list]
    batch_input_ids = torch.LongTensor(cur_batch, max_text_length).zero_()
    batch_mask = torch.LongTensor(cur_batch, max_text_length).zero_()
    batch_entity_list = torch.LongTensor(cur_batch, 2, max_length, 2).zero_()
    batch_head_list = torch.LongTensor(cur_batch,171, max_head_len, 2).zero_()
    batch_tail_list = torch.LongTensor(cur_batch,171, max_tail_len, 2).zero_()


    for i in range(cur_batch):
        batch_input_ids[i, :token_len[i]].copy_(torch.from_numpy(input_ids[i]))
        batch_mask[i, :token_len[i]].copy_(torch.from_numpy(attention_mask[i]))
        batch_entity_list[i, :, :length[i], :].copy_(torch.from_numpy(entity_list[i]))
        batch_head_list[i, :, :head_len[i], :].copy_(torch.from_numpy(head_list[i]))
        batch_tail_list[i, :, :tail_len[i], :].copy_(torch.from_numpy(tail_list[i]))
    batch_rel_list = torch.tensor(rel_list, dtype=torch.long)

    return {"text": text,
            "input_ids": batch_input_ids,
            "attention_mask": batch_mask,
            "entity_list": batch_entity_list,
            "head_list": batch_head_list,
            "tail_list": batch_tail_list,
            "spo_list": spo_list,
            "rel_list":batch_rel_list,
            "token": token}


if __name__ == '__main__':
    from config import Config
    from torch.utils.data import DataLoader
    config = Config()
    dataset = MyDataset(config, config.train_fn,is_test=False)
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
    for data in dataloader:
        print("*"*50)
        # print(data["entity_list"].shape)
        # print(data["head_list"].shape)
        # print(data["tail_list"].shape)