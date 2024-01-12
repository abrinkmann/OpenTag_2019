import json
import os
import sys
from time import strftime, localtime
from collections import Counter
from config import opt
from transformers import BertTokenizer
import random
import numpy as np 
import torch 
import models
from utils import get_dataloader
# from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from seqeval.metrics import f1_score, accuracy_score, classification_report, precision_score, recall_score
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def get_attributes(path):
    atts = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            if line:
                title, attribute, value, split = line.split('<$$$>')
                atts.append(attribute)
    return [item[0] for item in Counter(atts).most_common()]

def train(**kwargs):
    log_file = '{}-{}.log'.format(opt.model, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    dataset_name = 'ae-110k'
    percentage = '1_0'
    path = f'data/{dataset_name}_{percentage}.txt'
    att_list = get_attributes(path)
    preds, labels = [], []
    result_dict = {}

    print(f'Dataset: {dataset_name}')
    print(f'Percentage: {percentage}')
    print('---------------------------------')

    # for attribute in att_list:
    #     logger.info(f'Attribute: {attribute}')
    tokenizer = BertTokenizer.from_pretrained(opt.pretrained_bert_name)
    tags2id = {'':0,'B':1,'I':2,'O':3}
    id2tags = {v:k for k,v in tags2id.items()}

    opt._parse(kwargs)
    opt.pickle_path = f'./data/{dataset_name}_{percentage}.pkl'

    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # step1: configure model
    model = getattr(models, opt.model)(opt)
    model.model_name = f'{opt.model}_{dataset_name}_{percentage}'
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    # step2: data
    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(opt)

    # step3: criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    lr = opt.lr
    optimizer = model.get_optimizer(lr, opt.weight_decay)

    # step4 train
    model_f1 = 0
    attr_labels, attr_preds = [], []
    for epoch in range(opt.max_epoch):
        model.train()
        for ii, batch in tqdm(enumerate(train_dataloader)):
            # train model
            optimizer.zero_grad()
            x = batch['x'].to(opt.device)
            y = batch['y'].to(opt.device)
            att = batch['att'].to(opt.device)
            inputs = [x, att, y]
            loss = model.log_likelihood(inputs)
            loss.backward()
            #CRF
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=3)
            optimizer.step()
            if ii % opt.print_freq == 0:
                print('epoch:%04d,------------loss:%f'%(epoch,loss.item()))

        # validate every 20 epochs
        if epoch % 20 == 0:
            preds_eval, labels_eval = [], []
            model.eval()
            for index, batch in enumerate(test_dataloader):
                x = batch['x'].to(opt.device)
                y = batch['y'].to(opt.device)
                att = batch['att'].to(opt.device)
                inputs = [x, att, y]
                predict = model(inputs)

                if index % 5 == 0:
                    print(tokenizer.convert_ids_to_tokens([i.item() for i in x[0].cpu() if i.item() > 0]))
                    length = [id2tags[i.item()] for i in y[0].cpu() if i.item() > 0]
                    print(length)
                    print([id2tags[i] for i in predict[0][:len(length)]])

                # 统计非0的，也就是真实标签的长度
                leng = []
                for i in y.cpu():
                    tmp = []
                    for j in i:
                        if j.item() > 0:
                            tmp.append(j.item())
                    leng.append(tmp)

                for index, i in enumerate(predict):
                    preds_eval.append([id2tags[k] if k > 0 else id2tags[3] for k in i[:len(leng[index])]])
                    # preds += i[:len(leng[index])]

                for index, i in enumerate(y.tolist()):
                    labels_eval.append([id2tags[k] if k > 0 else id2tags[3] for k in i[:len(leng[index])]])

            eval_f1_score = f1_score(labels_eval, preds_eval)
            if eval_f1_score > model_f1:
                model_f1 = eval_f1_score
                model.save()
                logger.info(f'epoch:{epoch}, f1:{eval_f1_score}')
                #logger.info(classification_report(labels_eval, preds_eval, digits=4))

                # Evaluate intermediate best model on test set
                attr_labels, attr_preds = [], []
                for index, batch in enumerate(test_dataloader):
                    x = batch['x'].to(opt.device)
                    y = batch['y'].to(opt.device)
                    att = batch['att'].to(opt.device)
                    inputs = [x, att, y]
                    predict = model(inputs)

                    if index % 5 == 0:
                        print(tokenizer.convert_ids_to_tokens([i.item() for i in x[0].cpu() if i.item()>0]))
                        length = [id2tags[i.item()] for i in y[0].cpu() if i.item()>0]
                        print(length)
                        print([id2tags[i] for i in predict[0][:len(length)]])

                    # 统计非0的，也就是真实标签的长度
                    leng = []
                    for i in y.cpu():
                        tmp = []
                        for j in i:
                            if j.item()>0:
                                tmp.append(j.item())
                        leng.append(tmp)

                    for index, i in enumerate(predict):
                        attr_preds.append([id2tags[k] if k>0 else id2tags[3] for k in i[:len(leng[index])]])
                        # preds += i[:len(leng[index])]

                    for index, i in enumerate(y.tolist()):
                        attr_labels.append([id2tags[k] if k>0 else id2tags[3] for k in i[:len(leng[index])]])

    # Evaluate final best model on test set
    if len(attr_labels) == 0 and len(attr_preds) == 0:
        # Evaluate model on test set
        for index, batch in enumerate(test_dataloader):
            x = batch['x'].to(opt.device)
            y = batch['y'].to(opt.device)
            att = batch['att'].to(opt.device)
            inputs = [x, att, y]
            predict = model(inputs)

            if index % 5 == 0:
                print(tokenizer.convert_ids_to_tokens([i.item() for i in x[0].cpu() if i.item() > 0]))
                length = [id2tags[i.item()] for i in y[0].cpu() if i.item() > 0]
                print(length)
                print([id2tags[i] for i in predict[0][:len(length)]])

            # 统计非0的，也就是真实标签的长度
            leng = []
            for i in y.cpu():
                tmp = []
                for j in i:
                    if j.item() > 0:
                        tmp.append(j.item())
                leng.append(tmp)

            for index, i in enumerate(predict):
                attr_preds.append([id2tags[k] if k > 0 else id2tags[3] for k in i[:len(leng[index])]])
                # preds += i[:len(leng[index])]

            for index, i in enumerate(y.tolist()):
                attr_labels.append([id2tags[k] if k > 0 else id2tags[3] for k in i[:len(leng[index])]])

    else:
        # Evaluate intermediate best model on test set
        labels.extend(attr_labels)
        preds.extend(attr_preds)

    # Evaluate final best model on test set
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    result_dict['micro'] = {'Precision': precision, 'Recall': recall, 'F1': f1}
    report = {'Precision': precision, 'Recall': recall, 'F1': f1}
    logger.info(report)

    with open(f'results/result_dict_{dataset_name}_{percentage}.json', 'w') as f:
        json.dump(result_dict, f, indent=4)


if __name__=='__main__':
    #import fire
    #fire.Fire()

    train()