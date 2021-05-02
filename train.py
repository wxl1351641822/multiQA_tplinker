import random
import argparse
import os
import sys
import logging


import pickle
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
from transformers.optimization import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_

from model.model import MyModel,MrcTpLinkerModel
from evaluation import test_evaluation
from dataloader import load_data, load_eval_data, reload_data


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False



def args_parser():
    dt = datetime.now()
    id = dt.strftime("%Y%m%d-%H%M%S")
    # id = "20210416-163611"# 0.1 0.57、
    # id = "20210418-205134"
    # id = "20210419-102443"
    # id = "20210419-102443"
    # id = "20210421-102936"
    # id = "20210423-184319"
    # id = "20210426-200057"
    id = "20210426-200057"
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default=id)
    parser.add_argument("--log_path", default="./log/run_log")
    parser.add_argument("--model", default="MRCTPLinker",choices=["MRCTPLinker", "multiQA"])
    parser.add_argument("--dataset_tag", default='ace2005',
                        choices=['ace2005', 'ace2004', 'duie'])
    parser.add_argument("--train_path", default='data/cleaned_data/ACE2005/bert-base-uncased_overlap_15_window_100_threshold_1_max_distance_45_is_mq_False/train.json')
    parser.add_argument("--train_batch", type=int, default=24)
    parser.add_argument("--test_path", default='data/cleaned_data/ACE2005/bert-base-uncased_overlap_15_window_100_threshold_1_max_distance_45_is_mq_False/test.json')

    parser.add_argument("--test_batch", type=int, default=24)
    parser.add_argument("--dev_path", default='data/cleaned_data/ACE2005/bert-base-uncased_overlap_15_window_100_threshold_1_max_distance_45_is_mq_False/dev.json')
    parser.add_argument("--dev_batch", type=int, default=24)
    parser.add_argument("--max_len", default=200, type=int,
                        help="maximum length of input")
    parser.add_argument("--pretrained_model_path",default='../pretrained_models/bert-base-uncased')
    parser.add_argument("--max_epochs", default=32, type=int)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--theta", type=float,
                        help="weight of two tasks", default=0.2)
    parser.add_argument("--window_size", type=int,
                        default=100, help="size of the sliding window")
    parser.add_argument("--overlap", type=int, default=50,
                        help="overlap size of the two sliding windows")
    parser.add_argument("--threshold", type=int, default=5,
                        help="At least the number of times a possible relationship should appear in the training set (should be greater than or equal to the threshold in the data preprocessing stage)")
    parser.add_argument("--local_rank", type=int, default=-1, help="用于DistributedDataParallel")
    parser.add_argument("--max_grad_norm", type=float, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true",
                        help="whether to enable mixed precision")
    parser.add_argument("--not_save", action="store_true",
                        help="whether to save the model")
    parser.add_argument("--reload", action="store_true",
                        help="whether to reload data")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--dev_eval", action="store_true")
    parser.add_argument("--test_eval", action="store_true")
    parser.add_argument("--load_checkpoint_dir", type=str, default="-")
    parser.add_argument("--checkpoint_start", type=int, default=1)
    parser.add_argument("--num_questions", type=int, default=1)

    ## mrc_tplinker参数

    parser.add_argument("--train_ent", type=bool, default=True)
    parser.add_argument("--train_rel", type=bool, default=True)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--cross_valid", type=int, default=10)
    parser.add_argument("--shaking_type", type=str, default="cat", choices=["cat","cat_plus", "cln", "cln_plus"])
    parser.add_argument("--inner_enc_type", type=str, default="lstm", choices=["lstm", "mean_pooling", "max_pooling",  "mean_pooling"])
    parser.add_argument("--relH_theta", type=float,
                        help="weight of two tasks", default=0.4)
    parser.add_argument("--relT_theta", type=float,
                        help="weight of two tasks", default=0.4)
    parser.add_argument("--rel_loss_weight", type=float,
                       help="rel loss O's weight", default=0.25)

    parser.add_argument("--use_meg", type=bool,
                        help="megstudio?", default=False)
    args = parser.parse_args()

    args.test_eval = True
    # args.reload = True
    # args.train = True
    # args.dev_eval = True
    args.dataset_tag = "duie"
    if(args.dataset_tag == "duie"):
        args.train_batch = 16
        args.test_batch = 16
        args.dev_batch = 16
        args.lr = 5e-5
        args.pretrain_model = "../pretrained_models/bert-base-chinese"
        args.train_path = "data/cleaned_data/Duie/bert-base-chinese_is_mq_False/train.json"
        args.dev_path = "data/cleaned_data/Duie/bert-base-chinese_is_mq_False/dev.json"
        args.test_path = "data/cleaned_data/Duie/bert-base-chinese_is_mq_False/test1.json"
    # args.local_rank = torch.distributed.get_rank()
    # print(args.train)
    # id = dt.strftime("%Y%m%d-%H%M%S")
    # id = "2021_01_04_19_08_27"
    logger = get_logger(args.id, log_path=args.log_path, only_predict=not args.train and args.test_eval)

    return args,logger

def get_logger(id, log_path="./log/run_log", only_predict=False):
    sys.path.append("..")

    logger = logging.getLogger(__name__)

    logger.setLevel(level=logging.INFO)
    # log_path =
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_file_path = os.path.join(log_path, "log-{}".format(id))
    if(only_predict):
        log_file_path = os.path.join(log_path, "predict-log-{}".format(id))
    handler = logging.FileHandler(log_file_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s", "%Y%b%d-%H:%M:%S")
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

def train(args, train_dataloader):
    if(args.model=='MRCTPLinker'):
        model=MrcTpLinkerModel(args)
    else:
        model = MyModel(args)
    model.train()
    best_t1,best_t2,best=[-1,(0,0,0)],[-1,(0,0,0)],[-1,(0,0,0),(0,0,0)]
    if args.amp:
        scaler = GradScaler()
    device = args.local_rank if args.local_rank != -1 \
        else (torch.device('cuda:{}'.format(args.cuda)) if torch.cuda.is_available() and args.cuda >= 0 else torch.device('cpu'))
    # print(device)
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
    model.to(device)
    save_dir = './checkpoints/%s/%s/' % (args.dataset_tag, args.id)
    save_path = save_dir + "checkpoint_%d.cpt" % args.checkpoint_start
    # print(save_path)
    if (os.path.exists(save_path)):
        train=args.train
        dev=args.dev_eval
        test=args.test_eval
        cuda=args.cuda
        cs = args.checkpoint_start
        tb = args.train_batch
        local_rank = args.local_rank
        args = pickle.load(open(save_dir + "args", 'rb'))
        args.train=train
        args.dev_eval=dev
        args.test_eval=test
        args.cuda=cuda
        args.checkpoint_start = cs
        args.train_batch =tb
        args.local_rank = local_rank
        checkpoint = torch.load(save_path, map_location=device)
        model_state_dict = checkpoint['model_state_dict']
        model.load_state_dict(model_state_dict, strict=False)
        model.to(device)
        logger.info("加载{}".format(save_path))
    else:
        logger.info("加载模型失败...")

    for k,v in vars(args).items():
        logger.info("{}:{}".format(k,v))
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[
                                                          args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], "weight_decay":args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay":0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    if args.warmup_ratio > 0:
        num_training_steps = len(train_dataloader)*args.max_epochs
        warmup_steps = args.warmup_ratio*num_training_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, num_training_steps)
    if args.local_rank < 1:
        mid = args.id#time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
    # print(args.train, device, args.local_rank)
    if(args.train):
        for epoch in range( args.checkpoint_start+1,args.max_epochs):
            if args.local_rank != -1:
                train_dataloader.sampler.set_epoch(epoch)
            tqdm_train_dataloader = tqdm(
                train_dataloader, desc="epoch:%d" % epoch, ncols=150)
            for i, batch in enumerate(tqdm_train_dataloader):
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                txt_ids, attention_mask, token_type_ids, context_mask, turn_mask, tags, relH_tags, relT_tags = batch['txt_ids'], batch['attention_mask'], batch['token_type_ids'],\
                    batch['context_mask'], batch['turn_mask'], batch["tags"], batch['relH_tags'],batch['relT_tags']
                txt_ids, attention_mask, token_type_ids, context_mask, turn_mask = txt_ids.to(device), attention_mask.to(device), token_type_ids.to(device),\
                    context_mask.to(device), turn_mask.to(device)

                tags = tags.to(device) if tags is not None else None
                relH_tags = relH_tags.to(device) if relH_tags is not None else None
                relT_tags = relT_tags.to(device) if relT_tags is not None else None
                # print(tags)
                if args.amp:
                    with autocast():
                        loss, (loss_t1, loss_relH,loss_relT),acc = model(
                            txt_ids, attention_mask, token_type_ids, context_mask, turn_mask,
                            relH_tags=relH_tags, relT_tags=relT_tags, target_tags=tags)
                    scaler.scale(loss).backward()
                    if args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                    clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss, (loss_t1, loss_relH, loss_relT), acc = model(txt_ids, attention_mask,
                                                     token_type_ids, context_mask, turn_mask,  relH_tags=relH_tags,
                                                                       relT_tags=relT_tags,target_tags=tags)
                    loss.backward()
                    if args.max_grad_norm > 0:
                        clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                lr = optimizer.param_groups[0]['lr']
                named_parameters = [
                    (n, p) for n, p in model.named_parameters() if not p.grad is None]
                grad_norm = torch.norm(torch.stack(
                    [torch.norm(p.grad) for n, p in named_parameters])).item()
                if args.warmup_ratio > 0:
                    scheduler.step()
                postfix_str = "norm:{:.2f},lr:{:.1e},loss:{:.2e},t1:{:.2e},relh:{:.2e},relt:{:.2e},acc:{}".format(
                    grad_norm, lr, loss.item(), loss_t1, loss_relH,loss_relT,acc)
                tqdm_train_dataloader.set_postfix_str(postfix_str)
            logger.info(postfix_str)
            if args.local_rank in [-1, 0] and not args.not_save:
                if hasattr(model, 'module'):
                    model_state_dict = model.module.state_dict()
                else:
                    model_state_dict = model.state_dict()
                checkpoint = {"model_state_dict": model_state_dict}
                save_dir = './checkpoints/%s/%s/' % (args.dataset_tag, mid)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    pickle.dump(args, open(save_dir+'args', 'wb'))
                save_path = save_dir+"checkpoint_%d.cpt" % epoch
                torch.save(checkpoint, save_path)
                logger.info("model saved at:"+save_path)
                # 转存训练日志
                logname="log-{}".format(args.id)
                with open(save_dir+logname, "w", encoding="utf-8") as fw:
                    with open("./log/run_log/" + logname, "r", encoding="utf-8") as fr:
                        fw.write(fr.read())
            if args.dev_eval and args.local_rank in [-1, 0]:
                logger.info('****' * 10 + "Train Eval" + str(epoch) + "***" * 10)
                train_dataloader_eval = load_eval_data(args.dataset_tag, args.train_path, args.pretrained_model_path,
                                                batch_size=args.dev_batch,
                                                max_len=args.max_len, threshold=args.threshold,
                                                num_questions=args.num_questions,
                                                train_ent=args.train_ent)  # test_dataloader是第一轮问答的dataloder
                (p1, r1, f1), (p2, r2, f2) = test_evaluation(
                    model, train_dataloader_eval, args.threshold, args.amp, train_ent=args.train_ent, cuda=args.cuda)

                logger.info(
                    "Turn 1: precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(p1, r1, f1))
                logger.info(
                    "Turn 2: precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(p2, r2, f2))
                logger.info('****'*10+"DEV"+str(epoch)+"***"*10)
                dev_dataloader = load_eval_data(args.dataset_tag, args.dev_path, args.pretrained_model_path,
                                                batch_size=args.dev_batch,
                                                max_len=args.max_len, threshold=args.threshold,
                                                num_questions=args.num_questions,
                                                train_ent=args.train_ent)  # test_dataloader是第一轮问答的dataloder
                (p1, r1, f1), (p2, r2, f2) = test_evaluation(
                    model, dev_dataloader, args.threshold, args.amp, train_ent=args.train_ent,cuda=args.cuda)

                logger.info(
                    "Turn 1: precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(p1, r1, f1))
                logger.info(
                    "Turn 2: precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(p2, r2, f2))
                if(best_t1[1][-1]<f1):
                    best_t1=[epoch,(p1, r1, f1)]
                if (best_t2[1][-1] < f2):
                    best_t2 = [epoch,(p2, r2, f2)]
                if(best[1][-1] + best[2][-1] < f1 + f2):
                    best=[epoch,(p1, r1, f1),(p2, r2, f2)]
                # if(epoch-best[0]>3):
                #     break
                model.train()
            if args.local_rank != -1:
                torch.distributed.barrier()

            logger.info("***"*10+"DEV BEST"+"***"*10)
            logger.info(
                "Turn 1: [epoch,p,r,f]:{}".format(best_t1))
            logger.info(
                "Turn 2: [epoch,p,r,f]:{}".format(best_t2))
            logger.info(
                "all: [epoch,p1,r1,f1;p2,r2,f2]:{}".format(best))
    with open(args.dataset_tag+"dev.txt", "a", encoding="utf-8") as f:
        lis = [[], []]
        lis[1] = [best[0], best[1][0], best[1][1], best[1][2], best[2][0], best[2][1], best[2][2]]
        for k, v in vars(args).items():
            if(k == "id"):
                continue
            lis[0].append(k)
            lis[1].append(v)
        lis[1] = [str(s) for s in lis[1]]
        # f.write("id,epoch,p1,r1,f1,p2,r2,f2," + ",".join(lis[0])+"\n")
        f.write(str(args.id) + "," + ",".join(lis[1])+"\n")

    if args.test_eval and args.local_rank in [-1, 0]:
        logger.info('****' * 10 + "TEST" + "***" * 10)
        # dev_dataloader = load_eval_data(args.dataset_tag, args.dev_path, args.pretrained_model_path,max_len=args.max_len,
        #                                 args.local_rank != -1, num_questions=args.num_questions,
        #                                 train_ent=args.train_ent)  # test_dataloader是第一轮问答的dataloder
        dev_dataloader = load_eval_data(args.dataset_tag, args.test_path, args.pretrained_model_path,  batch_size=args.test_batch,
                                        max_len=args.max_len,threshold=args.threshold,num_questions=args.num_questions,
                                        train_ent=args.train_ent)  # test_dataloader是第一轮问答的dataloder
        (p1, r1, f1),(p2, r2, f2) = test_evaluation(
            model, dev_dataloader, args.threshold, args.amp,train_ent=args.train_ent,cuda=args.cuda)
        logger.info(
            "Turn 1: precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(p1, r1, f1))
        logger.info(
            "Turn 2: precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(p2, r2, f2))


def predict(args, beg, end):
    if(args.model=='MRCTPLinker'):
        model=MrcTpLinkerModel(args)
    else:
        model = MyModel(args)
    model.train()
    best_t1, best_t2, best = [-1, (0, 0, 0)], [-1, (0, 0, 0)], [-1, (0, 0, 0), (0, 0, 0)]
    if args.amp:
        scaler = GradScaler()
    device = args.local_rank if args.local_rank != -1 \
        else (torch.device('cuda:{}'.format(args.cuda)) if torch.cuda.is_available() and args.cuda >= 0 else torch.device('cpu'))
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
    model.to(device)
    for k, v in vars(args).items():
        logger.info("{}:{}".format(k, v))
    dev_dataloader = load_eval_data(args.dataset_tag, args.dev_path, args.pretrained_model_path,
                                    batch_size=args.test_batch,
                                    max_len=args.max_len, threshold=args.threshold, num_questions=args.num_questions,
                                    train_ent=args.train_ent)  # test_dataloader是第一轮问答的dataloder
    for i in range(beg, end):
        logger.info('****' * 10 + "TEST" + str(i) + "***" * 10)
        args.checkpoint_start = i
        save_dir = './checkpoints/%s/%s/' % (args.dataset_tag, args.id)
        save_path = save_dir + "checkpoint_%d.cpt" % args.checkpoint_start
        # print(save_path)
        if (os.path.exists(save_path)):
            train = args.train
            dev = args.dev_eval
            test = args.test_eval
            cuda = args.cuda
            checkpoint_start = args.checkpoint_start
            args = pickle.load(open(save_dir + "args", 'rb'))
            args.train = train
            args.dev_eval = dev
            args.test_eval = test
            args.cuda = cuda
            args.checkpoint_start = checkpoint_start
            checkpoint = torch.load(save_path, map_location=device)
            model_state_dict = checkpoint['model_state_dict']
            model.load_state_dict(model_state_dict, strict=False)
            model.to(device)
            logger.info("加载{}".format(save_path))
        else:
            logger.info("加载模型失败...")

        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[
                                                              args.local_rank], output_device=args.local_rank, find_unused_parameters=True)



        # dev_dataloader = load_eval_data(args.dataset_tag, args.dev_path, args.pretrained_model_path,max_len=args.max_len,
        #                                 args.local_rank != -1, num_questions=args.num_questions,
        #                                 train_ent=args.train_ent)  # test_dataloader是第一轮问答的dataloder

        (p1, r1, f1),(p2, r2, f2) = test_evaluation(
            model, dev_dataloader, args.threshold, args.amp,train_ent=args.train_ent,cuda=args.cuda)
        logger.info(
            "Turn 1: precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(p1, r1, f1))
        logger.info(
            "Turn 2: precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(p2, r2, f2))
        if (best_t1[1][-1] < f1):
            best_t1 = [i, (p1, r1, f1)]
        if (best_t2[1][-1] < f2):
            best_t2 = [i, (p2, r2, f2)]
        if (best[1][-1] + best[2][-1] < f1 + f2):
            best = [i, (p1, r1, f1), (p2, r2, f2)]
        logger.info(
            "Best Turn 1: [epoch,p,r,f]:{}".format(best_t1))
        logger.info(
            "Best Turn 2: [epoch,p,r,f]:{}".format(best_t2))
        logger.info(
            "Best all: [epoch,p1,r1,f1;p2,r2,f2]:{}".format(best))
        # 转存预测日志
        logname = "predict-log-{}".format(args.id)
        with open(save_dir + logname, "w", encoding="utf-8") as fw:
            with open("./log/run_log/" + logname, "r", encoding="utf-8") as fr:
                fw.write(fr.read())
    with open(args.dataset_tag+"test.txt", "a", encoding="utf-8") as f:
        lis = [[], []]
        lis[1] = [best[0], best[1][0], best[1][1], best[1][2], best[2][0], best[2][1], best[2][2]]
        for k, v in vars(args).items():
            if(k == "id"):
                continue
            lis[0].append(k)
            lis[1].append(v)
        lis[1] = [str(s) for s in lis[1]]
        # f.write("id,epoch,p1,r1,f1,p2,r2,f2," + ",".join(lis[0])+"\n")
        f.write(str(args.id) + "," + ",".join(lis[1])+"\n")

if __name__  ==  "__main__":
    args,logger = args_parser()
    set_seed(args.seed)
    # print(args)
    if(args.train):
        print(args.local_rank)
        if args.local_rank != -1:
            torch.distributed.init_process_group(backend='nccl')
        p = '{}_{}_{}'.format(args.dataset_tag, os.path.split(
            args.train_path)[-1].split('.')[0], os.path.split(args.pretrained_model_path)[-1])
        p1 = os.path.join(os.path.split(args.train_path)[0], p)
        # args.reload = True
        if not os.path.exists(p1) or args.reload:
            train_dataloader = load_data(args.dataset_tag, args.train_path, args.train_batch, args.max_len, args.pretrained_model_path,
                                         args.local_rank != -1, shuffle=True, threshold=args.threshold,num_questions=args.num_questions,train_ent=args.train_ent, train_rel=args.train_rel)
            pickle.dump(train_dataloader, open(p1, 'wb'))
            logger.info("training data saved at "+p1)
        else:
            logger.info("reload training data from "+p1)
            train_dataloader = pickle.load(open(p1, 'rb'))
            train_dataloader = reload_data(train_dataloader, args.train_batch, args.max_len,
                                           args.threshold, args.local_rank, True)
            pickle.dump(train_dataloader, open(p1, 'wb'))

            logger.info("reload success! " + p1)

            # args.checkpoint_start=i
        train(args, train_dataloader)

    if(not args.train and args.test_eval):
        beg, end = 1,31
        predict(args, beg, end)

