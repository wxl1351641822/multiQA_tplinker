import torch
from tqdm import tqdm

from dataloader import tag_idxs, load_t2_data


def get_score(gold_set, predict_set):
    TP = len(set.intersection(gold_set, predict_set))
    print("#TP:", TP, "#Gold:", len(gold_set), "#Predict:", len(predict_set))
    precision = TP/(len(predict_set)+1e-6)
    recall = TP/(len(gold_set)+1e-6)
    f1 = 2*precision*recall/(precision+recall+1e-6)
    return precision, recall, f1


def test_evaluation(model, dataloaders, threshold, amp=False, train_ent=False, cuda=1, filter_rel_sigma=0.5,
                    merge_rel_or=True):
    t1_dataloader, t2_dataloader=dataloaders
    if hasattr(model, 'module'):
        model = model.module
    model.eval()
    t1_predict = []
    t2_predict = []
    predict_dic = {}
    device = torch.device(
        "cuda:{}".format(cuda)) if torch.cuda.is_available() and cuda>=0 else torch.device("cpu")
    # window_offset_base = t1_dataloader.dataset.window_offset_base
    # query_offset1 = t1_dataloader.dataset.query_offset1
    # query_offset2 = t2_dataloader.dataset.query_offset2
    # turn 1
    if(train_ent):
        with (torch.no_grad() if not amp else torch.cuda.amp.autocast()):
            for i, batch in enumerate(tqdm(t1_dataloader, desc="predict")):
                txt_ids, attention_mask, token_type_ids, context_mask, turn_mask, p_id, type = \
                    batch['txt_ids'], batch['attention_mask'], batch['token_type_ids'], batch['context_mask'], batch[
                        'turn_mask'], batch['p_id'], batch['type']

                # txt_ids, attention_mask, token_type_ids, context_mask, = batch['txt_ids'], batch[
                #     'attention_mask'], batch['token_type_ids'], batch['context_mask']
                tag_idx, relH_tag_idx, relT_tag_idx, handshaking_context_mask = model(txt_ids.to(device), attention_mask.to(
                    device), token_type_ids.to(device),context_mask.to(device),turn_mask.to(device))
                predict_spans = tag_decode(tag_idx, context_mask, p_id=p_id, type=type, gold_dic=t1_dataloader.dataset.gold_dic, \
                                           predict_dic=predict_dic
                                           )


                t1_predict.extend(predict_spans)
                # break
    # print(len(t1_predict))
    # print(predict_dic)
    with (torch.no_grad() if not amp else torch.cuda.amp.autocast()):
        tqdm_test_dataloader=tqdm(t2_dataloader, desc="predict",ncols=150)
        for i, batch in enumerate(tqdm_test_dataloader):
            txt_ids, attention_mask, token_type_ids, context_mask, turn_mask, p_id, type = \
                batch['txt_ids'], batch['attention_mask'], batch['token_type_ids'], batch['context_mask'], batch['turn_mask'],batch['p_id'], batch['type']


            # txt_ids, attention_mask, token_type_ids, context_mask, = batch['txt_ids'], batch[
            #     'attention_mask'], batch['token_type_ids'], batch['context_mask']
            tag_idx, relH_tag_idx, relT_tag_idx, handshaking_context_mask = model(txt_ids.to(device), attention_mask.to(
                device), token_type_ids.to(device),context_mask.to(device),turn_mask.to(device))
            # print(relH_tag_idx)
            # print(ans)
            ents_spans,rels_spans,total_ents,total_rels = rel_tag_decode(relH_tag_idx, relT_tag_idx, handshaking_context_mask,
                                                                         context_mask, predict_dic=predict_dic,
                                                                         gold_dic=t2_dataloader.dataset.gold_dic,\
                                                                         p_id=p_id, type=type,
                                                                         filter_rel_sigma=filter_rel_sigma,
                                                                         merge_rel_or=merge_rel_or)
            # t2_predict.extend(predict_spans)
            if(len(t1_predict)==0):
                t1_predict.extend(ents_spans)
            t2_predict.extend(rels_spans)
            postfix_str = "total_ents:{},total_rels:{}".format(total_ents,total_rels)
            tqdm_test_dataloader.set_postfix_str(postfix_str)
            # break


    # get basic information
    t1_ids = t1_dataloader.dataset.t1_ids
    t1_gold = t1_dataloader.dataset.t1_gold
    # print(t1_predict)
    p1, r1, f1 = eval_t1(t1_predict, t1_gold, t1_ids)
    t2_ids = t2_dataloader.dataset.t2_ids
    #
    #
    t2_gold=t2_dataloader.dataset.t2_gold
    # # t2_gold = t2_dataloader.dataset.t2_gold
    #
    p2, r2, f2 = eval_t2(t2_predict, t2_gold, t2_ids)
    # p1,r1,f1,p2,r2,f2=0,0,0,0,0,0
    return (p1, r1, f1),(p2, r2, f2)


def eval_t1(predict, gold, ids):
    """
    Args:
        predict: [(s1,e1),(s2,e2),(s3,e3),...]
        gold: (passage_id,(entity_type,start_idx,end_idx,entity_str))
        ids: (passage_id, window_id,entity_type)
        query_offset: offset of [CLS]+title+[SEP]+query+[SEP]
        window_offset_base: value of window_size-overlap的值
    """
    predict1 = []
    for i, pre in enumerate(predict):
        # passage_id,entity_type = _id
        # for start, end in pre:
        #     # start1, end1 = start - query_offset[i]+window_offset, \
        #     #                 end - query_offset[i]+window_offset
        #     new = (passage_id, (entity_type, start, end))

        predict1.extend(pre)
        # print(pre)
    print(len(predict1))
    return get_score(set(gold), set(predict1))


def eval_t2(predict, gold, ids):
    """
    Args:
        predict: [(s1,e1),(s2,e2),(s3,e3),...]
        gold:  (passage_id,(head_entity,relation_type,end_entity))
        ids: (passage_id,window_id,head_entity,relation_type,end_entity_type)
        query_offset: offset of [CLS]+title+[SEP]+query+[SEP]
        window_offset_base: value of window_size-overlap
    """
    predict1 = []
    for i, pre in enumerate(predict):
        # passage_id, rel= _id
        # for s,e in pre:
        #     new=(passage_id,(s,rel,e))
        # window_offset = window_offset_base*window_id
        # head_start = head_entity[1]
        # for start, end in pre:
        #     #since we added two special tokens around the head entity for identification, there is a correction of 1.
        #     if head_start+query_offset[i]-window_offset+1 < start:
        #         start1, end1 = start - \
        #             query_offset[i]+window_offset-2, end - \
        #             query_offset[i]+window_offset-2
        #     else:
        #         start1, end1 = start - \
        #             query_offset[i]+window_offset, end - \
        #             query_offset[i]+window_offset
        #     new = (passage_id, (head_entity, relation_type,
        #                         (end_entity_type, start1, end1)))
        predict1.extend(pre)
        # print(pre)
    print(len(predict1))
    return get_score(set(gold), set(predict1))


def tag_decode(tags, context_mask=None, p_id=None, type=None, ans=None, gold_dic=None, predict_dic=None):
    # print(p_id)
    total_predict = 0
    TP = 0
    spans = [[]]*tags[1].shape[0]
    # print(tags.shape)
    tags = tags[1].tolist()
    if not context_mask is None:
        context_mask = context_mask.tolist()
    has_answer = []
    start_idxs = []
    end_idxs = []
    for i, t in enumerate(tags):
        # print(i,t)
        if t[0] != tag_idxs['S']:
            has_answer.append(i)
            if context_mask is None:
                mask = [1 if i != -1 else 0 for i in t]
            else:
                mask = context_mask[i]
            s = mask.index(1, 1)
            # print(s)
            e = mask.index(0, s)
            start_idxs.append(s)
            end_idxs.append(e)
            tags[i] = tags[i][s: e]
            # print(e-s)
    for i in has_answer:
        now_p_id = p_id[i].item()
        now_type = type[i]
        # gold = gold_dic[now_p_id]
        span = []
        j = 0
        while j < len(tags[i]):
            if tags[i][j] == tag_idxs['S']:
                span.append((now_p_id, (now_type, j, j + 1)))
                j += 1
            elif tags[i][j] == tag_idxs['B'] and j < len(tags[i]) - 1:
                for k in range(j + 1, len(tags[i])):
                    if tags[i][k] in [tag_idxs['B'], tag_idxs['S']]:
                        j = k
                        break
                    elif tags[i][k] == tag_idxs["E"]:
                        span.append((now_p_id, (now_type, j, k + 1)))
                        # print(tags[i][j:k+1])
                        j = k + 1
                        break
                    elif k == len(tags[i]) - 1:
                        j = k + 1
            else:
                j += 1
        spans[i] = span

        if(now_p_id in predict_dic):
            predict_dic[now_p_id].extend(span)
        else:
            predict_dic[now_p_id] = span
        # print(now_p_id, now_type, span)
        # print(gold)
    return spans

def rel_tag_decode(relH_tag_idx, relT_tag_idx, handshaking_context_mask,context_mask,overlap=15,ans = None, p_id=None,
                   type=None,gold_dic=None,predict_dic=None, filter_rel_sigma=0.5, merge_rel_or=True):
    ents_spans = [[]]*relH_tag_idx[1].shape[0]
    # print(ents_spans)
    rels_spans=[[]]*relH_tag_idx[1].shape[0]
    total_ents=0
    total_rels=0
    # relH_tag_idx=relH_tag_idx.tolist()
    # relT_tag_idx=relT_tag_idx.tolist()
    # print(relH_tag_idx[1].shape[0], relT_tag_idx[1].shape[0], handshaking_context_mask.shape[0])
    assert relH_tag_idx[1].shape[0] == relT_tag_idx[1].shape[0] == handshaking_context_mask.shape[0]
    for ii, (now_p_id, now_type, Hp, Tp, H, T, mask, c_mask) in enumerate(zip(p_id, type, relH_tag_idx[0],
                                                                              relT_tag_idx[0], relH_tag_idx[1], relT_tag_idx[1],
                                                                              handshaking_context_mask, context_mask)):
        now_p_id = now_p_id.item()
        gold = gold_dic[now_p_id]
        try:
            t1_predict = predict_dic[now_p_id]
        except:
            continue
        # H=H.reshape(-1,1)
        # T=T.reshape(-1,1)
        H = H[mask == 1].tolist()
        # print(H.nonzero())
        # H = H.tolist()
        T = T[mask == 1].tolist()
        Hp = Hp[mask == 1].tolist()
        Tp = Tp[mask == 1].tolist()
        # print(Tp)
        # print(Hp)
        seqlen=sum(c_mask)
        # dic={}
        k=0
        SH2OH,ST2OT=[],[]

        # print(seqlen)
        # print(H.nonzero())
        # print(H)
        H_ent_id=[]
        T_ent_id=[]
        # import time
        # s=time.time()
        for row in range(seqlen.item()):
            for col in range(row, seqlen.item()):
                if(H[k] != 0 and Hp[k] > filter_rel_sigma):
                    H_ent_id.append(row)
                    H_ent_id.append(col)
                    if(H[k] % 2 == 1):
                        SH2OH.append((row, col))
                    elif(H[k] % 2 == 0):
                        SH2OH.append((col, row))
                if(T[k]!=0 and Tp[k]>filter_rel_sigma):
                    # print(T[k])
                    T_ent_id.append(row+1)
                    T_ent_id.append(col+1)
                    if (T[k] % 2 == 1):
                        ST2OT.append((row+1, col+1))
                    elif (T[k] % 2 == 0):
                        ST2OT.append((col+1, row+1))
                k += 1
        # e=time.time()
        # print(e-s)
        # s=e

        # print(ii)
        # print(gold)
        # print(SH2OH)
        # print(ST2OT)
        # if(t1_predict is not None):
        # print(t1_predict)

        # e = time.time()
        # print(e - s)
        # s = e
        H_ents = {}
        T_ents = {}
        # if(len(SH2OH)!=0 and len(ST2OT)!=0):
        # print(len(SH2OH),len(ST2OT))
        ##这个再想一下
        if(len(t1_predict)==0):
            SH2OH.sort()
            ST2OT.sort()
            H_ent_id.sort()
            T_ent_id.sort()
            i,j=0,0
            ents=set()
            # print(H_ent_id)
            # print(T_ent_id)
            while(i<len(H_ent_id) and j<len(T_ent_id)):
                # print(H_ent_id[i],T_ent_id[j])
                if(H_ent_id[i]>T_ent_id[j] and H_ent_id[i]>T_ent_id[j]-overlap):
                    ents.add((H_ent_id[i],T_ent_id[j]))
                    H_ents[H_ent_id[i]] = (H_ent_id[i], T_ent_id[j])
                    T_ents[T_ent_id[j]] = (H_ent_id[i], T_ent_id[j])
                i+=1
                j+=1
        else:
            # ents=t1_predict[ii]
            for e in t1_predict:
                H_ents[e[1][1]] = e[1]
                T_ents[e[1][2]] = e[1]
            # print(H_ents)
        # e = time.time()
        # print(e - s)
        # s = e
        rels = set()
        rels_H = set()
        rels_T =set()
        # print(len(SH2OH),len(ST2OT))
        for s, o in SH2OH:
            try:
                rels_H.add((now_p_id, (now_type, H_ents[s], H_ents[o])))
            except:
                pass
        for s, o in ST2OT:
            try:
                rels_T.add((now_p_id, (now_type, T_ents[s], T_ents[o])))
            except:
                pass
        if(merge_rel_or):
            rels = rels_H | rels_T#合并
        else:
            rels = rels_H & rels_T
        # e = time.time()
        # print(e - s)
        # s = e
        # print(ii)
        # print(rels)
        # ents_spans[ii]=list(ents)
        rels_spans[ii]=list(rels)

        # total_ents +=len(ents_spans[-1]——)
        total_rels +=len(rels_spans[-1])

        # print(time.time()-start)
    return ents_spans,rels_spans,total_ents,total_rels

