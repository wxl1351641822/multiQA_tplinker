import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler
from transformers import BertTokenizer

from data.cleaned_data.ACE2005.preprocess import passage_blocks, get_question
from utils.constants import *

def tag_decode(tags, context_mask = None):

    spans = [[]]
    tags = [tags]
    context_mask = [context_mask]
    # print(tags.shape)
    # tags = tags.tolist()
    # if not context_mask is None:
    #     context_mask = context_mask.tolist()
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
            # print(s,e,e-s)
            tags[i] = tags[i][s: e]
    for i in has_answer:
        span = []
        j = 0
        while j < len(tags[i]):
            if tags[i][j] == tag_idxs['S']:
                span.append([j, j+1])
                j += 1
            elif tags[i][j] == tag_idxs['B'] and j < e-1:
                for k in range(j+1, e):
                    if tags[i][k] in [tag_idxs['B'], tag_idxs['S']]:
                        j = k
                        break
                    elif tags[i][k] == tag_idxs["E"]:
                        span.append([j, k+1])
                        # print(tags[i][j:k+1])
                        j = k+1
                        break
                    elif k == e-1:
                        j = k+1
            else:
                j += 1
        spans[i] = span
    # print(spans)
    return spans
def rel_tag_decode(relH_tag_idx, relT_tag_idx, context_mask,):
    ents_spans = [[]]
    # print(ents_spans)
    rels_spans=[[]]
    total_ents=0
    total_rels=0
    # relH_tag_idx=relH_tag_idx.tolist()
    # relT_tag_idx=relT_tag_idx.tolist()
    H = relH_tag_idx
    T = relT_tag_idx

    # H=H.reshape(-1,1)
    # T=T.reshape(-1,1)

    seqlen=sum(context_mask)
    print(seqlen)
    dic={}
    k=0
    SH2OH,ST2OT=[],[]

    # print(seqlen)
    # print(H.nonzero())
    H_ent_id=[]
    T_ent_id=[]
    # import time
    # s=time.time()
    for row in range(seqlen):
        for col in range(row,seqlen):
            if(H[k]!=0):
                H_ent_id.append(row)
                H_ent_id.append(col)
                if(H[k]==1):
                    SH2OH.append((row,col))
                elif(H[k]==2):
                    SH2OH.append((col,row))
            if(T[k]!=0):
                # print(T[k])
                T_ent_id.append(row+1)
                T_ent_id.append(col+1)
                if (T[k] == 1):
                    ST2OT.append((row+1, col+1))
                elif (T[k] == 2):
                    ST2OT.append((col+1, row+1))
            k+=1
    # e=time.time()
    # print(e-s)
    # s=e
    SH2OH.sort()
    ST2OT.sort()
    print(ST2OT)
    print(SH2OH)
    # e = time.time()
    # print(e - s)
    # s = e
    H_ents = {}
    T_ents = {}
    # if(len(SH2OH)!=0 and len(ST2OT)!=0):
    # print(len(SH2OH),len(ST2OT))

    H_ent_id.sort()
    T_ent_id.sort()
    i,j=0,0
    ents=set()
    # print(H_ent_id)
    # print(T_ent_id)
    while(i<len(H_ent_id) and j<len(T_ent_id)):
        print(H_ent_id[i],T_ent_id[j])
        if(H_ent_id[i]<=T_ent_id[j] and H_ent_id[i]>T_ent_id[j]-15):
            ents.add((H_ent_id[i],T_ent_id[j]))
            H_ents[H_ent_id[i]]=(H_ent_id[i],T_ent_id[j])
            T_ents[T_ent_id[j]] = (H_ent_id[i],T_ent_id[j])
        i+=1
        j+=1
    # print(H_ents,T_ents)
    # e = time.time()
    # print(e - s)
    # s = e
    rels=set()
    # print(len(SH2OH),len(ST2OT))
    for s,o in SH2OH:
        try:
            rels.add((H_ents[s],H_ents[o]))
        except:
            pass
    for s,o in ST2OT:
        try:
            rels.add((T_ents[s],T_ents[o]))
        except:
            pass
    # e = time.time()
    # print(e - s)
    # s = e
    # print(ii)
    ii = 0
    ents_spans[ii]=list(ents)
    rels_spans[ii]=list(rels)

    total_ents +=len(ents_spans[-1])
    total_rels +=len(rels_spans[-1])

        # print(time.time()-start)
    # print(rels_spans)
    return ents_spans,rels_spans,total_ents,total_rels
def collate_fn(batch):
    # for training
    nbatch = {}
    for b in batch:
        for k, v in b.items():
            try:
                nbatch[k] = nbatch.get(k, []) + [torch.tensor(v)]
            except:
                nbatch[k]=nbatch.get(k, []) + [v]
    txt_ids = nbatch['txt_ids']
    ntags, relH_ntags, relT_ntags = None, None, None
    if("tags" in nbatch):
        tags = nbatch['tags']
        ntags = pad_sequence(tags, batch_first=True, padding_value=-1)

    context_mask = nbatch['context_mask']
    if("batch_ans" in nbatch):
        batch_ans=nbatch['batch_ans']
        relH_tags_list=[]
        relT_tags_list=[]

        for i,(ans,mask) in enumerate(zip(batch_ans,context_mask)):
            seq_len=sum(mask)
            # print(seq_len)
            relH_tags = [relH_tag_idxs["O"]] * (seq_len * (seq_len + 1) // 2)
            relT_tags = [relT_tag_idxs["O"]] * (seq_len * (seq_len + 1) // 2)
            # print(len(relH_tags))
            # print(relH_tags)
            # print("seqlen",seq_len)
            num = 0
            for i, an in enumerate(ans):
                if(len(an)!=3):
                    continue
                num += 1
                rel, ent1, ent2 = an
                if (ent1[1] < ent2[1]):
                    index_s2o = convertrowcol2index(ent1[1], ent2[1], seq_len)
                    relH_tags[index_s2o] = relH_tag_idxs["SH2OH"]
                else:
                    index_s2o = convertrowcol2index(ent2[1], ent1[1], seq_len)
                    # print(len(relH_tags),seq_len,index_s2o,ent2[1],ent1[1])
                    relH_tags[index_s2o] = relH_tag_idxs["OH2SH"]
                if (ent1[2] < ent2[2]):
                    index_s2o = convertrowcol2index(ent1[2]-1, ent2[2]-1, seq_len)
                    # print(an)
                    # print(len(context),index_s2o, len(tags), ent1[2], ent2[2])
                    relT_tags[index_s2o] = relT_tag_idxs["ST2OT"]

                else:
                    index_s2o = convertrowcol2index(ent2[2]-1, ent1[2]-1,seq_len)
                    relT_tags[index_s2o] = relT_tag_idxs["OT2ST"]
            # if(num):
            #     print(ans)
            #     print(rel_tag_decode(relH_tags, relT_tags, mask))
            relH_tags = torch.tensor(relH_tags)
            # print(torch.where(relH_tags > 0))
            relH_tags_list.append(relH_tags)
            relT_tags_list.append(torch.tensor(relT_tags))
        # relH_tags=nbatch["relH_tags"]
        # relT_tags=nbatch["relT_tags"]
        # "relH_tags": tags[1], "relT_tags": tags[2],
        relH_ntags = pad_sequence(relH_tags_list, batch_first=True, padding_value=-1)
        relT_ntags = pad_sequence(relT_tags_list, batch_first=True, padding_value=-1)


    l=0
    l1=0
    # for i in range(10):
    #     l+=len(relH_tags[i])
    #     l1+=sum(context_mask[i])*(sum(context_mask[i])+1)//2
    #     print(len(relH_tags[i]),sum(context_mask[i]),sum(context_mask[i])*(sum(context_mask[i])+1)//2)
    # print(l,l1)
    token_type_ids = nbatch['token_type_ids']
    turn_mask = nbatch['turn_mask']
    ntxt_ids = pad_sequence(txt_ids, batch_first=True,
                            padding_value=0)

    ncontext_mask = pad_sequence(
        context_mask, batch_first=True, padding_value=0)
    ntoken_type_ids = pad_sequence(
        token_type_ids, batch_first=True, padding_value=1)
    attention_mask = torch.zeros(ntxt_ids.shape)
    for i in range(len(ntxt_ids)):
        txt_len = len(txt_ids[i])
        attention_mask[i, :txt_len] = 1
    nbatch['txt_ids'] = ntxt_ids

    nbatch['context_mask'] = ncontext_mask
    # l1 = 0
    # l=0
    # for i in range(10):
    #     l += len(relH_tags[i])
    #     l1 += sum(ncontext_mask[i]) * (sum(ncontext_mask[i]) + 1) // 2
    #     print(len(relH_tags[i]), sum(ncontext_mask[i]), sum(ncontext_mask[i]) * (sum(ncontext_mask[i]) + 1) // 2)
    # print(l, l1)
    nbatch['token_type_ids'] = ntoken_type_ids
    nbatch['attention_mask'] = attention_mask
    nbatch['turn_mask'] = torch.tensor(turn_mask, dtype=torch.uint8)
    nbatch['tags'] = ntags
    nbatch['relH_tags'] = relH_ntags
    nbatch['relT_tags'] = relT_ntags
    return nbatch


def collate_fn1(batch):
    # for testing
    nbatch = {}
    for b in batch:
        for k, v in b.items():
            try:
                nbatch[k] = nbatch.get(k, []) + [torch.tensor(v)]
            except:
                nbatch[k] = nbatch.get(k, []) + [v]
    txt_ids = nbatch['txt_ids']
    context_mask = nbatch['context_mask']
    token_type_ids = nbatch['token_type_ids']
    turn_mask = nbatch['turn_mask']
    ntxt_ids = pad_sequence(txt_ids, batch_first=True,
                            padding_value=0)
    ncontext_mask = pad_sequence(
        context_mask, batch_first=True, padding_value=0)
    ntoken_type_ids = pad_sequence(
        token_type_ids, batch_first=True, padding_value=1)
    attention_mask = torch.zeros(ntxt_ids.shape)
    for i in range(len(ntxt_ids)):
        txt_len = len(txt_ids[i])
        attention_mask[i, :txt_len] = 1
    nbatch['txt_ids'] = ntxt_ids
    nbatch['context_mask'] = ncontext_mask
    nbatch['token_type_ids'] = ntoken_type_ids
    nbatch['attention_mask'] = attention_mask
    nbatch['turn_mask'] = torch.tensor(turn_mask, dtype=torch.uint8)
    return nbatch
def convertrowcol2index(row,col,collength):
    return (collength+collength-row+1)*row//2+(col-row)
def get_inputs(context, q, tokenizer, title="", max_len=200, ans=[], type="ent"):
    query = tokenizer.tokenize(q)


    txt_len = len(query) + len(title) + len(context) + \
              4 if title else len(query) + len(context) + 3
    if txt_len > max_len:
        context = context[:max_len -
                           len(query) - 3] if not title else context[:max_len - len(query) - len(title) - 4]
    # print(len(context))
    tags = [tag_idxs["O"]] * (len(context))
    if(type=='ent'):
        # print(len(context))
        for i, an in enumerate(ans):
            # print(an)
            start, end = an[1:-1]
            end = end - 1
            if start != end:
                tags[start] = tag_idxs['B']
                tags[end] = tag_idxs['E']
                for i in range(start + 1, end):
                    tags[i] = tag_idxs['M']
            else:
                tags[start] = tag_idxs['S']
            # print(tags[start:end+1])
            # print(an,tags[start:end+1],context[start:end+1])
        # print(tags)
    if title:
        txt = ['[CLS]'] + query + ['[SEP]'] + \
              title + ['[SEP]'] + context + ['[SEP]']
    else:
        txt = ['[CLS]'] + query + ['[SEP]'] + context + ['[SEP]']
    txt_ids = tokenizer.convert_tokens_to_ids(txt)
    # [CLS] is used to judge whether there is an answe---是否有答案？
    if not title:
        if(type=='ent'):
            tags =[tag_idxs['O'] if len(
            ans) > 0 else tag_idxs['S']] + [-1] * (len(query) + 1) + tags + [-1]
            # print(len([tag_idxs['O'] if len(
            # ans) > 0 else tag_idxs['S']] + [-1] * (len(query) + 1)),len(tags)-1)
        context_mask = [0] + [0] * (len(query) + 1) + [1] * len(context) + [0]
        token_type_ids = [0] * (len(query) + 2) + [1] * (len(context) + 1)
    else:
        if (type == 'ent'):
            tags = [tag_idxs['O'] if len(
            ans) > 0 else tag_idxs['S']]+ [-1] * (len(query) + len(title) + 2) + tags + [-1]
            # print(len([tag_idxs['O'] if len(
            # ans) > 0 else tag_idxs['S']]+ [-1] * (len(query) + len(title) + 2)),len(tags)-1)
        context_mask = [0] + [0] * \
                       (len(query) + len(title) + 2) + [1] * len(context) + [0]  # title-->question
        token_type_ids = [0] * (len(query) + len(title) + 3) + [1] * (len(context) + 1)
    # if(sum(context_mask)*(sum(context_mask)+1)//2!=len(relH_tags)):
    #     print(sum(context_mask),len(context),sum(context_mask)*(sum(context_mask)+1)//2,len(relH_tags),len(relT_tags))
    # if(type == "ent"):
    #     span = tag_decode(tags, context_mask)
    #     print(span)
    #     print(ans)

    return txt_ids, (tags,ans), context_mask, token_type_ids


def get_inputs1(context, q, tokenizer, title="", max_len=200, ans=[], type="ent"):
    query = tokenizer.tokenize(q)
    tags = [tag_idxs['O']]*len(context)
    for i, an in enumerate(ans):
        start, end = an[1:-1]
        end = end-1
        if start != end:
            tags[start] = tag_idxs['B']
            tags[end] = tag_idxs['E']
            for i in range(start+1, end):
                tags[i] = tag_idxs['M']
        else:
            tags[start] = tag_idxs['S']
    if head_entity:
        h_start, h_end = head_entity[1], head_entity[2]
        context = context[:h_start]+['[unused0]'] + \
            context[h_start:h_end]+["[unused1]"]+context[h_end:]
        assert len(context) == len(tags)+2
        tags = tags[:h_start]+[tag_idxs['O']] + \
            tags[h_start:h_end]+[tag_idxs['O']]+tags[h_end:]
    txt_len = len(query)+len(title)+len(context) + \
        4 if title else len(query)+len(context)+3
    if txt_len > max_len:
        context = context[:max_len -
                          len(query) - 3] if not title else context[:max_len-len(query)-len(title)-4]
        tags = tags[:max_len -
                    len(query) - 3] if not title else tags[:max_len-len(query)-len(title)-4]
    if title:
        txt = ['[CLS]'] + query+['[SEP]'] + \
            title + ['[SEP]'] + context + ['[SEP]']
    else:
        txt = ['[CLS]'] + query + ['[SEP]'] + context + ['[SEP]']
    txt_ids = tokenizer.convert_tokens_to_ids(txt)
    # [CLS] is used to judge whether there is an answe---是否有答案？
    if not title:
        tags1 = [tag_idxs['O'] if len(
            ans) > 0 else tag_idxs['S']] + [-1] * (len(query) + 1) + tags + [-1]
        context_mask = [1] + [0] * (len(query) + 1) + [1] * len(context) + [0]
        token_type_ids = [0] * (len(query) + 2) + [1] * (len(context) + 1)
    else:
        tags1 = [tag_idxs['O'] if len(
            ans) > 0 else tag_idxs['S']] + [-1]*(len(query)+len(title)+2) + tags + [-1]
        context_mask = [1] + [0] * \
            (len(query)+len(title)+2) + [1] * len(context) + [0]#title-->question
        token_type_ids = [0]*(len(query)+len(title)+3)+[1]*(len(context) + 1)
    return txt_ids, tags1, context_mask, token_type_ids


def query2relation(question, question_templates):
    '''
    query -> <entity_type,relation_type,entity_type>
    '''
    turn2_questions = question_templates['qa_turn2']
    turn2_questions = {v: k for k, v in turn2_questions.items()}
    for k, v in turn2_questions.items():
        k1 = k.replace("XXX.", "")
        if question.startswith(k1):
            return eval(v)
    raise Exception("cannot find the relation type corresponding to the query, if the \
                 query template is changed, please re-implement this function according to the new template")




class MyDataset:
    def __init__(self, dataset_tag, path, tokenizer, max_len=512, threshold=5,num_questions=1,train_ent=False,is_test=False,train_rel=True):
        """
        Args:
            dataset_tag: type of dataset
            path:  path to training set file
            tokenizer： tokenizer of pretrained model
            max_len: max length of input
            threshold: only consider relationships where the frequency is greater than or equal to the threshold
        """
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.threshold = threshold
        self.dataset_tag = dataset_tag
        self.num_questions=num_questions
        self.train_ent=train_ent
        self.train_rel=train_rel
        self.is_test=is_test
        self.init_data()

    def init_data(self):
        self.all_t1 = []
        self.all_t2 = []

        # self.all_qas =[]
        if self.dataset_tag.lower() == "ace2004":
            idx1s = ace2004_idx1
            idx2s = ace2004_idx2
            dist = ace2004_dist
            if (self.num_questions==1):
                question_templates = ace2005_mq_question_templates
            else:
                question_templates = ace2005_question_templates
        elif self.dataset_tag.lower() == 'ace2005':
            idx1s = ace2005_idx1
            idx2s = ace2005_idx2
            dist = ace2005_dist
            if (self.num_questions==1):
                question_templates = ace2005_mq_question_templates
            else:
                question_templates = ace2005_question_templates
        else:
            raise Exception("this dataset is not yet supported")


        for p_id,d in enumerate(tqdm(self.data, desc="dataset")):

            context = d['context']
            title = d['title']
            qa_pairs = d['qa_pairs']
            t1 = qa_pairs[0]
            t2 = qa_pairs[1]

            t1_qas = []
            t2_qas = []

            if(self.train_ent):#未改动
                for i, (q, ans) in enumerate(t1.items()):
                    txt_ids, tags, context_mask, token_type_ids = get_inputs(
                        context, q, self.tokenizer, title, self.max_len, ans,type="ent")
                    # print(tags,ans)
                    # print(len(tags), tags)
                    t1_qas.append(
                        {"txt_ids": txt_ids, "tags": tags[0], "context_mask": context_mask, "token_type_ids": token_type_ids, 'turn_mask': 0})
                    if(self.train_rel):
                        t1_qas[-1]["batch_ans"]=tags[1]
                self.all_t1.extend(t1_qas)
            if(self.train_rel):
                #对关系提问
                # print(len(list(t2.keys())))
                for i,(q,ans) in enumerate(t2.items()):
                    # print(context)
                    txt_ids, tags, context_mask, token_type_ids = get_inputs(
                        context, q, self.tokenizer, title, self.max_len, ans,type="rel")
                    # print(tags)
                    t2_qas.append(
                        {"txt_ids": txt_ids, "batch_ans":tags[1], "context_mask": context_mask, "token_type_ids": token_type_ids,
                         'turn_mask': 1})
                    if (self.train_ent):
                        t2_qas[-1]["tags"]=tags[0]

            # for t in t2:
            #     head_entity = t[0]
            #     end_entity=t[1]
            #     for q, ans in t['qas'].items():
            #         rel = query2relation(q, question_templates)
            #         idx1, idx2 = rel[0], rel[1:]
            #         idx1, idx2 = idx1s[idx1], idx2s[idx2]
            #         if dist[idx1][idx2] >= self.threshold:
            #             txt_ids, tags, context_mask, token_type_ids = get_inputs(
            #                 context, q, self.tokenizer, title, self.max_len, ans, head_entity)
            #             t2_qas.append({"txt_ids": txt_ids, "tags": tags, "context_mask": context_mask,
            #                            "token_type_ids": token_type_ids, 'turn_mask': 1})

            self.all_t2.extend(t2_qas)
        self.all_qas = self.all_t2+self.all_t1
        print(len(self.all_qas))

    def __len__(self):
        return len(self.all_qas)

    def __getitem__(self, i):
        return self.all_qas[i]



class MyDatasetT2:
    def __init__(self, dataset_tag, path, tokenizer, max_len=512, threshold=5,num_questions=1,train_ent=False,is_test=True):
        """
        Args:
            dataset_tag: type of dataset
            path:  path to training set file
            tokenizer： tokenizer of pretrained model
            max_len: max length of input
            threshold: only consider relationships where the frequency is greater than or equal to the threshold
        """
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.threshold = threshold
        self.dataset_tag = dataset_tag
        self.num_questions=num_questions
        self.train_ent=train_ent
        self.is_test=is_test
        self.init_data()

    def init_data(self):
        # print(self.data[0])
        self.all_t1 = []
        self.all_t2 = []
        self.t1_ids=[]
        self.t2_ids=[]
        # self.all_qas =[]
        if self.dataset_tag.lower() == "ace2004":
            idx1s = ace2004_idx1
            idx2s = ace2004_idx2
            dist = ace2004_dist
            if (self.num_questions==1):
                question_templates = ace2005_mq_question_templates
            else:
                question_templates = ace2005_question_templates
        elif self.dataset_tag.lower() == 'ace2005':
            idx1s = ace2005_idx1
            idx2s = ace2005_idx2
            dist = ace2005_dist
            if (self.num_questions==1):
                question_templates = ace2005_mq_question_templates
            else:
                question_templates = ace2005_question_templates
        else:
            raise Exception("this dataset is not yet supported")

        self.t1_gold=[]
        self.t2_gold=[]
        self.gold_dic = {}
        # ace2005_q2type
        for p_id,d in enumerate(tqdm(self.data, desc="dataset")):
            start = len(self.t2_gold)
            context = d['context']
            title = d['title']
            qa_pairs = d['qa_pairs']
            t1 = qa_pairs[0]
            t2 = qa_pairs[1]

            t1_qas = []
            t2_qas = []

            #对关系提问
            for i,(q,ans) in enumerate(t2.items()):
                # print(context)
                
                txt_ids, tags, context_mask, token_type_ids = get_inputs(
                    context, q, self.tokenizer, title, self.max_len, ans,type="rel")
                # print(context,q)
                t2_qas.append(
                    {"txt_ids": txt_ids,  "context_mask": context_mask, "token_type_ids": token_type_ids,
                     'turn_mask': 1, "p_id": p_id, "type": ace2005_q2type["qa_turn2"][q]})

                if(self.is_test):
                    self.t2_ids.append((p_id,ace2005_q2type["qa_turn2"][q]))
                    # print(self.t2_ids[-1])
                    # t2_ids[-1]["type"] = ace2005_q2type["qa_turn2"][q]
                    # print(t2_qas[-1]["type"])
                    for an in ans:
                        rel,head,tail=an
                        if(self.train_ent):
                            self.t2_gold.append((p_id,(tuple(head[:-1]),rel,tuple(tail[:-1]))))
                        else:
                            self.t2_gold.append((p_id,(tuple(head[1:-1]),rel,tuple(tail[1:-1]))))
            end = len(self.t2_gold)
            self.gold_dic[p_id] = self.t2_gold[start + 1:end]

            # for t in t2:
            #     head_entity = t[0]
            #     end_entity=t[1]
            #     for q, ans in t['qas'].items():
            #         rel = query2relation(q, question_templates)
            #         idx1, idx2 = rel[0], rel[1:]
            #         idx1, idx2 = idx1s[idx1], idx2s[idx2]
            #         if dist[idx1][idx2] >= self.threshold:
            #             txt_ids, tags, context_mask, token_type_ids = get_inputs(
            #                 context, q, self.tokenizer, title, self.max_len, ans, head_entity)
            #             t2_qas.append({"txt_ids": txt_ids, "tags": tags, "context_mask": context_mask,
            #                            "token_type_ids": token_type_ids, 'turn_mask': 1})

            self.all_t2.extend(t2_qas)
        self.all_qas = self.all_t2
        print(len(self.all_qas),len(self.t2_gold),len(self.t2_ids))

    def __len__(self):
        return len(self.all_qas)

    def __getitem__(self, i):
        return self.all_qas[i]


class MyDatasetT1:
    def __init__(self, dataset_tag, path, tokenizer, max_len=512, threshold=5,num_questions=1,train_ent=False,is_test=True):
        """
        Args:
            dataset_tag: type of dataset
            path:  path to training set file
            tokenizer： tokenizer of pretrained model
            max_len: max length of input
            threshold: only consider relationships where the frequency is greater than or equal to the threshold
        """
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.threshold = threshold
        self.dataset_tag = dataset_tag
        self.num_questions=num_questions
        self.train_ent=train_ent
        self.is_test=is_test
        self.init_data()

    def init_data(self):
        self.all_t1 = []
        self.all_t2 = []
        self.t1_ids=[]

        # self.all_qas =[]
        if self.dataset_tag.lower() == "ace2004":
            idx1s = ace2004_idx1
            idx2s = ace2004_idx2
            dist = ace2004_dist
            if (self.num_questions==1):
                question_templates = ace2005_mq_question_templates
            else:
                question_templates = ace2005_question_templates
        elif self.dataset_tag.lower() == 'ace2005':
            idx1s = ace2005_idx1
            idx2s = ace2005_idx2
            dist = ace2005_dist
            if (self.num_questions==1):
                question_templates = ace2005_mq_question_templates
            else:
                question_templates = ace2005_question_templates
        else:
            raise Exception("this dataset is not yet supported")

        self.t1_gold=[]
        self.gold_dic = {}
        # print(self.data[0])

        for p_id,d in enumerate(tqdm(self.data, desc="dataset")):
            start = len(self.t1_gold)
            context = d['context']
            title = d['title']
            qa_pairs = d['qa_pairs']
            t1 = qa_pairs[0]
            t2 = qa_pairs[1]

            t1_qas = []
            t2_qas = []
            if(self.train_ent):#未改动
                for i, (q, ans) in enumerate(t1.items()):
                    txt_ids, tags, context_mask, token_type_ids = get_inputs(
                        context, q, self.tokenizer, title, self.max_len, ans,type="ent")
                    t1_qas.append(
                        {"txt_ids": txt_ids,  "context_mask": context_mask, "token_type_ids": token_type_ids, 'turn_mask': 0, 'p_id': p_id, "type": ace2005_q2type["qa_turn1"][q]})
                    if(self.is_test):
                        self.t1_ids.append((p_id,ace2005_q2type["qa_turn1"][q]))

                        for ent in ans:
                            self.t1_gold.append((p_id,tuple(ent[:-1])))

                self.all_t1.extend(t1_qas)
            end = len(self.t1_gold)
            self.gold_dic[p_id] = self.t1_gold[start + 1:end]


        self.all_qas = self.all_t1
        print(len(self.all_qas),len(self.t1_gold))

    def __len__(self):
        return len(self.all_qas)

    def __getitem__(self, i):
        return self.all_qas[i]



class T1Dataset:
    def __init__(self, dataset_tag, test_path, tokenizer, window_size, overlap, max_len=512):
        """
        Args:
            dataset_tag: type of dataset
            test_path: path to test set file
            tokenizer: tokenizer of pretrained model
            window_size: sliding window size
            overlap: overlap between two adjacent windows
            max_len: max length of input
        """
        with open(test_path, encoding="utf=8") as f:
            data = json.load(f)
        self.dataset_tag = dataset_tag
        if dataset_tag.lower() == 'ace2004':
            dataset_entities = ace2004_entities
            question_templates = ace2004_question_templates
        elif dataset_tag.lower() == 'ace2005':
            dataset_entities = ace2005_entities
            question_templates = ace2005_question_templates
        else:
            raise Exception("this data set is not yet supported")
        self.t1_qas = []
        self.passages = []
        self.entities = []
        self.relations = []
        self.titles = []
        self.window_size = window_size
        self.overlap = overlap
        self.t1_querys = []
        self.t1_ids = []
        self.t1_gold = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        # passage_windows[i][j] represents the j-th window of the i-th passage
        self.passage_windows = []
        self.query_offset1 = []
        self.window_offset_base = window_size-overlap
        for ent_type in dataset_entities:
            query = get_question(question_templates, ent_type)
            self.t1_querys.append(query)
        for p_id, d in enumerate(tqdm(data, desc="t1_dataset")):
            passage = d["passage"]
            entities = d['entities']
            relations = d['relations']
            title = d['title']
            self.passages.append(passage)
            self.entities.append(entities)
            self.relations.append(relations)
            self.titles.append(title)
            blocks, _ = passage_blocks(passage, window_size, overlap)
            self.passage_windows.append(blocks)
            for ent in entities:
                self.t1_gold.append((p_id, tuple(ent[:-1])))
            for b_id, block in enumerate(blocks):
                for q_id, q in enumerate(self.t1_querys):
                    txt_ids, _, context_mask, token_type_ids = get_inputs(
                        block, q, tokenizer, title, max_len)
                    self.t1_qas.append({"txt_ids": txt_ids, "context_mask": context_mask,
                                        "token_type_ids": token_type_ids})
                    self.t1_ids.append((p_id, b_id, dataset_entities[q_id]))
                    ofs = len(title)+len(tokenizer.tokenize(q))+3
                    self.query_offset1.append(ofs)
        print("t1:",len(self.t1_qas))

    def __len__(self):
        return len(self.t1_qas)

    def __getitem__(self, i):
        return self.t1_qas[i]


class T2Dataset:
    def __init__(self, t1_dataset, t1_predict, threshold=5):
        '''
        Args:
            t1_dataset: an instance of T1Dataset
            t1_predict: predictions of the first turn QA
            threshold: only consider relationships where the frequency is greater than or equal to the threshold
        '''
        if t1_dataset.dataset_tag.lower() == "ace2004":
            idx1s = ace2004_idx1
            idx2s = ace2004_idx2
            dist = ace2004_dist
            dataset_entities = ace2004_entities
            dataset_relations = ace2004_relations
            question_templates = ace2004_question_templates
        elif t1_dataset.dataset_tag.lower() == 'ace2005':
            idx1s = ace2005_idx1
            idx2s = ace2005_idx2
            dist = ace2005_dist
            dataset_entities = ace2005_entities
            dataset_relations = ace2005_relations
            question_templates = ace2005_question_templates
        else:
            raise Exception("this data set is not yet supported")
        tokenizer = t1_dataset.tokenizer
        max_len = t1_dataset.max_len
        t1_ids = t1_dataset.t1_ids
        passages = t1_dataset.passages
        titles = t1_dataset.titles
        passage_windows = t1_dataset.passage_windows
        self.t2_qas = []
        self.t2_ids = []
        self.t2_gold = []
        self.query_offset2 = []
        relations = t1_dataset.relations
        entities = t1_dataset.entities
        query_offset1 = t1_dataset.query_offset1
        window_offset_base = t1_dataset.window_offset_base
        for passage_id, (ents, rels) in enumerate(zip(entities, relations)):
            for re in rels:
                head, rel, end = ents[re[1]], re[0], ents[re[2]]
                self.t2_gold.append(
                    (passage_id, (tuple(head[:-1]), rel, tuple(end[:-1]))))
        for i, (_id, pre) in enumerate(zip(tqdm(t1_ids, desc="t2 dataset"), t1_predict)):
            passage_id, window_id, head_entity_type = _id
            window_offset = window_offset_base*window_id
            context = passage_windows[passage_id][window_id]
            title = titles[passage_id]
            head_entities = []
            for start, end in pre:
                start1, end1 = start - \
                    query_offset1[i]+window_offset, end - \
                    query_offset1[i]+window_offset
                ent_str = tokenizer.convert_tokens_to_string(
                    passages[passage_id][start1:end1])
                head_entity = (head_entity_type, start1, end1, ent_str)
                head_entities.append(head_entity)
            for head_entity in head_entities:
                for rel in dataset_relations:
                    for end_ent_type in dataset_entities:
                        idx1, idx2 = idx1s[head_entity[0]
                                           ], idx2s[(rel, end_ent_type)]
                        if dist[idx1][idx2] >= threshold:
                            query = get_question(
                                question_templates, head_entity, rel, end_ent_type)
                            window_head_entity = (
                                head_entity[0], head_entity[1]-window_offset, head_entity[2]-window_offset, head_entity[3])
                            txt_ids, _, context_mask, token_type_ids = get_inputs(
                                context, query, tokenizer, title, max_len, [], window_head_entity)
                            self.t2_qas.append({"txt_ids": txt_ids, "context_mask": context_mask,
                                                "token_type_ids": token_type_ids})
                            self.t2_ids.append(
                                (passage_id, window_id, head_entity[:-1], rel, end_ent_type))
                            ofs = len(title) + \
                                len(tokenizer.tokenize(query)) + 3
                            self.query_offset2.append(ofs)
        print(len(self.t2_qas))

    def __len__(self):
        return len(self.t2_qas)

    def __getitem__(self, i):
        return self.t2_qas[i]


def load_data(dataset_tag, file_path, batch_size, max_len, pretrained_model_path, dist=False, shuffle=False, threshold=5,num_questions=1, train_ent=False, train_rel=False):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
    dataset = MyDataset(dataset_tag, file_path, tokenizer,
                        max_len, threshold,num_questions=num_questions,train_ent=train_ent, train_rel=train_rel)
    sampler = DistributedSampler(dataset) if dist else None
    dataloader = DataLoader(dataset, batch_size, sampler=sampler, shuffle=shuffle if not sampler else False,
                            collate_fn=collate_fn)
    return dataloader


def reload_data(old_dataloader, batch_size, max_len, threshold, local_rank, shuffle=True):
    dataset = old_dataloader.dataset
    old_max_len, old_threshold = dataset.max_len, dataset.threshold
    if not(old_max_len == max_len and old_threshold == threshold):
        dataset.max_len = max_len
        dataset.threshold = threshold
        dataset.init_data()
    sampler = DistributedSampler(
        dataset, rank=local_rank) if local_rank != -1 else None
    dataloader = DataLoader(
        dataset, batch_size, sampler=sampler, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader


def load_eval_data(dataset_tag, test_path, pretrained_model_path,  batch_size=10, max_len=512,threshold=5,num_questions=1,train_ent=False):
    t1_dataloader=load_t1_data(dataset_tag, test_path, pretrained_model_path, threshold,num_questions, batch_size, max_len)
    t2_dataloader = load_t2_data(dataset_tag, test_path, pretrained_model_path, threshold, num_questions, batch_size,
                                 max_len,train_ent=train_ent)
    return t1_dataloader,t2_dataloader

def load_t1_data(dataset_tag, test_path, pretrained_model_path, threhold=5,num_questions=1, batch_size=10, max_len=512):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
    t1_dataset = MyDatasetT1(dataset_tag, test_path, tokenizer, max_len=max_len, threshold=threhold,num_questions=num_questions,train_ent=True)
    dataloader = DataLoader(t1_dataset, batch_size, collate_fn=collate_fn1)
    return dataloader

def load_t2_data(dataset_tag, test_path, pretrained_model_path, threhold=5,num_questions=1, batch_size=10, max_len=512,train_ent=False):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
    t2_dataset = MyDatasetT2(dataset_tag, test_path, tokenizer, max_len=max_len, threshold=threhold,num_questions=num_questions,train_ent=train_ent)
    dataloader = DataLoader(t2_dataset, batch_size, collate_fn=collate_fn1)
    return dataloader

# def convertrowcol2index(row,col,collength):
#     return (collength+collength-row+1)*row//2+(col-row)
# collength=8
# for i in range(collength):
#     for j in range(i,collength):
#         index=convertrowcol2index(i,j,collength)
#         print(i,j,index)