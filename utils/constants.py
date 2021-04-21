import json
import os

print(os.path.abspath(__file__))
question_templates_path = "/data/home/wuyuming/wxl/multiQA_tplinker/data/query_templates/"
print(os.path.abspath(question_templates_path))
ace2004_question_templates = json.load(
    open(question_templates_path+'ace2004.json'))
ace2005_question_templates = json.load(
    open(question_templates_path+'ace2005.json'))
duie_question_templates = json.load(
    open(question_templates_path + "duie.json")
)
duie_entity2rel = {'人物': ['毕业院校', '父亲', '母亲', '妻子', '丈夫', '祖籍', '国籍'],
                   '电视综艺': ['嘉宾', '主持人'],
                   '娱乐人物': ['配音', '获奖', '饰演'],
                   '影视作品': ['主题曲', '上映时间', '票房', '编剧', '制片人', '改编自', '出品公司', '导演', '主演'],
                   '企业/品牌': ['代言人'], '歌曲': ['所属专辑', '歌手', '作词', '作曲'],
                   '图书作品': ['作者'],
                   '学科专业': ['专业代码', '修业年限'], '机构': ['占地面积', '成立日期', '简称'],
                   '行政区': ['邮政编码', '气候', '面积', '人口数量'], '企业': ['注册资本', '创始人', '总部地点', '董事长'],
                   '文学作品': ['主角'], '学校': ['校长'], '国家': ['首都', '官方语言'], '历史人物': ['朝代', '号'], '景点': ['所在城市'], '地点': ['海拔']}

ace2005_q2type = {"qa_turn1":{v:k for k,v in ace2005_question_templates["qa_turn1"].items()},
                "qa_turn2":{v:k for k,v in ace2005_question_templates["qa_turn2"].items()}}
duie_q2type = {"qa_turn1":{v:k for k,v in duie_question_templates["qa_turn1"].items()},
                "qa_turn2":{v:k for k,v in duie_question_templates["qa_turn2"].items()}}
ace2004_mq_question_templates = json.load(
    open(question_templates_path+'ace2004_mq.json'))
ace2005_mq_question_templates = json.load(
    open(question_templates_path+'ace2005_mq.json'))

tag_idxs = {'O': 0,'B': 1, 'M': 2, 'E': 3, 'S': 4}
relH_tag_idxs = {'O': 0,"SH2OH":1,"OH2SH":2}
relT_tag_idxs = {'O': 0,"ST2OT":1,"OT2ST":2}

duie_relH_tag_idxs = {'O': 0, "SH2ValueH": 1, "ValueH2SH": 2, "SH2DateH": 3, "DateH2SH": 4, "SH2WorkH": 5, "WorkH2SH": 6,
                      "SH2AreaH": 7, "AreaH2SH": 8, "SH2PeriodH":9, "PeriodH2SH": 10}
role2duieH = {'onDate': {"SH2OH": 3, "OH2SH": 4}, 'period': {"SH2OH": 9, "OH2SH": 10},
              '@value': {"SH2OH": 1, "OH2SH": 2}, 'inArea': {"SH2OH": 7, "OH2SH": 8},
              'inWork': {"SH2OH": 5, "OH2SH": 6}}
duie_relT_tag_idxs = {'O': 0, "ST2ValueT": 1, "ValueT2ST": 2, "ST2DateT": 3, "DateT2ST": 4, "ST2WorkT": 5, "WorkT2ST": 6,
                      "ST2AreaT": 7, "AreaT2ST": 8, "ST2PeriodT":9, "PeriodT2ST": 10}
role2duieT = {'onDate': {"ST2OT": 3, "OT2ST": 4}, 'period': {"ST2OT": 9, "OT2ST": 10},
              '@value': {"ST2OT": 1, "OT2ST": 2}, 'inArea': {"ST2OT": 7, "OT2ST": 8},
              'inWork': {"ST2OT": 5, "OT2ST": 6}}
ace2004_entities = ['FAC', 'GPE', 'LOC', 'ORG', 'PER', 'VEH', 'WEA']
ace2004_entities_full = ["facility", "geo political",
                         "location", "organization", "person", "vehicle", "weapon"]
ace2004_relations = ['ART', 'EMP-ORG',
                     'GPE-AFF', 'OTHER-AFF', 'PER-SOC', 'PHYS']
ace2004_relations_full = ['artifact', 'employment, membership or subsidiary',
                          'geo political affiliation', 'person or organization affiliation', 'personal or social', 'physical']

ace2005_entities = ['FAC', 'GPE', 'LOC', 'ORG', 'PER', 'VEH', 'WEA']
ace2005_entities_full = ["facility", "geo political",
                         "location", "organization", "person", "vehicle", "weapon"]
ace2005_relations = ['ART', 'GEN-AFF',
                     'ORG-AFF', 'PART-WHOLE', 'PER-SOC', 'PHYS']
ace2005_relations_full = ["artifact", "gen affilliation",
                          'organization affiliation', 'part whole', 'person social', 'physical']

# index of ace2004 and ace2005 frequency matrix
ace2004_idx1 = {'FAC': 0, 'GPE': 1, 'LOC': 2,
                'ORG': 3, 'PER': 4, 'VEH': 5, 'WEA': 6}
ace2005_idx1 = {'FAC': 0, 'GPE': 1, 'LOC': 2,
                'ORG': 3, 'PER': 4, 'VEH': 5, 'WEA': 6}
ace2005_idx2t = {}
for i, rel in enumerate(ace2005_relations):
    for j, ent in enumerate(ace2005_entities):
        ace2005_idx2t[(rel, ent)] = i*len(ace2005_relations)+j+i
ace2005_idx2 = ace2005_idx2t
ace2004_idx2t = {}
for i, rel in enumerate(ace2004_relations):
    for j, ent in enumerate(ace2004_entities):
        ace2004_idx2t[(rel, ent)] = i*len(ace2004_relations)+j+i
ace2004_idx2 = ace2004_idx2t
# statistics on the training set
ace2005_dist = [[0,   0,   0,   0,   0,   0,   0,   0,   3,   1,   0,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,  33, 116,  39,   2,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0,  10,  22,  11,   0,
                 0,   0,   0],
                [30,   0,   0,   0,   0,  60,  61,   0,   0,   0,   0,   0,   0,
                 0,   0,   0,   0,  19,   0,   0,   0,   1, 143,  47,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0,   4,  14,   9,   0,
                 0,   0,   0],
                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0, 120,  31,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0,   3,  18,   8,   0,
                 0,   0,   0],
                [35,   0,   0,   0,   0,  35,  10,   0, 149,  20,   0,   0,   0,
                 0,   0,   5,   0,  12,   0,   0,   0,   0, 147,   1,  81,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0,   3,   5,   0,   0,
                 0,   0,   0],
                [67,   1,   0,   2,   0, 113,  77,   0, 270,  27,  10,  32,   0,
                 0,   0, 587,   0, 844,   5,   0,   0,   0,   0,   0,   4,   0,
                 0,   0,   0,   0,   0,   4, 434,   0,   0, 281, 494, 213,   4,
                 0,   1,   0],
                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 8,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 0,   0,   0],
                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 0,   0,   0]]
# statistics on the entire data set
ace2004_dist = [[0,    1,    0,    0,    0,    0,    0,    0,    2,    0,    0,
                 0,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,
                 0,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,
                 0,    0,   31,  104,   28,    0,    0,    0,    0],
                [8,    1,    1,    0,    0,   27,    8,    0,    8,    0,    3,
                 0,    0,    0,    1,   12,    5,    0,    0,    0,    0,    0,
                 1,    0,    3,   20,    0,    0,    0,    0,    0,    0,    1,
                 0,    0,    0,  236,   55,    0,    0,    0,    0],
                [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    1,
                 0,    0,    0,    0,    5,    0,    0,    0,    0,    0,    0,
                 1,    0,    1,    1,    0,    0,    0,    0,    0,    0,    0,
                 0,    0,    0,   65,   38,    0,    0,    0,    0],
                [28,    0,    0,    0,    0,   24,    2,    0,  132,    0,  120,
                 1,    0,    0,    0,  203,   16,    0,    0,    0,    0,    0,
                 4,    1,    6,    6,    0,    0,    0,    0,    0,    0,    1,
                 0,    0,    4,   19,    2,    1,    0,    0,    0],
                55,    0,    0,    0,    0,   29,   25,    3,  311,    1, 1035,
                [55,    0,    0,    0,    0,   29,   25,    3,  311,    1, 1035,
                 8,    0,    0,    0,  276,    9,    0,    1,    0,    0,    1,
                 5,    4,   18,   69,    0,    0,    0,    0,    0,    0,  363,
                 0,    0,  168,  328,   55,    8,    9,   16,    0],
                [0,    3,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                 0,    0,    8,   19,    5,    0,    0,    3,    0],
                [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                 0,    0,    3,    1,    3,    0,    0,    2,    3]]
# dic={}
# for i,ent_type in enumerate(ace2005_entities):
#     dic[ent_type] = []
#     for k,idx in ace2005_idx2t.items():
#         rel,ent2=k
#         print(k,idx)
#         if(ace2005_dist[i][idx]>0):
#             print(ent_type,rel,ent2,i,idx,ace2005_dist[i][idx])
#             dic[ent_type].append(str((ent_type,rel,ent2)))
# print(dic)
# print(ace2005_idx2t.keys())
