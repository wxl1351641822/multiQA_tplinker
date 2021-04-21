import json


ace2004_entities = ['FAC', 'GPE', 'LOC', 'ORG', 'PER', 'VEH', 'WEA']
ace2004_entities_full = ["facility","geo political","location","organization","person","vehicle","weapon"]
ace2004_relations = ['ART', 'EMP-ORG', 'GPE-AFF', 'OTHER-AFF', 'PER-SOC', 'PHYS']
ace2004_relations_full = ['artifact','employment, membership or subsidiary','geo political affiliation','person or organization affiliation','personal or social','physical']

ace2005_entities = ['FAC', 'GPE', 'LOC', 'ORG', 'PER', 'VEH', 'WEA']
ace2005_entities_full = ["facility","geo political","location","organization","person","vehicle","weapon"]
ace2005_relations = ['ART', 'GEN-AFF', 'ORG-AFF', 'PART-WHOLE', 'PER-SOC', 'PHYS']
ace2005_relations_full = ["artifact","gen affilliation",'organization affiliation','part whole','person social','physical']

rel2={}
rel2['artifact']=['artifact','is owned by','is inventor of']
rel2["gen affilliation"]=["gen affilliation","is a region of","is"]
rel2["organization affiliation"]=["organization affiliation","is a affiliation of","is a subsidiary of"]
rel2["part whole"]=["part whole","is part of","is belong to"]
rel2["person social"]=["person social","is family with","is business with"]
rel2["physical"]=["physical","is near","is located in"]
#ACE2004
rel2["employment, membership or subsidiary"]=["employment, membership or subsidiary","is employed by","is a subsidiary of"]
rel2["geo political affiliation"]=["geo political affiliation","is a region of","is belong to"]
rel2["person or organization affiliation"]=["person or rganization affiliation","is a affiliation of","organization affiliation"]
rel2["personal or social"]=["personal or social","is family with","is business with"]

ent2_dict = {"location": "location", "geo political": "state"}
with open("../cleaned_data/NYT_star/NYT-star/rel2id.json") as f:
    NYT_star_rel=list(json.load(f).keys())
    print(NYT_star_rel)
if __name__=="__main__":
    # templates = {"qa_turn1": {}, "qa_turn2": {}}
    # for rel in NYT_star_rel:
    #     templates['qa_turn2'][rel] = rel.replace("/"," ")
    #     templates['qa_turn2'][rel] = templates['qa_turn2'][rel].replace("_", " ")
    # with open("nyt_star.json",'w') as f:
    #     json.dump(templates,f)
    # #这里的模板应该保证我们的模型的query和对应的entity type或者(head_entity relation_type, end_entity_type)存在一一对应关系
    # templates = {"qa_turn1":{},"qa_turn2":{}}
    # for ent1,ent1f in zip(ace2005_entities,ace2005_entities_full):
    #     templates['qa_turn1'][ent1]="find all {} entities  in the context.".format(ent1f)
    # for rel, relf in zip(ace2005_relations, ace2005_relations_full):
    #     templates['qa_turn2'][rel] = rel2[relf][0]
    # with open("ace2005.json",'w') as f:
    #     json.dump(templates,f)
    # templates = {"qa_turn1": {}, "qa_turn2": {}}
    # for ent1,ent1f in zip(ace2004_entities,ace2004_entities_full):
    #     templates['qa_turn1'][ent1]="find all {} entities  in the context.".format(ent1f)
    # for rel, relf in zip(ace2004_relations, ace2004_relations_full):
    #     templates['qa_turn2'][rel] = rel2[relf][0]
    # with open("ace2004.json",'w') as f:
    #     json.dump(templates,f)
    #
    # # 这里的模板应该保证我们的模型的query和对应的entity type或者(head_entity relation_type, end_entity_type)存在一一对应关系
    # templates = {"qa_turn1": {}, "qa_turn2": {}}
    # for ent1, ent1f in zip(ace2005_entities, ace2005_entities_full):
    #     templates['qa_turn1'][ent1] = ["find all {} entities  in the context .".format(ent1f),"Which {} are mentioned in the text ?".format(ent1f),"{} .".format(ent1f)]
    # for rel, relf in zip(ace2005_relations, ace2005_relations_full):
    #     templates['qa_turn2'][rel] = rel2[relf]
    # with open("ace2005_mq.json", 'w') as f:
    #     json.dump(templates, f)
    # templates = {"qa_turn1": {}, "qa_turn2": {}}
    # for ent1, ent1f in zip(ace2004_entities, ace2004_entities_full):
    #     templates['qa_turn1'][ent1] = ["find all {} entities  in the context .".format(ent1f),
    #                                    "Which {} are mentioned in the text ?".format(ent1f), "{} .".format(ent1f)]
    # for rel, relf in zip(ace2004_relations, ace2004_relations_full):
    #         templates['qa_turn2'][rel] = rel2[relf]
    # with open("ace2004_mq.json", 'w') as f:
    #     json.dump(templates, f)

    templates = {"qa_turn1": {}, "qa_turn2": {}}
    with open("/data/home/wuyuming/wxl/数据集/baidu/duie_schema.json", "r") as f:
        schemas = f.readlines()
        # print(schemas)
        entity = set()
        subject2predicate = {}
        relation = set()
        for schema in schemas:
            schema = schema[:-1]
            schema = json.loads(schema)
            print(schema)
            if(schema["subject_type"] not in subject2predicate):
                subject2predicate[schema["subject_type"]] = []
            subject2predicate[schema["subject_type"]].append(schema["predicate"])
            for k, v in schema["object_type"].items():
                entity.add(v)
            entity.add(schema["subject_type"])
            relation.add(schema["predicate"])
        print(subject2predicate)
        print(len(list(subject2predicate.keys())))
        print(len(list(entity)), entity)
        print(len(list(relation)), relation)
        for e in list(entity):
            if(e == "Date"):
                templates["qa_turn1"][e] = "日期"
            elif(e == "Number"):
                templates["qa_turn1"][e] = "数字"
            elif(e == "Text"):
                templates["qa_turn1"][e] = "文本"
            else:
                templates["qa_turn1"][e] = e
        for r in list(relation):
            templates["qa_turn2"][r] = r
        with open("duie.json", "w") as f:
            json.dump(templates, f, sort_keys=True, indent=4, ensure_ascii=False)
        with open("duie.json", "r") as f:
            print(json.load(f))

        # 1.一个entity_type一个问题
        # 2.所有实体一个序列标注，26个-还可以，但是实体类别信息没有办法融入了,而且两个任务不一致
        # print(list(entity)+)