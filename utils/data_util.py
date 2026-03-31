import torch

# 读取实体编号文件
def read_entity_from_id(path):
    entity2id = {}
    with open(path + 'entity2id.txt', 'r',encoding='utf-8') as f:
        for line in f:
            instance = line.strip().split()
            entity2id[instance[0]] = int(instance[1])

    return entity2id


def read_relation_from_id(path):
    relation2id = {}
    with open(path + 'relation2id.txt', 'r') as f:
        for line in f:
            instance = line.strip().split()
            relation2id[instance[0]] = int(instance[1])

    return relation2id

# Calculate adjacency matrix
def get_adj(path, split):
    entity2id = read_entity_from_id(path)
    relation2id = read_relation_from_id(path)
    triples = []
    rows, cols, data = [], [], []
    unique_entities = set()
    with open(path+split+'.txt', 'r',encoding='utf-8') as f:
        for line in f:
            instance = line.strip().split(' ')
            e1, r, e2 = instance[0], instance[1], instance[2]
            unique_entities.add(e1)
            unique_entities.add(e2)
            triples.append((entity2id[e1], relation2id[r], entity2id[e2]))
            rows.append(entity2id[e2])
            cols.append(entity2id[e1])
            data.append(relation2id[r])

    return triples, (cols, rows, data), unique_entities


# Load data triples and adjacency matrix
def load_data(datasets):
    path = 'datasets/'+datasets+'/'
    #train_triples为三元组列表，train_adj为h,t,r三个列表的综合
    train_triples, train_adj, train_unique_entities = get_adj(path, 'train')
    val_triples, val_adj, val_unique_entities = get_adj(path, 'valid')
    test_triples, test_adj, test_unique_entities = get_adj(path, 'test')
    entity2id = read_entity_from_id(path)
    relation2id = read_relation_from_id(path)
    #note2
    img_features = torch.load(open(path+'img_features.pth', 'rb'))
    text_features = torch.load(open(path+'text_features.pth', 'rb'))
    # img_features = pickle.load(open(path+'img_features.pkl', 'rb'))
    # text_features = pickle.load(open(path+'text_features.pkl', 'rb'))

    return entity2id, relation2id, img_features, text_features, \
           (train_triples, train_adj, train_unique_entities), \
           (val_triples, val_adj, val_unique_entities), \
           (test_triples, test_adj, test_unique_entities)

