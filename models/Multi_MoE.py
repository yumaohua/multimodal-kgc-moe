from layers.layer1_moe import *
from layers.layer2_fuse import *
# from .model import BaseModel
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.device = args.device

    @staticmethod
    def format_metrics(metrics, split):
        return " ".join(
            ["{}_{}: {:.4f}".format(split, metric_name, metric_val) for metric_name, metric_val in metrics.items()])

    @staticmethod
    def has_improved(m1, m2):
        return (m1["Mean Rank"] > m2["Mean Rank"]) or (m1["Mean Reciprocal Rank"] < m2["Mean Reciprocal Rank"])

    @staticmethod
    def init_metric_dict():
        return {"Hits@100": -1, "Hits@10": -1, "Hits@3": -1, "Hits@1": -1,
                "MR": 100000, "MRR": -1}

class Multi_MoE(BaseModel):
    def __init__(self, args):
        super(Multi_MoE, self).__init__(args)
        self.dim=args.dim
        self.device=args.device
        # 1、初始化结构embedding
        self.entity_embeddings = nn.Embedding(
            len(args.entity2id),
            args.dim,
            padding_idx=None
        )
        nn.init.xavier_normal_(self.entity_embeddings.weight)

        self.relation_embeddings = nn.Embedding(
            2 * len(args.relation2id),
            args.r_dim,
            padding_idx=None
        )
        nn.init.xavier_normal_(self.relation_embeddings.weight)

        # 3、对这些预处理的图文embedding进行维度统一处理
        if args.dataset == "DB15K":  #这里就用了一个数据集
            img_pool = torch.nn.AvgPool2d(4, stride=4)
            img = img_pool(args.img.to(self.device).view(-1, 64, 64))
            img = img.view(img.size(0), -1)
            txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            txt = txt_pool(args.desp.to(self.device).view(-1, 12, 64))
            txt = txt.view(txt.size(0), -1)

        # 图像实体embedding使用图像本身的，图像关系embedding初始化一个
        # ps:区分CV训练的模型是为了得出图像embedding，而这里的图像embedding作为一个最底层存储embedding层，用于三元组对应匹配
        # 这几块儿一直都是拿到图像embedding，不涉及到图像embedding的CV模型
        self.img_entity_embeddings = nn.Embedding.from_pretrained(img, freeze=False)
        self.img_relation_embeddings = nn.Embedding(
            2 * len(args.relation2id),
            args.r_dim,
            padding_idx=None
        )
        nn.init.xavier_normal_(self.img_relation_embeddings.weight)
        # 文本同理
        self.txt_entity_embeddings = nn.Embedding.from_pretrained(txt, freeze=False)
        self.txt_relation_embeddings = nn.Embedding(
            2 * len(args.relation2id),
            args.r_dim,
            padding_idx=None
        )
        nn.init.xavier_normal_(self.txt_relation_embeddings.weight)

        # Score Functions
        self.dim = args.dim
        self.img_dim = self.img_entity_embeddings.weight.data.shape[1]
        self.txt_dim = self.txt_entity_embeddings.weight.data.shape[1]
        self.fuse_out_dim = self.dim

        # Multi-modal MOE layers
        self.visual_moe = MoEAdaptorLayer(n_exps=args.n_exp, layers=[self.img_dim, self.img_dim])
        self.text_moe = MoEAdaptorLayer(n_exps=args.n_exp, layers=[self.txt_dim, self.txt_dim])
        self.structure_moe = MoEAdaptorLayer(n_exps=args.n_exp, layers=[self.dim, self.dim])
        self.mm_moe = MoEAdaptorLayer(n_exps=args.n_exp, layers=[self.fuse_out_dim, self.fuse_out_dim])
        # Multi-modal fusion layers
        self.fuse_e = ModalFusionLayer(in_dim=args.dim, out_dim=self.fuse_out_dim, multi=2, img_dim=self.img_dim,
                                       txt_dim=self.txt_dim)
        self.fuse_r = ModalFusionLayer(in_dim=args.r_dim, out_dim=self.fuse_out_dim, multi=2, img_dim=args.r_dim,
                                       txt_dim=args.r_dim)

        self.bias = nn.Parameter(torch.zeros(len(args.entity2id)))
        self.bceloss=nn.BCELoss()

    def forward(self, batch_inputs, adj):
        #batch_inputs为triple三元组[[h1,r1,t1],[h2,r2,t2],...]
        head = batch_inputs[:, 0]  #[b]
        relation = batch_inputs[:, 1] #[b]

        stru_head = self.entity_embeddings(head) #[ b,dim]
        img_head=self.img_entity_embeddings(head) #[b,dim]
        txt_head=self.txt_entity_embeddings(head) #[b,dim]
        # e_embed, disen_str, atten_s = self.structure_moe(stru_head )
        # e_img_embed, disen_img, atten_i = self.visual_moe(img_head)
        # e_txt_embed, disen_txt, atten_t = self.text_moe(txt_head)
        e_embed, disen_str, atten_s, lb1 = self.structure_moe(stru_head)
        e_img_embed, disen_img, atten_i, lb2 = self.visual_moe(img_head)
        e_txt_embed, disen_txt, atten_t, lb3 = self.text_moe(txt_head)

        r_embed = self.relation_embeddings(relation)
        r_img_embed,r_txt_embed=r_embed,r_embed

        # 2、融合(三模态堆叠，经过dense变成三个gate并进行加权合并)
        # [b,dim] [b,3]
        e_mm_embed, attn_f = self.fuse_e(e_embed, e_img_embed, e_txt_embed)
        r_mm_embed, _ = self.fuse_r(r_embed, r_img_embed, r_txt_embed)

        # 3、各模态预测得分
        # e_embed [b,dim],r_embed=[b,dim]
        pred_s = e_embed*r_embed    #[b,dim]
        pred_i = e_img_embed* r_img_embed
        pred_d = e_txt_embed* r_txt_embed
        pred_mm = e_mm_embed* r_mm_embed

        # 4、预测所有类
        # self.entity_embeddings.weight=[num_entities, dim]
        # all_s [num,dim]
        all_s=self.entity_embeddings.weight  #[all_entity,dim]  -> [dim,all_entity]
        all_v=self.img_entity_embeddings.weight
        all_t = self.txt_entity_embeddings.weight
        all_f, _ = self.fuse_e(all_s, all_v, all_t)  # [num,dim]
        pred_s = torch.mm(pred_s, all_s.transpose(1, 0))  # [b,allentity]
        pred_i = torch.mm(pred_i, all_v.transpose(1, 0))  # [b,num]
        pred_d = torch.mm(pred_d, all_t.transpose(1, 0))  # [b,num]
        pred_mm = torch.mm(pred_mm, all_f.transpose(1, 0))  # [b,num]

        pred_s = torch.sigmoid(pred_s)  # [b,num]
        pred_i = torch.sigmoid(pred_i)  # [b,num]
        pred_d = torch.sigmoid(pred_d)  # [b,num]
        pred_mm = torch.sigmoid(pred_mm)  # [b,num]
        balance_loss = lb1 + lb2 + lb3
        if not self.training:
            return [pred_s, pred_i, pred_d, pred_mm], [atten_s, atten_i, atten_t, attn_f]
        else:
            return [pred_s, pred_i, pred_d, pred_mm], [disen_str, disen_img, disen_txt], balance_loss

    def get_batch_embeddings(self, batch_inputs):
        head = batch_inputs[:, 0]
        # _, disen_str, _ = self.structure_moe(self.entity_embeddings(head))
        # _, disen_img, _ = self.visual_moe(self.img_entity_embeddings(head))
        # _, disen_txt, _ = self.text_moe(self.txt_entity_embeddings(head))
        _, disen_str, _, _ = self.structure_moe(self.entity_embeddings(head))
        _, disen_img, _, _ = self.visual_moe(self.img_entity_embeddings(head))
        _, disen_txt, _, _ = self.text_moe(self.txt_entity_embeddings(head))
        return [disen_str, disen_img, disen_txt]

    # def loss_func(self, output, target):
    #     loss_s = self.bceloss(output[0], target)
    #     loss_i = self.bceloss(output[1], target)
    #     loss_d = self.bceloss(output[2], target)
    #     loss_mm = self.bceloss(output[3], target)
    #     return loss_s + loss_i + loss_d + loss_mm
    def loss_func(self, output, target, balance_loss):

        loss_s = self.bceloss(output[0], target)
        loss_i = self.bceloss(output[1], target)
        loss_d = self.bceloss(output[2], target)
        loss_mm = self.bceloss(output[3], target)

        main_loss = loss_s + loss_i + loss_d + loss_mm

        return main_loss + 0.01 * balance_loss
