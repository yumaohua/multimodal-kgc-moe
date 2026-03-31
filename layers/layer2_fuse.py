import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms
        return x * self.weight


class ModalFusionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, multi, img_dim, txt_dim):
        super(ModalFusionLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.multi = multi
        self.img_dim = img_dim
        self.text_dim = txt_dim

        modal1 = []
        for _ in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(in_dim, out_dim)
            modal1.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal1_layers = nn.ModuleList(modal1)

        modal2 = []
        for _ in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(self.img_dim, out_dim)
            modal2.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal2_layers = nn.ModuleList(modal2)

        modal3 = []
        for _ in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(self.text_dim, out_dim)
            modal3.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal3_layers = nn.ModuleList(modal3)

        self.ent_attn = nn.Linear(self.out_dim, 1, bias=False)
        self.ent_attn.requires_grad_(True)

        # ===== 修改1：query改为 multi × dim =====
        self.final_query = nn.Parameter(torch.randn(self.multi, self.out_dim))

        # ===== 修改2：norm =====
        self.final_norm = RMSNorm(self.out_dim)

    def forward(self, modal1_emb, modal2_emb, modal3_emb):

        batch_size = modal1_emb.size(0)

        x_mm = []

        for i in range(self.multi):

            x_modal1 = self.modal1_layers[i](modal1_emb)
            x_modal2 = self.modal2_layers[i](modal2_emb)
            x_modal3 = self.modal3_layers[i](modal3_emb)

            x_stack = torch.stack((x_modal1, x_modal2, x_modal3), dim=1)

            attention_scores = self.ent_attn(x_stack).squeeze(-1)

            attention_weights = torch.softmax(attention_scores, dim=-1)

            context_vectors = torch.sum(
                attention_weights.unsqueeze(-1) * x_stack,
                dim=1
            )

            x_mm.append(context_vectors)

        x_mm = torch.stack(x_mm, dim=1)   # [b, multi, dim]

        # ===== Kimi attention pooling =====

        values = x_mm                     # [b, multi, dim]

        keys = self.final_norm(values)

        # ===== 修改3：query reshape =====
        query = self.final_query.unsqueeze(0)   # [1, multi, dim]

        scores = (keys * query).sum(-1)         # [b, multi]

        weights = torch.softmax(scores, dim=-1)

        x_mm = torch.sum(weights.unsqueeze(-1) * values, dim=1)

        return x_mm, attention_weights
# import torch
# import torch.nn as nn
# class ModalFusionLayer(nn.Module):
#     def __init__(self, in_dim, out_dim, multi, img_dim, txt_dim):
#         super(ModalFusionLayer, self).__init__()

#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.multi = multi
#         self.img_dim = img_dim
#         self.text_dim = txt_dim

#         # self.multi是几个mlp叠加，与多模态无关
#         modal1 = []
#         for _ in range(self.multi):
#             do = nn.Dropout(p=0.2)
#             lin = nn.Linear(in_dim, out_dim)
#             modal1.append(nn.Sequential(do, lin, nn.ReLU()))
#         self.modal1_layers = nn.ModuleList(modal1)

#         modal2 = []
#         for _ in range(self.multi):
#             do = nn.Dropout(p=0.2)
#             lin = nn.Linear(self.img_dim, out_dim)
#             modal2.append(nn.Sequential(do, lin, nn.ReLU()))
#         self.modal2_layers = nn.ModuleList(modal2)

#         modal3 = []
#         for _ in range(self.multi):
#             do = nn.Dropout(p=0.2)
#             lin = nn.Linear(self.text_dim, out_dim)
#             modal3.append(nn.Sequential(do, lin, nn.ReLU()))
#         self.modal3_layers = nn.ModuleList(modal3)

#         self.ent_attn = nn.Linear(self.out_dim, 1, bias=False)
#         self.ent_attn.requires_grad_(True)

#     def forward(self, modal1_emb, modal2_emb, modal3_emb):
#         # emb:[b,dim]
#         batch_size = modal1_emb.size(0)
#         x_mm = []
#         for i in range(self.multi):  # self.multi是几个mlp叠加，与多模态无关
#             x_modal1 = self.modal1_layers[i](modal1_emb)  # [b,dim]
#             x_modal2 = self.modal2_layers[i](modal2_emb)  # [b,dim]
#             x_modal3 = self.modal3_layers[i](modal3_emb)  # [b,dim]
#             x_stack = torch.stack((x_modal1, x_modal2, x_modal3), dim=1)  # 沿dim=1进行堆叠[b,3,dim]
#             attention_scores = self.ent_attn(x_stack).squeeze(-1)  # [b,3]
#             attention_weights = torch.softmax(attention_scores, dim=-1)  # [b,3]
#             context_vectors = torch.sum(attention_weights.unsqueeze(-1) * x_stack, dim=1)  # [b,dim]
#             x_mm.append(context_vectors)
#         x_mm = torch.stack(x_mm, dim=1)  # [b,multi,dim]
#         x_mm = x_mm.sum(1).view(batch_size, self.out_dim)  # [b,dim]
#         # x_mm = torch.relu(x_mm)
#         return x_mm, attention_weights  # [b,dim]  [b,3]

