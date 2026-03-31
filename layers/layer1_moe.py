import torch
import torch.nn as nn
import torch.nn.functional as F


class PWLayer(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.lin = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.lin(self.dropout(x))


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.scale * x


class MoEAdaptorLayer(nn.Module):
    def __init__(self, n_exps, layers, dropout=0.0, noise=True):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.dim = layers[1]

        self.experts = nn.ModuleList([
            PWLayer(layers[0], layers[1], dropout)
            for _ in range(n_exps)
        ])

        # Kimi-style attention pooling
        self.query = nn.Parameter(torch.randn(1,self.n_exps, self.dim))
        self.norm = RMSNorm(self.dim)

    def forward(self, x):
        # x: [b, dim]

        expert_outputs = []

        for i in range(self.n_exps):
            tmp = self.experts[i](x)      # [b, dim]
            tmp = tmp.unsqueeze(-2)      # [b,1,dim]
            expert_outputs.append(tmp)

        expert_outputs = torch.cat(expert_outputs, dim=-2)   # [b, n_exps, dim]

        # ===== Kimi attention =====

        values = expert_outputs                     # value
        keys = self.norm(values)                    # key

        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)   # [b,n_exps,dim]

        scores = (keys * query).sum(-1)             # [b, n_exps]

        gates = torch.softmax(scores, dim=-1)       # attention weights

        multiple_outputs = gates.unsqueeze(-1) * values   # [b,n_exps,dim]

        output = multiple_outputs.sum(dim=-2)       # [b,dim]

        usage = gates.mean(dim=0)

        balance_loss = ((usage - 1/self.n_exps) ** 2).sum()

        return output, expert_outputs, gates, balance_loss

        # return output, expert_outputs, gates

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class PWLayer(nn.Module):
#     def __init__(self, input_size, output_size, dropout=0.0):
#         super(PWLayer, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         self.lin = nn.Linear(input_size, output_size)

#     def forward(self, x):
#         return self.lin(self.dropout(x))

# class MoEAdaptorLayer(nn.Module):
#     def __init__(self, n_exps, layers, dropout=0.0, noise=True):
#         super(MoEAdaptorLayer, self).__init__()
#         self.n_exps = n_exps
#         self.experts = nn.ModuleList([PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)])
#         self.gate=nn.Linear(layers[0],n_exps) #[dim,3]

#     def forward(self, x):
#         # x:[b,dim],  r=[b,1]
#         gates = F.softmax(self.gate(x),dim=-1)  # (B, n_exps) [b,3]
#         # expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)]  # [(B, 1, D)]*n_exps
#         expert_outputs=[]
#         for i in range(self.n_exps):
#             tmp=self.experts[i](x) #[b,dim]
#             tmp=tmp.unsqueeze(-2) #[b,1,dim]
#             expert_outputs.append(tmp)

#         expert_outputs = torch.cat(expert_outputs, dim=-2)  # (B, 3, D)
#         multiple_outputs = gates.unsqueeze(-1) * expert_outputs  # (b,n_exps,D)
#         return multiple_outputs.sum(dim=-2), expert_outputs, gates  # (b,D),(B, n_exps, D),(B, n_exps)


