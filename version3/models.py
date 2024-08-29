import math
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

def initialize_special_embs(cfg):
    special_embs = {}
    special_embs['USER'] = torch.normal(torch.zeros(cfg.emb_dim), torch.ones(cfg.emb_dim) / cfg.emb_dim**0.5).to(dtype=torch.float32)
    special_embs['BOT'] = torch.normal(torch.zeros(cfg.emb_dim), torch.ones(cfg.emb_dim) / cfg.emb_dim**0.5).to(dtype=torch.float32)
    special_embs['SOI'] = torch.normal(torch.zeros(cfg.emb_dim), torch.ones(cfg.emb_dim) / cfg.emb_dim**0.5).to(dtype=torch.float32)
    special_embs['EOI'] = torch.normal(torch.zeros(cfg.emb_dim), torch.ones(cfg.emb_dim) / cfg.emb_dim**0.5).to(dtype=torch.float32)

    special_embs['SOI'].requires_grad_()
    special_embs['EOI'].requires_grad_()
    special_embs['USER'].requires_grad_()
    special_embs['BOT'].requires_grad_()
    return special_embs




class ConvQ(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.Q = nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride = 1, padding = 1)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x):
        # B, S, D
        B, S, D = x.shape
        size = int(math.sqrt(x.shape[1]))

        x = x.view(B, size, size, D).permute(0, 3, 1, 2)

        out = self.Q(x)

        out = self.norm(out.flatten(2).permute(0, 2, 1))

        return out

class VisualToGPTMapping(nn.Module):
    def __init__(self, Wk_lengths, gpt_emb_dim):
        super(VisualToGPTMapping, self).__init__()

        self.levels = len(Wk_lengths)

        # H — целевой размер эмбеддинга, Hk - часть эмбеддинга H для k-уровня
        H = gpt_emb_dim

        # Создаем проекционные слои для каждого уровня и вычисляем Hk
        # Причем чем больше длина вектора Wk, тем меньше Hk
        self.projections = nn.ModuleList()
        self.Wk_lengths = Wk_lengths  # Сохраняем длины векторов для каждого уровня

        Hk_list = []
        # p - это доля эмбеддинга H для каждого уровня
        p = 1 / np.array(Wk_lengths)
        p = p / np.sum(p)

        for k in range(self.levels):
            length = Wk_lengths[k]
            Hk = max(round(p[k] * H), 1)
            Hk_list.append(Hk)

            if k < self.levels - 1:
                projection_layer = nn.Linear(length, Hk)
            else:
                # Корректируем последний Hk, чтобы сумма Hk была равна H
                Hk = H - sum(Hk_list[:-1])
                projection_layer = nn.Linear(length, Hk)
                Hk_list[-1] = Hk
            self.projections.append(projection_layer)

    def forward(self, Wk_batch):
        embeddings = []
        current_pos = 0

        for k in range(self.levels):
            # Извлекаем для каждого уровня вейвлет-преобразования
            # соответствующие им подмножества из входного батча
            Wk_length = self.Wk_lengths[k]
            Wk = Wk_batch[:, :, current_pos:current_pos + Wk_length]
            current_pos += Wk_length

            # Подаем каждый Wk на соответствующий линейный слой
            projection = self.projections[k]
            embedding = projection(Wk)
            embeddings.append(embedding)

        # Объединяем эмбеддинги всех уровней
        out = torch.cat(embeddings, dim=-1)
        return out



# class VisualToGPTMapping(nn.Module):
#     def __init__(self, Wk_lengths, gpt_emb_dim):
#         super(VisualToGPTMapping, self).__init__()

#         self.levels = len(Wk_lengths)

#         # H — целевой размер эмбеддинга, Hk - часть эмбеддинга H для k-уровня
#         H = gpt_emb_dim

#         # Создаем проекционные слои для каждого уровня и вычисляем Hk
#         # Причем чем больше длина вектора Wk, тем меньше Hk
#         self.projections = nn.ModuleList()
#         self.Wk_lengths = Wk_lengths  # Сохраняем длины векторов для каждого уровня

#         Hk_list = []
#         # p - это доля эмбеддинга H для каждого уровня
#         p = 1 / np.array(Wk_lengths)
#         p = p / np.sum(p)

#         for k in range(self.levels):
#             length = Wk_lengths[k]
#             Hk = round(p[k] * H)
#             Hk_list.append(Hk)

#             if k < self.levels - 1:
#                 projection_layer = nn.Linear(length, Hk)
#             else:
#                 # Корректируем последний Hk, чтобы сумма Hk была равна H
#                 Hk = H - sum(Hk_list[:-1])
#                 projection_layer = nn.Linear(length, Hk)
#                 Hk_list[-1] = Hk

#             self.projections.append(projection_layer)

#     def forward(self, Wk_batch):
#         embeddings = []
#         current_pos = 0

#         for k in range(self.levels):
#             # Извлекаем для каждого уровня вейвлет-преобразования
#             # соответствующие им подмножества из входного батча
#             Wk_length = self.Wk_lengths[k]
#             Wk = Wk_batch[:, current_pos:current_pos + Wk_length]
#             current_pos += Wk_length

#             # Подаем каждый Wk на соответствующий линейный слой
#             projection = self.projections[k]
#             embedding = projection(Wk)
#             embeddings.append(embedding)

#         # Объединяем эмбеддинги всех уровней
#         out = torch.cat(embeddings, dim=-1)
#         return out

'''
class VisualToGPTMapping(nn.Module):
    def __init__(self, visual_emb_dim, gpt_emb_dim):
        super(VisualToGPTMapping, self).__init__()

        self.Q = nn.Sequential(
                    nn.Linear(visual_emb_dim, visual_emb_dim), nn.GELU())

        # self.Q = ConvQ(visual_emb_dim, visual_emb_dim)

        self.pos = nn.Parameter(torch.randn(size = (1, 256, visual_emb_dim)))
        
        # transformer_layer = nn.TransformerEncoderLayer(d_model=visual_emb_dim, nhead=8, batch_first=True, norm_first=False)
        self.mlp = nn.Sequential(
                    nn.Linear(visual_emb_dim, gpt_emb_dim),
                    nn.GELU(),
                    nn.Linear(gpt_emb_dim, gpt_emb_dim))
        # self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)
        
        # self.linear = nn.Linear(visual_emb_dim, gpt_emb_dim)
        
    
    def forward(self, visual_embs0):
        B = visual_embs0.shape[0]
        E = self.Q(visual_embs0) + self.pos.expand(B, -1, -1)
        out = self.mlp(E)
        # out = self.linear(out)
        
        return out
'''




