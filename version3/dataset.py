import json
import torch
import torch.nn as nn
import numpy as np
import random
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pywt
import ptwt
import torchvision.transforms.v2 as T
from datasets import load_dataset

def expand2square(pil_img, background_color = 0):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def wavedec_with_filter(image, w = 'db1', l = 4, keep = 0.1):

    results = []
    for i in range(image.shape[-1]):
        coeffs = pywt.wavedec2(image[:,:, i], wavelet=w, level=l)
        coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
        thr = np.quantile(np.abs(coeff_arr), keep)
        ind = np.abs(coeff_arr) > thr
        Cfilt = coeff_arr * ind
        coeffs_filt = pywt.array_to_coeffs(Cfilt, coeff_slices, output_format='wavedec2')
        results.append(coeffs_filt)
        
    return results
    

def reshape_coeffs(coeffs):
    ch1, ch2, ch3 = coeffs
    size = ch1[0].shape
    results = []
    results.append(
        np.stack([ch1[0], ch2[0], ch3[0]])
    )
    
    results.append(
        np.concatenate([np.stack(ch1[1]), np.stack(ch2[1]), np.stack(ch3[1])])
    )
    
    for c1, c2, c3 in zip(ch1[2:], ch2[2:], ch3[2:]):
        c1 = np.stack(c1).reshape(-1, *(size))
        c2 = np.stack(c2).reshape(-1, *(size))
        c3 = np.stack(c3).reshape(-1, *(size))
        results.append(np.concatenate([c1,c2,c3]))
    
    return np.concatenate(results)

def pixel_space_embeddings(image, wavelet = 'db1', level = 5, keep = 0.2, level_limit=None):
    if level_limit is None:
        level_limit = level
    if (level_limit > level) or (level_limit < 1):
        level_limit = level

    A = []
    H = [[] for _ in range(level)]
    V = [[] for _ in range(level)]
    D = [[] for _ in range(level)]

    for i in range(image.shape[-1]):
        channel = image[:, :, i]
        channel_coeffs = pywt.wavedec2(channel, wavelet=wavelet, level=level, mode='periodization')

        # Фильтрация
        coeff_arr, coeff_slices = pywt.coeffs_to_array(channel_coeffs)
        thr = np.quantile(np.abs(coeff_arr), 1 - keep)
        ind = np.abs(coeff_arr) > thr
        Cfilt = coeff_arr * ind
        channel_coeffs = pywt.array_to_coeffs(Cfilt, coeff_slices, output_format='wavedec2')

        cA = channel_coeffs[0]  # Аппроксимация на последнем уровне
        N = cA.size
        A.append(np.reshape(cA, (N, -1)))

        for l in range(level):
            (cH, cV, cD) = channel_coeffs[l + 1]  # Коэффициенты на l-уровне (горизонтальные, вертикальные, диагональные)
            H[level-l-1].append(np.reshape(cH, (N,-1)))
            V[level-l-1].append(np.reshape(cV, (N,-1)))
            D[level-l-1].append(np.reshape(cD, (N,-1)))

    # Конкатенируем по всем каналам изображения
    A = np.concatenate(A, axis=1)
    H = [np.concatenate(H[i], axis=1) for i in range(level)]
    V = [np.concatenate(V[i], axis=1) for i in range(level)]
    D = [np.concatenate(D[i], axis=1) for i in range(level)]

    # Конкатенация массивов H, V и D для каждого уровня
    # Примечание: в статье это векторы D, формула (8)
    Wk = [np.concatenate((H[i], V[i], D[i]), axis=1) for i in range(level)]

    # Конкатенируем A (в статье это вектор P2, формула 7) с
    # Wk для последнего уровня (в статье это вектор D2)
    Wk[-1] = np.concatenate((A, Wk[-1]), axis=1) # Примечание: смотри формулу (10) статьи

    # Ограничим число учитываемых в модели уровней вейвлет преобразования:
    Wk = Wk[level - level_limit:]

    Wk_lengths = [Wk[i].shape[1] for i in range(len(Wk))]
    Wk = torch.from_numpy(np.concatenate(Wk, axis=1, dtype=np.float32)) # [36,16,16]
    return Wk, Wk_lengths


# def pixel_space_embeddings(image, wavelet='db1', level=5, keep=0.1, level_limit=None):
#     if level_limit is None:
#         level_limit = level
#     if (level_limit > level) or (level_limit < 1):
#         level_limit = level

#     A = []
#     H = [[] for _ in range(level)]
#     V = [[] for _ in range(level)]
#     D = [[] for _ in range(level)]

#     for i in range(image.shape[-1]):
#         channel = image[:, :, i]
#         channel_coeffs = pywt.wavedec2(channel, wavelet=wavelet, level=level)

#         # Фильтрация
#         coeff_arr, coeff_slices = pywt.coeffs_to_array(channel_coeffs)
#         thr = np.quantile(np.abs(coeff_arr), 1 - keep)
#         ind = np.abs(coeff_arr) > thr
#         Cfilt = coeff_arr * ind
#         channel_coeffs = pywt.array_to_coeffs(Cfilt, coeff_slices, output_format='wavedec2')

#         cA = channel_coeffs[0]  # Аппроксимация на последнем уровне
#         A.append(cA.flatten())

#         for l in range(level):
#             (cH, cV, cD) = channel_coeffs[
#                 l + 1]  # Коэффициенты на l-уровне (горизонтальные, вертикальные, диагональные)
#             H[level - l - 1].append(cH.flatten())
#             V[level - l - 1].append(cV.flatten())
#             D[level - l - 1].append(cD.flatten())

#     # Конкатенируем по всем каналам изображения
#     A = np.concatenate(A, axis=0)
#     H = [np.concatenate(H[i], axis=0) for i in range(level)]
#     V = [np.concatenate(V[i], axis=0) for i in range(level)]
#     D = [np.concatenate(D[i], axis=0) for i in range(level)]

#     # Конкатенация массивов H, V и D для каждого уровня
#     # Примечание: в статье это векторы D, формула (8)
#     Wk = [np.concatenate((H[i], V[i], D[i]), axis=0) for i in range(level)]

#     # Конкатенируем A (в статье это вектор P2, формула 7) с
#     # Wk для последнего уровня (в статье это вектор D2)
#     Wk[-1] = np.concatenate((A, Wk[-1]), axis=0)  # Примечание: смотри формулу (10) статьи

#     # Ограничим число учитываемых в модели уровней вейвлет преобразования:
#     Wk = Wk[level - level_limit:]

#     Wk_lengths = [len(Wk[i]) for i in range(len(Wk))]
#     Wk = torch.from_numpy(np.concatenate(Wk, axis=0, dtype=np.float32))

#     return Wk, Wk_lengths


class OmniDataset(Dataset):
    #  {Спецпромпт\n}[SOI][IMG][EOI][USER]{Q}[BOT]{A</s>}...
    def __init__(self, cfg, tokenizer):
        #with open(cfg.json_data_path) as f:
            #json_data = json.load(f)
        self.cfg = cfg
        #self.json_data = json_data
        self.json_data = load_dataset("lmms-lab/LLaVA-ReCap-118K")['train']
        self.tokenizer = tokenizer

        # 1. Добавил новые параметры
        self.image_size = 512
        self.wavelet = 'db1'
        self.level = 5
        self.level_limit = None    

        self.t = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

        # self.dwt = DWT_2D('haar')

    # 2. Добавил метод get_Wk_lengths, для того чтобы была возможность
    #   создать блочно-диагональную модель обучения (формула 15 статьи)
    def get_Wk_lengths(self):
        random_image = np.random.rand(self.image_size, self.image_size, 3)
        _ , Wk_lengths = pixel_space_embeddings(random_image, wavelet = self.wavelet, level = self.level, level_limit=self.level_limit)
        #Wk_lengths = [len(Wk[i]) for i in range(len(Wk))]
        return Wk_lengths

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        data_item = self.json_data[idx]
        tokens = []
        masks = []
        positions = []

        prompt_tokens = self.tokenizer.encode(f"{self.cfg.prompt}", add_special_tokens=False, return_tensors="pt")
        prompt_len = prompt_tokens.shape[-1]
        tokens.append(prompt_tokens)
        mask = prompt_len * [False]

        image = None
        if 'image' in data_item.keys():
            image_path = f"{self.cfg.image_folder}/{data_item['image']}"
            # 3. Заменил 512 на self.image_size
            # image = Image.open(image_path).convert("RGB").resize((512, 512))
            #image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size))
            image = data_item['image'].convert("RGB").resize((self.image_size, self.image_size))
            # 4. Добавил конвертацию в формат 'YCbCr' и убрал нормировку на 255
            image = image.convert('YCbCr')
            image = np.array(image)  # / 255

            # 5. Заменил следующий фрагмент на расчет Wk (формула 11 статьи)
            # coeffs = wavedec_with_filter(image, w = 'db1', keep=0.25, l = 5)
            # image = reshape_coeffs(coeffs)
            # image = torch.from_numpy(image).flatten(1).T
            # image = image.to(dtype=torch.bfloat16)
            Wk, _ = pixel_space_embeddings(image, wavelet=self.wavelet, level=self.level, keep=0.2, level_limit=self.level_limit)
            #Wk = Wk.to(dtype=torch.bfloat16)
            # Wk = [torch.tensor(w).to(dtype=torch.bfloat16) for w in Wk]

            tokens.append(
                torch.tensor(
                    [(self.cfg.vision_emb_num + 2) * [self.cfg.pad_id]],
                    dtype=torch.int64,
                )
            )

            positions += [
                {'type': 'SOI', 'position': prompt_len},
                {'type': 'IMG', 'position': (prompt_len + 1, prompt_len + 1 + self.cfg.vision_emb_num)},
                {'type': 'EOI', 'position': prompt_len + 1 + self.cfg.vision_emb_num}
            ]

            mask += (2 + self.cfg.vision_emb_num) * [False]

        for conversation in data_item['conversations']:
            if conversation['from'] == 'human':
                positions.append({'type': 'USER', 'position': len(mask)})
            else:  # from gpt
                positions.append({'type': 'BOT', 'position': len(mask)})
            mask += [False]
            tokens.append(
                torch.tensor(
                    [[self.cfg.pad_id]],
                    dtype=torch.int64,
                )
            )

            if conversation['from'] == 'human':
                text = conversation['value'].replace('\n<image>', '').replace('<image>\n', '').replace('<image>', '')
                text_tokens = self.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
                mask += text_tokens.shape[-1] * [False]

            else:
                text = conversation['value'].replace('</s>', '') + "</s>"
                text_tokens = self.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
                mask += text_tokens.shape[-1] * [True]

            tokens.append(text_tokens)

        tokens = torch.cat(tokens, dim=-1)[0]
        mask = torch.tensor(mask, dtype=bool)

        # 6. Заменил image на Wk
        # return image, tokens, mask, positions
        return Wk, tokens, mask, positions
        

def get_dataset(cfg, tokenizer):

    return OmniDataset(cfg, tokenizer)


def get_collate_function(cfg):

    def colate_fn(data):
        images, tokens, masks, positions = zip(*data)

        images_mask = torch.tensor([True if image is not None else False for image in images], dtype=bool)
        if images_mask.sum() > 0:
            images = torch.stack([image for image in images if image is not None])
            
            
        else:
            images = None
        
            
        tokens = list(tokens)
        masks = list(masks)
        positions = list(positions)
        max_len = max([token.shape[-1] for token in tokens])
        for i in range(len(tokens)):
            pad_len = max_len - tokens[i].shape[-1]
            masks[i] = torch.cat([masks[i], torch.tensor(pad_len*[False], dtype=bool)], dim=0)
            tokens[i] = torch.cat([tokens[i], torch.tensor(pad_len*[cfg.pad_id], dtype=int)], dim=0)
    
        
        masks = torch.stack(masks)
        tokens = torch.stack(tokens)
        return images, images_mask, tokens, masks, positions

    return colate_fn
