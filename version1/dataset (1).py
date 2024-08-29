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
    

class OmniDataset(Dataset):
    #  {Спецпромпт\n}[SOI][IMG][EOI][USER]{Q}[BOT]{A</s>}...
    def __init__(self, cfg, tokenizer):
        # with open(cfg.json_data_path) as f:
        #     json_data = json.load(f)
        self.cfg = cfg
        # self.json_data = json_data
        self.json_data = load_dataset("lmms-lab/LLaVA-ReCap-118K")['train']
        self.tokenizer = tokenizer
        
        self.t = T.Compose([
                            T.ToTensor(),
                            T.Normalize(mean = [0.5], std = [0.5]),
                            ])
        
        # self.dwt = DWT_2D('haar')

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
            # image_path = f"{self.cfg.image_folder}/{data_item['image']}"
            # image = Image.open(image_path).convert("RGB").resize((512, 512))
            image = data_item['image'].convert("RGB").resize((512, 512))
            image = np.array(image) / 255
            coeffs = wavedec_with_filter(image, w = 'db1', keep=0.25, l = 5)
            image = reshape_coeffs(coeffs)

            image = torch.from_numpy(image).flatten(1).T
            
            image = image.to(dtype=torch.float32)
            

            tokens.append(
                torch.tensor(
                    [(self.cfg.vision_emb_num + 2)*[self.cfg.pad_id]],
                    dtype=torch.int32,
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
            else: # from gpt
                positions.append({'type': 'BOT', 'position': len(mask)})
            mask += [False]
            tokens.append(
                torch.tensor(
                    [[self.cfg.pad_id]],
                    dtype=torch.int32,
                )
            )
                
            
            if conversation['from'] == 'human':
                text = conversation['value'].replace('\n<image>', '').replace('<image>\n', '').replace('<image>', '')
                text_tokens = self.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
                mask += text_tokens.shape[-1] * [False]
                
            else:
                text = conversation['value'].replace('</s>', '') + self.tokenizer.eos_token
                text_tokens = self.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
                mask += text_tokens.shape[-1] * [True]
                

            
            
            tokens.append(text_tokens)

        tokens = torch.cat(tokens, dim = -1)[0]
        mask = torch.tensor(mask, dtype=bool)
        return image, tokens, mask, positions
        

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
