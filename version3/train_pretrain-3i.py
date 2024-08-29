import random
import argparse
import json
import os
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import Adafactor, AdafactorSchedule, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.loggers import CSVLogger

from models import VisualToGPTMapping, initialize_special_embs
from dataset import get_dataset, get_collate_function
        

class Config:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            if isinstance(v, dict):
                setattr(self, k, Config(**v))
            else:
                setattr(self, k, v)
    
    def __str__(self):
        return '\n'.join(f'{key}: {value}' for key, value in self.__dict__.items())

    def __repr__(self):
        return self.__str__()

def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
        


class Model_pl(pl.LightningModule):
    def __init__(self, cfg, special_embs, model, projection, train_dataset, collate_function):
        super().__init__()
        self.cfg = cfg
        self.special_embs = special_embs
        self.projection = projection
        self.model = model
        self.n_embeddings = model.model.embed_tokens.weight.shape[0]
        self.loss_fct = nn.CrossEntropyLoss(reduction="none")
        self.train_dataset = train_dataset
        self.collate_function = collate_function
        self.n_iters = len(self.train_dataloader())
        self.save_hyperparameters('cfg')
        # self.automatic_optimization = False
        
    def configure_optimizers(self):
        
        # optimizer = Adafactor(list(self.special_embs.values()) + list(self.projection.parameters()), lr=self.cfg.learning_rate, relative_step=False)
        optimizer = torch.optim.AdamW(list(self.special_embs.values()) + list(self.projection.parameters()) + list(self.model.parameters()), lr=self.cfg.learning_rate)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.n_iters // self.cfg.grad_accum * 0.01, num_training_steps=self.n_iters // self.cfg.grad_accum)
        return {'optimizer': optimizer, 'lr_scheduler': {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
            
        }}
    

    def on_train_epoch_end(self):
        torch.save(self.projection, f"ckpts/{self.cfg.exp_name}/projection.pt")
        torch.save(self.special_embs, f"ckpts/{self.cfg.exp_name}/special_embeddings.pt")
        self.model.save_pretrained(f"ckpts/{self.cfg.exp_name}/tuned-model")


    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     print(batch_idx)


    def training_step(self, batch, batch_idx):
        images, images_mask, labels, mask, positions = batch
        labels = labels.long()
        if images_mask.sum() > 0:
            # image_embedding = self.clip(images) # preprocessing!!!
            projected_vision_embeddings = self.projection(images)
        
        embeddings = self.model.model.embed_tokens(labels)
        img_idx_counter = 0
        for i in range(len(embeddings)):
            for pos in positions[i]:
                if pos['type'] in self.special_embs.keys():
                    embeddings[i][pos['position']] = self.special_embs[pos['type']]
                if pos['type'] == 'IMG':
                    embeddings[i][pos['position'][0]:pos['position'][1]] = projected_vision_embeddings[img_idx_counter]
                    img_idx_counter += 1
        
        embeddings = embeddings[:, :self.cfg.max_context_len]
        labels = labels[:, :self.cfg.max_context_len]
        mask = mask[:, :self.cfg.max_context_len]
        
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            logits = self.model(inputs_embeds=embeddings.to(dtype = torch.float32), output_hidden_states=True).get("logits")[:, :-1]
        
        labels = labels[:, 1:]
        mask = mask[:, 1:]
          
        logits = logits[mask].contiguous().float()
        labels = labels[mask].contiguous()

        
        loss = self.loss_fct(logits.view(-1, self.n_embeddings), labels.view(-1)).mean()
            
        self.log("my_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        if batch_idx % self.cfg.ckp_iterations_step == 0 and self.global_rank == 0:
            os.makedirs(f"ckpts/{self.cfg.exp_name}/{batch_idx}", exist_ok=True)
            torch.save(self.projection, f"ckpts/{self.cfg.exp_name}/{batch_idx}/projection.pt")
            torch.save(self.special_embs, f"ckpts/{self.cfg.exp_name}/{batch_idx}/special_embeddings.pt")
        return loss
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, collate_fn=self.collate_function, num_workers = self.cfg.num_workers, shuffle = True)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/home/wavelet/configs/config-pretrain.json')
    args = parser.parse_args()
    with open(args.config, 'r') as config_file:
        config_dict = json.load(config_file)
    cfg = Config(**config_dict)
    
    ### Define models
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_ckp, use_fast=False)
    unk_id = tokenizer.encode("<unk>", add_special_tokens=False)[0]
    cfg.pad_id = unk_id
    os.makedirs(f"ckpts/{cfg.exp_name}", exist_ok=True)
    logger = CSVLogger("ckpts", name=cfg.exp_name)
    cfg.exp_name = os.path.join(cfg.exp_name, f'version_{logger.version}')
    
    
    model = AutoModelForCausalLM.from_pretrained(cfg.model_ckp, torch_dtype=torch.float32, device_map='cpu')

    ### Work with data
    train_dataset = get_dataset(cfg, tokenizer)
    collate_function = get_collate_function(cfg)

    Wk_lengths = train_dataset.get_Wk_lengths() # Определяем размеры Wk для каждого уровня вейвлет преобразования
    projection = VisualToGPTMapping(Wk_lengths, cfg.emb_dim).to(dtype=torch.float32)
    
    special_embs = initialize_special_embs(cfg)
    unfreeze(model)
    model.train()
    projection.train()

    module = Model_pl(cfg, special_embs, model, projection, train_dataset, collate_function)
    trainer = pl.Trainer(devices=[0, 1], max_epochs=cfg.n_epochs, logger = logger, accumulate_grad_batches = cfg.grad_accum, strategy='ddp_find_unused_parameters_true')
    trainer.fit(module)
    
    # run()






