from transformers import GPT2Tokenizer,GPT2Model
from datasets import load_dataset

from sklearn.decomposition import PCA
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token
gpt2 = GPT2Model.from_pretrained("openai-community/gpt2-medium").to(device)
wte = gpt2.wte

dataset = load_dataset("rotten_tomatoes", split="train")

# dataset = dataset.map(lambda e: tokenizer(e['text'], return_tensors="pt", truncation=True, padding='max_length'), batched=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4096)

for i, data in enumerate(dataloader):

    with torch.no_grad():
        input_ids = tokenizer(data['text'], return_tensors="pt", truncation=True, padding='max_length')["input_ids"].to(device)
        
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        inputs_embeds = wte(input_ids)
        print(inputs_embeds.shape) # B, L=1024, D=768/1024

    ## reshape
    inputs_embeds = torch.flatten(inputs_embeds, start_dim=0, end_dim=1) # 768 | 1024
    print(inputs_embeds.shape)

    ## PCA
    pca = PCA()
    inputs_embeds = inputs_embeds.detach().cpu().numpy()
    pca.fit(inputs_embeds)
    print(pca.components_.shape)
    with open('pca_med.npy', 'wb') as f:
        np.save(f, pca.components_)

    raise ValueError