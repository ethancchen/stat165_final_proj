import os
import json
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class StoryRLHF(Dataset):
    def __init__(self, data_path: str, actions_path: str):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.instruction_template = "Below is an instruction that describes a task. " + \
                          "Write a response that appropriately completes the request.\n\n" + \
                          "### Instruction:\n{instruction}\n\n### Response:"
        self.prompt_template = """Here is a story prompt: {prompt}\n\nHere is the first paragraph of the story:{init_paragpraph}.\n\n Here is a set of actions: {actions}.\n\nBased on the initial paragraph, choose the best action for the next paragraph. Only output the action you chose without any quotation marks."""
        self.actions_path = actions_path
        with open(actions_path, 'r') as file:
            self.actions = json.load(file)

    def __len__(self):
        return len(self.df)
    
    def format_prompt(self, story_idea, init_paragraph):
        return self.instruction_template.format(
                    instruction=self.prompt_template.format(
                        prompt=story_idea, 
                        init_paragpraph=init_paragraph,                     
                        actions=self.actions))
    
    def get_dataset(self):
        chosen = self.df['chosen'].tolist()
        rejected = self.df['rejected'].tolist()
        prompts = []
        for idx in range(len(self.df)):
            story_idea = self.df['prompt'][idx]
            init_paragraph = self.df['initial_paragraph'][idx]
            prompt = self.format_prompt(story_idea, init_paragraph)
            prompts.append(prompt)
    
        dataset = []
        for i in range(len(prompts)):
            data = {
                "prompt": prompts[i],
                "chosen": chosen[i],
                "rejected": rejected[i]
            }
            dataset.append(data)
        return dataset

