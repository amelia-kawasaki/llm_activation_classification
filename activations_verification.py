
import argparse
import os
import numpy as np
import transformers
from tqdm import tqdm
import torch
import pandas as pd
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Script for extracting model activations.")
    parser.add_argument("--model_link", type=str, required=True, help="Hugging Face model link")
    parser.add_argument("--attack_input_path", type=str, required=True, help="Path to attack input data")
    parser.add_argument("--benign_input_path", type=str, required=True, help="Path to benign input data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save activation data")
    parser.add_argument("--layer_size", type=int, required=True, help="Size of the model layer")
    parser.add_argument("--hidden_layer_size", type=int, required=True, help="Size of the hidden layer in the model")
    return parser.parse_args()

def activation_hook(module, input, output):
    activations[module.lidx,:] = output[0].detach().mean(1).flatten()


def setup_model(model):
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model_layers = pipeline.model.model.layers
    
    for lidx, layer in enumerate(model_layers):
        layer.lidx = lidx
        hook = layer.register_forward_hook(activation_hook)
        if lidx == (len(model_layers) - 1):
            break

    return pipeline, tokenizer

# create a new batch df
def init_batch(df, vector=True):
    batch_df = pd.DataFrame()

    # fill batch_df with columns from df
    for col in df.columns:
        batch_df[col] = None
    if vector:
        batch_df["vector"] = None

    return batch_df

def extract_activations():
    pipeline, tokenizer = setup_model(model_link)


    for dataset_name, output_dir_class in [(attack_input_path, output_dir_attack), (benign_input_path, output_dir_benign)]:
        print(f"Extracting activations for {dataset_name}")
        df = pd.read_csv(dataset_name, header=0)
        
        # shuffle the dataframe before batching
        df = df.sample(frac=1).reset_index(drop=True)
        
        batch_size = 256
        
        model = pipeline.model.model
        tokenizer.pad_token = tokenizer.eos_token

        batch_df = init_batch(df)
        batch_idx = 0

        for index, row in tqdm(df.iterrows(), total=len(df)):

            prompt = row['statement']
            # add row to batch_df
            batch_df.loc[index] = row
            
            tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048, padding="max_length")
        
            # copy tokens to gpu
            tokens = {k: v.cuda() for k, v in tokens.items()}

            # # send tokens through model
            with torch.no_grad():
                model(**tokens)

            # save activations to batch_df
            activations_bytes = activations.cpu().numpy().copy().tobytes()
            batch_df.loc[index, "vector"] = activations_bytes

            if len(batch_df) < batch_size:
                continue

            # save batch_df
            print(f"Saving batch {batch_idx}")
            batch_df.to_parquet(f"{output_dir_class}/activations_batch_{batch_idx:04d}.parquet")
            
            # reset batch_df
            batch_df = init_batch(df)
            

            batch_idx += 1
            
if __name__ == "__main__":
    args = parse_args()
    model_link = args.model_link
    attack_input_path = args.attack_input_path
    benign_input_path = args.benign_input_path
    output_dir = args.output_dir
    layer_size = args.layer_size
    hidden_layer_size = args.hidden_layer_size
    
    # make output directory if it doesn't exist
    output_dir_benign = os.path.join(output_dir, "benign")
    output_dir_attack = os.path.join(output_dir, "attack")
    os.makedirs(output_dir_benign, exist_ok=True)
    os.makedirs(output_dir_attack, exist_ok=True)
    
    # global activations var
    activations = torch.zeros(layer_size,hidden_layer_size).pin_memory().cuda()

    extract_activations()