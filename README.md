# Model Activations Project

This is the repository for the code that accompanies [Defending Large Language Models Against Attacks With Residual Stream Activation Analysis](https://arxiv.org/abs/2406.03230) This repository provides scripts for extracting and classifying model activations for custom datasets of prompts.

## Setup

To use these scripts, ensure you have installed the required Python packages. You can install them using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Scripts

### 1. `activation_extraction.py`

This script extracts model activations for each prompt provided and saves outputs in specified directories.

#### Usage

```bash
python activation_extraction.py --model_link <model_link> --benign_input_path <input_directory> --attack_input_path <input_directory> --output_dir <output_directory> --layer_size <layer_size> --hidden_layer_size <hidden_layer_size>
```

#### Arguments

- `--model_link` : Hugging Face model link
- `--benign_input_path` : path to csv file of benign prompts
- `--attack_input_path` : path to csv file of attack prompts
- `--output_dir` : Directory to save activations.
- `--layer_size` : Number of layers in the model.
- `--hidden_layer_size` : Size of the hidden layer in the model.

#### Example

```bash
python activation_extraction.py --model_link meta-llama/Llama-2-7b-chat-hf --benign_input_path sample_data/benign_open_test_1000.csv --attack_input_path sample_data/jb_open_test_1000.csv --output_dir output --layer_size 32 --hidden_layer_size 4096
```

### 2. `activation_classification.py`

This script verifies model activations and saves verification outputs as per the specified arguments.

#### Usage

```bash
python activation_classification.py --benign_input_dir <input_directory> --attack_input_dir <input_directory> --output_dir <output_directory> --layer_size <layer_size> --hidden_layer_size <hidden_layer_size>
```

#### Arguments
- `--benign_input_dir` : Path to the input directory containing activations for the benign prompts.
- `--attack_input_dir` : Path to the input directory containing data for processing.
- `--output_dir` : Path to the output directory where results will be saved.
- `--layer_size` : Size of the model layer.
- `--hidden_layer_size` : Size of the hidden layer in the model.

## Example

To run the activation processing script, you might use:

```bash
python activation_classification.py --benign_input_dir "output/benign" --attack_input_dir "output/attack" --output_dir classification_results --layer_size 32 --hidden_layer_size 4096```
