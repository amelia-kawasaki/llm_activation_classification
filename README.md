# Model Activations Project

This repository provides scripts for processing and verifying model activations using specified parameters. The scripts are designed to allow flexible usage, saving each output array into files as specified by user-defined arguments.

## Setup

To use these scripts, ensure you have installed the required Python packages. You can install them using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Scripts

### 1. `llama2_activations_amelia_final.py`

This script processes model activations and saves outputs in specified directories.

#### Usage

```bash
python llama2_activations_amelia_final.py --model_link <model_link> --input_dir <input_directory> --output_dir <output_directory> --layer_size <layer_size> --hidden_layer_size <hidden_layer_size>
```

#### Arguments

- `--model_link` : Hugging Face model link.
- `--input_dir` : Path to the input directory containing data for processing.
- `--output_dir` : Path to the output directory where results will be saved.
- `--layer_size` : Size of the model layer.
- `--hidden_layer_size` : Size of the hidden layer in the model.

### 2. `activations_verification_final.py`

This script verifies model activations and saves verification outputs as per the specified arguments.

#### Usage

```bash
python activations_verification_final.py --model_link <model_link> --input_dir <input_directory> --output_dir <output_directory> --layer_size <layer_size> --hidden_layer_size <hidden_layer_size>
```

#### Arguments

- `--model_link` : Hugging Face model link for verification.
- `--input_dir` : Directory of input data for verification.
- `--output_dir` : Directory to save verification outputs.
- `--layer_size` : Size of the model layer.
- `--hidden_layer_size` : Size of the hidden layer in the model.

## Example

To run the activation processing script, you might use:

```bash
python llama2_activations_amelia_final.py --model_link 'your_model_link' --input_dir './input' --output_dir './output' --layer_size 1024 --hidden_layer_size 512
```
