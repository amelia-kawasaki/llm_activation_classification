
import argparse
import os
import numpy as np
import pandas as pd
import ast
import torch
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb

def parse_args():
    parser = argparse.ArgumentParser(description="Train lightgbm to classify activations and save outputs.")
    parser.add_argument("--benign_input_dir", type=str, required=True, help="Input directory path for benign activations")
    parser.add_argument("--attack_input_dir", type=str, required=True, help="Input directory path for attack activations")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory path")
    parser.add_argument("--layer_size", type=int, required=True, help="Size of the model layer")
    parser.add_argument("--hidden_layer_size", type=int, required=True, help="Size of the hidden layer in the model")
    return parser.parse_args()

def count_files_in_directory(directory_path):
    try:
        # List all items in the directory and count only files
        return len([f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))])
    except FileNotFoundError:
        print("The directory does not exist.")
        return None
    except PermissionError:
        print("You do not have permission to access this directory.")
        return None


def get_vectors_from_parquet(s3_uri, layers, hidden):
    # get the dataframe
    df = pd.read_parquet(s3_uri)

    # grab all vectors from df
    vecs = np.array([np.frombuffer(v, dtype=np.float32).reshape(layers, hidden) for v in df['vector']])
    
    prompts = df['statement'].values

    return vecs, prompts

def load_activations(benign_input_dir, attack_input_dir, layer_size, hidden_layer_size):
    
    datasets = [(attack_input_dir, 1), (benign_input_dir, 0)]
    benign_batch_count = count_files_in_directory(benign_input_dir)
    attack_batch_count = count_files_in_directory(attack_input_dir)
    
    # balance class sizes
    number_of_batches = min(benign_batch_count, attack_batch_count)
    
    l = np.arange(number_of_batches)
    np.random.shuffle(l)
    train_idx = l[:round(len(l)/10 * 8)]
    val_idx = l[round(len(l)/10 * 8):]

    train_vecs = np.array([])
    train_labels = np.array([])

    val_vecs = np.array([])
    val_labels = np.array([])
    
    print("Starting to load activations...")

    for path, label in datasets:
        print(f"Loading activations for {path}")
        print("Loading activations for training")
        for j in range(len(train_idx)):
            i = train_idx[j]
            vecs_, _ = get_vectors_from_parquet(f'{path}/activations_batch_{i:04d}.parquet', layer_size, hidden_layer_size)
            labels = np.full(vecs_.shape[0], label)
            if train_vecs.shape[0] == 0:
                train_vecs = vecs_
                train_labels = labels
            else:
                train_vecs = np.concatenate([train_vecs, vecs_])
                train_labels = np.concatenate([train_labels, labels])
        print("Loading activations for validation")
        for j in range(len(val_idx)):
            i = val_idx[j]
            vecs_, _ = get_vectors_from_parquet(f'{path}/activations_batch_{i:04d}.parquet', layer_size, hidden_layer_size)
            labels = np.full(vecs_.shape[0], label)
            if val_vecs.shape[0] == 0:
                val_vecs = vecs_
                val_labels = labels
            else:
                val_vecs = np.concatenate([val_vecs, vecs_])
                val_labels = np.concatenate([val_labels, labels])
    print("Activations loaded.")
    # train_vecs.shape, train_labels.shape, val_vecs.shape, val_labels.shape
    return train_vecs, train_labels, val_vecs, val_labels

def make_lgb(data):
    train_vecs, train_labels, val_vecs, val_labels = data
    
    dtrain = lgb.Dataset(train_vecs, label=train_labels)
    dtest = lgb.Dataset(val_vecs, label=val_labels)

    param = {'num_leaves': 64, 'objective': 'binary', 'max_depth':6, 'learning_rate':0.3}
    param['metric'] = ['binary_logloss', 'auc']
    # Fit the model, test sets are used for early stopping.

    num_round = 100
    bst = lgb.train(param, dtrain, num_round, valid_sets=[dtest], callbacks=[lgb.early_stopping(stopping_rounds=10)])
    # bst.save_model('lgb_model_{layer}.txt', num_iteration=bst.best_iteration)
    ypred = bst.predict(val_vecs, num_iteration=bst.best_iteration, type='class')
    ypred_class = (ypred > 0.5).astype("int")
    # score the model
    print('testing score')
    cm = confusion_matrix(val_labels, ypred_class)
   
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    
    classification_dict = classification_report(val_labels, ypred_class, target_names=['benign', 'harmful'], output_dict=True)
    return classification_dict['benign']['precision'], classification_dict['harmful']['precision'], classification_dict['benign']['recall'], classification_dict['harmful']['recall'], classification_dict['accuracy'], (fpr, fnr)


def train_all_layers(data, layers):
    accuracy_l = []
    b_precision_l = []
    b_recall_l = []
    h_precision_l = []
    h_recall_l = []
    fpr_l = []
    fnr_l = []
    for l in range(0, layers):
        print(f'layer {l}')
        layer_train_data = data[0][:, l, :] # train_vecs
        layer_val_data = data[2][:, l, :] # val_vecs
        layer_data = (layer_train_data, data[1], layer_val_data, data[3])
        b_precision, h_precision, b_recall, h_recall, accuracy, rates = make_lgb(layer_data)
        b_precision_l.append(b_precision)
        h_precision_l.append(h_precision)
        b_recall_l.append(b_recall)
        h_recall_l.append(h_recall)
        accuracy_l.append(accuracy)
        fpr, fnr = rates
        fpr_l.append(fpr)
        fnr_l.append(fnr)
    return accuracy_l, b_precision_l, h_precision_l, b_recall_l, h_recall_l, fpr_l, fnr_l

def main():
    args = parse_args()
    benign_input_dir = args.benign_input_dir
    attack_input_dir = args.attack_input_dir
    output_dir = args.output_dir
    layer_size = args.layer_size
    hidden_layer_size = args.hidden_layer_size
    
    # make output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    train_vecs, train_labels, val_vecs, val_labels = load_activations(benign_input_dir, attack_input_dir, layer_size, hidden_layer_size)
    data = [train_vecs, train_labels, val_vecs, val_labels]
    accuracy_l, b_precision_l, h_precision_l, b_recall_l, h_recall_l, fpr_l, fnr_l = train_all_layers(data, layer_size)
    
    # Save the output arrays to the output directory
    pd.Series(accuracy_l).to_csv(f'{output_dir}/accuracy.csv', index=False)
    pd.Series(b_precision_l).to_csv(f'{output_dir}/benign_precision.csv', index=False)
    pd.Series(h_precision_l).to_csv(f'{output_dir}/attack_precision.csv', index=False)
    pd.Series(b_recall_l).to_csv(f'{output_dir}/benign_recall.csv', index=False)
    pd.Series(h_recall_l).to_csv(f'{output_dir}/attack_recall.csv', index=False)
    pd.Series(fpr_l).to_csv(f'{output_dir}/fpr.csv', index=False)
    pd.Series(fnr_l).to_csv(f'{output_dir}/fnr.csv', index=False)

if __name__ == "__main__":
    main()
    


