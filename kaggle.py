import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
import pickle as pkl
from torch import nn
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm
import math
import os
import sys
from pytorch_pretrained_bert.optimization import BertAdam
from eval_metrics import get_final_metric, compute_bias_metrics_for_model, calculate_overall_auc

def load_data(path):
    print("Loading data on {}...".format(path))
    with open(path, "rb") as f:
        data = pkl.load(f)
    return data


class Dataset(torch.utils.data.Dataset): 
    def __init__(self, tokenizer, identity_columns, df):
        if "target" in df:
            self.labels = [int(y>=0.5) for y in df['target']]
        else:
            self.labels = None
        self.texts = [tokenizer(comment_text, 
                               padding='max_length', max_length = 317, truncation=True,
                                return_tensors="pt") for comment_text in tqdm(df['comment_text'], ncols=200)]
        self.id = df['id']
        self.df = df
        self.identity_columns = identity_columns
        

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.texts)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return self.labels[idx]

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def get_batch_id(self, idx):
        # Fetch a batch of inputs
        return self.id.iloc[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_x = self.get_batch_id(idx)
        if self.labels is not None:
            batch_y = self.get_batch_labels(idx)
            return batch_texts, batch_y, batch_x
        else:
            return batch_texts, batch_x

        
class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        # self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        output = self.linear(dropout_output)
        # final_layer = self.relu(linear_output)
        # output = nn.Softmax(dim=-1)(output)
        return output
    
def evaluate(model, val_dataloader, best_auc, save_path, criterion, val_df, identity_columns):
    model.eval()
    all_preds = []
    all_ids = []
    with torch.no_grad():
        loss_val = []
        acc_val = []
        for val_input, val_label, val_ids in tqdm(val_dataloader, desc="evaluate", ncols=100):
            val_label = val_label.cuda()
            mask = val_input['attention_mask'].cuda()
            input_id = val_input['input_ids'].squeeze(1).cuda()
            # output = model(input_id, mask).squeeze(-1)
            output = model(input_id, mask)#[:,1]
            # batch_loss = criterion(output, val_label.to(output.dtype))
            batch_loss = criterion(output, val_label)
            pred = nn.Softmax(-1)(output)[:,1]
            all_preds.extend(pred.detach().cpu().numpy().tolist())
            all_ids.extend(val_ids)
            val_acc = (output.argmax(dim=1) == val_label).sum().item() / len(val_label)
            acc_val.append(val_acc) 
            loss_val.append(batch_loss.item())
    total_val_loss = sum(loss_val) / len(loss_val)
    total_val_acc = sum(acc_val) / len(acc_val)
    MODEL_NAME = "model1"
    val_df[MODEL_NAME] = all_preds
    print(val_df["id"][:10], all_ids[:10]) 
    bias_metrics_df = compute_bias_metrics_for_model(val_df, identity_columns, MODEL_NAME, "target")
    val_auc = get_final_metric(bias_metrics_df, calculate_overall_auc(val_df, MODEL_NAME, "target"))
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), os.path.join(save_path, "best_model.ckpt"))
        print("Saving best model.")
            
    return total_val_loss, total_val_acc, val_auc, best_auc

def train(model, train_set, val_set, learning_rate, epochs, batch_size, save_path):
    num_train_optimization_steps = math.ceil(len(train_X) / batch_size) * epochs
   
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)
    # criterion = nn.MSELoss()
    weights = torch.tensor([1.0, 1.0]).cuda()
    criterion = nn.CrossEntropyLoss(weight=weights)
    param_optimizer = list(model.named_parameters())   
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
     ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=learning_rate,
                         t_total=num_train_optimization_steps)
    # optimizer = Adam(model.parameters(), lr=learning_rate)

    model = model.cuda()
    criterion = criterion.cuda()
    
    total_steps = 1
    best_acc, best_auc = 0.0, 0.0
    
    total_val_loss, total_val_acc, total_val_auc = float("inf"), 0.0, 0.0
    for epoch_num in range(epochs):
        
        loss_train = []
        acc_train = []
        # for train_input, train_label, train_ids in tqdm(train_dataloader,desc="training", ncols=100):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for train_input, train_label, train_ids in tepoch:
                tepoch.set_description(f"Epoch {epoch_num}")
                train_label = train_label.cuda()
                mask = train_input['attention_mask'].cuda()
                input_id = train_input['input_ids'].squeeze(1).cuda()
                output = model(input_id, mask)
                train_acc = (output.argmax(dim=1) == train_label).sum().item() / len(train_label)
                
                acc_train.append(train_acc)

                batch_loss = criterion(output, train_label)
                loss_train.append(batch_loss.item())
                
                acc_train_temp = sum(acc_train)/len(acc_train)
                loss_train_temp = sum(loss_train)/len(loss_train)
                
                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if total_steps % 30000 == 0:
                    total_val_loss, total_val_acc, val_auc, best_auc = evaluate(model, val_dataloader, best_acc, save_path, criterion, val_set.df, val_set.identity_columns)
                tepoch.set_postfix(train_loss=loss_train_temp, train_acc=acc_train_temp, val_loss=total_val_loss, val_acc=total_val_acc)
                total_steps += 1
        total_val_loss, total_val_acc, val_auc, best_auc = evaluate(model, val_dataloader, best_acc, save_path, criterion, val_set.df, val_set.identity_columns)
        total_train_loss = sum(loss_train) / len(loss_train)
        total_train_acc = sum(acc_train) / len(acc_train)
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_train_loss: .5f} | Val Loss: {total_val_loss : .5f} | Train acc: {total_train_acc : .5f}| Val acc: {total_val_acc : .5f} | Val auc: {val_auc : .5f} | Best auc: {best_auc : .5f}\n\n') 

        
def test(test_set, model_path, batch_size):
    model = BertClassifier()
    model_state_dict = torch.load(os.path.join(model_path, "best_model.ckpt"))
    model.load_state_dict(model_state_dict)
    model.cuda()
    model.eval()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
    all_outputs = []
    all_ids = []
    cnt = 100
    with torch.no_grad():  
        for test_input, test_ids in tqdm(test_dataloader, desc="testing", ncols=100):
            mask = test_input['attention_mask'].cuda()
            input_id = test_input['input_ids'].squeeze(1).cuda()
            # output = model(input_id, mask).squeeze(-1)
            output = model(input_id, mask)#[:,1]
            output = nn.Softmax(-1)(output)
            all_outputs.extend(output.detach().cpu().numpy().tolist())
            all_ids.extend(test_ids.detach().cpu().numpy().tolist())
            cnt -= 1
#             if cnt == 0:
#                 break
    return all_outputs, all_ids           


def write_csv(predictions, test_ids, prediction_path):
    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path, exist_ok=True)
    with open(os.path.join(prediction_path, "submission.csv"), "w") as f:
              f.write("id,prediction\n")
              for tid, pred in zip(test_ids, predictions):
                f.write("{},{}\n".format(int(tid),float(pred[1])))

def main():              
    do_train, do_test, epochs, batch_size, lr, model_path, prediction_path = sys.argv[1:]
    do_train, do_test, batch_size, epochs, lr = do_train == "True", do_test == "True", int(batch_size), int(epochs), float(lr)
    print("do_train:{}, do_test:{}, epochs:{}, batch_size:{}, lr:{}, model_path:{}, prediction_path:{}".format(do_train, do_test, epochs, batch_size, lr, model_path, prediction_path))
    do_eval = True
    train_data_path = "train_set_v4.0.pkl"
    val_data_path = "val_set_v4.0.pkl"
    test_data_path = "test_set_v4.0.pkl"
    if do_train:
        print("Training...")
        model = BertClassifier()
        train_set = load_data(train_data_path)
        val_set = load_data(val_data_path)
        train(model,train_set, val_set, lr, epochs, batch_size, model_path)
        
    if do_eval:
        val_set = load_data(val_data_path)
        val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)
        best_auc = float("inf")
        model = BertClassifier()
        model_state_dict = torch.load(os.path.join(model_path, "best_model.ckpt"))
        model.load_state_dict(model_state_dict)
        model.cuda()
        criterion = nn.CrossEntropyLoss()
        total_val_loss, total_val_acc, val_auc, _ = evaluate(model, val_dataloader, best_auc, model_path, criterion, val_set.df, val_set.identity_columns)
        print(
            f'Val Loss: {total_val_loss : .5f} | Val acc: {total_val_acc : .5f} | Val auc: {val_auc : .5f}') 

    if do_test:
        print("Testing...")
        test_set = load_data(test_data_path)
        predictions, test_ids = test(test_set, model_path, batch_size)
        write_csv(predictions, test_ids, prediction_path)

if __name__ == "__main__":
    main()

