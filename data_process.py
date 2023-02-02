import pandas as pd
from kaggle import Dataset
import pickle
import numpy as np
from transformers import BertTokenizer

identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

def save_data(data, path):
    print("save data to {}...".format(path))
    with open(path, "wb") as f:
        pickle.dump(data, f)
        
def create_data(path, is_train=False):
    print("create data for {}...".format(path))
    data = pd.read_pickle(path)
    if is_train:
        train_data, val_data = np.split(data.sample(frac=1, random_state=42), 
                                     [int(.8*len(data))])
        print(train_data.keys())
        train_X = train_data[['id','comment_text', "target"]+identity_columns]
        val_X = val_data[['id','comment_text', "target"]+identity_columns]
#         train_Y = train_data[['target']]
#         val_Y = val_data[['target']]
        return train_X, val_X#, val_X, val_Y
    else:
        print(data.keys())
        test_X = data[['id','comment_text']+identity_columns]
        return test_X
        
# train_X, train_Y, val_X, val_Y = create_data("train_df.pkl", is_train=True)
train_df, val_df = create_data("train_df.pkl", is_train=True)
# test_df = create_data("test_df.pkl") 1 test public label "rating":T/F->1/0 2 "subgroup":fillna
train_df_sub = train_df[:1000]
val_df_sub = val_df[:10000]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# train_set = Dataset(tokenizer, train_X, train_Y)
# val_set = Dataset(tokenizer, val_X, val_Y)
# test_set = Dataset(tokenizer, test_X)
# train_set = Dataset(tokenizer, train_df)
# val_set = Dataset(tokenizer, val_df)
# test_set = Dataset(tokenizer, test_df)

train_set = Dataset(tokenizer, identity_columns, train_df)
val_set = Dataset(tokenizer, identity_columns, val_df)

save_data(train_set, "train_set_v4.0.pkl")
save_data(val_set, "val_set_v4.0.pkl")
# save_data(test_set, "test_set_v4.0.pkl")