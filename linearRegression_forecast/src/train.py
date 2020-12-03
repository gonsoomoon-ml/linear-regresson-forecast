
from __future__ import print_function

import argparse
import joblib
import os
import pandas as pd
import logging
import numpy as np

from sklearn.linear_model import Ridge, RidgeCV, LassoCV, Lasso
from sklearn.model_selection import cross_val_score

def train(train_X, train_y, model):
    '''
    아래 모델의 타입에 따라 해당 모델을 학습하고 모델을 리턴한다.
    ridge_model = train(train_X, train_y, model="ridge")
    '''
    if model =='ridge':
        model = Ridge(alpha=3.0)
        print("Train Ridge model")
    elif model == 'lasso':
        model = Lasso(alpha=3.0)
        print("Train Lasso model")        
        
    model.fit(train_X, train_y)        
    
    return model



def save_model(model, model_folder, model_name):
    '''
    모델을 해당 위치에 저장한다.
    '''
    save_path = os.path.join(model_folder, model_name)
    joblib.dump(model, save_path)
    print(f'{save_path} is saved')
    
def handle_input_data(args):
    '''
    훈련 데이터를 X, y 로 변환하여 리턴
    '''
    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    raw_data = [ pd.read_csv(file, header=None, engine="python") for file in input_files ]
    train_data = pd.concat(raw_data)

    # labels are in the first column
    train_data = train_data.astype('float64')
    # print(train_data.info())
    train_y = train_data.iloc[:, 0]
    train_X = train_data.iloc[:, 1:].to_numpy()
    
    print("train_y: ", train_y.shape)
    print("train_X: ", train_X.shape)    
    
    return train_y, train_X

    
    
    
def parse_args():   
    '''
    커맨드 인자로 넘어온 정보를 파싱한다.
    '''
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    # parser.add_argument('--max_leaf_nodes', type=int, default=-1)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()
    print("args: ", args)
    
    return args

    
    

if __name__ == '__main__':
    args = parse_args()    
    train_y, train_X = handle_input_data(args)
    
    ridge_model = train(train_X, train_y, model="ridge")
    model_name = 'model.joblib'
    save_model(ridge_model, args.model_dir, model_name)

    