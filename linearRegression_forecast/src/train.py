
from __future__ import print_function

import argparse
import joblib
import os
import pandas as pd
import logging
import numpy as np

from sklearn.linear_model import Ridge, RidgeCV, LassoCV, Lasso
from sklearn.model_selection import cross_val_score

def train(args, train_X, train_y, model):
    '''
    아래 모델의 타입에 따라 해당 모델을 학습하고 모델을 리턴한다.
    ridge_model = train(train_X, train_y, model="ridge")
    '''
    print("alpha value in train: ", args.alpha)
    if model =='ridge':
        model = Ridge(alpha= args.alpha)
        print("Train Ridge model")
    elif model == 'lasso':
        model = Lasso(alpha=1.0)
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
    아래 args를 파싱한 결과는 아래와 같습니다.
    args:  Namespace(alpha=1.5,
    model_dir='/opt/ml/model',train='/opt/ml/input/data/train')
    '''
    parser = argparse.ArgumentParser()

    # Hyperparameters 의 데이터를 파싱하여 args 오브젝트에 저장 합니다.
    parser.add_argument('--alpha', type=float, default=3)

    # 모델은 훈련후에 model_dir='/opt/ml/model' 로 저장이 됩니다. 이후에 자동으로 S3에 업로드가 됩니다.
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    
    # sklearn_estimator.fit({'train': s3_train_path}, logs=False) 에서 기술한
    # 입력 데이터는 S3로 부터 train='/opt/ml/input/data/train' 경로로 다운로드 됩니다.
    # 이후에 이 로컬 경로에서 데이터를 가져와서 훈련을 합니다.
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()
    print("args: ", args)
    
    return args

if __name__ == '__main__':
    args = parse_args()  # 파라미터 얻음  
    train_y, train_X = handle_input_data(args) # 훈련 데이터를 traiy_y, train_X 로 분리 합니다.
    
    ridge_model = train(args, train_X, train_y, model="ridge") # 데이터를 가지고 훈련 합니다.
    model_name = 'model.joblib' #저장할 모델 이름 입니다.
    save_model(ridge_model, args.model_dir, model_name) #args.model_dir 에 명시된 경로에 모델 저장

    