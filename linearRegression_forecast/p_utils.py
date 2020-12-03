import boto3, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


#########################
# 전처리
#########################

def upload_s3(bucket, file_path, prefix):
    '''
    bucket = sagemaker.Session().default_bucket()
    prefix = 'comprehend'
    train_file_name = 'test/train/train.csv'
    s3_train_path = upload_s3(bucket, train_file_name, prefix)
    '''
    
    prefix_path = os.path.join(prefix, file_path)
    # prefix_test_path = os.path.join(prefix, 'infer/test.csv')

    boto3.Session().resource('s3').Bucket(bucket).Object(prefix_path).upload_file(file_path)
    s3_path = "s3://{}/{}".format(bucket, prefix_path)
    print("s3_path: ", s3_path)

    return s3_path

#########################
# 평가
#########################

def evaluate(test_y, pred, metric = 'MdAPE'):
    '''
    # Naive MdAPE = 0.03687277085962615
    # One-step-ahead MdAPE =  0.020360223119258346    
    '''
    if metric == 'MdAPE':
        MdAPE = np.median(np.abs(test_y - pred) / test_y)    
    return MdAPE

def input_fn(input_data, request_content_type='text/csv'):
    """
    """
    n_feature = input_data.shape[1]
    sample = input_data.reshape(-1,n_feature)
    return sample

def show_chart(test_y, pred):
    plt.plot(np.array(test_y), label='actual')
    plt.plot(pred, label='naive')
    plt.legend()
    plt.show()





