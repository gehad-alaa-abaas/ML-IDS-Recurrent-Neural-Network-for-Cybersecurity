import os 
import pandas as pd  
import numpy as np  
import torch
from time import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split 
import torch.nn as nn 
from torchvision import transforms, utils
from tqdm import tqdm
torch.manual_seed(1)
import logging
from sklearn.model_selection import KFold
# Set up logging


# EPOCH = 1000 
# EPOCH = 2
BATCH_SIZE = 64
TIME_STEP = None
INPUT_SIZE = 80
LR = 0.01


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_working_directory = os.getcwd()
dataset_filename = 'combinedDataSet.csv'




class My_Custom_Data_Set(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        feature_row = self.features.iloc[index].values.astype(np.float32)
        label = self.labels.iloc[index]
        return feature_row, label

class IdsRnn(nn.Module):
    def __init__(self,hidden_size, output_size):
        super(IdsRnn, self).__init__()
        self.rnn = nn.LSTM(
            input_size= INPUT_SIZE,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out

def preProcessDataSet(df, batch_size, label_encoding_method='lambda'):
    df.columns = df.columns.str.strip()
    df = df.drop(columns=['Flow ID', 'Source IP','Destination IP','Timestamp'])
    pd.set_option('mode.use_inf_as_na', True)
    df['Flow Bytes/s'] = df['Flow Bytes/s'].astype('float64')
    df['Flow Packets/s'] = df['Flow Packets/s'].astype('float64')
    df['Flow Bytes/s'].fillna(df['Flow Bytes/s'].mean(), inplace=True)
    df['Flow Packets/s'].fillna(df['Flow Packets/s'].mean(), inplace=True)

    if label_encoding_method == 'lambda':
        df['Label'] = df['Label'].apply(lambda x: 0 if 'BENIGN' in x else 1)
        output_size = 2
         
    elif label_encoding_method == 'mapping':
        attack_mapping = {
            'BENIGN': 0,
            'DoS Hulk': 1,
            'PortScan': 2,
            'DDoS': 3,
            'DoS GoldenEye': 4,
            'FTP-Patator': 5,
            'SSH-Patator': 6,
            'DoS slowloris': 7,
            'DoS Slowhttptest': 8,
            'Bot': 9,
            'Web Attack � Brute Force': 10,
            'Web Attack � XSS': 11,
            'Infiltration': 12,
            'Web Attack � Sql Injection': 13,
            'Heartbleed': 14,
        }
        df['Label'] = df['Label'].map(attack_mapping)
        output_size = 15
         
    else:
        raise ValueError("Invalid label_encoding_method. Choose 'lambda' or 'mapping'.")

    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, 0:-1], df['Label'], test_size=0.3, random_state=42)
    train_data = My_Custom_Data_Set(x_train, y_train)
    test_data = My_Custom_Data_Set(x_test, y_test)
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return trainloader, testloader,output_size




dataSetPath = os.path.join(current_working_directory, dataset_filename)
# dataSetPath =  r'C:\Users\gehad\Documents\HBO ICT\Semester 5\ML-dataset\CICIDS2017GeneratedLabelledFlows\TrafficLabelling\combinedDataSet.csv'
print("Reading DATASET -------------------------------------------")
df = pd.read_csv(dataSetPath,low_memory=False) 
print(" ------------------------------------------- DATASET read SUcc")
# df.columns = df.columns.str.strip()
hidden_size = 512
print(" ------------------------------------------- Start Training Module---------------------------------")

def training_modul(df, methoud, outputname, use_cross_validation=True):
    log_file = os.path.join(os.getcwd(), 'logs', f'training_log_{outputname}.txt')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO)

    print(f'Starting Traininig Module {outputname}')
    logging.info(f'Starting Traininig Module {outputname}')
    if use_cross_validation:

        num_folds = 5
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

        for fold, (train_index, test_index) in enumerate(kf.split(df)):
            train_data, test_data = df.iloc[train_index], df.iloc[test_index]
            EPOCH = 100
            early_stopping_threshold = 5

            trainloader, testloader,output_size = preProcessDataSet(train_data, BATCH_SIZE, methoud)
            rnn = IdsRnn(hidden_size, output_size).to(device)
            optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
            loss_func = nn.CrossEntropyLoss()
            early_stopping_counter = 0
            best_test_accuracy = 0

            for epoch in range(EPOCH):

                total_loss = 0
                for step, (b_x, b_y) in enumerate(tqdm(trainloader, desc=f'| Fold: {fold} Epoch {epoch}', unit='batch')):
                    b_x = b_x.view(-1, 1, INPUT_SIZE).float().to(device)
                    b_y = b_y.to(device)
                    output = rnn(b_x)
                    loss = loss_func(output, b_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                average_loss = total_loss / (step + 1)
                sum = 0
                accuracy = 0
                for step, (a_x, a_y) in enumerate(tqdm(testloader, desc='Testing', unit='batch')):
                    a_x = a_x.view(-1, 1, INPUT_SIZE).float().to(device)
                    a_y = a_y.to(device)
                    test_output = rnn(a_x)
                    pred_y = torch.max(test_output, 1)[1]
                    sum += (pred_y == a_y).sum().item()

                accuracy = sum / len(testloader.dataset)

                print(f'Epoch: {epoch} | Fold: {fold} | Average train loss: {average_loss:.4f} | Test accuracy: {accuracy:.2%}')
                logging.info(f'Epoch: {epoch} | Fold: {fold} | Average train loss: {average_loss:.4f} | Test accuracy: {accuracy:.2%}')

                if accuracy > best_test_accuracy:
                    best_test_accuracy = accuracy
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= early_stopping_threshold:
                    print(f'Early stopping after{epoch} | Fold: {fold}  epochs without improvement.')
                    logging.info(f'Early stopping after {epoch} | Fold: {fold}  epochs without improvement.')

                    break

            model_name = f"{outputname}_fold_{fold}.pth"
            saveFile = os.path.join(current_working_directory, model_name)
            torch.save(rnn.state_dict(), saveFile)


        model_name = f"{outputname}_all.pth"
        saveFile = os.path.join(current_working_directory, model_name)
        torch.save(rnn.state_dict(), saveFile)
    else:
        EPOCH = 500
        early_stopping_threshold = 25
        trainloader, testloader, output_size = preProcessDataSet(df, BATCH_SIZE, methoud)

        rnn = IdsRnn(  hidden_size, output_size).to(device)


        optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
        loss_func = nn.CrossEntropyLoss()


        early_stopping_counter = 0
        best_test_accuracy = 0

        for epoch in range(EPOCH):
            # Training
            total_loss = 0
            for step, (b_x, b_y) in enumerate(tqdm(trainloader, desc=f'Epoch {epoch}', unit='batch')):
                b_x = b_x.view(-1, 1,  INPUT_SIZE).float().to(device)
                b_y = b_y.to(device)
                output = rnn(b_x)
                loss = loss_func(output, b_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            average_loss = total_loss / (step + 1)
            sum = 0
            accuracy = 0
            for step, (a_x, a_y) in enumerate(tqdm(testloader, desc='Testing', unit='batch')):
                a_x = a_x.view(-1, 1,  INPUT_SIZE).float().to(device)
                a_y = a_y.to(device)
                test_output = rnn(a_x)
                pred_y = torch.max(test_output, 1)[1]
                sum += (pred_y == a_y).sum().item()

            accuracy = sum / len(testloader.dataset)

            print(f'Epoch: {epoch} | Average train loss: {average_loss:.4f} | Test accuracy: {accuracy:.2%}')
            logging.info(f'Epoch: {epoch} | Average train loss: {average_loss:.4f} | Test accuracy: {accuracy:.2%}')


            if accuracy > best_test_accuracy:
                best_test_accuracy = accuracy
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= early_stopping_threshold:
                print(f'Early stopping after {epoch} | epochs without improvement.')
                logging.info(f'Early stopping after {epoch}  epochs without improvement.')

                break


        model_name = f"{outputname}.pth"
        saveFile = os.path.join(current_working_directory, model_name)
        torch.save(rnn.state_dict(), saveFile)

    return rnn


training_lambdawith_val = training_modul(df, 'lambda', "lambda_with_valid", use_cross_validation=True)
training_lambda_no_vali = training_modul(df, 'lambda', "lambda_no_valid", use_cross_validation=False)
training_mapping_with_val = training_modul(df, 'mapping', "mapping_with_valid", use_cross_validation=True)
training_mapping_no_vali = training_modul(df, 'mapping', "mapping_no_valid", use_cross_validation=False)