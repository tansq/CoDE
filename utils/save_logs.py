import datetime
import os

def save_logs(save_path, message):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'val_logs.txt'), mode='a+') as f:
        f.write('{:^17}: {}\n'.format('CREATE_TIME', datetime.datetime.now()))
        f.write('----------------------------------------------------------\n')
        f.write(message+'\n')
        f.write('----------------------------------------------------------\n\n')
        f.close()

def save_train_logs(save_path, message):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'train_logs.txt'), mode='a+') as f:
        f.write('{:^17}: {}\n'.format('CREATE_TIME', datetime.datetime.now()))
        f.write('----------------------------------------------------------\n')
        f.write(message+'\n')
        f.write('----------------------------------------------------------\n\n')
        f.close()
