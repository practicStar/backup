import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.io as sio
from collections import defaultdict

label = [[1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
        [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
        [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]]


def list_files(directory, sorted_dir=True):
    files = defaultdict(list)
    directory = os.path.join(directory, "eeg_raw_data")
    list_dir = os.listdir(directory)
    for session_id in list_dir:
        session = os.path.join(directory, session_id)
        if sorted_dir:
            session_dir = sorted(os.listdir(session), key=lambda filename: int(filename.split('_')[0]))
        else:
            session_dir = os.listdir(session)
        for trial in session_dir:
            single = os.path.join(session, trial)
            files[trial.split("_")[0]].append(single)
    return files

def preprocess(sessions_dir, data_path, save_path):
    for dir_id in tqdm(sessions_dir):
        # print(dir_id)
        dir = sessions_dir[dir_id]

        for trial in dir:
            # 解析 eeg.mat 文件
            eeg_data = sio.loadmat(trial)
            # eeg_data = eeg_data['data']

            # 解析 eye.mat 文件
            eye_path = os.path.join(data_path, "eye_raw_data")

            # 计算注视点 (这里用分散度代替?)
            PD_filename = os.path.join(eye_path, trial.split("\\")[-1].split(".")[0] + "_PD.mat")
            PD_data = sio.loadmat(PD_filename)
            gaze_coord = PD_data['PD']




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../dataset/SEED_IV',
                        help='Path to Sessions folder')
    parser.add_argument('--save_path', type=str, default='../dataset/seed_preproc_data',
                        help='Path to save preprocessed data')
    args = parser.parse_args()

    sessions_dir = list_files(args.data_path)
    preprocess(sessions_dir, args.data_path, args.save_path)

