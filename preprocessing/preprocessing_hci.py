import argparse
import os
import xml.etree.ElementTree as ET
import mne
import pandas as pd
import numpy as np
from tqdm import tqdm 

def list_files(directory, sorted_dir):
    files = []
    if sorted_dir:
        list_dir = sorted(os.listdir(directory), key=lambda filename: int(filename.split('.')[0]))
    else:
        list_dir = os.listdir(directory)
    for filename in list_dir:
        if int(filename) <= 1470 and int(filename) >= 1432:
            continue
        if int(filename) <= 1854 and int(filename) >= 1822:
            continue
        if int(filename) == 1984:
            continue
        # if filename.endswith(".csv"):
        single = os.path.join(directory, filename)
        files.append(single)
    return files

def preprocess(sessions_dir, save_path, verbose=False):
    for dir_id in tqdm(range(len(sessions_dir)), desc='Preprocessing'):
        # print(dir_id)
        dir = sessions_dir[dir_id]

        # 解析 session.xml 文件 --------------------------------------------------------------
        labels_file = os.path.join(dir, "session.xml")
        tree = ET.parse(labels_file)
        root = tree.getroot()

        # feltEmo--情绪   feltArsl--唤起度   feltVlnc--效价
        feltEmo = root.attrib['feltEmo']
        feltArsl = root.attrib['feltArsl']
        feltVlnc = root.attrib['feltVlnc']

        # 会话编号  受试者ID
        session = root.attrib['cutNr']
        subject = root[0].attrib['id']
        
        # 生理数据 -------------------------------------------------------
        # 根据受试者ID和会话编号构建生理数据文件路径
        physio_file = os.path.join(dir, "Part_{}_S_Trial{}_emotion.bdf".format(subject, int(session)//2))
        raw = mne.io.read_raw_bdf(physio_file, preload=True, verbose=verbose)

        # 获取生理数据的时间采样数、时间轴、通道名称、通道数、采样频率、高通滤波和低通滤波信息
        n_time_samps = raw.n_times
        time_secs = raw.times
        ch_names = raw.ch_names
        n_chan = len(ch_names)
        sfreq = raw.info['sfreq']
        highpass = raw.info['highpass']
        lowpass = raw.info['lowpass']

        # 将生理数据重采样到128Hz
        raw = raw.resample(sfreq=128, stim_picks=46, verbose=verbose)

        # 提取视频开始和结束的索引
        status_ch, time = raw[-1]  # extract last channel
        video_indices = np.where(np.diff(status_ch) > 0)[1]
        video_indices = video_indices + (1,1)

        # 提取EEG、ECG和GSR数据，并进行滤波处理
        EEG_CH = ch_names[0:32]
        # SELECTED_EEG_CH = ["F3", "F7", "FC5", "T7", "P7",
        #                 "F4", "F8", "FC6", "T8", "P8"]
        ECG_CH = ch_names[32:35]
        GSR_CH = ch_names[40]

        # EEG ------------------------------------------------------------------------------------------- 
        raw_eeg = raw.copy().pick_channels(EEG_CH, verbose=verbose)
        raw_eeg = raw_eeg.set_eeg_reference(ref_channels='average', verbose=verbose)
        raw_eeg = raw_eeg.notch_filter(50, verbose=verbose)
        raw_eeg = raw_eeg.filter(l_freq=1,  h_freq=45, verbose=verbose)

        # ECG -------------------------------------------------------------------------------------------
        raw_ecg = raw.copy().pick_channels(ECG_CH, verbose=verbose)
        raw_ecg = raw_ecg.notch_filter(50, verbose=verbose)
        raw_ecg = raw_ecg.filter(l_freq=0.5, h_freq=45, verbose=verbose)

        # GSR -------------------------------------------------------------------------------------------
        raw_gsr = raw.copy().pick_channels([GSR_CH], verbose=verbose)
        raw_gsr = raw_gsr.notch_filter(50, verbose=verbose)
        raw_gsr = raw_gsr.filter(l_freq=None, h_freq=60, verbose=verbose)

        # 提取视频期间的生理数据
        # eeg_data, time = raw_eeg[SELECTED_EEG_CH, video_indices[0]:video_indices[1]]
        eeg_data, time = raw_eeg[:, video_indices[0]:video_indices[1]]
        ecg_data, time = raw_ecg[:, video_indices[0]:video_indices[1]]
        gsr_data, time = raw_gsr[:, video_indices[0]:video_indices[1]]

        # 提取基线数据并进行基线校正
        baseline, _ = raw_gsr[:, video_indices[0]-26:video_indices[0]]
        gsr_data = gsr_data - np.mean(baseline)

        # 眼动数据 ----------------------------------------------------------------
        gaze_file = os.path.join(dir, "P{}-Rec1-All-Data-New_Section_{}.tsv".format(subject, session))
        gaze_df = pd.read_csv(gaze_file, sep = '\t', skiprows=23)

        """
        Number: 数据点的编号或时间戳
        GazePointXLeft: 左眼注视点的X坐标
        GazePointYLeft: 左眼注视点的Y坐标。
        DistanceLeft: 左眼到屏幕的距离
        PupilLeft: 左眼瞳孔直径
        ValidityLeft: 左眼数据的有效性
        GazePointXRight: 右眼注视点的X坐标
        GazePointYRight: 右眼注视点的Y坐标
        DistanceRight: 右眼到屏幕的距离
        PupilRight: 右眼瞳孔直径
        ValidityRight: 右眼数据的有效性
        Event: 事件标记
        """
        gaze_df = gaze_df[["Number", "GazePointXLeft", "GazePointYLeft", "DistanceLeft", "PupilLeft", "ValidityLeft",
                            "GazePointXRight", "GazePointYRight", "DistanceRight", "PupilRight", "ValidityRight", "Event"]]

        # 提取视频期间的眼动数据
        video_start = gaze_df.index[gaze_df["Event"] == "MovieStart"].tolist()
        video_start = video_start[0] + 1

        video_end = gaze_df.index[gaze_df["Event"] == "MovieEnd"].tolist()
        video_end = video_end[0]

        gaze_df = gaze_df.iloc[video_start:video_end, :]

        # # 删除按键和鼠标点击事件的数据
        key_indices = gaze_df.index[gaze_df["Event"] == "KeyPress"].tolist()
        r_mouse_indices = gaze_df.index[gaze_df["Event"] == "RightMouseClick"].tolist()
        l_mouse_indices = gaze_df.index[gaze_df["Event"] == "LeftMouseClick"].tolist()
        gaze_df = gaze_df.drop(key_indices + r_mouse_indices + l_mouse_indices)

        gaze_df = gaze_df.reset_index(drop=True)  

        samples_id = [int(x) for x in gaze_df["Number"].tolist()]

        # 计算左右眼瞳孔直径
        l_pupil_dim = [float(x) for x in gaze_df["PupilLeft"].tolist()]
        r_pupil_dim = [float(x) for x in gaze_df["PupilRight"].tolist()]
        mean_pupil_dim = []
        for i in range(len(l_pupil_dim)):
            if l_pupil_dim[i] >= 0  and r_pupil_dim[i] >= 0 and l_pupil_dim[i] < 9  and r_pupil_dim[i] < 9:
                mean_pupil_dim.append(round((l_pupil_dim[i] + r_pupil_dim[i])/2, 7) )
            elif l_pupil_dim[i] >= 0 and l_pupil_dim[i] < 9 and r_pupil_dim[i] < 0:
                mean_pupil_dim.append(l_pupil_dim[i])
            elif l_pupil_dim[i] < 0 and r_pupil_dim[i] >= 0 and r_pupil_dim[i] < 9:
                mean_pupil_dim.append(r_pupil_dim[i])
            else:
                mean_pupil_dim.append(-1)

        # 计算左右眼注视点坐标x
        l_gaze_x = [float(x) for x in gaze_df["GazePointXLeft"].tolist()]
        r_gaze_x = [float(x) for x in gaze_df["GazePointXRight"].tolist()]
        mean_gaze_x = []
        for i in range(len(l_gaze_x)):
            if l_gaze_x[i] >= 0  and r_gaze_x[i] >= 0  and l_pupil_dim[i] < 9  and r_pupil_dim[i] < 9:
                mean_gaze_x.append( round((l_gaze_x[i] + r_gaze_x[i])/2, 7) )
            elif l_gaze_x[i] >= 0 and l_pupil_dim[i] < 9 and r_gaze_x[i] < 0:
                mean_gaze_x.append(l_gaze_x[i])
            elif l_gaze_x[i] < 0 and r_gaze_x[i] >= 0 and r_pupil_dim[i] < 9:
                mean_gaze_x.append(r_gaze_x[i])
            else:
                mean_gaze_x.append(-1)

        # 计算左右眼注视点坐标y
        l_gaze_y = [float(x) for x in gaze_df["GazePointYLeft"].tolist()]
        r_gaze_y = [float(x) for x in gaze_df["GazePointYRight"].tolist()]
        mean_gaze_y = []
        for i in range(len(l_gaze_y)):
            if l_gaze_y[i] >= 0  and r_gaze_y[i] >= 0  and l_pupil_dim[i] < 9  and r_pupil_dim[i] < 9:
                mean_gaze_y.append( round((l_gaze_y[i] + r_gaze_y[i])/2, 7) )
            elif l_gaze_y[i] >= 0 and l_pupil_dim[i] < 9 and r_gaze_y[i] < 0:
                mean_gaze_y.append(l_gaze_y[i])
            elif l_gaze_y[i] < 0 and r_gaze_y[i] >= 0 and r_pupil_dim[i] < 9:
                mean_gaze_y.append(r_gaze_y[i])
            else:
                mean_gaze_y.append(-1)

        # 计算左右眼睛距离的平均值
        l_eye_dist = [float(x) for x in gaze_df["DistanceLeft"].tolist()]
        r_eye_dist = [float(x) for x in gaze_df["DistanceRight"].tolist()]
        mean_eye_dist = []
        for i in range(len(l_eye_dist)):
            if l_eye_dist[i] >= 0  and r_eye_dist[i] >= 0  and l_pupil_dim[i] < 9  and r_pupil_dim[i] < 9:
                mean_eye_dist.append( round((l_eye_dist[i] + r_eye_dist[i])/2, 7) )
            elif l_eye_dist[i] >= 0 and l_pupil_dim[i] < 9 and r_eye_dist[i] < 0:
                mean_eye_dist.append(l_eye_dist[i])
            elif l_eye_dist[i] < 0 and r_eye_dist[i] >= 0 and r_pupil_dim[i] < 9:
                mean_eye_dist.append(r_eye_dist[i])
            else:
                mean_eye_dist.append(-1)

        # 创建保存路径并保存预处理后的数据
        path_name = os.path.join(save_path, 'S'+f"{int(subject):02}")
        if not os.path.exists(path_name):
                os.makedirs(path_name)
                
        # trial_data = [eeg_data.T, ecg_data.T, gsr_data.T, mean_pupil_dim, np.stack((mean_gaze_x, mean_gaze_y), axis=1), mean_eye_dist]
        # signals_types = ["EEG", "ECG", "GSR", "PUPIL", "GAZE_COORD", "EYE_DIST"]
        trial_data = [eeg_data.T, mean_pupil_dim, np.stack((mean_gaze_x, mean_gaze_y), axis=1), mean_eye_dist]
        signals_types = ["EEG", "PUPIL", "GAZE_COORD", "EYE_DIST"]
        for i in range(len(signals_types)):
            file_name = os.path.join(path_name, '{}_{}.csv'.format(int(session)//2 - 1, signals_types[i]))
            np.savetxt(file_name, trial_data[i], delimiter=",")

        # 保存情感标签到CSV文件
        trial_labels = [feltEmo, feltArsl, feltVlnc]
        labels_types = ["feltEmo", "feltArsl", "feltVlnc"]
        for i in range(len(labels_types)):
            label_file_name = os.path.join(path_name, 'labels_{}.csv'.format(labels_types[i]))
            f = open(label_file_name, 'a+')
            f.write(trial_labels[i] + "\n")
            f.close()
        # label_file_name = os.path.join(path_name, 'labels_feltEmo.csv')
        # f = open(label_file_name, 'a+')
        # f.write(feltEmo + "\n")
        # f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sessions_path', type=str, default='../dataset/hci-tagging-database/Sessions')
    parser.add_argument('--save_path', type=str, default='../dataset/hci_preproc_data')
    parser.add_argument('--verbose',  type=bool, action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    sessions_dir = list_files(args.sessions_path, sorted_dir=True)
    preprocess(sessions_dir, args.save_path, args.verbose)
