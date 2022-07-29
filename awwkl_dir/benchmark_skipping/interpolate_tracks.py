import cv2
import numpy as np
import pandas as pd
from yolox.utils.visualize import plot_tracking

def interpolate_track_results(input_result_path, output_result_path):
    df_orig = pd.read_csv(input_result_path)
    df_ret = df_orig.copy(deep=True)
    new_rows = []

    # Get list of unique objects (tid)
    unique_tid_list = list( df_orig['tid'].unique() )
    for tid in unique_tid_list:
        frame_ids_with_tracks = list ( df_orig[ df_orig['tid'] == tid ]['frame_id'].unique() )
        if len(frame_ids_with_tracks) < 2:
            continue
        
        tid_rows = df_orig[ df_orig['tid'] == tid ]
        tid_rows = tid_rows.reset_index()

        for ind in range(len(frame_ids_with_tracks) - 1):
            fid_start = frame_ids_with_tracks[ind]
            fid_end =   frame_ids_with_tracks[1 + ind]
            fid_diff = fid_end - fid_start
            for frame_id in range(1+fid_start, fid_end):
                print(f'interpolate {frame_id} using start:{fid_start}, end:{fid_end}')
                tmp = [0] * 5
                tmp[0] = tid_rows.loc[ind]['tlwh[0]'] + (tid_rows.loc[1+ind]['tlwh[0]'] - tid_rows.loc[ind]['tlwh[0]']) / fid_diff
                tmp[1] = tid_rows.loc[ind]['tlwh[1]'] + (tid_rows.loc[1+ind]['tlwh[1]'] - tid_rows.loc[ind]['tlwh[1]']) / fid_diff
                tmp[2] = tid_rows.loc[ind]['tlwh[2]'] + (tid_rows.loc[1+ind]['tlwh[2]'] - tid_rows.loc[ind]['tlwh[2]']) / fid_diff
                tmp[3] = tid_rows.loc[ind]['tlwh[3]'] + (tid_rows.loc[1+ind]['tlwh[3]'] - tid_rows.loc[ind]['tlwh[3]']) / fid_diff
                tmp[4] = (tid_rows.loc[ind]['tscore'] + tid_rows.loc[ind]['tscore']) / 2
                new_rows.append([frame_id, tid, tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], -1, -1, -1])
        
    print('# new rows:', len(new_rows))
    df_new = pd.DataFrame(new_rows, columns=df_orig.columns)
    df_ret = pd.concat([df_orig, df_new], ignore_index=True)
    df_ret = df_ret.sort_values(by=['frame_id', 'tid'])
    df_ret = df_ret.round(decimals=2)
    df_ret.to_csv(output_result_path, index=False)

def plot_tracking_video(input_video_path, input_result_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    df_orig = pd.read_csv(input_result_path)

    frame_id = 0
    while True:
        return_value, frame = cap.read()
        frame_id += 1
        if not return_value:
            break
        
        frame_rows = df_orig[ df_orig['frame_id'] == frame_id ]
        online_tlwhs = []
        online_ids = []
        online_scores = []
        for ind, row in frame_rows.iterrows():
            tlwh = [ row['tlwh[0]'], row['tlwh[1]'], row['tlwh[2]'], row['tlwh[3]'] ]
            tid = row['tid']
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(row['tscore'])
        online_im = plot_tracking(
            frame, online_tlwhs, online_ids, frame_id=frame_id, fps=0.0
        )
        out.write(np.asarray(online_im))
    out.release()
        

if __name__ == '__main__':
    # input_result_path = 'C:/Users/Khai_Loong/Documents/ByteTrack/awwkl_dir/benchmark_skipping/results_folder/tracks_skip_2/MOT17-04_frames150.txt'
    # output_result_path = 'C:/Users/Khai_Loong/Documents/ByteTrack/awwkl_dir/benchmark_skipping/results_folder/tracks_skip_2/MOT17-04_frames150_interpolated.txt'

    # input_result_path = 'C:/Users/Khai_Loong/Documents/ByteTrack/awwkl_dir/benchmark_skipping/results_folder/tracks_skip_2/MOT17-04.txt'
    # output_result_path = 'C:/Users/Khai_Loong/Documents/ByteTrack/awwkl_dir/benchmark_skipping/results_folder/tracks_skip_2/MOT17-04_interpolated.txt'
    # interpolate_track_results(input_result_path, output_result_path)

    # input_video_path = 'C:/Users/Khai_Loong/Documents/ByteTrack/awwkl_dir/MOT17-04_frames15.mp4'
    # input_result_path = 'C:/Users/Khai_Loong/Documents/ByteTrack/awwkl_dir/benchmark_skipping/results_folder/tracks_skip_2/MOT17-04_interpolated.txt'
    # output_video_path = 'C:/Users/Khai_Loong/Documents/ByteTrack/awwkl_dir/benchmark_skipping/results_folder/tracks_skip_2/MOT17-04.mp4'
    # plot_tracking_video(input_video_path, input_result_path, output_video_path)
    

    input_result_path = 'C:/Users/Khai_Loong/Documents/ByteTrack/awwkl_dir/benchmark_skipping/results_folder/tracks_skip_2/MOT17-04_frames150.txt'
    output_result_path = 'C:/Users/Khai_Loong/Documents/ByteTrack/awwkl_dir/benchmark_skipping/results_folder/tracks_skip_2/MOT17-04_frames150_interpolated.txt'
    interpolate_track_results(input_result_path, output_result_path)

    input_video_path = 'C:/Users/Khai_Loong/Documents/ByteTrack/awwkl_dir/MOT17-04_frames150.mp4'
    input_result_path = output_result_path
    output_video_path = 'C:/Users/Khai_Loong/Documents/ByteTrack/awwkl_dir/benchmark_skipping/results_folder/tracks_skip_2/MOT17-04_frames150.mp4'
    plot_tracking_video(input_video_path, input_result_path, output_video_path)