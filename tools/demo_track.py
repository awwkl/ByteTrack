import argparse
import os
import os.path as osp
import time
import cv2
import numpy as np
import pandas as pd
import torch

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer

from PIL import Image, ImageFont, ImageDraw

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="./videos/palace.mp4", help="path to images or video"
    )
    parser.add_argument("--output_dir", default="./awwkl_dir/output")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info


def image_demo(predictor, vis_folder, current_time, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    tracker = BYTETracker(args, frame_rate=args.fps)
    timer = Timer()
    results = []

    for frame_id, img_path in enumerate(files, 1):
        outputs, img_info = predictor.inference(img_path, timer)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if args.save_result:
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_folder = osp.join(vis_folder, timestamp)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    if args.save_result:
        res_file = osp.join(vis_folder, timestamp, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = osp.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
            if args.save_result:
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")

def convert_dets_to_boxes(dets):
    ret_boxes = []

    dets = dets.cpu().numpy() # Bring to CPU, then convert to numpy
    for i, det in enumerate(dets):
        # ret_boxes[i] = list(det[:4])
        ret_boxes.append( list(det[:4]) )

    return ret_boxes

def convert_boxes_to_dets(boxes):
    n_boxes = len(boxes)
    dets = np.empty((n_boxes, 5))
    for i in range(n_boxes):
        dets[i][0] = boxes[i][0]
        dets[i][1] = boxes[i][1]
        dets[i][2] = boxes[i][2]
        dets[i][3] = boxes[i][3]
        dets[i][4] = 0.8
        
    return dets

def calculate_iou(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        yA = max(boxA[0], boxB[0])
        xA = max(boxA[1], boxB[1])
        yB = min(boxA[2], boxB[2])
        xB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

def interpolate_det_boxes(start_frame_boxes, end_frame_boxes, frames=[]):

    matched_boxes = []
    steps = len(frames)
    iou_match_threshold = 0.4

    # Match boxes between start frame & end frame
    for start_box in start_frame_boxes:
        is_start_box_matched = False

        for end_box in end_frame_boxes:
            iou = calculate_iou(start_box, end_box)
            if iou > iou_match_threshold:
                step = [(end_box[0]-start_box[0])/steps,(end_box[1]-start_box[1])/steps,(end_box[2]-start_box[2])/steps,(end_box[3]-start_box[3])/steps]
                matched_boxes.append((start_box, end_box, step))
                is_start_box_matched = True

            if is_start_box_matched:
                break
                
    match_percent = 100.0 * len(matched_boxes) / len(start_frame_boxes)
    print(f'[DEBUG] # matched boxes: {len(matched_boxes)}, percent: {match_percent:.2f}')

    # Start interpolation
    frames_boxes = []
    for image in frames:
        frame_boxes = []
        for info in matched_boxes:
            new_box = [info[0][0]+info[2][0], info[0][1]+info[2][1], info[0][2]+info[2][2], info[0][3]+info[2][3]]
            frame_boxes.append(new_box)               
        frames_boxes.append(frame_boxes)

    return frames_boxes

def interpolate_track_results(input_result_path, output_result_path):
    print('[DEBUG] Interpolate track results, using tracked results with skipped frames')
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
                tmp = [0] * 5
                tmp[0] = tid_rows.loc[ind]['tlwh[0]'] + (tid_rows.loc[1+ind]['tlwh[0]'] - tid_rows.loc[ind]['tlwh[0]']) / fid_diff
                tmp[1] = tid_rows.loc[ind]['tlwh[1]'] + (tid_rows.loc[1+ind]['tlwh[1]'] - tid_rows.loc[ind]['tlwh[1]']) / fid_diff
                tmp[2] = tid_rows.loc[ind]['tlwh[2]'] + (tid_rows.loc[1+ind]['tlwh[2]'] - tid_rows.loc[ind]['tlwh[2]']) / fid_diff
                tmp[3] = tid_rows.loc[ind]['tlwh[3]'] + (tid_rows.loc[1+ind]['tlwh[3]'] - tid_rows.loc[ind]['tlwh[3]']) / fid_diff
                tmp[4] = (tid_rows.loc[ind]['tscore'] + tid_rows.loc[ind]['tscore']) / 2
                new_rows.append([frame_id, tid, tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], -1, -1, -1])
        
    df_new = pd.DataFrame(new_rows, columns=df_orig.columns)
    df_ret = pd.concat([df_orig, df_new], ignore_index=True)
    df_ret = df_ret.sort_values(by=['frame_id', 'tid'])
    df_ret = df_ret.round(decimals=2)
    df_ret.to_csv(output_result_path, index=False)

def plot_tracking_video(input_video_path, input_result_path, output_video_path):
    print('[DEBUG] Output tracking video, using input video and results file')
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
    
def imageflow_demo_skip_frames(predictor, save_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = osp.join(save_folder, args.path.split("/")[-1])
    logger.info(f"video save_path is {save_path}")
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = ["frame_id,tid,tlwh[0],tlwh[1],tlwh[2],tlwh[3],tscore,-1,-1,-1\n"]

    interpolate_frames = []
    start_frame_boxes = None
    end_frame_boxes = None
    
    while True:
        timer.tic()
        return_value, frame = cap.read()
        frame_id += 1
        if not return_value:
            break

        if frame_id == 1:
            print(f'[DEBUG] Frame {frame_id} - Predict')
            outputs, img_info = predictor.inference(frame, timer)
            image_height, image_width = img_info['height'], img_info['width']
            _, results = update_tracker(tracker, outputs[0], frame, image_height, image_width, exp, results, frame_id, timer)

            if args.interpolate_dets:
                start_frame_boxes = convert_dets_to_boxes(outputs[0])

            elif args.interpolate_tracks:
                pass

        elif len(interpolate_frames) < int(args.skip_frame) and not (frame_id == int(cap. get(cv2. CAP_PROP_FRAME_COUNT))):
            print(f'[DEBUG] Frame {frame_id} - Skip & add to list of frames to interpolate')

            if args.interpolate_dets:
                interpolate_frames.append(frame)

            elif args.interpolate_tracks:
                interpolate_frames.append(frame)
            
        else:
            print(f'[DEBUG] Frame {frame_id} - Predict')
            outputs, img_info = predictor.inference(frame, timer)
            image_height, image_width = img_info['height'], img_info['width']

            if args.interpolate_dets:
                # Interpolate frames, and for each interpolated frame, pass to ByteTracker
                if len(interpolate_frames) > 0:
                    end_frame_boxes = convert_dets_to_boxes(outputs[0])
                    frames_boxes = interpolate_det_boxes(start_frame_boxes, end_frame_boxes, interpolate_frames)
                    for ind, frame_box in enumerate(frames_boxes):
                        frame_cnt = frame_id - len(frames_boxes) + ind
                        dets = convert_boxes_to_dets(frame_box)
                        _, results = update_tracker(tracker, dets, frame, image_height, image_width, exp, results, frame_cnt, timer)

                # Pass detection output of current frame to ByteTracker too
                _, results = update_tracker(tracker, outputs[0], frame, image_height, image_width, exp, results, frame_id, timer)

            elif args.interpolate_tracks:
                _, results = update_tracker(tracker, outputs[0], frame, image_height, image_width, exp, results, frame_id, timer)

            else:
                _, results = update_tracker(tracker, outputs[0], frame, image_height, image_width, exp, results, frame_id, timer)

            # Reset variables for next interpolation
            interpolate_frames.clear()
            start_frame_boxes = end_frame_boxes
            end_frame_boxes = None

    # Generate output text file
    output_txt_path = os.path.join(save_folder, args.path.split("/")[-1].replace('.mp4', '.txt'))
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, 'w') as f:
        f.write
        f.writelines(results)

    # Interpolate track results for skipped frames
    if args.interpolate_tracks:
        interpolate_track_results(output_txt_path, output_txt_path)

    # Generate output video, using the input video & tracking results
    plot_tracking_video(args.path, output_txt_path, save_path)


def update_tracker(tracker, dets, original_image, image_height, image_width, exp, results, frame_id, timer):
    online_targets = tracker.update(dets, [image_height, image_width], exp.test_size)
    online_tlwhs = []
    online_ids = []
    online_scores = []
    for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id
        vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
        if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(t.score)
            results.append(
                f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
            )
    timer.toc()
    # online_im = plot_tracking(
    #     original_image, online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
    # )
    return None, results

def main(exp, args):
    
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    # output_dir = osp.join(exp.output_dir, args.experiment_name)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        # vis_folder = osp.join(output_dir, "track_vis")
        vis_folder = output_dir
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, current_time, args)
    elif args.demo == "video" or args.demo == "webcam":
        # imageflow_demo(predictor, vis_folder, current_time, args)
        pass

    for vid in ['MOT17-04', 'MOT17-10', 'MOT17-13']:
        args.path = 'C:/Users/Khai_Loong/Documents/ByteTrack/awwkl_dir/benchmark_skipping/input_videos/'
        args.path = os.path.join(args.path, vid + '.mp4')
        args.output_dir = 'C:/Users/Khai_Loong/Documents/ByteTrack/awwkl_dir/benchmark_skipping/results_folder'
        
        args.output_dir = os.path.join(args.output_dir, 'dets_skip_2')
        args.skip_frame = 2

        args.interpolate_dets = True
        args.interpolate_tracks = not (args.interpolate_dets)
        imageflow_demo_skip_frames(predictor, args.output_dir, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
