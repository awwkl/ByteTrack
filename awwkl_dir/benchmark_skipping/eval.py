from loguru import logger

from yolox.utils import setup_logger
import os
import glob
import motmetrics as mm
from collections import OrderedDict
from pathlib import Path

def eval_result_folder(gt_folder, results_folder):
    # Set up logger
    setup_logger(results_folder, distributed_rank=0, filename="eval_result.log", mode="a")
    
    # Get text files for Ground Truth (gt) and Results to compare (ts)
    gtfiles = [f for f in glob.glob(os.path.join(gt_folder, '*.txt'))]
    tsfiles = [f for f in glob.glob(os.path.join(results_folder, '*.txt'))]
    logger.info("\n" + 'Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
    gt = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in gtfiles])
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in tsfiles])

    # Compare dataframes
    accs, names = compare_dataframes(gt, ts)

    # Compute metrics
    mm.lap.default_solver = 'lap'
    mh = mm.metrics.create()    
    metrics = mm.metrics.motchallenge_metrics + ['num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    logger.info("\n" + mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))

def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:            
            print('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            print('No ground truth for {}, skipping.'.format(k))
    return accs, names

if __name__ == '__main__':

    gt_folder = 'C:/Users/Khai_Loong/Documents/ByteTrack/awwkl_dir/benchmark_skipping/gt_folder'
    results_folder = 'dets_skip_2'
    # results_folder = 'orig_bytetrack'
    results_folder = os.path.join('C:/Users/Khai_Loong/Documents/ByteTrack/awwkl_dir/benchmark_skipping/results_folder/', 
                            results_folder)

    eval_result_folder(gt_folder, results_folder)