import argparse


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input",
                    type=str,
                    default="example_videos/janie.mp4",
                    help="path to (optional) input video file")
    ap.add_argument("-o", "--output-dir",
                    type=str,
                    default="public",
                    help="path to (optional) output video file")
    ap.add_argument('--step',
                    type=int,
                    default=4,
                    help='indicates how many frames have to be skipped in between')
    ap.add_argument("-y", "--yolo",
                    default='yolo-coco',
                    help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence",
                    type=float,
                    default=0.5,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold",
                    type=float,
                    default=0.3,
                    help="threshold when applyong non-maxima suppression")
    ap.add_argument("-u", "--use-gpu",
                    type=bool,
                    default=0,
                    help="boolean indicating if CUDA GPU should be used")
    args = ap.parse_args()
    return args
