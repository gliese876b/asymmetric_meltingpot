import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ray.tune.logger import UnifiedLogger
from datetime import datetime
import tempfile


def custom_log_creator(custom_path, custom_str):

    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    def logger_creator(config):

        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator

def save_img(rgb_arr, path, name):
    plt.imshow(rgb_arr, interpolation="nearest")
    plt.savefig(path + name)


def make_video_from_image_dir(vid_path, img_folder, video_name="trajectory", fps=5):
    """
    Create a video from a directory of images
    """
    images = [img for img in os.listdir(img_folder) if img.endswith(".png")]
    images.sort()

    rgb_imgs = []
    for i, image in enumerate(images):
        img = cv2.imread(os.path.join(img_folder, image))
        rgb_imgs.append(img)

    make_video_from_rgb_imgs(rgb_imgs, vid_path, video_name=video_name, fps=fps)


def make_video_from_rgb_imgs(
    rgb_arrs, vid_path, video_name="trajectory", fps=5, format="mp4v", resize=None  # FIXME: There is a color issue here!
):
    """
    Create a video from a list of rgb arrays
    """
    print("Rendering video...")
    if vid_path[-1] != "/":
        vid_path += "/"
    video_path = vid_path + video_name + ".mp4"

    if resize is not None:
        width, height = resize
    else:
        frame = rgb_arrs[0]
        height, width, _ = frame.shape
        resize = width, height

    fourcc = cv2.VideoWriter_fourcc(*format)
    video = cv2.VideoWriter(video_path, fourcc, float(fps), (width, height))

    for i, image in enumerate(rgb_arrs):
        percent_done = int((i / len(rgb_arrs)) * 100)
        if percent_done % 20 == 0:
            print("\t...", percent_done, "% of frames rendered")
        # Always resize, without this line the video does not render properly.
        image = cv2.resize(image.astype(np.uint8), resize, interpolation=cv2.INTER_NEAREST)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if i == 0:
            cv2.imwrite(vid_path + video_name + '.png' , image)
        video.write(image)

    video.release()
    print("Video is created as", video_path)
