import argparse
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import sys
from openai import OpenAI
import numpy as np
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

sys.path.insert(0, 'third_party/CenterNet2/projects/CenterNet2/')
from third_party.CenterNet2.projects.CenterNet2.centernet.config import add_centernet_config
from grit.config import add_grit_config

from grit.predictor import VisualizationDemo

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



# constants
WINDOW_NAME = "GRiT"


def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_grit_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    if args.test_task:
        cfg.MODEL.TEST_TASK = args.test_task
    cfg.MODEL.BEAM_SIZE = 1
    cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
    cfg.USE_ACT_CHECKPOINT = False
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--test-task",
        type=str,
        default='',
        help="Choose a task to have GRiT perform",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        results = {}
        input_dir = args.input[0]
        for path in tqdm.tqdm(os.listdir(input_dir), desc="Processing images"):
            img = read_image(os.path.join(input_dir, path), format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            # 中身をcpuに移動
            instances = predictions["instances"].to("cpu")
            # 検出したboxの座標を取得
            boxes = instances.pred_boxes.tensor.numpy().tolist()
            # それへのキャプション
            pred_object_description = instances.pred_object_descriptions.data
            formatted_results = []
            for box, description in zip(boxes, pred_object_description):
                x1, y1, x2, y2 = box
                width = x2-x1
                height = y2-y1
                formatted_results.append({
                    "description": description,
                    "width": width,
                    "height": height
                })
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages = [
                    {
                        "role":"system", "content":"You should describe the scene the scene in detail."
                    },
                    {
                        "role":"user", "content": f"""You are given a description of the object recognition model for the driver's view, a description of the objects in the image, which is {formatted_results}. 
                                                    Assume you are a blind but intelligent image caption creator.
                                                    Note that the positions given  are the coodinates of the upper left corner of the object and the size of the object.
                                                    Describe this image in about 150words in a way that an elementry school student could understand."""
                    }
                ],


            )
            output = completion.choices[0].message.content
            response = client.embeddings.create(
                input=output,
                model="text-embedding-3-small"
            )
            embedding = response.data[0].embedding

            # embeddingを{path}.npyとして保存する
            if not os.path.exists("embeddings"):
                os.makedirs("embeddings")
            np.save(f"embeddings/{path}.npy", embedding)
            # もとの文字列もtxtとして保存する
            if not os.path.exists("txt_data"):
                os.mkdir("txt_data")
            with open(f"txt_data/{path}.txt", "w") as f:
                f.write(output)

            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if not os.path.exists(args.output):
                    os.mkdir(args.output)
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit