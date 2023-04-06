import argparse
import subprocess

import cv2
import supervision as sv
from ultralytics import YOLO


class YouTubeVideoStreamer:
    def __init__(self, url, key, width=640, height=480):
        self.url = url
        self.key = key
        self.width = width
        self.height = height
        self.writer = None

    def start_streaming(self):
        command = ['ffmpeg',
                   '-use_wallclock_as_timestamps', '1',  ###
                   '-y',  # overwrite output file if it exists
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-pixel_format', 'bgr24',
                   '-s', "{}x{}".format(self.width, self.height),
                   '-re',  # Fix 2
                   # '-r', str(20),

                   '-i', '-',  # input comes from a pipe
                   '-vsync', 'cfr',  # Fix
                   '-r', '25',  # Fix
                   # '-re',
                   '-f', 'lavfi',  # <<< YouTube Live requires a audio stream
                   '-i', 'anullsrc',  # <<< YouTube Live requires a audio stream
                   '-c:v', 'libx264',
                   '-c:a', 'aac',  # <<< YouTube Live requires a audio stream
                   # '-x264opts', "keyint=40:min-keyint=40:no-scenecut",
                   "-crf", "24",
                   '-pix_fmt', 'yuv420p',
                   '-preset', 'medium',
                   '-f', 'flv',
                   f'{self.url}/{self.key}']

        self.writer = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)

    def stream_frame(self, frame):
        self.writer.stdin.write(frame.tobytes())

    def stop_streaming(self):
        self.writer.stdin.close()
        self.writer.wait()


class AiStreamer:
    def __init__(self, source, model_path, url, key, width=640, height=480, external_streamer=None):
        self.source = source

        if external_streamer is None:
            external_streamer = YouTubeVideoStreamer

        self.model_path = model_path
        
        self.streamer = external_streamer(url=url, key=key, width=width, height=height)
        self.streamer.start_streaming()

    def stream(self):
        box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=1,
            text_scale=0.5
        )
        class_exclude = ['traffic light']
        while True:
            self.model = YOLO(self.model_path)
            classes = [class_id for class_id, class_name in self.model.names.items() if class_name not in class_exclude]
            for result in self.model.track(source=self.source, stream=True, agnostic_nms=True,
                                     device=0, verbose=False, batch=1, classes=classes):
                frame = result.orig_img
                detections = sv.Detections.from_yolov8(result)
                if result.boxes.id is not None:
                    detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
                labels = [
                    f"{(tracker_id + ' ') if tracker_id else ''}{self.model.model.names[class_id]} {confidence:0.2f}"
                    for _, confidence, class_id, tracker_id
                    in detections
                ]
                frame = box_annotator.annotate(
                    scene=frame,
                    detections=detections,
                    labels=labels
                )
                fps = 1000 / (result.speed['preprocess'] + result.speed['inference'] + result.speed['postprocess'])
                text = f"FPS: {fps:.2f}, Pr: {result.speed['preprocess']:.2f}ms, In: {result.speed['inference']:.2f}ms, Post: {result.speed['postprocess']:.2f}ms"
                cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                self.streamer.stream_frame(frame)
    
    def __del__(self):
        self.streamer.stop_streaming()


class ArgumentsParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--source', default=0, help='video source')
        self.parser.add_argument('--model', type=str, default='yolov8n.pt', help='model path')
        self.parser.add_argument('--url', type=str, default='rtmp://a.rtmp.youtube.com/live2', help='streaming url')
        self.parser.add_argument('--key', type=str, default='xxxx-xxxx-xxxx-xxxx-xxxx', help='streaming key')
        self.parser.add_argument('--width', type=int, default=640, help='streaming width')
        self.parser.add_argument('--height', type=int, default=480, help='streaming height')

    def parse(self):
        return self.parser.parse_args()


if __name__ == "__main__":
    args = ArgumentsParser().parse()
    streamer_ai = AiStreamer(source=args.source,
                             model_path=args.model,
                             url=args.url,
                             key=args.key,
                             width=args.width, height=args.height,
                             external_streamer=YouTubeVideoStreamer)
    streamer_ai.stream()
