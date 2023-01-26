from flask import Flask, render_template, request, redirect, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import os

import cv2
import torch
import pickle
from PIL import Image

from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

# Flask configuration
app = Flask(__name__)
app.config['UPLOAD_DIRECTORY'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['ALLOWED_EXTENSIONS'] = ['.jpg', '.jpeg']
app.config['PREDICTION_DIR_YOLO'] = 'prediction_yolo/'
app.config['PREDICTION_DIR_DT2'] = 'prediction_dt2/'

# Detectron2 configuration
det2_conf_save_file = 'detectron2/OD_cfg.pickle'
det2_out_dir_path = 'detectron2/output/object_detection'

with open(det2_conf_save_file, 'rb') as f:
  cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = os.path.join(det2_out_dir_path, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75
cfg.MODEL.DEVICE = 'cpu'
predictor = DefaultPredictor(cfg)

yolo_answers = dict()
dt2_answers = dict()


@app.route('/')
def index():
  files = os.listdir(app.config['UPLOAD_DIRECTORY'])
  files.sort(key=lambda f: os.path.getmtime(os.path.join(app.config['UPLOAD_DIRECTORY'], f)))
  images = []

  for file in files:
    extension = os.path.splitext(file)[1].lower()
    if extension in app.config['ALLOWED_EXTENSIONS']:
        images.append(file)

  return render_template('index.html', images=images, dt2_answers=dt2_answers, yolo_answers=yolo_answers)


@app.route('/upload', methods=['POST'])
def upload():
  file = request.files['file']
  extension = os.path.splitext(file.filename)[1].lower()

  try:
    if file:

      if extension not in app.config['ALLOWED_EXTENSIONS']:
        return 'File is not a jpg image'

      file_name = secure_filename(file.filename)
      upload_file_name = os.path.join(app.config['UPLOAD_DIRECTORY'], file_name)

      file.save(upload_file_name)

      img = cv2.imread(upload_file_name)
      outputs = predictor(img)
      if outputs["instances"].has("scores"):
        if torch.numel(outputs["instances"].scores[outputs["instances"].scores > 0.75]) != 0:
          dt2_answers[file_name] = 'dirty'
        else:
          dt2_answers[file_name] = 'clean'
      else:
        dt2_answers[file_name] = 'clean'

      v = Visualizer(img[:, :, ::-1], metadata={}, scale=1.0, instance_mode=ColorMode.IMAGE)
      out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
      pred_img = out.get_image()
      image = Image.fromarray(pred_img.astype('uint8')).convert('RGB')
      image.save(os.path.join(app.config['PREDICTION_DIR_DT2'], file_name))

      yolo_command = 'python3 yolov7/detect.py --weights yolov7/runs/train/yolov733/weights/best.pt --conf 0.5 --no-trace --save-txt --img-size 640 --source ' + upload_file_name
      os.system(yolo_command)
      move_command = 'mv runs/detect/exp/{0} {1}'.format(file_name,  os.path.join(app.config['PREDICTION_DIR_YOLO'], file_name))
      os.system(move_command)

      os.system('ls -l runs/detect/exp/labels')

      x = file_name.split('.')

      if os.path.isfile('runs/detect/exp/labels/' + x[0] + '.txt'):
        yolo_answers[file_name] = 'dirty'
      else:
        yolo_answers[file_name] = 'clean'

      os.system('rm -rf runs/detect/exp')

      os.system('ls -l prediction_yolo')
      os.system('ls -l prediction_dt2')

  except RequestEntityTooLarge:
    return 'File is larger than 16MB limit.'

  return redirect('/')


@app.route('/serve-image/<filename>', methods=['GET'])
def serve_image(filename):
  return send_from_directory(app.config['UPLOAD_DIRECTORY'], filename)

@app.route('/serve-dt2/<filename>', methods=['GET'])
def serve_dt2(filename):
  return send_from_directory(app.config['PREDICTION_DIR_DT2'], filename)

@app.route('/serve-yolo/<filename>', methods=['GET'])
def serve_yolo(filename):
  return send_from_directory(app.config['PREDICTION_DIR_YOLO'], filename)


if __name__ == '__main__':
  app.run(port=5000, debug=True)
