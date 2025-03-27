import socketio
import eventlet
import numpy as np
import cv2
import base64
import torch
import sys
from pathlib import Path
from models.common import DetectMultiBackend
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

# Add YOLOv5 root directory to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.general import (check_img_size, non_max_suppression, scale_boxes)

# Check if CUDA is available and print information
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# Set up Socket.IO server
sio = socketio.Server(cors_allowed_origins='*')
app = socketio.WSGIApp(sio)

# Force CUDA if available
device = select_device('0' if torch.cuda.is_available() else 'cpu')
# Verify device is set to CUDA
print(f"Using device: {device}")

# Load YOLOv5 model
model_path = 'models/firesmoke.pt'
model = DetectMultiBackend(model_path, device=device)
stride, names, pt = model.stride, model.names, model.pt

# Set model parameters
imgsz = check_img_size((640, 640), s=stride)  # check image size
model.warmup(imgsz=(1, 3, *imgsz))  # warmup
conf_thres = 0.25  # confidence threshold
iou_thres = 0.45  # NMS IOU threshold

print(f"Model loaded on {device}")
print(f"Class names: {names}")

@sio.event
def connect(sid, environ):
    print(f'Client connected: {sid}')

@sio.event
def disconnect(sid):
    print(f'Client disconnected: {sid}')

@sio.event
def frame(sid, data):
    try:
        # Decode the base64 image
        encoded_data = data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Prepare image for model
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).to(device)
        img_tensor = img_tensor.float() / 255.0  # 0 - 255 to 0.0 - 1.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            pred = model(img_tensor)
            pred = non_max_suppression(pred, conf_thres, iou_thres)
        
        # Process predictions
        for i, det in enumerate(pred):
            annotator = Annotator(img.copy(), line_width=2)
            
            if len(det):
                # Rescale boxes from img_size to img size
                det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], img.shape).round()
                
                # Draw boxes and labels
                for *xyxy, conf, cls in det:
                    c = int(cls)
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                
                # Log detections
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    print(f"Detected {n} {names[int(c)]}")
            
            # Get annotated image
            result_img = annotator.result()
        
        # Convert back to base64 to send to client
        _, buffer = cv2.imencode('.jpg', result_img)
        result_encoded = base64.b64encode(buffer).decode('utf-8')
        result_data = f'data:image/jpeg;base64,{result_encoded}'
        
        # Emit the processed frame back to the client
        sio.emit('processed_frame', result_data, room=sid)
        
    except Exception as e:
        print(f"Error processing frame: {e}")

if __name__ == '__main__':
    # Start the server
    port = 5001
    print(f"Starting server on port {port}")
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', port)), app)