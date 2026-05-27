import cv2
import numpy as np
import os
import sys

COCO_CLASSES = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "street sign",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack",
    "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk",
    "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "blender", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype="uint8")

MODEL_CONFIG  = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
MODEL_WEIGHTS = "frozen_inference_graph.pb"


def load_image(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)
    image = cv2.imread(path)
    if image is None:
        print(f"Could not read image: {path}")
        sys.exit(1)
    print(f"Image loaded — Shape: {image.shape}")
    return image


def load_model():
    if not os.path.exists(MODEL_CONFIG) or not os.path.exists(MODEL_WEIGHTS):
        print("Model files not found. Place frozen_inference_graph.pb and "
              "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt in this folder.")
        sys.exit(1)
    net = cv2.dnn.readNetFromTensorflow(MODEL_WEIGHTS, MODEL_CONFIG)
    print("MobileNet-SSD model loaded.")
    return net


def build_blob(image):
    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=1.0 / 127.5,
        size=(300, 300),
        mean=(127.5, 127.5, 127.5),
        swapRB=True,
        crop=False
    )
    print(f"Blob constructed — Shape: {blob.shape}")
    return blob


def detect_objects(net, blob, image, confidence_threshold=0.5, nms_threshold=0.4):
    net.setInput(blob)
    detections = net.forward()

    (h, w) = image.shape[:2]
    boxes, confidences, class_ids = [], [], []

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < confidence_threshold:
            continue
        class_id = int(detections[0, 0, i, 1])
        x1 = max(0, int(detections[0, 0, i, 3] * w))
        y1 = max(0, int(detections[0, 0, i, 4] * h))
        x2 = min(w - 1, int(detections[0, 0, i, 5] * w))
        y2 = min(h - 1, int(detections[0, 0, i, 6] * h))
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        confidences.append(confidence)
        class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            label = COCO_CLASSES[class_ids[i]] if class_ids[i] < len(COCO_CLASSES) else "unknown"
            results.append({
                "class_id": class_ids[i],
                "label": label,
                "confidence": confidences[i],
                "box": (x, y, x + bw, y + bh)
            })

    return results


def annotate_image(image, detections):
    annotated = image.copy()
    for det in detections:
        (x1, y1, x2, y2) = det["box"]
        color = [int(c) for c in COLORS[det["class_id"] % len(COLORS)]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label_text = f"{det['label']}: {det['confidence']*100:.1f}%"
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - text_h - baseline - 4), (x1 + text_w, y1), color, -1)
        cv2.putText(annotated, label_text, (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated


def run_pipeline(image_path, confidence_threshold=0.5):
    print("\n" + "=" * 50)
    print("  PROJECT 4 — OBJECT DETECTION PIPELINE")
    print("=" * 50)

    image = load_image(image_path)
    net = load_model()
    blob = build_blob(image)
    detections = detect_objects(net, blob, image, confidence_threshold)
    annotated = annotate_image(image, detections)
    cv2.imwrite("detection_output.jpg", annotated)
    print("Annotated image saved → detection_output.jpg")

    print("\n--- Detection Results ---")
    if not detections:
        print("No objects detected above the confidence threshold.")
    else:
        print(f"{'#':<4} {'Label':<20} {'Confidence':>10}  {'Box (X,Y,W,H)'}")
        print("-" * 55)
        for idx, det in enumerate(detections, 1):
            (x1, y1, x2, y2) = det["box"]
            print(f"{idx:<4} {det['label']:<20} {det['confidence']*100:>9.1f}%  ({x1},{y1},{x2-x1},{y2-y1})")

        avg_conf = sum(d["confidence"] for d in detections) / len(detections)
        print(f"\nObjects detected : {len(detections)}")
        print(f"Avg confidence   : {avg_conf*100:.1f}%")

    print("=" * 50 + "\n")
    return detections


if __name__ == "__main__":
    IMAGE_PATH = "sample_image.jpg"
    CONFIDENCE_THRESHOLD = 0.5
    run_pipeline(IMAGE_PATH, confidence_threshold=CONFIDENCE_THRESHOLD)