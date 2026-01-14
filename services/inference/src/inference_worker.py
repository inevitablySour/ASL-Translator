'''
The inferenc worker is where the model actually works. Basic design prinicple is going to be that the system
can call one function with the corresponding data and the rest gets fixed!!
'''
from hand_detector import HandDetector
from gesture_classifier import GestureClassifier
import numpy as np
import cv2
import base64


def base64_to_bgr(base64_str: str) -> np.ndarray:
    """
    Converts a base64-encoded image (data URL or raw) to OpenCV BGR image
    """
    # Remove data URL prefix if present
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    img_bytes = base64.b64decode(base64_str)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Failed to decode base64 image")

    return image

def predict_gesture_from_base64(base64_image: str) -> dict:
    """
    Input: base64 webcam frame
    Output: {gesture, confidence}
    """

    # 1. Decode image
    image = base64_to_bgr(base64_image)

    # 2. Detect hand + landmarks
    with HandDetector() as detector:
        _, hands = detector.detect_hands(image)

        if not hands:
            return {
                "gesture": "NO_HAND",
                "confidence": 0.0
            }

        hand = hands[0]
        features = detector.extract_features(hand)

        # 4. Predict gesture
    classifier = GestureClassifier()
    prediction = classifier.predict(features, image=image)

    # ✅ handle tuple output
    label, conf = prediction[0], prediction[1]

    return {
        "gesture": label,
        "confidence": float(conf)
    }