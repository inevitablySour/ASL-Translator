'''
The inferenc worker is where the model actually works. Basic design prinicple is going to be that the system
can call one function with the corresponding data and the rest gets fixed!!
'''
from hand_detector import HandDetector
from gesture_classifier import GestureClassifier
from translator import Translator
import numpy as np
import cv2
import base64
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Initialize classifier once globally
_classifier = None

def get_classifier():
    """Get or initialize the gesture classifier (singleton pattern)"""
    global _classifier
    if _classifier is None:
        logger.info("Initializing gesture classifier (first time)...")
        _classifier = GestureClassifier()
    return _classifier


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

def predict_gesture_from_base64(base64_image: str, language: str = "english") -> dict:
    """
    Input: base64 webcam frame, language
    Output: {gesture, confidence, translation, language}
    """
    logger.info(f"Starting prediction pipeline for language: {language}")

    # 1. Decode image
    try:
        image = base64_to_bgr(base64_image)
        logger.debug(f"Image decoded. Shape: {image.shape}")
    except Exception as e:
        logger.error(f"Failed to decode image: {e}", exc_info=True)
        raise

    # 2. Detect hand + landmarks
    # 3. Get classifier (initialized once, reused across predictions)
    classifier = get_classifier()
    
    with HandDetector() as detector:
        _, hands = detector.detect_hands(image)
        logger.info(f"Hand detection complete. Hands found: {len(hands)}")

        if not hands:
            logger.warning("No hand detected in image")
            return {
                "gesture": "NO_HAND",
                "confidence": 0.0,
                "translation": "No hand detected",
                "language": language,
                "landmarks": None
            }

        hand = hands[0]
        
        # Use enhanced features if model expects 78, otherwise use basic 63
        if classifier.expected_features == 78:
            logger.debug("Extracting enhanced features (78)")
            features = detector.extract_features_enhanced(hand)
        else:
            logger.debug("Extracting basic features (63)")
            features = detector.extract_features(hand)
        
        logger.debug(f"Features extracted. Shape: {features.shape}")

    # 4. Predict gesture
    prediction = classifier.predict(features, image=image)

    # ✅ handle tuple output
    label, conf = prediction[0], prediction[1]
    logger.info(f"Prediction: {label} with confidence {conf:.4f}")

    # 5. Translate gesture
    translator = Translator()
    translation = translator.translate(label, language)
    logger.info(f"Translation: {label} -> {translation}")

    # Include landmarks if confidence is high enough (for feedback collection)
    landmarks_list = None
    if conf >= 0.9:  # Threshold for feedback collection
        landmarks_list = features.tolist()  # Convert numpy array to list
        logger.debug(f"Including landmarks for feedback (conf={conf:.2f})")

    result = {
        "gesture": label,
        "confidence": float(conf),
        "translation": translation,
        "language": language,
        "landmarks": landmarks_list
    }
    logger.info(f"Final result: {result}")
    return result
