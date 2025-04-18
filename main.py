from flask import Flask, request, jsonify, Response
from google import genai
import PIL.Image
import os
import json
import re
import cv2
import numpy as np
import io
from dotenv import load_dotenv
from datetime import datetime, timedelta
import threading
import time

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize the Gemini client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Global variables for webcam
camera = None
camera_lock = threading.Lock()
last_frame = None
is_camera_running = False


def initialize_camera():
    """Initialize the webcam"""
    global camera, is_camera_running
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0)  # 0 is usually the default webcam
            if not camera.isOpened():
                return False
            is_camera_running = True
            return True
        return camera.isOpened()


def release_camera():
    """Release the webcam resources"""
    global camera, is_camera_running
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
            is_camera_running = False


def get_frame():
    """Get a frame from the webcam"""
    global camera, last_frame
    with camera_lock:
        if camera is not None and camera.isOpened():
            success, frame = camera.read()
            if success:
                last_frame = frame
                return frame
    return last_frame


def analyze_food_image(image_array):
    """
    Analyzes a food image using Gemini to detect the type of food
    and determine if it's fresh, stale, or spoiled.

    Args:
        image_array: The image as a numpy array

    Returns:
        dict: Structured analysis of the food
    """
    try:
        # Convert the OpenCV BGR image to RGB for PIL
        rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        # Convert numpy array to PIL Image
        pil_image = PIL.Image.fromarray(rgb_image)

        # Create a BytesIO object
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)

        # Create a new PIL Image from BytesIO
        image = PIL.Image.open(img_byte_arr)

        # Create the prompt that specifically asks about food type and freshness
        prompt = """
        Please analyze this food image and provide the following information in JSON format:
        {
          "food_name": "The specific name of the food shown in the image",
          "food_category": "One of: pork, chicken, vegetable, dairy, poultry, fruits, pastry",
          "food_condition": "One of: fresh, stale, spoiled",
        }

        Please be specific and base your analysis on visual indicators.
        If you cannot identify food in the image, use "unknown" for the values.
        """

        # Send request to Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=[prompt, image]
        )

        # Extract structured data from the response
        response_text = response.text

        # Try to parse JSON directly from Gemini response
        try:
            # Find JSON object in the response text
            json_match = re.search(r"({[\s\S]*})", response_text)
            if json_match:
                json_str = json_match.group(1)
                food_info = json.loads(json_str)

                # Add missing fields if needed
                if "food_name" not in food_info:
                    food_info["food_name"] = extract_food_name(response_text)
                if "food_category" not in food_info:
                    food_info["food_category"] = extract_food_category(response_text)
                if "food_condition" not in food_info:
                    food_info["food_condition"] = extract_food_condition(response_text)
                # if "best_before" not in food_info:
                #     food_info["best_before"] = estimate_best_before_date(response_text)
            else:
                # Fallback to extraction methods if JSON parsing fails
                food_info = {
                    "food_name": extract_food_name(response_text),
                    "food_category": extract_food_category(response_text),
                    "food_condition": extract_food_condition(response_text),
                    # "best_before": estimate_best_before_date(response_text),
                }
        except json.JSONDecodeError:
            # Fallback to extraction methods if JSON parsing fails
            food_info = {
                "food_name": extract_food_name(response_text),
                "food_category": extract_food_category(response_text),
                "food_condition": extract_food_condition(response_text),
                # "best_before": estimate_best_before_date(response_text),
            }

        # Add date_added field (current date)
        food_info["date_added"] = datetime.now().strftime("%Y-%m-%d")

        # Add full analysis for reference/debugging
        food_info["analysis"] = response_text

        return food_info

    except Exception as e:
        return {"error": f"Error analyzing image: {str(e)}"}


def extract_food_name(text):
    """Extract the specific food name from the text"""
    # Look for specific food type mentions
    food_patterns = [
        r"food (?:is|appears to be|shown is)(?:\s\w+){1,4}((?:\w+\s){1,3}\w+)",
        r"image shows (?:a|an)(?:\s\w+){1,3}((?:\w+\s){1,3}\w+)",
        r"(?:this is|appears to be) (?:a|an)(?:\s\w+){1,3}((?:\w+\s){1,3}\w+)",
        r"food_name[\"']?\s*[:=]\s*[\"']([^\"']+)[\"']",
    ]

    for pattern in food_patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1).strip()

    # Fallback: Look for sentences with "food" and get neighboring words
    sentences = text.split(".")
    for sentence in sentences:
        if "food" in sentence.lower() and "type" in sentence.lower():
            words = sentence.split()
            try:
                food_index = words.index("food")
                if food_index + 3 < len(words):
                    return " ".join(words[food_index + 2 : food_index + 4])
            except ValueError:
                pass

    return "unknown"


def extract_food_category(text):
    """Extract food category matching the required categories"""
    # Specified categories to look for
    categories = [
        "pork",
        "chicken",
        "vegetable",
        "dairy",
        "poultry",
        "fruits",
        "pastry",
    ]

    # Direct match in the text
    text_lower = text.lower()
    for category in categories:
        if category in text_lower:
            # Check if it's actually referring to the category
            if (
                f"food_category" in text_lower
                and category in text_lower.split("food_category")[1][:50]
            ):
                return category
            if (
                f"category" in text_lower
                and category in text_lower.split("category")[1][:30]
            ):
                return category
            if (
                f"belongs to" in text_lower
                and category in text_lower.split("belongs to")[1][:30]
            ):
                return category

    # Look for category in potential JSON format
    category_pattern = r"food_category[\"']?\s*[:=]\s*[\"']([^\"']+)[\"']"
    match = re.search(category_pattern, text_lower)
    if match and match.group(1).strip() in categories:
        return match.group(1).strip()

    # Mapping broader categories to the specified ones
    category_mapping = {
        "meat": "pork",  # Default meat to pork unless specified
        "beef": "pork",
        "veal": "pork",
        "lamb": "pork",
        "vegetable": "vegetable",
        "vegetables": "vegetable",
        "produce": "vegetable",
        "dairy": "dairy",
        "milk": "dairy",
        "cheese": "dairy",
        "yogurt": "dairy",
        "poultry": "poultry",
        "bird": "poultry",
        "duck": "poultry",
        "turkey": "poultry",
        "fruit": "fruits",
        "fruits": "fruits",
        "pastry": "pastry",
        "baked": "pastry",
        "bread": "pastry",
        "cake": "pastry",
        "dessert": "pastry",
    }

    # Check for broader category mentions and map them
    for broader, specific in category_mapping.items():
        if broader in text_lower:
            return specific

    return "unknown"


def extract_food_condition(text):
    """Extract food condition as fresh, stale, or spoiled"""
    conditions = ["fresh", "stale", "spoiled"]
    text_lower = text.lower()

    # Check for direct condition statements in potential JSON format
    condition_pattern = r"food_condition[\"']?\s*[:=]\s*[\"']([^\"']+)[\"']"
    match = re.search(condition_pattern, text_lower)
    if match and match.group(1).strip() in conditions:
        return match.group(1).strip()

    # Direct mentions of condition
    for condition in conditions:
        if (
            f"food is {condition}" in text_lower
            or f"appears to be {condition}" in text_lower
            or f"condition: {condition}" in text_lower
            or f"condition is {condition}" in text_lower
        ):
            return condition

    # Count mentions of each condition
    condition_counts = {
        condition: text_lower.count(condition) for condition in conditions
    }
    if condition_counts:
        return max(condition_counts.items(), key=lambda x: x[1])[0]

    return "unknown"


def estimate_best_before_date(text):
    """Estimate the best before date based on the food condition"""
    # Try to find a date pattern in the text first (YYYY-MM-DD)
    date_pattern = r"(\d{4}-\d{2}-\d{2})"
    best_before_pattern = r"best[_\s]before[\"\']?\s*[:=]\s*[\"\']?([^\"\'\s]+)"

    # Look for a date in the best_before field first
    match = re.search(best_before_pattern, text.lower())
    if match:
        date_str = match.group(1).strip()
        # Check if it's in YYYY-MM-DD format
        if re.match(date_pattern, date_str):
            return date_str

    # Look for any date in the text
    match = re.search(date_pattern, text)
    if match:
        return match.group(1)

    # Current date
    current_date = datetime.now()

    # Logic based on food condition
    text_lower = text.lower()
    if "spoiled" in text_lower or "rotten" in text_lower:
        # Already spoiled - past date (only adjust days)
        best_before = datetime(
            current_date.year, current_date.month, max(1, current_date.day - 7)
        )
    elif "stale" in text_lower or "not fresh" in text_lower:
        # Stale - consume immediately
        best_before = current_date
    elif "fresh" in text_lower:
        # Fresh - estimate based on type (only adjust days)
        days_to_add = 3  # Default

        if any(
            food in text_lower
            for food in ["fruit", "fruits", "vegetable", "vegetables", "produce"]
        ):
            days_to_add = 5
        elif any(
            food in text_lower
            for food in ["meat", "pork", "beef", "poultry", "chicken"]
        ):
            days_to_add = 2
        elif any(food in text_lower for food in ["dairy", "milk", "yogurt"]):
            days_to_add = 7
        elif any(food in text_lower for food in ["pastry", "bread", "cake"]):
            days_to_add = 4

        # Calculate the new date while handling month/year transitions properly
        best_before = current_date + timedelta(days=days_to_add)
    else:
        best_before = current_date + timedelta(days=3)  # Default

    return best_before.strftime("%Y-%m-%d")


@app.route("/detect", methods=["GET"])
def detect_food():
    """API endpoint to capture a frame from webcam and analyze it"""
    if not initialize_camera():
        return (
            jsonify(
                {"error": "Camera not available. Please check your webcam connection."}
            ),
            500,
        )

    # Capture a frame
    frame = get_frame()
    if frame is None:
        return jsonify({"error": "Failed to capture image from webcam."}), 500

    # Analyze the captured frame
    result = analyze_food_image(frame)

    # Return the analysis result as JSON
    return jsonify(result)


@app.route("/test", methods=["GET"])
def test_connection():
    """Simple endpoint to test if the API is running"""
    return jsonify(
        {"status": "API is running", "timestamp": datetime.now().isoformat()}
    )


@app.route("/analyze_folder", methods=["GET"])
def analyze_folder():
    """Analyze all food images in the specified folder"""
    folder_path = request.args.get("folder", "images")  # Default to 'images' folder

    # Check if folder exists
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return (
            jsonify(
                {"error": f"Folder '{folder_path}' not found or is not a directory"}
            ),
            404,
        )

    # Get all image files in the folder
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    image_files = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
        and os.path.splitext(f)[1].lower() in image_extensions
    ]

    if not image_files:
        return jsonify({"error": f"No image files found in '{folder_path}'"}), 404

    # Process each image
    results = []
    for img_file in image_files:
        file_path = os.path.join(folder_path, img_file)
        try:
            # Read the image with OpenCV
            img = cv2.imread(file_path)
            if img is None:
                results.append(
                    {"filename": img_file, "error": "Failed to read image file"}
                )
                continue

            # Analyze the image
            analysis = analyze_food_image(img)

            # Add filename to the result
            analysis["filename"] = img_file
            results.append(analysis)

        except Exception as e:
            results.append(
                {"filename": img_file, "error": f"Error processing image: {str(e)}"}
            )

    return jsonify(
        {"folder": folder_path, "images_processed": len(results), "results": results}
    )


@app.route("/analyze_image/<filename>", methods=["GET"])
def analyze_single_image(filename):
    """Analyze a single food image from the images folder"""
    folder_path = request.args.get("folder", "images")  # Default to 'images' folder
    file_path = os.path.join(folder_path, filename)

    # Check if file exists
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return (
            jsonify({"error": f"Image file '{filename}' not found in '{folder_path}'"}),
            404,
        )

    try:
        # Read the image with OpenCV
        img = cv2.imread(file_path)
        if img is None:
            return jsonify({"error": f"Failed to read image file '{filename}'"}), 500

        # Analyze the image
        analysis = analyze_food_image(img)

        # Add filename to the result
        analysis["filename"] = filename

        return jsonify(analysis)

    except Exception as e:
        return (
            jsonify(
                {"filename": filename, "error": f"Error processing image: {str(e)}"}
            ),
            500,
        )


@app.route("/list_images", methods=["GET"])
def list_images():
    """List all image files in the images folder"""
    folder_path = request.args.get("folder", "images")  # Default to 'images' folder

    # Check if folder exists
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return (
            jsonify(
                {"error": f"Folder '{folder_path}' not found or is not a directory"}
            ),
            404,
        )

    # Get all image files in the folder
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    image_files = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
        and os.path.splitext(f)[1].lower() in image_extensions
    ]

    return jsonify(
        {"folder": folder_path, "image_count": len(image_files), "images": image_files}
    )


@app.teardown_appcontext
def cleanup(exception=None):
    """Cleanup resources when the application shuts down"""
    release_camera()


if __name__ == "__main__":
    # Clean up camera resources on exit
    try:
        app.run(debug=True)
    finally:
        release_camera()
