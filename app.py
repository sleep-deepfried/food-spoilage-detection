from flask import Flask, request, jsonify, session
from google import genai
import PIL.Image
import os
import json
import cv2
import io
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv
from datetime import datetime, timedelta
import uuid
from functools import wraps
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", os.urandom(24))  # Add secret key for sessions

# Initialize the Gemini client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Global camera variable
camera = None

# In-memory temporary storage for detection results with timestamp
detection_cache = {}

# Database configuration
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT", "5432")  # Default PostgreSQL port


def get_db_connection():
    """Create and return a database connection to PostgreSQL"""
    return psycopg2.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME,
        port=DB_PORT,
        cursor_factory=DictCursor,
    )


def initialize_camera():
    """Initialize the webcam"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)  # 0 is usually the default webcam
    return camera is not None and camera.isOpened()


def release_camera():
    """Release the webcam resources"""
    global camera
    if camera is not None:
        camera.release()
        camera = None


def get_frame():
    """Get a frame from the webcam"""
    global camera
    if camera is not None and camera.isOpened():
        success, frame = camera.read()
        if success:
            return frame
    return None


def analyze_food_image(image_array):
    """
    Analyzes a food image using Gemini to detect food type and condition,
    prioritizing Filipino food names

    Args:
        image_array: The image as a numpy array

    Returns:
        dict: Structured analysis of the food
    """
    try:
        # Convert OpenCV BGR image to RGB for PIL
        rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = PIL.Image.fromarray(rgb_image)

        # Create a BytesIO object
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)

        # Load image from BytesIO
        image = PIL.Image.open(img_byte_arr)

        # Create a prompt that asks for structured output with Filipino food names
        prompt = """
        Please analyze this food image and provide the following information in JSON format:
        {
          "food_name": "The specific name of the food shown in the image IN FILIPINO language (e.g., Adobo, Sinigang, Lechon, Pancit). If not a Filipino dish, provide the closest Filipino equivalent or translation.",
          "english_name": "The English name or closest equivalent of the food",
          "food_category": "One of: pork, chicken, vegetable, dairy, poultry, fruits, pastry", 
          "food_condition": "One of: fresh, spoiled"
        }
        
        Be specific and base your analysis on visual indicators.
        If you cannot identify food in the image, use "unknown" for the values.
        
        IMPORTANT: Priority should be given to identifying the food with Filipino names.
        """

        # Send request to Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=[prompt, image]
        )

        # Extract the response text
        response_text = response.text

        # Find and parse JSON from the response
        try:
            # Extract JSON object from response text
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1

            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                food_info = json.loads(json_str)
            else:
                # Fallback if JSON parsing fails
                food_info = {
                    "food_name": "hindi kilala",  # "unknown" in Filipino
                    "english_name": "unknown",
                    "food_category": "unknown",
                    "food_condition": "unknown",
                }

            # Add date_added field
            food_info["date_added"] = datetime.now().strftime("%Y-%m-%d")
            # Set timestamp for cache management
            food_info["timestamp"] = datetime.now().isoformat()
            # Set confidence to high if we got a proper detection
            food_info["confidence"] = (
                0.95 if food_info["food_name"] != "hindi kilala" else 0.1
            )

            return food_info

        except json.JSONDecodeError:
            return {
                "food_name": "hindi kilala",  # "unknown" in Filipino
                "english_name": "unknown",
                "food_category": "unknown",
                "food_condition": "unknown",
                "date_added": datetime.now().strftime("%Y-%m-%d"),
                "timestamp": datetime.now().isoformat(),
                "confidence": 0.1,
                "error": "Could not parse response",
            }

    except Exception as e:
        return {"error": f"Error analyzing image: {str(e)}"}


def add_food_to_db(food_data, quantity=1):
    """
    Add a food item to the database

    Args:
        food_data: Dictionary containing food details
        quantity: Quantity to add (default 1)

    Returns:
        dict: Result of the operation
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Calculate best before date (default 7 days for most items)
        entry_date = datetime.strptime(food_data["date_added"], "%Y-%m-%d")

        # Set different expiration times based on food category
        expiry_days = {
            "fruits": 5,
            "vegetable": 4,
            "dairy": 7,
            "pork": 3,
            "chicken": 3,
            "poultry": 3,
            "pastry": 5,
        }

        days = expiry_days.get(food_data["food_category"], 7)
        best_before = (entry_date + timedelta(days=days)).strftime("%Y-%m-%d")

        # Check if food already exists
        cursor.execute(
            "SELECT inventoryid, quantity FROM public.food_inventory WHERE food_name = %s AND food_type = %s",
            (food_data["food_name"], food_data["food_category"]),
        )
        existing_food = cursor.fetchone()

        if existing_food:
            # Update quantity
            cursor.execute(
                "UPDATE public.food_inventory SET quantity = quantity + %s WHERE inventoryid = %s",
                (quantity, existing_food["inventoryid"]),
            )
            result = {
                "status": "updated",
                "new_quantity": existing_food["quantity"] + quantity,
            }
        else:
            # Insert new food
            cursor.execute(
                """INSERT INTO public.food_inventory 
                   (food_name, food_type, entry_date, best_before, confidence, quantity) 
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (
                    food_data["food_name"],
                    food_data["food_category"],
                    food_data["date_added"],
                    best_before,
                    food_data["confidence"],
                    quantity,
                ),
            )
            result = {"status": "added", "quantity": quantity}

        conn.commit()
        return result

    except Exception as e:
        if conn:
            conn.rollback()
        return {"error": f"Database error: {str(e)}"}

    finally:
        if conn:
            conn.close()


def remove_food_from_db(food_data, quantity=1):
    """
    Remove a food item from the database

    Args:
        food_data: Dictionary containing food details
        quantity: Quantity to remove (default 1)

    Returns:
        dict: Result of the operation
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if food exists
        cursor.execute(
            "SELECT inventoryid, quantity FROM public.food_inventory WHERE food_name = %s AND food_type = %s",
            (food_data["food_name"], food_data["food_category"]),
        )
        existing_food = cursor.fetchone()

        if not existing_food:
            return {"status": "error", "message": "Food not found in inventory"}

        if existing_food["quantity"] <= quantity:
            # Remove the entire entry
            cursor.execute(
                "DELETE FROM public.food_inventory WHERE inventoryid = %s",
                (existing_food["inventoryid"],),
            )
            result = {
                "status": "removed",
                "message": "Food completely removed from inventory",
            }
        else:
            # Decrease quantity
            cursor.execute(
                "UPDATE public.food_inventory SET quantity = quantity - %s WHERE inventoryid = %s",
                (quantity, existing_food["inventoryid"]),
            )
            result = {
                "status": "updated",
                "remaining_quantity": existing_food["quantity"] - quantity,
            }

        conn.commit()
        return result

    except Exception as e:
        if conn:
            conn.rollback()
        return {"error": f"Database error: {str(e)}"}

    finally:
        if conn:
            conn.close()


def requires_detection(f):
    """Decorator to ensure detection was done first"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Log all request data for debugging
        logger.debug(f"Request headers: {request.headers}")
        logger.debug(f"Request form: {request.form}")
        logger.debug(f"Request JSON: {request.get_json(silent=True)}")
        logger.debug(f"Current detection cache keys: {list(detection_cache.keys())}")

        # Try to get detection_id from different sources
        detection_id = None

        # Check if JSON in request
        if request.is_json:
            detection_id = request.json.get("detection_id")

        # If not found in JSON, check form data
        if not detection_id and request.form:
            detection_id = request.form.get("detection_id")

        # If still not found, check query parameters
        if not detection_id:
            detection_id = request.args.get("detection_id")

        logger.debug(f"Extracted detection_id: {detection_id}")

        if not detection_id:
            return (
                jsonify(
                    {
                        "error": "No detection ID provided. Please include 'detection_id' in your request.",
                        "status": "error",
                    }
                ),
                400,
            )

        if detection_id not in detection_cache:
            return (
                jsonify(
                    {
                        "error": "No valid detection found. Please use /detect endpoint first.",
                        "status": "error",
                        "available_ids": list(detection_cache.keys()),  # For debugging
                    }
                ),
                400,
            )

        return f(*args, **kwargs)

    return decorated_function


@app.route("/detect", methods=["POST"])
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

    # Generate a unique ID for this detection
    detection_id = str(uuid.uuid4())
    logger.debug(f"Generated new detection_id: {detection_id}")

    # Store the result in the cache
    detection_cache[detection_id] = result
    logger.debug(
        f"Detection cache updated. Current keys: {list(detection_cache.keys())}"
    )

    # Add detection_id to the result
    result["detection_id"] = detection_id

    # Create a proper response structure with detections array
    response = {
        "status": "success",
        "detections": [
            # Include the current detection
            result
        ],
    }

    # If you want to include previous detections (optional, limited to most recent 5)
    # This shows most recent detections first
    previous_detections = []
    for prev_id, prev_data in list(detection_cache.items())[-5:]:
        if prev_id != detection_id:  # Don't duplicate the current detection
            prev_data_copy = prev_data.copy()
            prev_data_copy["detection_id"] = prev_id
            previous_detections.append(prev_data_copy)

    # Add previous detections to the response if there are any
    if previous_detections:
        response["detections"].extend(previous_detections)

    # Return analysis result with all detections
    return jsonify(response)


@app.route("/detection_status", methods=["GET"])
def detection_status():
    """API endpoint to check detection cache status (for debugging)"""
    return jsonify(
        {
            "detection_cache_size": len(detection_cache),
            "detection_ids": list(detection_cache.keys()),
        }
    )


@app.route("/add", methods=["POST", "GET"])
@requires_detection
def add_food():
    """API endpoint to add food to inventory"""
    try:
        # Get detection_id from various sources
        detection_id = None
        quantity = 1

        if request.is_json:
            data = request.json
            detection_id = data.get("detection_id")
            quantity = data.get("quantity", 1)
        elif request.form:
            detection_id = request.form.get("detection_id")
            quantity = int(request.form.get("quantity", 1))
        else:
            detection_id = request.args.get("detection_id")
            quantity = int(request.args.get("quantity", 1))

        # Get food data from the detection cache
        food_data = detection_cache.get(detection_id, {})
        if not food_data:
            return (
                jsonify({"error": "Detection data not found", "status": "error"}),
                404,
            )

        # Log the data we're using
        logger.debug(f"Using detection_id: {detection_id}")
        logger.debug(f"Food data: {food_data}")
        logger.debug(f"Quantity: {quantity}")

        # Add food to database
        result = add_food_to_db(food_data, quantity)

        # Clean up the cache after successful operation
        if result.get("status") not in ["error"]:
            detection_cache.pop(detection_id, None)
            logger.debug(
                f"Detection {detection_id} removed from cache after successful add"
            )

        return jsonify(
            {
                "status": "success",
                "operation": "add",
                "food": food_data,
                "result": result,
            }
        )

    except Exception as e:
        logger.exception("Error in add_food endpoint")
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/remove", methods=["POST", "GET"])
@requires_detection
def remove_food():
    """API endpoint to remove food from inventory"""
    try:
        # Get detection_id from various sources
        detection_id = None
        quantity = 1

        if request.is_json:
            data = request.json
            detection_id = data.get("detection_id")
            quantity = data.get("quantity", 1)
        elif request.form:
            detection_id = request.form.get("detection_id")
            quantity = int(request.form.get("quantity", 1))
        else:
            detection_id = request.args.get("detection_id")
            quantity = int(request.args.get("quantity", 1))

        # Get food data from the detection cache
        food_data = detection_cache.get(detection_id, {})
        if not food_data:
            return (
                jsonify({"error": "Detection data not found", "status": "error"}),
                404,
            )

        # Log the data we're using
        logger.debug(f"Using detection_id: {detection_id}")
        logger.debug(f"Food data: {food_data}")
        logger.debug(f"Quantity: {quantity}")

        # Remove food from database
        result = remove_food_from_db(food_data, quantity)

        # Clean up the cache after successful operation
        if result.get("status") not in ["error"]:
            detection_cache.pop(detection_id, None)
            logger.debug(
                f"Detection {detection_id} removed from cache after successful remove"
            )

        return jsonify(
            {
                "status": "success",
                "operation": "remove",
                "food": food_data,
                "result": result,
            }
        )

    except Exception as e:
        logger.exception("Error in remove_food endpoint")
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/test", methods=["GET"])
def test_connection():
    """Simple endpoint to test if the API is running"""
    return jsonify(
        {
            "status": "API is running",
            "timestamp": datetime.now().isoformat(),
            "detection_cache_size": len(detection_cache),
        }
    )


@app.teardown_appcontext
def cleanup(exception=None):
    """Cleanup resources when the application shuts down"""
    release_camera()


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=True)
    finally:
        release_camera()
