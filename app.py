from flask import Flask, request, jsonify
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

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Initialize the Gemini client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Global camera variable
camera = None

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
    # Close the camera after taking the picture
    release_camera()
    if frame is None:
        return jsonify({"error": "Failed to capture image from webcam."}), 500
    
    # Analyze the captured frame
    result = analyze_food_image(frame)
    
    # Return analysis result
    return jsonify(result)

@app.route("/add", methods=["POST"])
def add_food():
    """API endpoint to add food to inventory"""
    try:
        # Get food data from request
        if request.is_json:
            food_data = request.json
        else:
           # If no JSON provided, try to detect the food
            try:
                if not initialize_camera():
                    return jsonify({"error": "Camera not available"}), 500
    
                frame = get_frame()
                if frame is None:
                    return jsonify({"error": "Failed to capture image"}), 500
    
                food_data = analyze_food_image(frame)
            finally:
                # Close the camera after taking the picture
                release_camera()

        # Extract quantity if provided, default to 1
        quantity = (
            food_data.pop("quantity", 1)
            if isinstance(food_data, dict) and "quantity" in food_data
            else 1
        )

        # Add food to database
        result = add_food_to_db(food_data, quantity)

        return jsonify(
            {
                "status": "success",
                "operation": "add",
                "food": food_data,
                "result": result,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/remove", methods=["POST"])
def remove_food():
    """API endpoint to remove food from inventory"""
    try:
        # Get food data from request
        if request.is_json:
            food_data = request.json
        else:
           # If no JSON provided, try to detect the food
            try:
                if not initialize_camera():
                    return jsonify({"error": "Camera not available"}), 500
    
                frame = get_frame()
                if frame is None:
                    return jsonify({"error": "Failed to capture image"}), 500
    
                food_data = analyze_food_image(frame)
            finally:
                # Close the camera after taking the picture
                release_camera()

        # Extract quantity if provided, default to 1
        quantity = (
            food_data.pop("quantity", 1)
            if isinstance(food_data, dict) and "quantity" in food_data
            else 1
        )

        # Remove food from database
        result = remove_food_from_db(food_data, quantity)

        return jsonify(
            {
                "status": "success",
                "operation": "remove",
                "food": food_data,
                "result": result,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/test", methods=["GET"])
def test_connection():
    """Simple endpoint to test if the API is running"""
    return jsonify(
        {"status": "API is running", "timestamp": datetime.now().isoformat()}
    )


@app.teardown_appcontext
def cleanup(exception=None):
    """Cleanup resources when the application shuts down"""
    release_camera()


if __name__ == "__main__":
    try:
        app.run(debug=True)
    finally:
        release_camera()