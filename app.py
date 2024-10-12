from flask import Flask, request, jsonify
from PIL import Image  # Để xử lý ảnh
import tensorflow as tf  # Sử dụng TensorFlow
from io import BytesIO
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from flask_cors import CORS

app = Flask(__name__)
# Them cors vao flask app
CORS(app)
# Tải mô hình đã huấn luyện
model_path = "model/base_model_trained.h5"
model = tf.keras.models.load_model(model_path)

# Tai du lieu file csv data recommend40Food
file_path = "data/40FoodRec.csv"
recipe_df = pd.read_csv(file_path)

print(recipe_df.shape)

# Preprocess Ingredients
vectorizer = TfidfVectorizer()
X_ingredients = vectorizer.fit_transform(recipe_df["ingredients_en"])

# Full list of numerical columns
numerical_columns = [
    "carbohydrates",
    "protein",
    "cholesterol",
    "sodium",
    "fiber",
]

# Normalize numerical features
scaler = StandardScaler()


# Định nghĩa route nhận ảnh
@app.route("/recognize", methods=["POST"])
def recognize_image():
    print("0ke")
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    # Lấy ảnh từ request
    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # List class dish
    classes = [
        "Banh Beo",
        "Banh Bot Loc",
        "Banh Can",
        "Banh Canh",
        "Banh Chung",
        "Banh Cuon",
        "Banh Duc",
        "Banh Gio",
        "Banh Khot",
        "Banh Mi",
        "Banh Pia",
        "Banh Tet",
        "Banh Trang Nuong",
        "Banh Xeo",
        "Hue Beef Noodles",
        "Vermicelli with Fried Tofu and Fermented Shrimp Paste",
        "Fermented Fish Noodle Soup",
        "Crab Noodle Soup",
        "Grilled Pork Noodles",
        "Bánh Cu Dơ",
        "Bánh Dau Xanh",
        "Braised Fish",
        "Sour Soup",
        "Cao Lau",
        "Liver Porridge",
        "Broken Rice",
        "Crispy Rice",
        "Spring Rolls",
        "Hu Tieu",
        "Mi Quang",
        "Fermented Pork",
        "Grilled Pork Sausage",
        "Pho",
        "Sticky Rice with Mung Beans",
        "Banh Bo",
        "Banh Cong",
        "Pork Skin Cake",
        "Pig Ear Cake",
        "Banh Tieu",
        "Moon Cake",
    ]

    # Chuyển ảnh sang định dạng mà mô hình cần (ví dụ: mảng numpy)
    try:
        image = Image.open(file)
        image_array = preprocess_image(image)  # Hàm xử lý ảnh

        # Thực hiện dự đoán
        prediction = model.predict(image_array)

        # Lấy kết quả dự đoán
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Trả về kết quả dự đoán
        return jsonify({"prediction": classes[int(predicted_class)]})
        # print(classes[int(predicted_class)])
        # print(int(predicted_class))
        # return jsonify({"prediction": int(predicted_class)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Function to recommend recipes based on input features and ingredients
@app.route("/recommend", methods=["POST"])
def recommend_recipes():
    data = request.get_json()
    print(data)
    input_features = data.get("input_features", {})
    list_ingredients = data.get("list_ingredients", [])
    # Construct the model and get valid columns
    knn, valid_columns = construct_model(input_features, list_ingredients)

    # Scale the input numerical features for the valid columns
    valid_numerical_input = [input_features[col] for col in valid_columns]

    if valid_numerical_input:
        input_numerical_scaled = scaler.transform([valid_numerical_input])
    else:
        input_numerical_scaled = np.array([]).reshape(
            1, 0
        )  # Empty array if no valid numerical features

    # Transform the ingredients list for the input
    input_ingredients_transformed = vectorizer.transform([", ".join(list_ingredients)])

    # Combine the numerical and ingredient features for the input
    if input_numerical_scaled.size > 0:
        input_combined = np.hstack(
            [input_numerical_scaled, input_ingredients_transformed.toarray()]
        )
    else:
        input_combined = input_ingredients_transformed.toarray()  # Only ingredients

    # Get recommendations using the KNN model
    distances, indices = knn.kneighbors(input_combined)

    # Fetch and return recommendations
    recommendations = recipe_df.iloc[indices[0]]
    # result = recommendations[["id"]].to_dict(orient="records")

    # return list id recommend food
    result = recommendations["id"].tolist()
    return jsonify(result)


def preprocess_image(image):
    """
    Hàm xử lý ảnh để chuyển đổi từ PIL Image sang mảng numpy phù hợp với mô hình TensorFlow.
    """
    image = image.resize(
        (224, 224)
    )  # Resize ảnh về kích thước 224x224 (tuỳ theo mô hình yêu cầu)
    image_array = np.array(image)  # Chuyển đổi ảnh sang mảng numpy
    image_array = image_array / 255.0  # Chuẩn hóa giá trị pixel về khoảng [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Thêm batch dimension
    return image_array


# Function to construct the model by dynamically removing zero-value features
def construct_model(input_features, list_ingredients):
    # Determine which numerical features are non-zero
    valid_numerical = {k: v for k, v in input_features.items() if v != 0}

    # Get the valid columns based on the non-zero features
    valid_columns = list(valid_numerical.keys())

    # If there are valid columns, filter the dataset based on those columns
    if valid_columns:
        X_numerical_filtered = scaler.fit_transform(recipe_df[valid_columns])
    else:
        X_numerical_filtered = np.array([]).reshape(
            len(recipe_df), 0
        )  # Empty array if all inputs are zero

    # Process the ingredient list
    X_ingredients_transformed = vectorizer.fit_transform(recipe_df["ingredients_en"])

    # Combine the filtered numerical features and the ingredient features
    if X_numerical_filtered.size > 0:
        X_combined = np.hstack([X_numerical_filtered, X_ingredients.toarray()])
    else:
        X_combined = (
            X_ingredients.toarray()
        )  # Only ingredients if no numerical features are present

    # Re-train the KNN model
    knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
    knn.fit(X_combined)

    return knn, valid_columns


if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)
