from recommend2.preprocessing import preprocess_data
from recommend2.train_model import train_model
from recommend2.session_based_recommend import recommend_products
import os

# Step 1: Specify the CSV file path directly
csv_file_path = "part-00000-782d7740-2eee-412f-9c1e-98a0ef8c80fe-c000.csv"

if not os.path.exists(csv_file_path):
    raise FileNotFoundError(f"Dataset file not found at {csv_file_path}")

# Step 2: Preprocess data
print("Preprocessing data...")
df, df_weighted = preprocess_data(csv_file_path)

# Step 3: Train model only if it doesn't already exist
model_file = "model.pkl"

if not os.path.exists(model_file):
    print("Training model...")
    train_model(df_weighted, model_file)
else:
    print(f"Model already exists at {model_file}. Skipping training.")

# Step 5: Generate recommendations
print("Generating recommendations...")
# target_user_id = 4811111111  # cold-start user
target_user_id = 1
target_product_id = 1
event_type = input("Enter Event Type (view/cart/purchase): ").lower()

recommendations = recommend_products(model_file, df, target_user_id, target_product_id, event_type= event_type)

# Display recommendations
print("Top 10 Recommendations:")
print(recommendations)
