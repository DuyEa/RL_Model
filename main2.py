from recommend2.preprocessing import preprocess_data
from recommend2.train_model import train_model
from recommend2.session_based_recommend import recommend_products
import os
import random

# Path to CSV dataset
csv_file_path = "dataset.csv"

if not os.path.exists(csv_file_path):
    raise FileNotFoundError(f"Dataset file not found at {csv_file_path}")

print("Preprocessing data...")
df, df_weighted = preprocess_data(csv_file_path)

model_file = "model.pkl"

if not os.path.exists(model_file):
    print("Training model...")
    train_model(df_weighted, model_file)
else:
    print(f"Model already exists at {model_file}. Skipping training.")

# Log how many unique items we have
unique_items = df["product_id"].unique()
print(f"Dataset loaded with {len(unique_items)} unique items.")

# User ID input only once
target_user_id = int(input("Enter User ID: "))
if target_user_id < 0:
    raise ValueError("User ID cannot be negative.")

while True:
    typed_item_id = int(input("Enter Item ID (or -1 to exit): "))
    if typed_item_id == -1:
        print("Exiting recommendation system.")
        break

    event_type = input("Enter Event Type (view/cart/purchase): ").lower()

    # Basic validation of event_type
    if event_type not in ["view", "cart", "purchase"]:
        print("Invalid event type. Please enter 'view', 'cart', or 'purchase'.")
        continue

    # Check if typed_item_id is in the dataset
    if typed_item_id not in unique_items:
        print(f"Item ID {typed_item_id} does NOT exist in the dataset.")
        print("Falling back to a random item ID from the dataset so we don't get empty candidates.")
        target_product_id = random.choice(unique_items)
        print(f"Using random item: {target_product_id}")
    else:
        target_product_id = typed_item_id

    print("Generating recommendations...")
    recommendations = recommend_products(
        model_file=model_file,
        df=df,
        target_user_id=target_user_id,
        target_product_id=target_product_id,
        event_type=event_type
    )

    if recommendations.empty:
        print("Top Recommendations: [Empty DataFrame]")
    else:
        print("Top Recommendations:")
        print(recommendations[["product_id", "category_code", "brand"]])
