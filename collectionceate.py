from pymongo import MongoClient

# Replace with your connection string
client = MongoClient("mongodb+srv://rebeccaadrian13:pass45edx@cluster0.mongodb.net/?retryWrites=true&w=majority")

try:
    # Access the database
    db = client["epidemic_predictions"]
    print("Connected to MongoDB Atlas successfully!")
except Exception as e:
    print(f"Error: {e}")