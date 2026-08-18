import os
from pymongo import MongoClient
from pymongo.errors import PyMongoError, ConfigurationError

# -------------------------------
# Connect to MongoDB
# -------------------------------
mongo_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
try:
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
    db = client["StudentDB"]
    collection = db["Students"]
except (PyMongoError, ConfigurationError) as e:
    print(f"MongoDB Connection Warning: {e}")
    client = None
    db = None
    collection = None



# -------------------------------
# CREATE
# -------------------------------
def create_student():
    if collection is None:
        print("MongoDB connection unavailable. Set MONGODB_URI to connect.")
        return
    student = {
        "name": "John",
        "age": 20,
        "course": "Computer Science"
    }

    try:
        result = collection.insert_one(student)
        print("Student Added!")
        print("ID:", result.inserted_id)
    except Exception as e:
        print(f"Error creating student: {e}")


# -------------------------------
# READ
# -------------------------------
def read_students():
    if collection is None:
        print("MongoDB connection unavailable. Set MONGODB_URI to connect.")
        return
    print("\nStudent List:\n")

    try:
        for student in collection.find():
            print(student)
    except Exception as e:
        print(f"Error reading students: {e}")


# -------------------------------
# UPDATE
# -------------------------------
def update_student():
    if collection is None:
        print("MongoDB connection unavailable. Set MONGODB_URI to connect.")
        return
    try:
        collection.update_one(
            {"name": "John"},
            {"$set": {"age": 21}}
        )
        print("Student Updated!")
    except Exception as e:
        print(f"Error updating student: {e}")


# -------------------------------
# DELETE
# -------------------------------
def delete_student():
    if collection is None:
        print("MongoDB connection unavailable. Set MONGODB_URI to connect.")
        return
    try:
        collection.delete_one({"name": "John"})
        print("Student Deleted!")
    except Exception as e:
        print(f"Error deleting student: {e}")



# -------------------------------
# MENU
# -------------------------------
while True:

    print("\n===== CRUD MENU =====")
    print("1. Create")
    print("2. Read")
    print("3. Update")
    print("4. Delete")
    print("5. Exit")

    choice = input("Enter choice: ")

    if choice == "1":
        create_student()

    elif choice == "2":
        read_students()

    elif choice == "3":
        update_student()

    elif choice == "4":
        delete_student()

    elif choice == "5":
        print("Program Ended")
        break

    else:
        print("Invalid Choice")