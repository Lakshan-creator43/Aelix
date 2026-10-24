from pymongo import MongoClient

# -------------------------------
# Connect to MongoDB
# -------------------------------
client = MongoClient("YOUR_MONGODB_CONNECTION_STRING")

# Create Database
db = client["StudentDB"]

# Create Collection
collection = db["Students"]


# -------------------------------
# CREATE
# -------------------------------
def create_student():
    student = {
        "name": "John",
        "age": 20,
        "course": "Computer Science"
    }

    result = collection.insert_one(student)
    print("Student Added!")
    print("ID:", result.inserted_id)


# -------------------------------
# READ
# -------------------------------
def read_students():
    print("\nStudent List:\n")

    for student in collection.find():
        print(student)


# -------------------------------
# UPDATE
# -------------------------------
def update_student():
    collection.update_one(
        {"name": "John"},
        {"$set": {"age": 21}}
    )

    print("Student Updated!")


# -------------------------------
# DELETE
# -------------------------------
def delete_student():
    collection.delete_one({"name": "John"})
    print("Student Deleted!")


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