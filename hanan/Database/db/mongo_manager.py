from bson import ObjectId
from pymongo import MongoClient
import json

class MongoManager:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            config = json.load(file)
        self.client = MongoClient(config["mongo_uri"])
        self.db = self.client[config["database_name"]]

    def store_user_data(self, user_data):
        collection = self.db['users']
        result = collection.insert_one(user_data)
        return result.inserted_id

    def get_user_data(self, user_id):
        collection = self.db['users']
        user_data = collection.find_one({"_id": ObjectId(user_id)})
        return user_data