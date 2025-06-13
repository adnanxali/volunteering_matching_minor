import requests
import json

BASE_URL = "http://127.0.0.1:3001"

# Sample volunteer data
volunteer = {
    "name": "John Doe",
    "email": "john@example.com",
    "skills": ["teaching", "first aid", "event management"],
    "location": {
        "latitude": 28.7041,
        "longitude": 77.1025
    }
}

# Sample list of projects
projects = [
    {
        "title": "Community Teaching Program",
        "description": "Teach children in underprivileged areas.",
        "skillsReq": ["teaching", "patience"],
        "location": {
            "label": "Delhi",
            "latitude": 28.7041,
            "longitude": 77.1025
        }
    },
    {
        "title": "Disaster Relief Team",
        "description": "Provide aid in natural disasters.",
        "skillsReq": ["first aid", "physical stamina"],
        "location": {
            "label": "Mumbai",
            "latitude": 19.0760,
            "longitude": 72.8777
        }
    },
    {
        "title": "Cultural Event Planner",
        "description": "Organize local cultural events.",
        "skillsReq": ["event management", "communication"],
        "location": {
            "label": "Jaipur",
            "latitude": 26.9124,
            "longitude": 75.7873
        }
    },
    {
        "title": "Tree Plantation Drive",
        "description": "Help plant trees and spread awareness.",
        "skillsReq": ["gardening", "teamwork"],
        "location": {
            "label": "Chandigarh",
            "latitude": 30.7333,
            "longitude": 76.7794
        }
    }
]

# Send request to the recommendation API
def test_recommendation():
    data = {
        "volunteer": volunteer,
        "projects": projects
    }
    response = requests.post(f"{BASE_URL}/api/recommend", json=data)
    print("Recommendation Results:")
    try:
        print(json.dumps(response.json(), indent=4))
    except Exception as e:
        print("Failed to parse JSON:", e)
        print(response.text)

if __name__ == "__main__":
    test_recommendation()
