# Training data for volunteer-project recommender system
# This data emphasizes location proximity while also considering skill matching

import json

# Create volunteers with different skills and locations
volunteers = [
    {
        "id": "v1",
        "name": "Alex Johnson",
        "skills": ["python", "data analysis", "web development"],
        "interest": "machine learning",
        "location": {"lat": 37.7749, "lng": -122.4194}  # San Francisco
    },
    {
        "id": "v2",
        "name": "Priya Sharma",
        "skills": ["graphic design", "marketing", "social media"],
        "interest": "community outreach",
        "location": {"lat": 40.7128, "lng": -74.0060}  # New York
    },
    {
        "id": "v3",
        "name": "Marcus Lee",
        "skills": ["teaching", "tutoring", "mathematics"],
        "interest": "education",
        "location": {"lat": 34.0522, "lng": -118.2437}  # Los Angeles
    },
    {
        "id": "v4",
        "name": "Fatima Ali",
        "skills": ["healthcare", "first aid", "elderly care"],
        "interest": "nursing",
        "location": {"lat": 29.7604, "lng": -95.3698}  # Houston
    },
    {
        "id": "v5",
        "name": "Jordan Taylor",
        "skills": ["gardening", "sustainability", "community organizing"],
        "interest": "environmental conservation",
        "location": {"lat": 47.6062, "lng": -122.3321}  # Seattle
    }
]

# Create projects with different skills and locations
projects = [
    # San Francisco area projects (close to volunteer 1)
    {
        "id": "p1",
        "name": "Tech for Good",
        "description": "Developing apps for local nonprofits",
        "skillsReq": ["python", "web development", "mobile app development"],
        "location": {"lat": 37.7849, "lng": -122.4294}  # 1km from v1
    },
    {
        "id": "p2",
        "name": "Data Analysis Workshop",
        "description": "Teaching data skills to underserved communities",
        "skillsReq": ["data analysis", "teaching", "statistics"],
        "location": {"lat": 37.7649, "lng": -122.4194}  # 1.1km from v1
    },
    {
        "id": "p3",
        "name": "AI Ethics Committee",
        "description": "Discussing ethical implications of AI",
        "skillsReq": ["machine learning", "ethics", "public speaking"],
        "location": {"lat": 37.7749, "lng": -122.3994}  # 1.7km from v1
    },
    {
        "id": "p4",
        "name": "Coding Bootcamp",
        "description": "Teaching coding to beginners",
        "skillsReq": ["python", "web development", "teaching"],
        "location": {"lat": 37.7849, "lng": -122.4094}  # 1.2km from v1
    },
    
    # New York area projects (close to volunteer 2)
    {
        "id": "p5",
        "name": "Nonprofit Branding",
        "description": "Creating visual identities for nonprofits",
        "skillsReq": ["graphic design", "marketing", "branding"],
        "location": {"lat": 40.7228, "lng": -74.0060}  # 1.1km from v2
    },
    {
        "id": "p6",
        "name": "Social Media Campaign",
        "description": "Raising awareness for homeless shelters",
        "skillsReq": ["social media", "content creation", "marketing"],
        "location": {"lat": 40.7128, "lng": -74.0160}  # 0.83km from v2
    },
    
    # Los Angeles area projects (close to volunteer 3)
    {
        "id": "p7",
        "name": "Math Tutoring Program",
        "description": "Tutoring underprivileged students",
        "skillsReq": ["mathematics", "tutoring", "education"],
        "location": {"lat": 34.0622, "lng": -118.2437}  # 1.1km from v3
    },
    {
        "id": "p8",
        "name": "STEM Workshop",
        "description": "Introducing STEM to middle school students",
        "skillsReq": ["teaching", "science", "mathematics"],
        "location": {"lat": 34.0522, "lng": -118.2637}  # 1.7km from v3
    },
    
    # Houston area projects (close to volunteer 4)
    {
        "id": "p9",
        "name": "Senior Care Center",
        "description": "Assisting elderly residents",
        "skillsReq": ["elderly care", "healthcare", "compassion"],
        "location": {"lat": 29.7704, "lng": -95.3698}  # 1.1km from v4
    },
    {
        "id": "p10",
        "name": "First Aid Training",
        "description": "Teaching first aid to community members",
        "skillsReq": ["first aid", "teaching", "healthcare"],
        "location": {"lat": 29.7604, "lng": -95.3798}  # 0.9km from v4
    },
    
    # Seattle area projects (close to volunteer 5)
    {
        "id": "p11",
        "name": "Community Garden",
        "description": "Building and maintaining community gardens",
        "skillsReq": ["gardening", "community organizing", "sustainability"],
        "location": {"lat": 47.6162, "lng": -122.3321}  # 1.1km from v5
    },
    {
        "id": "p12",
        "name": "Environmental Cleanup",
        "description": "Organizing beach and park cleanups",
        "skillsReq": ["environmental conservation", "community organizing", "project management"],
        "location": {"lat": 47.6062, "lng": -122.3421}  # 0.83km from v5
    },
    
    # Projects far from any volunteer but with matching skills
    {
        "id": "p13",
        "name": "Remote Data Analysis",
        "description": "Analyzing data for international NGO",
        "skillsReq": ["python", "data analysis", "statistics"],
        "location": {"lat": 51.5074, "lng": -0.1278}  # London (far from v1)
    },
    {
        "id": "p14",
        "name": "Virtual Tutoring",
        "description": "Online tutoring for rural students",
        "skillsReq": ["teaching", "mathematics", "online communication"],
        "location": {"lat": 19.0760, "lng": 72.8777}  # Mumbai (far from v3)
    },
    
    # Projects close to volunteers but with non-matching skills
    {
        "id": "p15",
        "name": "SF Symphony Volunteer",
        "description": "Helping with music events",
        "skillsReq": ["music", "event planning", "customer service"],
        "location": {"lat": 37.7749, "lng": -122.4205}  # Close to v1 but skills don't match
    },
    {
        "id": "p16",
        "name": "NY Legal Aid",
        "description": "Administrative support for legal clinic",
        "skillsReq": ["legal", "administrative", "database management"],
        "location": {"lat": 40.7138, "lng": -74.0070}  # Close to v2 but skills don't match
    }
]

# Create interaction data
# 1 = matched/successful, 0 = not matched/unsuccessful
interactions = [
    # Volunteer 1 (Alex - SF) successful matches - close by AND skill match
    {"volunteer_id": "v1", "project_id": "p1", "matched": 1},  # High skill match, close distance
    {"volunteer_id": "v1", "project_id": "p2", "matched": 1},  # High skill match, close distance
    {"volunteer_id": "v1", "project_id": "p3", "matched": 1},  # Matches interest, close distance
    {"volunteer_id": "v1", "project_id": "p4", "matched": 1},  # High skill match, close distance
    {"volunteer_id": "v1", "project_id": "p13", "matched": 0.5},  # High skill match, far distance
    {"volunteer_id": "v1", "project_id": "p15", "matched": 0.3},  # Low skill match, close distance
    
    # Volunteer 2 (Priya - NY) successful matches
    {"volunteer_id": "v2", "project_id": "p5", "matched": 1},  # High skill match, close distance
    {"volunteer_id": "v2", "project_id": "p6", "matched": 1},  # High skill match, close distance
    {"volunteer_id": "v2", "project_id": "p16", "matched": 0.4},  # Low skill match, close distance
    
    # Volunteer 3 (Marcus - LA) successful matches
    {"volunteer_id": "v3", "project_id": "p7", "matched": 1},  # High skill match, close distance
    {"volunteer_id": "v3", "project_id": "p8", "matched": 1},  # High skill match, close distance
    {"volunteer_id": "v3", "project_id": "p14", "matched": 0.5},  # High skill match, far distance
    
    # Volunteer 4 (Fatima - Houston) successful matches
    {"volunteer_id": "v4", "project_id": "p9", "matched": 1},  # High skill match, close distance
    {"volunteer_id": "v4", "project_id": "p10", "matched": 1},  # High skill match, close distance
    
    # Volunteer 5 (Jordan - Seattle) successful matches
    {"volunteer_id": "v5", "project_id": "p11", "matched": 1},  # High skill match, close distance
    {"volunteer_id": "v5", "project_id": "p12", "matched": 1},  # High skill match, close distance
    
    # Cross-volunteer unsuccessful matches (wrong skills AND far away)
    {"volunteer_id": "v1", "project_id": "p5", "matched": 0},  # Low skill match, far distance (SF to NY)
    {"volunteer_id": "v1", "project_id": "p7", "matched": 0},  # Low skill match, far distance (SF to LA)
    {"volunteer_id": "v2", "project_id": "p9", "matched": 0},  # Low skill match, far distance (NY to Houston)
    {"volunteer_id": "v3", "project_id": "p11", "matched": 0},  # Low skill match, far distance (LA to Seattle)
    {"volunteer_id": "v4", "project_id": "p1", "matched": 0},  # Low skill match, far distance (Houston to SF)
    {"volunteer_id": "v5", "project_id": "p5", "matched": 0},  # Low skill match, far distance (Seattle to NY)
    
    # Cross-volunteer mixed matches (good skills but far away) - medium match score
    {"volunteer_id": "v1", "project_id": "p14", "matched": 0.3},  # Some skill match (python), far distance
    {"volunteer_id": "v3", "project_id": "p4", "matched": 0.3},  # Some skill match (teaching), far distance
    {"volunteer_id": "v3", "project_id": "p10", "matched": 0.3},  # Some skill match (teaching), far distance
    
    # Cross-volunteer mixed matches (close by but wrong skills) - medium-low match score
    {"volunteer_id": "v1", "project_id": "p15", "matched": 0.3},  # Low skill match, close distance
    {"volunteer_id": "v2", "project_id": "p16", "matched": 0.3}   # Low skill match, close distance
]

# Create test data for validation
test_volunteers = [
    {
        "id": "tv1",
        "name": "Samantha Wilson",
        "skills": ["python", "machine learning", "data visualization"],
        "interest": "AI ethics",
        "location": {"lat": 37.7849, "lng": -122.4194}  # Near SF
    },
    {
        "id": "tv2",
        "name": "Robert Chen",
        "skills": ["healthcare", "nutrition", "public health"],
        "interest": "community wellness",
        "location": {"lat": 29.7604, "lng": -95.3798}  # Near Houston
    }
]

test_projects = [
    {
        "id": "tp1",
        "name": "Data Science for Social Good",
        "description": "Using data science to solve social problems",
        "skillsReq": ["python", "data analysis", "machine learning"],
        "location": {"lat": 37.7849, "lng": -122.4294}  # Near SF
    },
    {
        "id": "tp2",
        "name": "Public Health Workshop",
        "description": "Educating communities about health",
        "skillsReq": ["healthcare", "teaching", "public health"],
        "location": {"lat": 29.7704, "lng": -95.3798}  # Near Houston
    },
    {
        "id": "tp3",
        "name": "Remote Machine Learning Project",
        "description": "Developing ML models for charity",
        "skillsReq": ["python", "machine learning", "statistics"],
        "location": {"lat": 51.5074, "lng": -0.1278}  # London (far away)
    }
]

# Expected recommendations:
# For tv1 (Samantha): tp1 should be first (close + skills match), tp3 second (skills but far)
# For tv2 (Robert): tp2 should be first (close + skills match)

# Save to files for the API
with open('training_volunteers.json', 'w') as f:
    json.dump(volunteers, f, indent=2)

with open('training_projects.json', 'w') as f:
    json.dump(projects, f, indent=2)

with open('training_interactions.json', 'w') as f:
    json.dump(interactions, f, indent=2)

with open('test_volunteers.json', 'w') as f:
    json.dump(test_volunteers, f, indent=2)

with open('test_projects.json', 'w') as f:
    json.dump(test_projects, f, indent=2)

print("Training data generated:")
print(f"- {len(volunteers)} volunteers")
print(f"- {len(projects)} projects")
print(f"- {len(interactions)} interactions")
print(f"- {len(test_volunteers)} test volunteers")
print(f"- {len(test_projects)} test projects")

# Sample API call for training
"""
curl -X POST http://localhost:3001/api/train \
  -H "Content-Type: application/json" \
  -d @- << EOF
{
  "volunteers": $(cat training_volunteers.json),
  "projects": $(cat training_projects.json),
  "interactions": $(cat training_interactions.json),
  "test_size": 0.2
}
EOF
"""

# Sample API call for recommendation testing
"""
curl -X POST http://localhost:3001/api/recommend \
  -H "Content-Type: application/json" \
  -d @- << EOF
{
  "volunteer": $(cat test_volunteers.json | jq '.[0]'),
  "projects": $(cat test_projects.json),
  "top_n": 3
}
EOF
"""