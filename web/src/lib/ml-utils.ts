// import axios from "axios";

// // Base URL for ML service
// const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:3001';

// /**
//  * Check if the ML service is available
//  */
// export async function checkMLServiceAvailability() {
//   try {
//     const response = await axios.get(`${ML_SERVICE_URL}/api/status`, { 
//       timeout: 2000 // 2 second timeout
//     });
//     return {
//       available: true,
//       data: response.data
//     };
//   } catch (error) {
//     console.error("ML service unavailable:", error);
//     return {
//       available: false,
//       error: String(error)
//     };
//   }
// }

// /**
//  * Get recommendations from the ML service
//  */
// export async function getRecommendations(volunteer, projects, topN = 10) {
//   try {
//     const response = await axios.post(`${ML_SERVICE_URL}/api/recommend`, {
//       volunteer,
//       projects,
//       top_n: topN
//     }, {
//       timeout: 5000 // 5 second timeout
//     });
    
//     return {
//       success: true,
//       recommendations: response.data.recommendations,
//       metadata: response.data.metadata
//     };
//   } catch (error) {
//     console.error("ML recommendation error:", error);
//     return {
//       success: false,
//       error: String(error)
//     };
//   }
// }

// /**
//  * Format volunteer data for ML service
//  */
// export function formatVolunteerForML(volunteer) {
//   return {
//     id: volunteer.id,
//     name: volunteer.name,
//     email: volunteer.email,
//     skills: volunteer.skills || [],
//     interest: volunteer.interest || "",
//     location: {
//       lat: volunteer.location?.lat || 0,
//       lng: volunteer.location?.lng || 0
//     }
//   };
// }

// /**
//  * Format project data for ML service
//  */
// export function formatProjectForML(project) {
//   return {
//     id: project.id,
//     title: project.title,
//     description: project.description,
//     skillsReq: project.skillsReq ? 
//       project.skillsReq.map(skill => skill.toLowerCase()) : [],
//     location: {
//       label: project.location?.label || "",
//       lat: parseFloat(project.location?.latitude) || 0,
//       lng: parseFloat(project.location?.longitude) || 0
//     }
//   };
// }