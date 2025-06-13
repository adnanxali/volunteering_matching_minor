import { db } from "@/lib/firebase";
import axios from "axios";
import { addDoc, query, getDocs, where, collection, getDoc, doc } from "firebase/firestore";
import { NextRequest, NextResponse } from "next/server";

type Project = {
  id: string;
  orgId: string;
  location: { label: string, longitude: string, latitude: string };
  description: string;
  title: string;
  skillsReq: string[];
  duration: string;
};

export async function GET(req: NextRequest) {
  try {
    const volId = req.headers.get('volunteerId');
    if (!volId) {
      return NextResponse.json({ error: 'Missing volunteerId in headers' }, { status: 400 });
    }

    // Get all projects
    const projectsRef = collection(db, "Project");
    const docSnap = await getDocs(projectsRef);
    
    // Get volunteer data
    const volRef = doc(db, 'volunteer', volId);
    const volSnap = await getDoc(volRef);
    const volData = volSnap.data();

    if (!volData) {
      return NextResponse.json({ success: false, msg: "Volunteer not found" }, { status: 404 });
    }

    if (docSnap.empty) {
      return NextResponse.json({ success: false, msg: "No projects found!" }, { status: 404 });
    }

    // Format volunteer data for ML model
    const mappedVolData = {
      id: volId,
      name: volData?.name,
      email: volData?.email,
      skills: volData?.skills || [],
      interest: volData?.interest || "",
      location: {
        lat: volData?.location?.lat,
        lng: volData?.location?.lng
      }
    };

    // Format projects data
    const projects: Project[] = docSnap.docs.map(doc => ({
      id: doc.id,
      ...(doc.data() as Omit<Project, 'id'>)
    }));

    // Map projects to the format expected by the ML model
    const mappedProjects = projects.map(project => ({
      id: project.id,
      title: project.title,
      description: project.description,
      skillsReq: project.skillsReq ? project.skillsReq.map(skill => skill.toLowerCase()) : [],
      location: {
        label: project.location?.label || "",
        lat: parseFloat(project.location?.latitude) || 0,
        lng: parseFloat(project.location?.longitude) || 0
      }
    }));

    try {
      // Make API call to ML recommender system
      const matchedProjects = await axios.post('http://localhost:3001/api/recommend', {
        volunteer: mappedVolData,
        projects: mappedProjects,
        top_n: 10 // Get top 10 recommendations
      });

      if (matchedProjects.data.success) {
        // Extract recommended projects and their scores
        const recommendations = matchedProjects.data.recommendations;
        
        // Map back to original project data with score and explanation
        //@ts-ignore
        const enhancedProjects = recommendations.map(rec => {
          // Find the original project
          const originalProject = projects.find(p => p.id === rec.project.id);
          
          return {
            ...originalProject,
            matchScore: rec.score * 100, // Convert to percentage
            matchExplanation: rec.explanation
          };
        });

        return NextResponse.json({ 
          success: true, 
          data: enhancedProjects,
          metadata: matchedProjects.data.metadata
        }, { status: 200 });
      } else {
        // Fallback to returning all projects if the ML service failed but returned a response
        console.warn("ML service returned unsuccessful response, falling back to all projects");
        return NextResponse.json({ success: true, data: projects }, { status: 200 });
      }
    } catch (mlError) {
      // If ML service is unavailable, return all projects as fallback
      console.error("ML service error:", mlError);
      return NextResponse.json({ 
        success: true, 
        data: projects, 
        metadata: { error: "ML service unavailable, showing all projects" } 
      }, { status: 200 });
    }

  } catch (error) {
    console.error("Server error:", error);
    return NextResponse.json({ success: false, msg: "Server error", error: String(error) }, { status: 500 });
  }
}