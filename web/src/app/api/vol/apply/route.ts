import { db } from "@/lib/firebase";
import { addDoc, arrayUnion, collection, doc, getDoc, getDocs, query, setDoc, updateDoc, where } from "firebase/firestore";
import { NextRequest ,NextResponse} from "next/server";


export async function POST(req: NextRequest) {
    try {
      const body = await req.json();
      const { projectId, projectTitle, volunteerId } = body;
  
      if (!projectId || !volunteerId) {
        return NextResponse.json({ success: false, msg: "Volunteer ID and Project ID are required" }, { status: 400 });
      }
  
      
      const projectRef = doc(db, "Project", projectId);
      const projectSnap = await getDoc(projectRef);
  
      if (!projectSnap.exists()) {
        return NextResponse.json({ success: false, msg: "Project does not exist" }, { status: 404 });
      }
  
     
      await updateDoc(projectRef, {
        appliedVolunteers: arrayUnion(volunteerId)
      });
  
      
      const volunteerProjectRef = doc(db, "volunteer", volunteerId, "appliedProjects", projectId);
      await setDoc(volunteerProjectRef, {
        projectId,
        projectTitle,
        appliedAt: new Date()
      });
  
      
      const volunteerDoc = await getDoc(doc(db, "volunteer", volunteerId));
      if (!volunteerDoc.exists()) {
        return NextResponse.json({ success: false, msg: "Volunteer does not exist" }, { status: 404 });
      }
  
      const volunteerData = volunteerDoc.data();
      const projectVolunteerRef = doc(db, "Project", projectId, "appliedVolunteers", volunteerId);
  
      await setDoc(projectVolunteerRef, {
        volunteerId,
        name: volunteerData.name || "",
        email: volunteerData.email || "",
        skills: volunteerData.skills || [],
        appliedAt: new Date()
      });
  
      return NextResponse.json({ success: true, msg: "Successfully applied to the project." }, { status: 200 });
  
    } catch (error) {
      console.error("Error applying to project:", error);
      return NextResponse.json({ success: false, msg: "Internal Server Error", error: (error as Error).message }, { status: 500 });
    }
  }


  export async function GET(req: NextRequest, { params }: { params: { id: string } }) {
    try {
      const { id: volunteerId } = params;
  
      if (!volunteerId) {
        return NextResponse.json({ success: false, msg: "Volunteer ID is required" }, { status: 400 });
      }
  
      // Reference to the subcollection "appliedProjects" of this volunteer
      const appliedProjectsRef = collection(db, "Volunteer", volunteerId, "appliedProjects");
  
      const snapshot = await getDocs(appliedProjectsRef);
  
      if (snapshot.empty) {
        return NextResponse.json({ success: true, data: [], msg: "No applied projects found" }, { status: 200 });
      }
  
      const appliedProjects = snapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      }));
  
      return NextResponse.json({ success: true, data: appliedProjects }, { status: 200 });
  
    } catch (error) {
      console.error("Error fetching applied projects:", error);
      return NextResponse.json({ success: false, msg: "Internal Server Error" }, { status: 500 });
    }
  }