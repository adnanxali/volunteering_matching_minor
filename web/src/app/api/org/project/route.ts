import { Project } from "@/app/types/types";
import { db } from "@/lib/firebase";
import { addDoc,query,getDocs, where,collection } from "firebase/firestore";
import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
    const { title, description, skillsReq, duration, location, latitude, longitude, orgId } = await req.json();

    // Check for required fields
    if (!title || !description || !skillsReq || !duration || !location || !latitude || !longitude) {
        return NextResponse.json({ success: false, msg: "All fields are required!" }, { status: 400 });
    }

    try {
        // Get a reference to the Firestore collection
        const projectRef = collection(db, "Project");

        // Add the project document with location and coordinates
        const docRef = await addDoc(projectRef, {
            title,
            description,
            skillsReq,
            duration,
            location:{
                label:location,
                latitude,
                longitude
            },
            orgId,
        });

        return NextResponse.json({
            success: true,
            data: docRef.id,
        });
    } catch (e) {
        console.error("Error creating project: ", e);
        return NextResponse.json({ success: false, msg: "Internal Server Error!" }, { status: 500 });
    }
}


export async function GET(req: NextRequest) {
    try {
        const reqHeaders = new Headers(req.headers);
        const orgId = reqHeaders.get("orgId");

        if (!orgId) {
            return NextResponse.json({ success: false, msg: "Organization ID is required!" }, { status: 400 });
        }

        
        const projectsRef = collection(db, "Project");

        
        const q = query(projectsRef, where("orgId", "==", orgId));
        const querySnapshot = await getDocs(q);

        if (querySnapshot.empty) {
            return NextResponse.json({ success: false, msg: "No projects found for this organization!" }, { status: 404 });
        }

        const projects = querySnapshot.docs.map(doc => ({ id: doc.id, ...doc.data() }));

        return NextResponse.json({ success: true, data: projects }, { status: 200 });

    } catch (error) {
        return NextResponse.json({ success: false, msg: "Server error" }, { status: 500 });
    }
}
