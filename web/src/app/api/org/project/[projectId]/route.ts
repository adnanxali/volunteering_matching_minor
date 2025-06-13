import { Project } from "@/app/types/types";
import { db } from "@/lib/firebase";
import { collection, doc, getDoc, getDocs, query, where } from "firebase/firestore";
import { NextRequest, NextResponse } from "next/server";


export async function GET(req: NextRequest, { params }: { params: { projectId: string } }) {
  try {
    const headers = new Headers(req.headers);
    const orgId = headers.get("orgId");
    const { projectId } = await params;

    if (!orgId) {
      return NextResponse.json({ success: false, msg: "Organization ID is required in headers!" }, { status: 400 });
    }

    if (!projectId) {
      return NextResponse.json({ success: false, msg: "Project ID is missing!" }, { status: 400 });
    }

    const projectRef = doc(db, "Project", projectId);
    const projectSnap = await getDoc(projectRef);

    if (!projectSnap.exists()) {
      return NextResponse.json({ success: false, msg: "Project not found!" }, { status: 404 });
    }

    const projectData = projectSnap.data();

    if (projectData.orgId !== orgId) {
      return NextResponse.json({ success: false, msg: "Unauthorized access to project!" }, { status: 403 });
    }

    return NextResponse.json({ success: true, data: { id: projectSnap.id, ...projectData } }, { status: 200 });

  } catch (error) {
    console.error("Error fetching project:", error);
    return NextResponse.json({ success: false, msg: "Server error" }, { status: 500 });
  }
}
