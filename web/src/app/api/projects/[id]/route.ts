import { NextRequest, NextResponse } from "next/server";
import { doc, getDoc } from "firebase/firestore";
import { db } from "@/lib/firebase";


export async function GET(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } =await  params;

    const projectRef = doc(db, "Project", id);
    const projectSnap = await getDoc(projectRef);

    if (!projectSnap.exists()) {
      return NextResponse.json(
        { success: false, msg: "Project not found!" },
        { status: 404 }
      );
    }

    return NextResponse.json(
      { success: true, data: { id: projectSnap.id, ...projectSnap.data() } },
      { status: 200 }
    );
  } catch (error) {
    return NextResponse.json(
      { success: false, msg: "Server error", error: (error as Error).message },
      { status: 500 }
    );
  }
}
