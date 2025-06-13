
import { db } from "@/lib/firebase";
import { collection, doc, getDoc, getDocs } from "firebase/firestore";
import { NextRequest, NextResponse } from "next/server";


export async function GET(req: NextRequest, { params }: { params: { id: string } }) {
  try {
    const volId = params.id;
    const volRef = doc(db, "volunteer", volId);
    const volSnap = await getDoc(volRef);

    if (!volSnap.exists()) {
      return NextResponse.json({ success: false, msg: "Volunteer not found" }, { status: 404 });
    }

    const volData = volSnap.data();

    // Fetch appliedProjects subcollection
    const appliedProjectsRef = collection(volRef, "appliedProjects");
    const appliedProjectsSnap = await getDocs(appliedProjectsRef);

    const appliedProjects = appliedProjectsSnap.docs.map((doc) => doc.id);

    return NextResponse.json(
      {
        success: true,
        data: {
          ...volData,
          appliedProjects, 
        },
      },
      { status: 200 }
    );
  } catch (e) {
    console.error(e);
    return NextResponse.json({ success: false, msg: "Internal Server Error" }, { status: 500 });
  }
}

