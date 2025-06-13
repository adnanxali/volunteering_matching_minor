import { db } from "@/lib/firebase";
import { collection, getDocs, query, where } from "firebase/firestore";
import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { email } = body;

    if (!email) {
      return NextResponse.json(
        { success: false, error: "Email is required" },
        { status: 400 }
      );
    }

    const volRef = collection(db, "volunteer");
    const q = query(volRef, where("email", "==", email));
    const snapshot = await getDocs(q);

    if (snapshot.empty) {
      return NextResponse.json(
        { success: false, error: "Volunteer not found. Please sign up." },
        { status: 404 }
      );
    }

    const volunteerData = snapshot.docs[0].data();
    const id = snapshot.docs[0].id;

    return NextResponse.json(
      {
        success: true,
        msg: "Login successful",
        volunteer: { id, ...volunteerData },
      },
      { status: 200 }
    );
  } catch (e) {
    return NextResponse.json(
      { success: false, error: "Internal Server Error: " + e },
      { status: 500 }
    );
  }
}
