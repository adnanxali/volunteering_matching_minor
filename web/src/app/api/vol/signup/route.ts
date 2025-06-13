import { db } from "@/lib/firebase";
import { addDoc, collection, getDocs, query, where } from "firebase/firestore";
import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {
      name,
      email,
      skills,
      interest,
      location, 
      latitude,
      longitude,
    } = body;

    console.log("Received:", name, email, latitude, longitude);

    if (!name || !email || !skills || !interest || !location) {
      return NextResponse.json(
        { success: false, error: "All fields are required" },
        { status: 400 }
      );
    }

    const volRef = collection(db, "volunteer");
    const q = query(volRef, where("email", "==", email));
    const existsDoc = await getDocs(q);

    if (!existsDoc.empty) {
      return NextResponse.json(
        { success: false, error: "User Already Exists!" },
        { status: 400 }
      );
    }

    const docRef = await addDoc(volRef, {
      name,
      email,
      skills,
      interest,
      location: {
        label: location, // e.g. "Mumbai, India"
        lat: latitude || null,
        lng: longitude || null,
      },
      createdAt: new Date().toISOString(),
    });

    return NextResponse.json(
      {
        success: true,
        msg: "User added to database",
        id: docRef.id,
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
