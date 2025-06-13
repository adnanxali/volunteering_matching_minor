import { db } from "@/lib/firebase";
import {
  collection,
  doc,
  getDoc,
  setDoc
} from "firebase/firestore";
import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { name, email, userId } = body;

    if (!name || !email || !userId) {
      return NextResponse.json(
        { success: false, error: "Missing required fields" },
        { status: 400 }
      );
    }
    
    const docRef = doc(db, "organization", userId);
    const docSnap = await getDoc(docRef);

    if (!docSnap.exists()) {
      // Create new org document if not exists
      await setDoc(docRef, {
        uid: userId,
        name,
        email,
        createdAt: new Date().toISOString()
      });
    }

    return NextResponse.json(
      { success: true, msg: "Proceed to dashboard", id: userId },
      { status: 200 }
    );
  } catch (e) {
    return NextResponse.json(
      { success: false, error: "Internal Server Error: " + e },
      { status: 500 }
    );
  }
}
