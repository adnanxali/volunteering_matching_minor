import { db } from "@/lib/firebase";
import {
    addDoc,
  collection,
  doc,
  getDoc,
  getDocs,
  query,
  setDoc,
  where
} from "firebase/firestore";
import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { name, email} = body;

    if (!name || !email) {
      return NextResponse.json(
        { success: false, error: "Missing required fields" },
        { status: 400 }
      );
    }

    
    const q = query(collection(db,'organization'),where('email','==',email))

    const docSnap = await getDocs(q);
    if(docSnap.empty){
        const org = await addDoc(collection(db,'organization'),{name,email,createdAt:Date.now()})
        return NextResponse.json({msg:"Added to database",success:true,organizationId:org.id})
    }
    const organizationId = docSnap.docs[0].id
    return NextResponse.json({msg:"Proceeds to login",success:true,organizationId})

    }catch(e){
        return NextResponse.json({msg:"Internal Server Error !"+e,success:false},{status:500})
    }
}   
