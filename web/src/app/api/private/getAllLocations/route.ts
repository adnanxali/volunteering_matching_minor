import { db } from "@/lib/firebase";
import axios from "axios";
import { addDoc,query,getDocs, where,collection, getDoc, doc } from "firebase/firestore";
import { NextRequest, NextResponse } from "next/server";

export async function GET(req:NextRequest) {
    const docRef = collection(db,'volunteer');
    const docSnap = await getDocs(docRef);
    let locations:string [] = [];
    docSnap.forEach((doc)=>{
        if(!locations.includes(doc.data().location)){
            locations.push(doc.data().location);
        }
    })
    const docRef2 = collection(db,'Project');
    const docSnap2 = await getDocs(docRef2);
    
    docSnap2.forEach((doc)=>{
        if(!locations.includes(doc.data().location)){
            locations.push(doc.data().location);
        }
    })
    console.log(locations);
    return NextResponse.json({msg:"Private IP No response"},{status:403})

}