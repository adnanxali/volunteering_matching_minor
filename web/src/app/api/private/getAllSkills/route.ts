import { db } from "@/lib/firebase";
import axios from "axios";
import { addDoc,query,getDocs, where,collection, getDoc, doc } from "firebase/firestore";
import { NextRequest, NextResponse } from "next/server";

export async function GET(req:NextRequest) {
    const docRef = collection(db,'volunteer');
    const docSnap = await getDocs(docRef);
    let skill:string [] = [];
    docSnap.forEach((doc)=>{
        if(!skill.includes(doc.data().skills)){
            skill.push(doc.data().skills);
        }
    })
    const docRef2 = collection(db,'Project');
    const docSnap2 = await getDocs(docRef2);
    
    docSnap2.forEach((doc)=>{
        if(!skill.includes(doc.data().skillsReq)){
            skill.push(doc.data().skillsReq);
        }
    })
    const flatSkills = Array.from(
        new Set(
          skill.flatMap(item => {
            if (Array.isArray(item)) {
              return item.map(skill => skill.trim());
            } else if (typeof item === 'string') {
              return item.split(',').map(skill => skill.trim());
            }
            return [];
          })
        )
      );
    console.log(flatSkills);
    return NextResponse.json({msg:"Private IP No response"},{status:403})


}