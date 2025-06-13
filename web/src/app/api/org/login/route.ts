import { db } from "@/lib/firebase";

import { doc, getDoc, updateDoc } from "firebase/firestore";

import { NextRequest, NextResponse } from "next/server";

export async function POST(req:NextRequest) {
    const body = await req.json();
    const {email,password}=body;
}
export async function PUT(req:NextRequest) {
    try{
        const body = await req.json();
        const {orgId,mission,location} =body;
        if(!mission || !location){
            return NextResponse.json({success:false,error:"Invalid Input fields"},{status:401})
        }
        const orgRef = doc(db,'organization',orgId);
        
        const docSnap = getDoc(orgRef);
        if(!(await docSnap).exists()){
            return NextResponse.json({success:false,error:"Organization Not Found !"},{status:400});
        }

        const result = await updateDoc(orgRef,{mission,location})
        return NextResponse.json({success:true,msg:"Organization Details Updated !",data:result},{status:200});
    }catch(e){
        return NextResponse.json({success:false,error:"Internal Server Error !",e},{status:500})
    }

}