import axios from "axios";
import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  try {
    // Try to connect to the ML service
    const response = await axios.get('http://localhost:3001/api/status', { 
      timeout: 2000 // Set a short timeout
    });
    
    return NextResponse.json({ 
      available: true,
      status: response.data
    });
  } catch (error) {
    console.error("ML service health check failed:", error);
    return NextResponse.json({ 
      available: false,
      error: "ML service is currently unavailable"
    });
  }
}