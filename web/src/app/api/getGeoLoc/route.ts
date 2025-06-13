import axios from "axios";
import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  const city = req.headers.get("city");
  const country = req.headers.get("country");

  if (!city || !country) {
    return NextResponse.json({ error: "City and country headers are required" }, { status: 400 });
  }

  try {
    const location = `${city}, ${country}`;
    const url = `https://api.opencagedata.com/geocode/v1/json?q=${encodeURIComponent(location)}&key=a944d12f69484c21bbb57b36d14228e9`;

    const res = await axios.get(url);
    const result = res.data.results[0];

    if (!result) {
      return NextResponse.json({ error: "No results found" }, { status: 404 });
    }

    const { lat, lng } = result.geometry;

    return NextResponse.json({ lat, lng });
  } catch (error) {
    console.error("Geocoding error:", error);
    return NextResponse.json({ error: "Failed to fetch coordinates" }, { status: 500 });
  }
}
