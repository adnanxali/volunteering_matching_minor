"use client";
import { useState } from "react";
import axios from "axios";

const skillsOptions = [
  "Teaching",
  "Healthcare",
  "Environmental Awareness",
  "Fundraising",
  "Event Planning",
  "Animal Care",
  "Community Outreach"
];

export default function AddPostPage() {
  const [formData, setFormData] = useState({
    title: "",
    description: "",
    duration: "",
    location: "",
    latitude: "",
    longitude: "",
    skillsReq: [] as string[],
  });

  const [message, setMessage] = useState("");
  const [loadingLocation, setLoadingLocation] = useState(false);
  const orgId = typeof window !== "undefined" ? localStorage.getItem("organizationId") : null;

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleMultiSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const selected = Array.from(e.target.selectedOptions).map(option => option.value);
    setFormData(prev => ({ ...prev, skillsReq: selected }));
  };

  // Function to get the current location using GPS
  const handleGetCurrentLocation = async () => {
    setLoadingLocation(true);
    setMessage('');
    try {
      if (!navigator.geolocation) {
        setMessage("Geolocation is not supported by your browser.");
        setLoadingLocation(false);
        return;
      }

      navigator.geolocation.getCurrentPosition(async (position) => {
        const { latitude, longitude } = position.coords;

        // Reverse geocode using OpenCage API
        const res = await axios.get(
          `https://api.opencagedata.com/geocode/v1/json?q=${latitude}+${longitude}&key=a944d12f69484c21bbb57b36d14228e9`
        );

        const components = res.data.results[0]?.components;
        const city = components.city || components.town || components.village || '';
        const country = components.country || '';
        const locationString = `${city}, ${country}`;

        setFormData((prev) => ({
          ...prev,
          location: locationString,
          latitude: latitude.toString(),
          longitude: longitude.toString(),
        }));

        setMessage("Location filled using GPS.");
        setLoadingLocation(false);
      });
    } catch (error) {
      console.error(error);
      setMessage("Failed to fetch GPS location.");
      setLoadingLocation(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    let latitude = formData.latitude;
    let longitude = formData.longitude;

    // If the coordinates are not provided, try to get them from location
    if (!latitude || !longitude) {
      try {
        const geoRes = await axios.get(
          `https://api.opencagedata.com/geocode/v1/json?q=${encodeURIComponent(formData.location)}&key=a944d12f69484c21bbb57b36d14228e9`
        );

        const geometry = geoRes.data.results[0]?.geometry;
        if (geometry) {
          latitude = geometry.lat.toString();
          longitude = geometry.lng.toString();
        } else {
          throw new Error("Invalid location");
        }
      } catch (geoErr) {
        console.error("Error fetching coordinates:", geoErr);
        setMessage("Invalid location. Please try again or use GPS.");
        return;
      }
    }

    const payload = { ...formData, orgId, latitude, longitude };

    try {
      const res = await axios.post("/api/org/project", payload);
      if (res.data.success) {
        alert("Project posted successfully!");
        setFormData({
          title: "",
          description: "",
          duration: "",
          location: "",
          latitude: "",
          longitude: "",
          skillsReq: []
        });
      } else {
        alert("Failed to post project");
      }
    } catch (err) {
      console.error("Error submitting post", err);
      alert("Server Error");
    }
  };

  return (
    <div className="min-h-screen pt-24 px-4 md:px-10 bg-gray-50">
      <div className="max-w-3xl mx-auto bg-white p-8 rounded-xl shadow-lg">
        <h2 className="text-3xl font-bold mb-6 text-gray-800">Create a Volunteer Opportunity</h2>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block mb-2 font-medium text-gray-700">Title</label>
            <input
              name="title"
              value={formData.title}
              onChange={handleChange}
              type="text"
              placeholder="Post title"
              className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 focus:outline-none"
              required
            />
          </div>

          <div>
            <label className="block mb-2 font-medium text-gray-700">Description</label>
            <textarea
              name="description"
              value={formData.description}
              onChange={handleChange}
              placeholder="Describe the opportunity..."
              className="w-full border border-gray-300 rounded-lg px-4 py-2 h-32 resize-none focus:ring-2 focus:ring-blue-500 focus:outline-none"
              required
            />
          </div>

          <div>
            <label className="block mb-2 font-medium text-gray-700">Duration</label>
            <input
              name="duration"
              value={formData.duration}
              onChange={handleChange}
              type="text"
              placeholder="e.g. 2 weeks"
              className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 focus:outline-none"
              required
            />
          </div>

          <div>
            <label className="block mb-2 font-medium text-gray-700">Location</label>
            <div className="flex gap-2">
              <input
                name="location"
                value={formData.location}
                onChange={handleChange}
                type="text"
                placeholder="Enter the location, Eg: Delhi, India"
                className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 focus:outline-none"
                required
              />
              <button
                type="button"
                onClick={handleGetCurrentLocation}
                className="px-4 py-2 bg-white text-black rounded hover:bg-gray-300 transition"
              >
                {loadingLocation ? "Getting..." : "Use GPS"}
              </button>
            </div>
          </div>

          <div>
            <label className="block mb-2 font-medium text-gray-700">Skills Required</label>
            <select
              name="skillsReq"
              multiple
              value={formData.skillsReq}
              onChange={handleMultiSelect}
              className="w-full h-32 border border-gray-300 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 focus:outline-none"
              required
            >
              {skillsOptions.map(skill => (
                <option key={skill} value={skill}>{skill}</option>
              ))}
            </select>
            <p className="text-sm text-gray-500 mt-1">Hold Ctrl (Windows) or Cmd (Mac) to select multiple.</p>
          </div>

          <button
            type="submit"
            className="w-full bg-blue-600 text-white py-3 rounded-lg font-medium hover:bg-blue-700 transition"
          >
            Post Opportunity
          </button>
        </form>
        {message && <p className="text-center text-sm text-gray-500 mt-4">{message}</p>}
      </div>
    </div>
  );
}
