"use client";
import { useEffect, useState } from "react";
import axios from "axios";
import Link from "next/link";

export default function OrgDashboard() {
  const [projects, setProjects] = useState<any[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchProjects = async () => {
      const orgId = localStorage.getItem("organizationId");
      if (!orgId) return setError("Organization ID not found.");

      try {
        const res = await axios.get("http://localhost:3000/api/org/project", {
          headers: { orgId },
        });
        setProjects(res.data.data);
      } catch (err) {
        setError(`No Projects found ${err}`);
      }
    };

    fetchProjects();
  }, []);

  return (
    <main className="min-h-screen p-10 bg-gray-100">
      <h1 className="text-3xl font-bold mb-6">Organization Dashboard</h1>
      {error && <p className="text-red-600">{error}</p>}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {projects.map((project) => (
          <div key={project.id} className="p-6 bg-white rounded shadow">
            <h2 className="text-xl font-semibold">{project.title}</h2>
            <p className="text-gray-600">{project.description}</p>
            <p className="text-sm mt-1 text-gray-500">
              Location: {project.location.label}
            </p>
            <Link href={`/org/projects/${project.id}`}>
              <button className="mt-4 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">
                View Details
              </button>
            </Link>
          </div>
        ))}
      </div>
    </main>
  );
}
