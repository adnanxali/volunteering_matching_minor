"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import axios from "axios";

interface Project {
  id: string;
  title: string;
  description: string;
  skillsReq: string[];
  duration: string;
  location: {
    label: string;
    latitude: number;
    longitude: number;
  };
}

const ProjectDetailsPage = () => {
  const { id } = useParams();
  const [project, setProject] = useState<Project | null>(null);
  const [loading, setLoading] = useState(true);
  const [alreadyApplied, setAlreadyApplied] = useState(false);

  useEffect(() => {
    const fetchProjectAndApplicationStatus = async () => {
      try {
        const volunteerId = localStorage.getItem("volunteerId");

        if (!id) return;

        // Fetch project details
        const projectRes = await axios.get(`/api/projects/${id}`);
        setProject(projectRes.data.data);

        if (volunteerId) {
          // Check if the volunteer has already applied
          const volRes = await axios.get(`/api/vol/${volunteerId}`);
          const appliedProjects = volRes.data.data?.appliedProjects || [];
          console.log(volRes);
          if (appliedProjects.includes(id)) {
            setAlreadyApplied(true);
          }
        }
      } catch (error) {
        console.error("Error fetching data", error);
      } finally {
        setLoading(false);
      }
    };

    fetchProjectAndApplicationStatus();
  }, [id]);

  const handleApply = async () => {
    const volunteerId = localStorage.getItem("volunteerId");

    if (!volunteerId || !id) {
      alert("Unable to apply. Missing volunteer ID or project ID.");
      return;
    }

    try {
      await axios.post("/api/vol/apply", {
        volunteerId,
        projectId: id,
        projectTitle: project?.title,
      });
      alert("Application submitted successfully!");
      setAlreadyApplied(true);
    } catch (error) {
      console.error("Application failed", error);
      alert("Something went wrong. Please try again.");
    }
  };

  if (loading) return <p className="p-10">Loading...</p>;
  if (!project) return <p className="p-10">Project not found</p>;

  return (
    <div className="max-w-4xl mx-auto mt-28 px-6 py-10">
      <h1 className="text-4xl font-bold text-gray-900 mb-6">{project.title}</h1>

      <p className="text-gray-800 mb-6 text-lg leading-relaxed">
        {project.description}
      </p>

      <div className="space-y-3 text-gray-900 text-base">
        {/* Updated location display */}
        <p>
          <span className="font-semibold">Location:</span> {project.location.label}

        </p>
        <p>
          <span className="font-semibold">Duration:</span> {project.duration}
        </p>
        <p>
          <span className="font-semibold">Required Skills:</span>{" "}
          {project.skillsReq.join(", ")}
        </p>
      </div>

      <button
        onClick={handleApply}
        disabled={alreadyApplied}
        className={`mt-10 px-8 py-3 rounded-lg text-lg transition ${
          alreadyApplied
            ? "bg-gray-400 text-white cursor-not-allowed"
            : "bg-blue-600 text-white hover:bg-blue-500"
        }`}
      >
        {alreadyApplied ? "Already Applied" : "Apply Now"}
      </button>
    </div>
  );
};

export default ProjectDetailsPage;
