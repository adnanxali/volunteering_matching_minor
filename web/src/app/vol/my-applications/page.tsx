"use client";
import { useEffect, useState } from "react";
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

export default function MyApplicationsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [withdrawing, setWithdrawing] = useState<string | null>(null);

  useEffect(() => {
    const fetchAppliedProjects = async () => {
      try {
        const volunteerId = localStorage.getItem("volunteerId");
        if (!volunteerId) {
          setError("Volunteer ID not found.");
          return;
        }

        const volRes = await axios.get(`/api/vol/${volunteerId}`);
        const appliedProjectIds = volRes.data.data.appliedProjects;

        if (!appliedProjectIds.length) {
          setProjects([]);
          return;
        }

        const projectPromises = appliedProjectIds.map((id: string) =>
          axios.get(`/api/projects/${id}`).then((res) => res.data.data)
        );

        const fetchedProjects = await Promise.all(projectPromises);
        setProjects(fetchedProjects);
      } catch (err) {
        console.error(err);
        setError("Failed to load your applications.");
      } finally {
        setLoading(false);
      }
    };

    fetchAppliedProjects();
  }, []);

  const handleWithdraw = async (projectId: string) => {
    setWithdrawing(projectId);
    
    // Placeholder for actual implementation
    setTimeout(() => {
      setWithdrawing(null);
      alert("Not implemented");
    }, 500);
    
    // Actual implementation (commented out)
    // const volunteerId = localStorage.getItem("volunteerId");
    // if (!volunteerId) return;
    // try {
    //   await axios.delete("/api/vol/withdraw", {
    //     data: { volunteerId, projectId },
    //   });
    //   setProjects((prev) => prev.filter((proj) => proj.id !== projectId));
    //   toast.success("Application withdrawn successfully.");
    // } catch (err) {
    //   console.error("Withdraw failed", err);
    //   toast.error("Could not withdraw. Please try again.");
    // } finally {
    //   setWithdrawing(null);
    // }
  };

  if (loading) {
    return (
      <div className="max-w-5xl mx-auto mt-20 px-4 py-12 flex justify-center items-center">
        <div className="animate-pulse flex flex-col items-center">
          <div className="h-8 w-64 bg-gray-200 rounded mb-8"></div>
          {[1, 2, 3].map((i) => (
            <div key={i} className="w-full max-w-3xl bg-gray-100 rounded-lg p-6 mb-4">
              <div className="h-6 bg-gray-200 rounded w-3/4 mb-4"></div>
              <div className="h-4 bg-gray-200 rounded w-1/2 mb-3"></div>
              <div className="h-20 bg-gray-200 rounded w-full mb-3"></div>
              <div className="h-4 bg-gray-200 rounded w-2/3 mb-4"></div>
              <div className="h-10 bg-gray-200 rounded w-32"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto mt-20 px-4 py-8">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">My Applications</h1>
        <p className="text-gray-600 mt-2">
          Track and manage your project applications
        </p>
      </header>

      {error ? (
        <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-red-700">
          <p className="font-medium mb-1">Error</p>
          <p>{error}</p>
        </div>
      ) : projects.length === 0 ? (
        <div className="bg-blue-50 border border-blue-100 rounded-lg p-8 text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-100 rounded-full mb-4">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" className="w-8 h-8 text-blue-600">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h2 className="text-xl font-semibold text-gray-800 mb-2">No Applications Yet</h2>
          <p className="text-gray-600 mb-6">You haven't applied to any volunteer projects.</p>
          <a href="/projects" className="inline-flex items-center px-5 py-3 bg-black text-white rounded-lg hover:bg-gray-800 transition">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
            </svg>
            Explore Projects
          </a>
        </div>
      ) : (
        <div className="grid gap-6 md:grid-cols-2">
          {projects.map((project) => (
            <div
              key={project.id}
              className="border border-gray-200 rounded-xl overflow-hidden bg-white shadow-sm hover:shadow-md transition-shadow"
            >
              <div className="p-6">
                <h2 className="text-xl font-semibold text-gray-800 mb-1">
                  {project.title}
                </h2>
                <div className="flex items-center text-gray-600 mb-3">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                  <span>{project.location.label}</span>
                  <span className="mx-2">•</span>
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span>{project.duration}</span>
                </div>
                
                <p className="text-gray-700 mb-4 line-clamp-3">
                  {project.description}
                </p>
                
                <div className="mb-5">
                  <h3 className="text-sm font-medium text-gray-700 mb-2">Required Skills:</h3>
                  <div className="flex flex-wrap gap-2">
                    {project.skillsReq.map((skill, index) => (
                      <span 
                        key={index} 
                        className="inline-block bg-gray-100 px-3 py-1 rounded-full text-sm text-gray-700"
                      >
                        {skill}
                      </span>
                    ))}
                  </div>
                </div>
                
                <div className="flex justify-between items-center">
                  <button
                    onClick={() => handleWithdraw(project.id)}
                    disabled={withdrawing === project.id}
                    className="flex items-center px-4 py-2 bg-black text-white rounded-lg hover:bg-gray-800 transition disabled:opacity-70"
                  >
                    {withdrawing === project.id ? (
                      <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                    ) : (
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    )}
                    Withdraw Application
                  </button>
                  <a 
                    href={`/projects/${project.id}`} 
                    className="text-blue-600 hover:text-blue-800 font-medium text-sm"
                  >
                    View Details
                  </a>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}