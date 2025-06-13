"use client";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import axios from "axios";

interface Volunteer {
  id: string;
  name: string;
  email: string;
  skills: string[];
  interest: string;
  location: {
    label: string;
    latitude?: number;
    longitude?: number;
  };
}

interface Project {
  id: string;
  title: string;
  description: string;
  skillsReq: string[];
  duration: string;
  location: {
    label: string;
    latitude?: number;
    longitude?: number;
  };
  appliedVolunteers?: string[];
  status?: string;
  createdAt?: string;
}

export default function ProjectDetails() {
  const { id } = useParams();
  const [project, setProject] = useState<Project | null>(null);
  const [volunteers, setVolunteers] = useState<Volunteer[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [selectedVolunteer, setSelectedVolunteer] = useState<string | null>(null);

  useEffect(() => {
    const fetchProject = async () => {
      try {
        const orgId = localStorage.getItem("organizationId");
        if (!orgId) {
          setError("Organization ID missing from localStorage.");
          setLoading(false);
          return;
        }

        const res = await axios.get(`http://localhost:3000/api/org/project/${id}`, {
          headers: { orgId },
        });

        const projectData = res.data?.data;
        if (!projectData) {
          setError("Project not found.");
          setLoading(false);
          return;
        }

        setProject(projectData);

        // Fetch volunteer details
        const vols = await Promise.all(
          projectData.appliedVolunteers?.map(async (volId: string) => {
            try {
              const volRes = await axios.get(`/api/vol/${volId}`);
              return volRes.data?.data;
            } catch (err) {
              console.warn(`Failed to fetch volunteer ${volId}`);
              return null;
            }
          }) || []
        );

        // Filter out null responses
        setVolunteers(vols.filter(Boolean));
        setLoading(false);
      } catch (err) {
        console.error(err);
        setError("Failed to load project details.");
        setLoading(false);
      }
    };

    fetchProject();
  }, [id]);

  const handleApproveVolunteer = (volunteerId: string) => {
    // Implementation placeholder
    setSelectedVolunteer(volunteerId);
    setTimeout(() => {
      alert("Approval functionality not implemented");
      setSelectedVolunteer(null);
    }, 500);
  };

  const handleRejectVolunteer = (volunteerId: string) => {
    // Implementation placeholder
    setSelectedVolunteer(volunteerId);
    setTimeout(() => {
      alert("Rejection functionality not implemented");
      setSelectedVolunteer(null);
    }, 500);
  };

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto mt-20 px-6 py-8">
        <div className="animate-pulse">
          <div className="h-10 bg-gray-200 rounded w-3/4 mb-6"></div>
          <div className="h-4 bg-gray-200 rounded w-full mb-4"></div>
          <div className="h-4 bg-gray-200 rounded w-full mb-4"></div>
          
          <div className="grid grid-cols-3 gap-4 mb-8">
            <div className="h-8 bg-gray-200 rounded"></div>
            <div className="h-8 bg-gray-200 rounded"></div>
            <div className="h-8 bg-gray-200 rounded"></div>
          </div>
          
          <div className="h-8 bg-gray-200 rounded w-1/3 mb-6 mt-12"></div>
          
          {[1, 2, 3].map((i) => (
            <div key={i} className="border border-gray-200 rounded-lg p-6 mb-4 bg-gray-50">
              <div className="grid grid-cols-2 gap-4">
                <div className="h-6 bg-gray-200 rounded"></div>
                <div className="h-6 bg-gray-200 rounded"></div>
                <div className="h-6 bg-gray-200 rounded"></div>
                <div className="h-6 bg-gray-200 rounded"></div>
              </div>
              <div className="mt-4 flex gap-2">
                <div className="h-10 bg-gray-200 rounded w-24"></div>
                <div className="h-10 bg-gray-200 rounded w-24"></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-6xl mx-auto mt-20 px-6 py-8">
        <div className="bg-red-50 border border-red-200 text-red-700 px-6 py-8 rounded-lg">
          <div className="flex items-center mb-4">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <h2 className="text-xl font-semibold">Error</h2>
          </div>
          <p>{error}</p>
          <button 
            onClick={() => window.location.href = "/org/dashboard"}
            className="mt-6 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition"
          >
            Back to Projects
          </button>
        </div>
      </div>
    );
  }

  if (!project) {
    return null; // This should never happen due to loading state, but added for type safety
  }

  const formatDate = (dateString?: string) => {
    if (!dateString) return "N/A";
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  return (
    <div className="max-w-6xl mx-auto mt-20 px-6 py-8">
      <nav className="mb-8 flex" aria-label="Breadcrumb">
        <ol className="inline-flex items-center space-x-1 md:space-x-3">
          <li className="inline-flex items-center">
            <a href="/org/dashboard" className="text-gray-700 hover:text-blue-600">
              Dashboard
            </a>
          </li>
          <li>
            <div className="flex items-center">
              <svg className="w-6 h-6 text-gray-400" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd"></path>
              </svg>
              <a href="/projects" className="text-gray-700 hover:text-blue-600 ml-1 md:ml-2">
                Projects
              </a>
            </div>
          </li>
          <li aria-current="page">
            <div className="flex items-center">
              <svg className="w-6 h-6 text-gray-400" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd"></path>
              </svg>
              <span className="text-gray-500 ml-1 md:ml-2 font-medium truncate">
                {project.title}
              </span>
            </div>
          </li>
        </ol>
      </nav>

      <div className="bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden">
        <div className="p-8">
          <div className="flex justify-between items-start mb-6">
            <h1 className="text-3xl font-bold text-gray-900">{project.title}</h1>
            <span className={`px-3 py-1 rounded-full text-sm font-medium ${
              project.status === 'active' ? 'bg-green-100 text-green-800' :
              project.status === 'completed' ? 'bg-blue-100 text-blue-800' :
              project.status === 'pending' ? 'bg-yellow-100 text-yellow-800' :
              'bg-gray-100 text-gray-800'
            }`}>
              {project.status || 'Active'}
            </span>
          </div>
          
          <p className="text-gray-700 text-lg mb-8 leading-relaxed">
            {project.description}
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="flex items-center mb-2">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                <h3 className="text-gray-700 font-medium">Location</h3>
              </div>
              <p className="text-gray-800">{project.location?.label || 'Not specified'}</p>
            </div>
            
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="flex items-center mb-2">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <h3 className="text-gray-700 font-medium">Duration</h3>
              </div>
              <p className="text-gray-800">{project.duration}</p>
            </div>
            
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="flex items-center mb-2">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <h3 className="text-gray-700 font-medium">Created</h3>
              </div>
              <p className="text-gray-800">{formatDate(project.createdAt)}</p>
            </div>
          </div>
          
          <div className="mb-8">
            <h3 className="text-lg font-semibold text-gray-800 mb-3">Required Skills</h3>
            <div className="flex flex-wrap gap-2">
              {project.skillsReq.map((skill, idx) => (
                <span
                  key={idx}
                  className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800"
                >
                  {skill}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="mt-10">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-gray-900">Applied Volunteers</h2>
          <span className="bg-blue-100 text-blue-800 text-sm font-medium px-3 py-1 rounded-full">
            {volunteers.length} Applicant{volunteers.length !== 1 ? 's' : ''}
          </span>
        </div>

        {volunteers.length === 0 ? (
          <div className="bg-gray-50 border border-gray-200 rounded-lg p-8 text-center">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-gray-100 rounded-full mb-4">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-gray-800 mb-2">No Volunteers Yet</h3>
            <p className="text-gray-600 mb-6">
              There are currently no volunteers who have applied to this project.
            </p>
            <a href="/org/dashboard" className="px-4 py-2 bg-black text-white rounded-md hover:bg-gray-800 transition">
              Back to Projects
            </a>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4">
            {volunteers.map((vol) => (
              <div key={vol.id} className="bg-white border border-gray-200 rounded-lg shadow-sm overflow-hidden">
                <div className="p-6">
                  <div className="sm:flex sm:justify-between sm:items-start">
                    <div className="mb-4 sm:mb-0">
                      <h3 className="text-xl font-semibold text-gray-900 mb-1">{vol.name}</h3>
                      <div className="text-gray-600 mb-2">
                        <a href={`mailto:${vol.email}`} className="hover:text-blue-600 transition">
                          {vol.email}
                        </a>
                      </div>
                      <div className="flex items-center text-gray-600">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                        </svg>
                        {vol.location?.label || 'Location not specified'}
                      </div>
                    </div>
                    
                  </div>
                  
                  <div className="mt-6">
                    <div className="mb-4">
                      <h4 className="text-sm font-medium text-gray-700 mb-2">Skills:</h4>
                      <div className="flex flex-wrap gap-2">
                        {vol.skills.map((skill, idx) => (
                          <span
                            key={idx}
                            className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800"
                          >
                            {skill}
                          </span>
                        ))}
                      </div>
                    </div>
                    
                    <div>
                      <h4 className="text-sm font-medium text-gray-700 mb-2">Interest Statement:</h4>
                      <blockquote className="italic text-gray-600 border-l-4 border-gray-200 pl-4">
                        {vol.interest || "No interest statement provided."}
                      </blockquote>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}