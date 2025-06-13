'use client'

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import axios from 'axios';

interface Project {
  id: string;
  title: string;
  description: string;
  duration: string;
  location: {
    label: string;
    latitude: string;
    longitude: string;
  };
  skillsReq: string[];
  matchScore?: number;
  matchExplanation?: {
    matching_skills: string[];
    distance_km: number;
    skill_match_percent: number;
    location_match_percent: number;
  };
}

interface MetaData {
  total_projects: number;
  returned_results: number;
  model_status: string;
  timestamp: string;
  error?: string;
}

export default function VolunteerDashboard() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<MetaData | null>(null);
  const router = useRouter();
  
  useEffect(() => {
    const volId = localStorage.getItem('volunteerId');
    if(!volId){
      router.replace('/vol/login');
      return;
    }

    const fetchProjects = async () => {
      setIsLoading(true);
      try {
        const res = await axios.get('/api/projects', {
          headers: { volunteerId: volId }
        });
        
        if (res.data.success) {
          setProjects(res.data.data);
          if (res.data.metadata) {
            setMetadata(res.data.metadata);
          }
        } else {
          setError(res.data.msg || 'Failed to fetch projects');
        }
      } catch (err) {
        console.error('Error fetching projects:', err);
        setError('Failed to fetch projects. Please try again later.');
      } finally {
        setIsLoading(false);
      }
    };

    fetchProjects();
  }, [router]);

  // Helper function to format match score
  const formatMatchScore = (score?: number) => {
    if (!score && score !== 0) return null;
    return Math.round(score);
  };

  return (
    <div className="min-h-screen pt-24 px-4 bg-gray-100">
      <h1 className="text-3xl font-bold text-center mb-6">Volunteering Opportunities</h1>
      
      {metadata && !metadata.error && (
        <div className="text-center mb-6 text-sm text-gray-600">
          <p>Showing best {metadata.returned_results} matches from {metadata.total_projects} available projects</p>
          <p>Matching method: {metadata.model_status === 'trained' ? 'AI-powered recommendations' : 'Similarity matching'}</p>
        </div>
      )}
      
      {metadata?.error && (
        <div className="text-center mb-6 text-sm text-amber-600">
          <p>{metadata.error}</p>
        </div>
      )}

      {isLoading ? (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
        </div>
      ) : error ? (
        <div className="text-center text-red-500 p-4 bg-red-50 rounded-lg">
          {error}
        </div>
      ) : projects.length === 0 ? (
        <div className="text-center text-gray-500 p-4 bg-gray-50 rounded-lg">
          No projects found. Check back later for new opportunities!
        </div>
      ) : (
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {projects.map((project) => (
            <div key={project.id} className="bg-white rounded-2xl shadow-md p-6 flex flex-col justify-between">
              <div>
                <div className="flex justify-between items-start mb-2">
                  <h2 className="text-xl font-semibold">{project.title}</h2>
                  {/* {project.matchScore !== undefined && (
                    <div className="bg-blue-100 text-blue-800 text-xs font-semibold px-2.5 py-0.5 rounded-full">
                      {formatMatchScore(project.matchScore)}% match
                    </div>
                  )} */}
                </div>
                
                <p className="text-gray-600 text-sm line-clamp-3">{project.description}</p>
                
                <div className="mt-3">
                  <p className="text-sm text-gray-500">📍 {project.location.label}</p>
                  <p className="text-sm text-gray-500">⏳ {project.duration}</p>
                  
                  {project.skillsReq && project.skillsReq.length > 0 && (
                    <div className="mt-2">
                      <p className="text-xs text-gray-500 mb-1">Skills required:</p>
                      <div className="flex flex-wrap gap-1">
                        {project.skillsReq.slice(0, 3).map((skill, idx) => (
                          <span key={idx} className="bg-gray-100 text-gray-800 text-xs px-2 py-0.5 rounded">
                            {skill}
                          </span>
                        ))}
                        {project.skillsReq.length > 3 && (
                          <span className="bg-gray-100 text-gray-800 text-xs px-2 py-0.5 rounded">
                            +{project.skillsReq.length - 3} more
                          </span>
                        )}
                      </div>
                    </div>
                  )}
                  
                  {project.matchExplanation && (
                    <div className="mt-3 text-xs">
                      {project.matchExplanation.matching_skills && project.matchExplanation.matching_skills.length > 0 && (
                        <div className="mt-1">
                          <p className="text-green-600">✓ {project.matchExplanation.matching_skills.length} matching skills</p>
                        </div>
                      )}
                      {project.matchExplanation.distance_km < 20 && (
                        <p className="text-green-600">✓ {project.matchExplanation.distance_km.toFixed(1)} km away</p>
                      )}
                    </div>
                  )}
                </div>
              </div>
              
              <button
                className="mt-4 bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-xl text-sm"
                onClick={() => router.push(`/vol/projects/${project.id}`)}
              >
                Apply Now
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}