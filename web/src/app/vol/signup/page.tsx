'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import axios from 'axios';

export default function Login() {
  const [form, setForm] = useState({
    name: '',
    email: '',
    skills: '',
    interest: '',
    location: '',
    latitude: '',
    longitude: '',
  });

  const [message, setMessage] = useState('');
  const [loadingLocation, setLoadingLocation] = useState(false);
  const router = useRouter();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleGetCurrentLocation = async () => {
    setLoadingLocation(true);
    setMessage('');
    try {
      if (!navigator.geolocation) {
        setMessage('Geolocation is not supported by your browser.');
        setLoadingLocation(false);
        return;
      }

      navigator.geolocation.getCurrentPosition(async (position) => {
        const { latitude, longitude } = position.coords;

        // Reverse geocode using OpenCage
        const res = await axios.get(
          `https://api.opencagedata.com/geocode/v1/json?q=${latitude}+${longitude}&key=a944d12f69484c21bbb57b36d14228e9`
        );

        const components = res.data.results[0]?.components;
        const city = components.city || components.town || components.village || '';
        const country = components.country || '';
        const locationString = `${city}, ${country}`;

        setForm((prev) => ({
          ...prev,
          location: locationString,
          latitude: latitude.toString(),
          longitude: longitude.toString(),
        }));

        setMessage('Location filled using GPS.');
        setLoadingLocation(false);
      });
    } catch (error) {
      console.error(error);
      setMessage('Failed to fetch GPS location.');
      setLoadingLocation(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setMessage('');

    let latitude = form.latitude;
    let longitude = form.longitude;

    
    if (!latitude || !longitude) {
      try {
        const geoRes = await axios.get(
          `https://api.opencagedata.com/geocode/v1/json?q=${encodeURIComponent(
            form.location
          )}&key=a944d12f69484c21bbb57b36d14228e9`
        );

        const geometry = geoRes.data.results[0]?.geometry;
        if (geometry) {
          latitude = geometry.lat.toString();
          longitude = geometry.lng.toString();
        } else {
          throw new Error('Invalid location');
        }
      } catch (geoErr) {
        console.error('Error fetching coordinates:', geoErr);
        setMessage('Invalid location. Please try again or use GPS.');
        return;
      }
    }

    const formattedForm = {
      ...form,
      skills: form.skills
        .split(',')
        .map((skill) => skill.trim())
        .filter((skill) => skill.length > 0),
      latitude,
      longitude,
    };

    try {
      const res = await axios.post('http://localhost:3000/api/vol/signup', formattedForm);

      if (res.data.id) {
        localStorage.setItem('volunteerId', res.data.id);
        router.push('/vol/dashboard');
      }

      setMessage(res.data.msg || 'Signed up successfully!');
      setForm({
        name: '',
        email: '',
        skills: '',
        interest: '',
        location: '',
        latitude: '',
        longitude: '',
      });
    } catch (err: any) {
      setMessage(err.response?.data?.error || 'Something went wrong.');
    }
  };

  return (
    <main className="min-h-screen bg-black text-white flex items-center justify-center px-4">
      <div className="w-full max-w-md p-8 bg-zinc-900 rounded-2xl shadow-xl space-y-6">
        <h1 className="text-3xl font-bold text-center">Volunteer Signup</h1>
        <form className="space-y-4" onSubmit={handleSubmit}>
          <input
            name="name"
            value={form.name}
            onChange={handleChange}
            placeholder="Name"
            className="w-full p-3 rounded bg-zinc-800 text-white border border-zinc-700"
          />
          <input
            name="email"
            type="email"
            value={form.email}
            onChange={handleChange}
            placeholder="Email"
            className="w-full p-3 rounded bg-zinc-800 text-white border border-zinc-700"
          />
          <input
            name="skills"
            value={form.skills}
            onChange={handleChange}
            placeholder="e.g., design, teaching, coding"
            className="w-full p-3 rounded bg-zinc-800 text-white border border-zinc-700"
          />
          <input
            name="interest"
            value={form.interest}
            onChange={handleChange}
            placeholder="Interest"
            className="w-full p-3 rounded bg-zinc-800 text-white border border-zinc-700"
          />
          <div className="flex gap-2">
            <input
              name="location"
              value={form.location}
              onChange={handleChange}
              placeholder="Location: Eg: Mumbai, India"
              className="w-full p-3 rounded bg-zinc-800 text-white border border-zinc-700"
            />
            <button
              type="button"
              onClick={handleGetCurrentLocation}
              className="px-4 py-2 bg-white text-black rounded hover:bg-gray-300 transition"
            >
              {loadingLocation ? 'Getting...' : 'Use GPS'}
            </button>
          </div>
          <button
            type="submit"
            className="w-full p-3 rounded bg-white text-black font-semibold hover:bg-gray-300 transition"
          >
            Sign Up
          </button>
        </form>
        {message && (
          <p className="text-center text-sm text-gray-400">{message}</p>
        )}
      </div>
    </main>
  );
}
