'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import axios from 'axios';

export default function VolunteerLogin() {
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState('');
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setMessage('');

    try {
      const res = await axios.post('/api/vol/login', { email });
      const { volunteer } = res.data;

      localStorage.setItem('volunteerId', volunteer.id);
      router.push('/vol/dashboard');
    } catch (err: any) {
      setMessage(err.response?.data?.error || 'Something went wrong.');
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-black via-zinc-900 to-black text-white flex items-center justify-center px-4">
      <div className="w-full max-w-md p-8 bg-zinc-900 rounded-2xl shadow-2xl space-y-6">
        <h1 className="text-3xl font-bold text-center">Volunteer Login</h1>

        <form onSubmit={handleSubmit} className="space-y-4">
          <label className="block text-sm font-medium text-zinc-300">
            Email Address
          </label>
          <input
            type="email"
            placeholder="Enter your email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full p-3 rounded-lg bg-zinc-800 text-white border border-zinc-700 focus:outline-none focus:ring-2 focus:ring-white"
            required
          />

          <button
            type="submit"
            className="w-full p-3 rounded-lg bg-white text-black font-semibold hover:bg-gray-300 transition-all duration-200"
          >
            Login
          </button>
        </form>

        {message && (
          <p className="text-center text-sm text-red-400">{message}</p>
        )}

        <div className="flex items-center justify-center gap-2 text-sm text-zinc-400 pt-4 border-t border-zinc-700">
          <span>New here?</span>
          <Link href="/vol/signup" className="text-blue-400 hover:underline">
            Sign up as a volunteer
          </Link>
        </div>
      </div>
    </main>
  );
}
