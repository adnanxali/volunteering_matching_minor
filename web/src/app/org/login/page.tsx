"use client";
import axios from 'axios';
import { useEffect, useState } from "react";
import { signInWithPopup, onAuthStateChanged, GoogleAuthProvider } from "firebase/auth";
import { auth } from "../../../lib/firebase";
import { useRouter } from "next/navigation";
import { FcGoogle } from "react-icons/fc";

export default function LoginPage() {
  const googleProvider = new GoogleAuthProvider();
  const [user, setUser] = useState<any>(null);
  const [message, setMessage] = useState('');
  const [orgName, setOrgName] = useState('');
  const [orgEmail, setOrgEmail] = useState('');
  const router = useRouter();

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      if (user) setUser(user);
    });
    return () => unsubscribe();
  }, []);

  const handleGoogleLogin = async () => {
    try {
      const result = await signInWithPopup(auth, googleProvider);
      setUser(result.user);
      await handleRegisterOrg(result.user);
    } catch (error) {
      console.error("Login failed:", error);
    }
  };

  const handleRegisterOrg = async (user: any) => {
    try {
      const response = await axios.post('http://localhost:3000/api/org/signup-oauth', {
        userId: user.uid,
        name: user.displayName || orgName,
        email: user.email || orgEmail,
      });

      if (response.data.success) {
        localStorage.setItem('organizationId', user.uid);
        router.push('/org/dashboard');
      } else {
        setMessage(response.data.error || "Registration failed.");
      }
    } catch (error: any) {
      setMessage(error.response?.data?.error || 'Something went wrong.');
    }
  };

  const handleEmailRegister = async () => {
    if (!orgName || !orgEmail) {
      setMessage("Please fill in both name and email.");
      return;
    }

    try {
      const response = await axios.post('http://localhost:3000/api/org/signup-details', {
        name: orgName,
        email: orgEmail,
      });

      if (response.data.success) {
        localStorage.setItem('organizationId', response.data.organizationId); 
        router.push('/org/dashboard');
      } else {
        setMessage(response.data.error || "Registration failed.");
      }
    } catch (error: any) {
      setMessage(error.response?.data?.error || 'Something went wrong.');
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100 px-4">
      <div className="w-full max-w-md bg-white rounded-2xl shadow-lg p-8">
        <h1 className="text-2xl font-bold text-center mb-6 text-gray-800">Organization Login</h1>

        {!user && (
          <div className="space-y-4">
            <input
              type="text"
              placeholder="Organization Name"
              value={orgName}
              onChange={(e) => setOrgName(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <input
              type="email"
              placeholder="Organization Email"
              value={orgEmail}
              onChange={(e) => setOrgEmail(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />

            <button
              onClick={handleEmailRegister}
              className="w-full bg-green-600 text-white py-2 rounded-lg hover:bg-green-700 transition"
            >
              Continue with Email
            </button>

            <div className="flex items-center justify-center text-sm text-gray-500">OR</div>

            <button
              onClick={handleGoogleLogin}
              className="w-full flex items-center justify-center gap-2 bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition"
            >
              <FcGoogle className="text-xl bg-white rounded-full" />
              Sign in with Google
            </button>

            {message && <p className="text-red-500 text-sm mt-2 text-center">{message}</p>}
          </div>
        )}
      </div>
    </div>
  );
}
