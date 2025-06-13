"use client";

import React, { useEffect, useState } from "react";
import { HoveredLink, Menu, MenuItem } from "@/components/ui/navbar-menu";
import { cn } from "@/lib/util";
import { useRouter } from "next/navigation";
import { auth } from "@/lib/firebase";

export function ClientNavbar({ className, type = "org" }: { className?: string, type?: string }) {
  // Set a stable initial state that matches on both server and client
  const [isClient, setIsClient] = useState(false);
  const [login, setLogin] = useState(false);
  const router = useRouter();

  // Check login state only after component mounts on client
  useEffect(() => {
    setIsClient(true);
    
    // Check login state based on type
    if (type === "org") {
      const orgId = localStorage.getItem('organizationId');
      setLogin(!!orgId);
    } else if (type === "vol") {
      const volId = localStorage.getItem('volunteerId');
      setLogin(!!volId);
    }
  }, [type]);

  // Only render complete navbar after client-side hydration
  if (!isClient) {
    // Return minimal markup during SSR to prevent hydration mismatch
    return (
      <div className={cn("fixed top-0 inset-x-0 z-50 bg-white shadow-md px-4 py-7", className)}>
        <div className="max-w-6xl mx-auto flex justify-between items-center">
          <h1 className="text-xl font-bold text-black">ServeTogether.</h1>
          <div></div> {/* Placeholder for menu */}
        </div>
      </div>
    );
  }

  // Render appropriate navbar based on type and login state
  if (type === "org" && login) {
    return <NavbarOrg className={className} />;
  } else if (type === "vol" && login) {
    return <NavbarVol className={className} />;
  } else {
    return <Navbar className={className} />;
  }
}

export function Navbar({ className }: { className?: string }) {
  const [active, setActive] = useState<string | null>(null);
  
  return (
    <div className={cn("fixed top-0 inset-x-0 z-50 bg-white shadow-md px-4 py-1", className)}>
      <div className="max-w-6xl mx-auto flex justify-between items-center">
        <h1 className="text-xl font-bold">ServeTogether.</h1>
        <Menu setActive={setActive}>
          <MenuItem setActive={setActive} active={active} item="Login">
            <div className="flex flex-col space-y-4 text-sm">
              <HoveredLink href="/vol/login">Login As Volunteer</HoveredLink>
              <HoveredLink href="/org/login">Login As Organization</HoveredLink>
            </div>
          </MenuItem>
        </Menu>
      </div>
    </div>
  );
}

export function NavbarVol({ className }: { className?: string }) {
  const router = useRouter();
  
  return (
    <div className={cn("fixed top-0 inset-x-0 z-50 bg-white shadow-md px-4 py-7", className)}>
      <div className="max-w-6xl mx-auto flex justify-between items-center">
        <h1 className="text-xl font-bold text-black">ServeTogether.</h1>
        <div className="flex items-center gap-8 text-sm">
          <HoveredLink href="/vol/my-applications">My Applications</HoveredLink>
          <HoveredLink href="/vol/dashboard">Home</HoveredLink>
          <button
            className="cursor-pointer"
            onClick={() => {
              localStorage.removeItem('volunteerId');
              router.push('/');
            }}
          >
            Logout
          </button>
        </div>
      </div>
    </div>
  );
}

export function NavbarOrg({ className }: { className?: string }) {
  const router = useRouter();

  const handleLogout = async () => {
    await auth.signOut();
    localStorage.removeItem('organizationId');
  };

  return (
    <div className={cn("fixed top-0 inset-x-0 z-50 bg-white shadow-md px-4 py-7", className)}>
      <div className="max-w-6xl mx-auto flex justify-between items-center">
        <h1 className="text-xl font-bold text-black">ServeTogether.</h1> {/* Fixed typo in "ServeTogether" */}
        <div className="flex items-center gap-8 text-sm">
          <HoveredLink href="/org/projects/create">Create Post</HoveredLink>
          <HoveredLink href="/org/dashboard">Home</HoveredLink>
          <button
            className="cursor-pointer"
            onClick={() => {
              handleLogout();
              router.push('/');
            }}
          >
            Logout
          </button>
        </div>
      </div>
    </div>
  );
}