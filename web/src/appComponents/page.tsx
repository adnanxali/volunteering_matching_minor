"use client";
import React, { use, useState } from "react";
import { HoveredLink, Menu, MenuItem, ProductItem } from "@/components/ui/navbar-menu";
import { cn } from "@/lib/util";
import { useRouter } from "next/navigation";
import { auth } from "@/lib/firebase";

export function Navbar({ className,type }: { className?: string ,type?:string }) {
    const [active, setActive] = useState<string | null>(null);
    const [login,setLogin] = useState<boolean>(false);
    const router = useRouter();
    return (
      <div
        className={cn("fixed top-0 inset-x-0 z-50 bg-white shadow-md px-4 py-1 ", className)}
      >
        <div className="max-w-6xl mx-auto flex justify-between items-center">
          <h1 className="text-xl font-bold">ServeTogether.</h1>
          <Menu setActive={setActive}>
            {login && type=="org" &&<HoveredLink href="/#">Add Post for Volunteer</HoveredLink>}
            {login && type=="vol" &&<HoveredLink href="/#">Apply for Volunteer</HoveredLink>}
            
           {!login&& <MenuItem setActive={setActive} active={active} item="Login">
              <div className="flex flex-col space-y-4 text-sm">
                <HoveredLink href="http://localhost:3000/vol/login" >Login As Volunteer</HoveredLink>
                <HoveredLink href="http://localhost:3000/org/login">Login As Organization</HoveredLink>
              </div>
            </MenuItem>}
          </Menu>
        </div>
      </div>
    );
}
export function NavbarVol({ className, type }: { className?: string; type?: string }) {
  const [active, setActive] = useState<string | null>(null);
  const [login, setLogin] = useState<boolean>(false);
  const router = useRouter();
  return (
    <div
      className={cn(
        "fixed top-0 inset-x-0 z-50 bg-white shadow-md px-4 py-7", // increased py for more height
        className
      )}
    >
      <div className="max-w-6xl mx-auto flex justify-between items-center">
        <h1 className="text-xl font-bold text-black">ServeTogether.</h1>

        <div className="flex items-center gap-8 text-sm">
          <HoveredLink href="http://localhost:3000/vol/my-applications">My Applications</HoveredLink>
          <HoveredLink href="http://localhost:3000/vol/dashboard">Home</HoveredLink>
          <button  onClick={()=>{
            localStorage.removeItem('volunteerId');
            router.push('/')
          }}>Logout</button>
        </div>
      </div>
    </div>
  );
}
export function NavbarOrg({ className, type }: { className?: string; type?: string }) {
  const [active, setActive] = useState<string | null>(null);
  const [login, setLogin] = useState<boolean>(false);
  const router = useRouter();

  const handleLogout = async () => {
      await auth.signOut();
      localStorage.removeItem('organizationId');
    };


  return (
    <div
      className={cn(
        "fixed top-0 inset-x-0 z-50 bg-white shadow-md px-4 py-7", // increased py for more height
        className
      )}
    >
      <div className="max-w-6xl mx-auto flex justify-between items-center">
        <h1 className="text-xl font-bold text-black">ServerTogether.</h1>

        <div className="flex items-center gap-8 text-sm">
          <HoveredLink href="http://localhost:3000/org/projects/create">Create Post</HoveredLink>
          <HoveredLink href="http://localhost:3000/org/dashboard">Home</HoveredLink>
          <button onClick={()=>{
            handleLogout();
            router.push('/')
          }}>Logout</button>
        </div>
      </div>
    </div>
  );
}





