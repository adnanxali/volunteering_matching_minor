"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import Link from "next/link";
import { cn } from "@/lib/util";

export const MenuItem = ({
  setActive,
  active,
  item,
  children,
}: {
  setActive: (item: string | null) => void;
  active: string | null;
  item: string;
  children?: React.ReactNode;
}) => {
  return (
    <div className="relative">
      <motion.button
        onHoverStart={() => setActive(item)}
        className={cn(
          "relative px-4 py-2 text-sm hover:opacity-100",
          active === item ? "text-black" : "text-neutral-500"
        )}
      >
        <span className="relative z-10">{item}</span>
        {active === item && (
          <motion.div
            layoutId="bg"
            className="absolute inset-0 rounded-md bg-neutral-100"
            style={{ zIndex: 0 }}
            transition={{ type: "spring", bounce: 0.25, duration: 0.3 }}
          />
        )}
      </motion.button>
      <div className="absolute top-full mt-2">
        {active === item && children}
      </div>
    </div>
  );
};

export const Menu = ({
  setActive,
  children,
}: {
  setActive: (item: string | null) => void;
  children: React.ReactNode;
}) => {
  return (
    <nav
      onMouseLeave={() => setActive(null)}
      className="relative rounded-full border border-neutral-200 bg-white px-4 py-2 shadow-md"
    >
      <div className="flex gap-2">
        {children}
      </div>
    </nav>
  );
};

export const HoveredLink = ({ children, href, className }: { children: React.ReactNode; href: string; className?: string }) => {
  return (
    <Link href={href} className={cn("flex items-center py-2 px-3 whitespace-nowrap rounded-md transition-colors hover:bg-neutral-100", className)}>
      {children}
    </Link>
  );
};

export const ProductItem = ({
  title,
  description,
  href,
  src,
}: {
  title: string;
  description: string;
  href: string;
  src: string;
}) => {
  return (
    <Link href={href} className="flex gap-4 rounded-md p-3 hover:bg-neutral-100">
      <div className="flex-shrink-0">
        <div className="flex h-12 w-12 items-center justify-center rounded-md bg-neutral-100">
          <img src={src} alt={title} width={24} height={24} className="h-6 w-6" />
        </div>
      </div>
      <div>
        <h3 className="font-medium">{title}</h3>
        <p className="text-sm text-neutral-500">{description}</p>
      </div>
    </Link>
  );
};