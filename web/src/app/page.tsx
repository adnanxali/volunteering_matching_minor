import Image from "next/image";
import { TextGenerateEffect } from "@/components/ui/text-generator-effect";
import Link from "next/link";

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen justify-between">
      {/* Hero Section */}
      <main className="text-center mt-32 px-4">
        <TextGenerateEffect words="Together, We Make a Difference!" />
        <p className="mt-4 text-gray-600 text-lg max-w-xl mx-auto">
          Join us in building a better world through meaningful volunteer opportunities and impactful projects.
        </p>
        <Link
          href="/volunteer"
          className="inline-block mt-8 px-6 py-3 bg-blue-600 text-white rounded-full hover:bg-blue-700 transition"
        >
          Explore Opportunities
        </Link>
      </main>

    </div>
  );
}
