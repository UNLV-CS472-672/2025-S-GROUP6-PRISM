"use client";

import { usePathname } from "next/navigation";
import HeaderBar from "@/components/HeaderBar";
import React, { useEffect, useState } from "react";

const routeTitles: Record<string, string> = {
  "/account/": "Account",
  "/alerts/": "Alerts",
  "/student_comparison/": "Student Comparison",
  // Add more paths and titles as needed
};

const staticUsername = "Alex"; // Replace with your auth logic or context

export default function ClientLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const hideHeader = pathname === "/login/";
  console.log(location.pathname);
  
  const [username, setUsername] = useState<string | null>(null);

  useEffect(() => {
    const fetchUsername = async () => {
      try {
        const res = await fetch("http://localhost:8000/api/user/users", {
          method: "GET",
          credentials: "include",
          headers: {
            "Content-Type": "application/json",
          },
        });

        //if (!res.ok) throw new Error("Failed to fetch username");

        const data = await res.json();
        console.log(JSON.stringify(data));
        setUsername(data.first_name);
      } catch (error) {
        console.error("Error fetching username:", error);
        setUsername("User");
      }
    };

    fetchUsername();
  }, []);

  const getTitle = () => {
    if (pathname === "/dashboard/") return `Welcome, ${username}`;
    const match = Object.keys(routeTitles).find((key) => pathname.startsWith(key));
    return routeTitles[match || ""] || "Your App";
  };

  return (
    <>
      {!hideHeader && <HeaderBar title={getTitle()} />}
      <main>{children}</main>
    </>
  );
}
