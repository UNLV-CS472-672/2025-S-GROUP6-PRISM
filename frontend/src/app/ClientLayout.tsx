"use client";

import { usePathname } from "next/navigation";
import HeaderBar from "@/components/HeaderBar";
import React, { useEffect, useState } from "react";
import { useAuth } from "@/context/AuthContext"
import { easyFetch } from "@/utils/fetchWrapper";

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
  // console.log(location.pathname);
  const { user } = useAuth();
  
  const [username, setUsername] = useState<string | null>(null);

  useEffect(() => {
    const fetchUsername = async () => {
      try {
        const res = await easyFetch('http://localhost:8001/api/user/users', {
          method: "GET",
        });

        if (!res.ok) throw new Error("Failed to fetch username");

        const data = await res.json();
        const username = data[0]
        // console.log("Raw data:", JSON.stringify(data, null, 2));
        setUsername(username?.first_name);
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
