// src/app/account/page.tsx
"use client";

import React, { useEffect, useState } from "react";
import { useAuth } from "@/context/AuthContext"

export default function AccountPage() {
  const [editing, setEditing] = useState(false);
  const [message, setMessage] = useState("");
  const { user } = useAuth()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setMessage("");
    try {
      const res = await fetch("/api/users/user", {
        method: "PUT",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(user),
      });

      if (!res.ok) throw new Error("Update failed");

      const updated = await res.json();
      setUser({ email: updated.email, username: updated.username });
      setEditing(false);
      setMessage("Information updated successfully!");
    } catch (err) {
      setMessage("Error updating user information.");
      console.error(err);
    }
  };

  return (
    <div>
      <h1>Account Information</h1>
      {message && <p>{message}</p>}
      <form onSubmit={handleSubmit}>
        <label>
          Email:
          <input
            name="email"
            value={user.email}
            onChange={handleChange}
            disabled={!editing}
            type="email"
          />
        </label>
        <br />
        <label>
          Username:
          <input
            name="username"
            value={user.username}
            onChange={handleChange}
            disabled={!editing}
          />
        </label>
        <br />
        {editing ? (
          <>
            <button type="submit">Save</button>
            <button type="button" onClick={() => setEditing(false)}>
              Cancel
            </button>
          </>
        ) : (
          <button type="button" onClick={() => setEditing(true)}>
            Edit
          </button>
        )}
      </form>
    </div>
  );
}
