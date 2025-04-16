"use client"

import React, { useEffect, useState } from "react"
import { useAuth } from "@/context/AuthContext"
import { easyFetch } from "@/utils/fetchWrapper"
import { autocompleteClasses } from "@mui/material"
import { match } from "assert"

interface UserInfo {
	first_name: string
	last_name: string
	email: string
}

export default function AccountPage() {
	const { user } = useAuth()
	const [userInfo, setUserInfo] = useState<UserInfo>({
    first_name: "temp_first",
    last_name: "temp_last",
    email: "temp_email",
  })
	const [isEditing, setIsEditing] = useState(false)
	const [message, setMessage] = useState("")

	useEffect(() => {
    console.log("Current user:", user)
    if (!user?.professor_id) return
  
    const fetchUser = async () => {
      try {
		console.log("Inside of the try.")
        const res = await easyFetch(`http://localhost:8001/api/user/users`, {
			method: "GET"
		})
        if (!res.ok) throw new Error("Failed to fetch user info")
  
        console.log("Raw response:", res)
  
        const data = await res.json()
        console.log("Parsed data:", data)
  
        const matchedUser = data.find((u: any) => u.id === user.professor_id)
  
        if (!matchedUser) throw new Error("User not found in response")
  
        console.log("Matched user:", matchedUser)
        setUserInfo({
          first_name: matchedUser.first_name,
          last_name: matchedUser.last_name,
          email: matchedUser.email,
        })
      } catch (err) {
        console.error("Error loading user data:", err)
      }
    }
  
    fetchUser()
  }, [user?.professor_id])
  

	const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
		if (!userInfo) return
		setUserInfo({ ...userInfo, [e.target.name]: e.target.value })
	}

	const handleSubmit = async (e: React.FormEvent) => {
		e.preventDefault()
		if (!user?.professor_id || !userInfo) return

		try {
			const res = await easyFetch(`http://localhost:8001/api/user/users/${user.professor_id}`, {
				method: "PUT",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify(userInfo),
			})
			if (!res.ok) throw new Error("Update failed")
			setMessage("User information updated successfully!")
			setIsEditing(false)
		} catch (err) {
			console.error(err)
			setMessage("Failed to update user information.")
		}
	}

	//if (!userInfo) return <p>Loading...</p>

	return (
		<div className="container" >
			{message && <p>{message}</p>}
			<form onSubmit={handleSubmit}>
				<label>
					First Name: &nbsp;&nbsp;&nbsp;&nbsp;
					<input
						type="text"
						name="first_name"
						value={userInfo?.first_name}
						onChange={handleChange}
						disabled={!isEditing}
					/>
				</label>
				<br />
				<label>
					Last Name: &nbsp;&nbsp;&nbsp;&nbsp;
					<input
						type="text"
						name="last_name"
						value={userInfo?.last_name}
						onChange={handleChange}
						disabled={!isEditing}
					/>
				</label>
				<br />
				<label>
					Email: &nbsp;&nbsp;&nbsp;&nbsp;
					<input
						type="email"
						name="email"
						value={userInfo?.email}
						onChange={handleChange}
						disabled={!isEditing}
					/>
				</label>
				<br />
				{isEditing ? (
					<>
						<button type="submit">Save</button>
						<button type="button" onClick={() => setIsEditing(false)}>
							Cancel
						</button>
					</>
				) : (
					<button type="button" onClick={() => setIsEditing(true)}>
						Edit
					</button>
				)}
			</form>
		</div>
	)
}
