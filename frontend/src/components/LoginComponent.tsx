/*
    This login component renders the login screen in the browser.
    It asks for username/password for local account verification
    and includes a sign in with Google option.
*/

"use client"

import React, { useState, useEffect } from "react"
import {
	TextField,
	Button,
	Container,
	Typography,
	Box,
	Alert,
	Divider,
} from "@mui/material"
import { SignInButton } from "@/components/AuthenticationMethod" // Use SignInButton component
import { useRouter } from "next/navigation"
import { useAuth } from "@/context/AuthContext"
import { easyFetch } from "@/utils/fetchWrapper"

const LoginComponent: React.FC = () => {
	const router = useRouter()
	const [username, setUsername] = useState<string>("")
	const [password, setPassword] = useState<string>("")
	const [message, setMessage] = useState<{
		type: "success" | "error"
		text: string
	} | null>(null)
	const { user, login } = useAuth()

	// Hydrated statee added to handle mismatched rendering
	const [hydrated, setHydrated] = useState(false)

	useEffect(() => {
		setHydrated(true)

		const errorMessage = sessionStorage.getItem("loginError")
		if (errorMessage) {
			setMessage({ type: "error", text: errorMessage })
			sessionStorage.removeItem("loginError")
		}
	}, [])

	useEffect(() => {
		if (hydrated && user?.isLoggedIn) router.push("/courses/")
	})

	if (!hydrated) return null // Prevents SSR mismatches

	const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
		event.preventDefault()

		try {
			const response = await easyFetch("http://localhost:8000/api/login", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ username, password }),
			})

			const data = await response.json()

			if (response.ok) {
				login(data["user"])
				router.push("/courses/")
			} else {
				setMessage({
					type: "error",
					text:
						data?.non_field_errors?.[0] || "Login failed. Please try again.",
				})
			}
		} catch (err) {
			console.error("Login error:", err)
			setMessage({
				type: "error",
				text: "Unexpected error. Please try again later.",
			})
		}
	}

	return (
		<Container maxWidth="xs">
			<Box
				sx={{
					display: "flex",
					flexDirection: "column",
					alignItems: "center",
					mt: 8,
					p: 4,
					boxShadow: 3,
					borderRadius: 2,
					bgcolor: "background.paper",
				}}
			>
				<Typography variant="h5" component="h1" gutterBottom>
					PRISM
				</Typography>

				{/* Username & Password Login */}
				<form onSubmit={handleSubmit} style={{ width: "100%" }}>
					<TextField
						id="username-input"
						label="Username"
						variant="outlined"
						fullWidth
						margin="normal"
						value={username}
						onChange={(e) => setUsername(e.target.value)}
						required
						InputLabelProps={{ shrink: true }}
						inputProps={{ "aria-label": "Username" }}
					/>
					<label
						id="username-label"
						htmlFor="username-input"
						style={{ display: "none" }}
					>
						Username
					</label>
					<TextField
						id="password-input"
						label="Password"
						type="password"
						variant="outlined"
						fullWidth
						margin="normal"
						value={password}
						onChange={(e) => setPassword(e.target.value)}
						required
						InputLabelProps={{ shrink: true }}
						inputProps={{ "aria-labelledby": "password-label" }}
					/>
					<label
						id="password-label"
						htmlFor="password-input"
						style={{ display: "none" }}
					>
						Password
					</label>
					<Button type="submit" variant="contained" fullWidth sx={{ mt: 2 }}>
						Login
					</Button>
				</form>

				{/* OR Divider */}
				<Divider sx={{ width: "100%", my: 2 }}></Divider>

				{/* NextAuth Google Login Button */}
				<Button variant="contained">
					<SignInButton />
				</Button>

				{/* Display Messages */}
				{message && (
					<Alert severity={message.type} sx={{ mt: 2, width: "100%" }}>
						{message.text}
					</Alert>
				)}
			</Box>
		</Container>
	)
}

export default LoginComponent
