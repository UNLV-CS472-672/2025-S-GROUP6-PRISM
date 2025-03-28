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
	const context = useAuth()

	const [loading, setLoading] = useState(false);
	// Hydrated statee added to handle mismatched rendering
	const [hydrated, setHydrated] = useState(false)

	useEffect(() => {
		setHydrated(true)
	}, [])

	if (!hydrated) return null // Prevents SSR mismatches

	const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
		event.preventDefault()

		// Validate there's input
		if (!username || !password) {
			setMessage({ type: "error", text: "Username and password are required." });
			return;
		}

		setLoading(true);
		try {
			const response = await easyFetch("http://localhost:8000/api/login", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ username, password }),
			})

			const data = await response.json()

			if (response.ok) {
				console.log(data["user"])
				context?.login(data["user"])
				router.push("/dashboard")
			} else {
				setMessage({ type: "error", text: ` ${data.error}`});
			}
		} catch (err) {
			setMessage({ type: "error", text: " Server error. Please try again."});
		} finally {
			setLoading(false);
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
					<Button
						type="submit"
						variant="contained"
						fullWidth
						sx={{ mt: 2 }}
						disabled={loading}
					>
						{loading ? "Logging in..." : "Login"}
					</Button>
				</form>

				{/* OR Divider */}
				<Divider sx={{ width: "100%", my: 2 }}>OR</Divider>

				{/* NextAuth Google Login Button */}
				<SignInButton />

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
