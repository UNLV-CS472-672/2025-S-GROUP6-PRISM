"use client"
import { Button } from "@mui/material"
import { signIn } from "next-auth/react"
import { useRouter } from "next/navigation"

export function SignInButton() {
	return (
		<Button
			variant="contained"
			data-testid="google-sign-in"
			onClick={() =>
				signIn("google", { callbackUrl: "http://localhost:3000/callback" })
			}
		>
			Sign In with Google
		</Button>
	)
}

export function SignOutButton() {
	const router = useRouter()

	return (
		<Button variant="outlined" onClick={() => router.push("/logout")}>
			Sign Out
		</Button>
	)
}
