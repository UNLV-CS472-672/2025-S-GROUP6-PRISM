"use client"
import { signIn } from "next-auth/react"
import { useRouter } from "next/navigation"
import Button from "@mui/material/Button"

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

	return <Button onClick={() => router.push("/logout")}>Sign Out</Button>
}
