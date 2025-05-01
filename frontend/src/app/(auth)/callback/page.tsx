"use client"

import { useSession } from "next-auth/react"
import { useRouter } from "next/navigation"
import { useEffect } from "react"

export default function OAuthCallbackHandler() {
	const { data: session } = useSession()
	const router = useRouter()

	useEffect(() => {
		if (session?.idToken) {
			const sendToDjango = async () => {
				try {
					const res = await fetch("http://localhost:8000/api/google/verify", {
						method: "POST",
						headers: {
							"Content-Type": "application/json",
						},
						body: JSON.stringify({ id_token: session.idToken }),
						credentials: "include",
					})

					const data = await res.json()

					if (res.ok) {
						router.push("/dashboard")
					} else {
						sessionStorage.setItem(
							"loginError",
							data?.non_field_errors?.[0] || "Authentication failed."
						)
						router.push("/login")
					}
				} catch (err) {
					console.log(err)
					sessionStorage.setItem(
						"loginError",
						"Unexpected error during authentication."
					)
					router.push("/login")
				}
			}

			sendToDjango()
		}
	}, [session, router])

	return <p>Authenticating with backend...</p>
}
