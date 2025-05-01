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
						console.error("Django auth failed", data)
					}
				} catch (err) {
					console.error("Error sending token to Django", err)
				}
			}

			sendToDjango()
		}
	}, [session, router])

	return <p>Authenticating with backend...</p>
}
