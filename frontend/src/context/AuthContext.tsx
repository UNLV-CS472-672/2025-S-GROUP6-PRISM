import { createContext, useContext, useEffect, useState } from "react"
import { User } from "@/types"

// This just defines the structure of the context
interface AuthContextType {
	user: User | null
	login: (userData: User) => void
	logout: () => void
}

// Creation of context
export const AuthContext = createContext<AuthContextType | undefined>(undefined)

// This is the global provider that is set in Providers.tsx to allow for the application to use context fields
const AuthProvider = ({ children }: { children: React.ReactNode }) => {
	const [user, setUser] = useState<User | null>(null)

	// Loading the user from local storage on page load
	useEffect(() => {
		const storedUser = localStorage.getItem("user")
		if (storedUser) setUser(JSON.parse(storedUser))
	}, [])

	// Set user with login
	const login = (userData: User) => {
		userData.isLoggedIn = true
		setUser(userData)
		localStorage.setItem("user", JSON.stringify(userData))
	}

	// Set the user to null on logout
	const logout = () => {
		setUser(null)
		localStorage.removeItem("user")
	}

	return (
		<AuthContext.Provider value={{ user, login, logout }}>
			{children}
		</AuthContext.Provider>
	)
}

// Other components can use this to access context fields
export const useAuth = () => {
	const context = useContext(AuthContext)
	if (context) return context
	throw new Error("useAuth must be used within an AuthProvider")
}

export default AuthProvider
