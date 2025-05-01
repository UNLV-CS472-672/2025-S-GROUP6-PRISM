"use client"
import {
	createContext,
	useContext,
	useState,
	useEffect,
	ReactNode,
} from "react"

interface CourseContextType {
	courseInstanceId: number | null
	setCourseInstanceId: (id: number | null) => void
	semesterId: number | null
	setSemesterId: (id: number | null) => void
}

const CourseContext = createContext<CourseContextType | undefined>(undefined)

export function CourseProvider({ children }: { children: ReactNode }) {
	const [courseInstanceId, setCourseInstanceIdState] = useState<number | null>(
		null
	)
	const [semesterId, setSemesterIdState] = useState<number | null>(null)

	// Load from localStorage on mount
	useEffect(() => {
		const storedCourseInstanceId = localStorage.getItem("course_instance_id")
		if (storedCourseInstanceId) {
			setCourseInstanceIdState(Number(storedCourseInstanceId))
		}
		const storedSemesterId = localStorage.getItem("semester_id")
		if (storedSemesterId) {
			setSemesterIdState(Number(storedSemesterId))
		}
	}, [])

	// Sync courseInstanceId
	const setCourseInstanceId = (id: number | null) => {
		setCourseInstanceIdState(id)
		if (id !== null) {
			localStorage.setItem("course_instance_id", id.toString())
		} else {
			localStorage.removeItem("course_instance_id")
		}
	}

	// Sync semesterId
	const setSemesterId = (id: number | null) => {
		setSemesterIdState(id)
		if (id !== null) {
			localStorage.setItem("semester_id", id.toString())
		} else {
			localStorage.removeItem("semester_id")
		}
	}

	return (
		<CourseContext.Provider
			value={{
				courseInstanceId,
				setCourseInstanceId,
				semesterId,
				setSemesterId,
			}}
		>
			{children}
		</CourseContext.Provider>
	)
}

export function useCourseContext() {
	const context = useContext(CourseContext)
	if (!context) {
		throw new Error("useCourseContext must be used within CourseProvider")
	}
	return context
}
