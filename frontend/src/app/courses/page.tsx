"use client"
import { Typography, Box } from "@mui/material"
import CourseCards from "@/components/CourseCards"
import { useRouter } from "next/navigation"
import { GetCourses } from "@/controllers/courses"
import { Course, CourseCatalog, CourseResponse } from "@/types/coursesTypes"
import { useAuth } from "@/context/AuthContext"
import { useEffect, useState } from "react"
import { useCourseContext } from "@/context/CourseContext"

export default function Courses() {
	const router = useRouter()
	const { user } = useAuth()
	const { setCourseInstanceId, semesterId } = useCourseContext()
	const [courses, setCourses] = useState<Course[]>([])

	const courseClick = (courseInstanceId: number, catalogId: number) => {
		setCourseInstanceId(courseInstanceId)
		router.push(`/courses/${catalogId}/assignments`)
	}

	useEffect(() => {
		if (!(semesterId && user?.professor_id)) return
		console.log(semesterId)
		const loadCourses = async () => {
			const data = await GetCourses(
				Number(semesterId),
				Number(user?.professor_id)
			)
			if ("results" in data) {
				setCourses(data.results)
			} else {
				console.error("Error fetching courses: ", data)
			}
		}

		loadCourses()
	}, [semesterId, user?.professor_id])

	return (
		<Box
			sx={(theme) => ({
				backgroundColor: theme.palette.background.paper,
				height: "100%",
				p: 2,
			})}
		>
			<Box>
				<Box sx={{ mb: 2 }}>
					<Typography variant="h4">Welcome {user?.first_name}</Typography>
				</Box>
				<Box className="sections">
					{courses.map((course) => (
						<CourseCards
							key={course.id}
							onClick={() => courseClick(course.id, course.course_catalog.id)}
						>
							<Typography>{course.course_catalog.name}</Typography>
							<Typography>{course.section_number}</Typography>
						</CourseCards>
					))}
				</Box>
			</Box>
		</Box>
	)
}
