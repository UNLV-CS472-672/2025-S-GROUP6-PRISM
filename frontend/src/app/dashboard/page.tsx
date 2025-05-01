/* 
    Variables: Class_Name, Section_Name, Display_Name
    Buttons: Menu, Student Comparison, Alerts, n boxes with Class_Names and Section_Names
    Links: Student Comparison Tool, Alerts, Each box
    Header takes Display_Name
*/

"use client"
import { useEffect, useState } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import { Typography } from "@mui/material"
// import styled, { keyframes } from "styled-components"
import { GetCourses } from "@/controllers/courses"
import { Course } from "@/types/coursesTypes"
import { useAuth } from "@/context/AuthContext"
import { useCourseContext } from "@/context/CourseContext"

export default function Dashboard() {
	return <div>Dashboard</div>
}

// interface for course and user

/*
	Pulls semesterId from URL
	Pulls professorId from user.id
	Calls your GetCourses() API wrapper
	Updates course and semester ID into global context
	Routes to course detail page on click 
*/

// function Dashboard() {
// 	const router = useRouter()
// 	const searchParams = useSearchParams()
// 	const { user } = useAuth()
// 	const { setCourseInstanceId, setSemesterId } = useCourseContext()
// 	const semesterId = Number(searchParams.get("semester"))
// 	const [courses, setCourses] = useState<Course[]>([])

// 	// gets semesterid from URL and userId from context
// 	useEffect(() => {
// 		if (!semesterId || !user?.id) return

// 		setSemesterId(semesterId) // update context

// 		// calls API wrapper to get courses
// 		// and sets them into state
// 		// if error, log it to console
// 		GetCourses(semesterId, user.id).then((res) => {
// 			if ("courses" in res) {
// 				setCourses(res.courses)
// 			} else {
// 				console.error(res.detail || res.message)
// 			}
// 		})
// 	}, [semesterId, user])

// 	// function to handle course click
// 	// sets courseInstanceId in context and routes to course detail page
// 	const handleCourseClick = (courseId: number) => {
// 		setCourseInstanceId(courseId) // update context
// 		router.push(`/courses/${courseId}`)
// 	}

// 	return (
// 		<Container>
// 			<CompareButtons>
// 				<Button onClick={() => router.push("/student_comparison")}>
// 					Compare Students
// 				</Button>
// 				<Button onClick={() => router.push("/alerts")}>Alerts</Button>
// 			</CompareButtons>

// 			<Sections>
// 				{courses.map((course) => (
// 					<Button key={course.id} onClick={() => handleCourseClick(course.id)}>
// 						<Typography>
// 							{course.course_catalog.name} — {course.section_number}
// 						</Typography>
// 						<Typography>{course.course_catalog.course_title}</Typography>
// 					</Button>
// 				))}
// 			</Sections>
// 		</Container>
// 	)
// }

// // Center everything inside the main Container
// const Container = styled.div`
// 	display: flex;
// 	flex-direction: column;
// 	align-items: center;
// 	justify-content: center;
// 	padding: 2rem;
// 	text-align: center;
// `

// const pastelDiagonal = keyframes`
//   0% {
//     background-position: bottom left;
//   }
//   100% {
//     background-position: top right;
//   }
// `

// const CompareButtons = styled.div`
// 	Button {
// 		width: 250px; /* Increase width */
// 		height: 60px; /* Increase height */
// 		padding: 1.5rem; /* More internal spacing */
// 		font-size: 1.2rem; /* Larger text */
// 		display: inline-flex;
// 		justify-content: center;
// 		align-items: center;
// 		border: 1px solid #ccc;
// 		border-radius: 8px;
// 		box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
// 		margin: 1rem;
// 	}
// `

// const Button = styled.button`
// 	&:hover {
// 		background: linear-gradient(
// 		135deg,
// 		#ffe7e0,
// 		#e0f7fa,
// 		#e8f0fe,
// 		#f3e5f5,
// 		#ffe7e0
// 		);
// 		background-size: 400% 400%;
// 		animation: ${pastelDiagonal} 1s forwards;
// 		transform: scale(1.05);
// 		box-shadow: 0 6px 18px rgba(0, 0, 0, 0.1);
// 	}

// 	&:not(:hover) {
// 		background-color: white; /* resets back to white */
// 	}
// }`

// const Sections = styled.div`
// 	margin-top: 5rem; /* Add spacing above the section buttons */

// 	Button {
// 		width: 300px;
// 		height: 250px;
// 		padding: 1.5rem;
// 		font-size: 1.2rem;
// 		display: inline-flex;
// 		justify-content: center;
// 		align-items: center;
// 		border: 1px solid #ccc;
// 		border-radius: 8px;
// 		box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
// 		margin: 1rem;
// 	}
// `

// export default Dashboard
