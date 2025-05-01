"use client"
import Image from "next/image"
import styles from "./page.module.css"
import { useState, useEffect, useCallback } from "react"
import {
	FormControl,
	InputLabel,
	MenuItem,
	Select,
	Box,
	Stack,
	Typography,
	Table,
	TableBody,
	TableCell,
	TableContainer,
	TableHead,
	TableRow,
	Autocomplete,
	Button,
	TextField,
} from "@mui/material"
import Dropdown from "@/components/Dropdown"
import Paper from "@mui/material/Paper"
import { easyFetch } from "@/utils/fetchWrapper"
import { RepeatOneSharp } from "@mui/icons-material"
import { Course, CourseCatalog, CourseResponse } from "@/types/coursesTypes"
import { useAuth } from "@/context/AuthContext"
import { useCourseContext } from "@/context/CourseContext"
import { GetCourses } from "@/controllers/courses"
import { GetSemesters } from "@/controllers/semesters"
import { Semester } from "@/types/semesterTypes"
import { StudentEnrollments } from "@/types/studentTypes"
import { GetStudents } from "@/controllers/students"
import { SubmissionSimilarity } from "@/types/similarity"
import { grey } from "@mui/material/colors"
import { AssignmentResponse, AssignmentItem } from "@/types/assignmentTypes"

//todo: Just implement the text fields and button on page
export default function StudentComparison() {
	const { user } = useAuth()
	const { semesterId } = useCourseContext()
	const [courses, setCourses] = useState<Course[]>([])
	const [courseId, setCourseId] = useState<number | null>(null)
	const [semesters, setSemesters] = useState<Semester[]>([])
	const [semId, setSemId] = useState<number | null>(null)
	const [students, setStudents] = useState<StudentEnrollments[]>([])
	const [assignments, setAssignments] = useState<AssignmentItem[]>([])
	const [asID, setasID] = useState<number | null>(null)

	// Prepping the student comparison query
	const [studentAID, setStudentAID] = useState<number | null>(null)
	const [studentBID, setStudentBID] = useState<number | null>(null)

	// Similarity Scores
	const [similarity, setSimilarity] = useState<SubmissionSimilarity[] | null>(
		null
	)

	// Fetching semesters
	useEffect(() => {
		const fetchSemesters = async () => {
			const data = await GetSemesters()
			if ("results" in data) {
				setSemesters(data.results)
			} else {
				console.error("Failed to load semesters: ", data)
			}
		}

		fetchSemesters()
	}, [])

	useEffect(() => {
		if (!(semId && user?.isLoggedIn)) return
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
	}, [semId, user?.isLoggedIn])

	// Fetch assignments
	useEffect(() => {
		if (!courseId) return
		const fetchAssignments = async () => {
			try {
				const queryParams = new URLSearchParams({
					course_id: String(courseId),
				})

				const response = await easyFetch(
					`http://localhost:8000/api/assignment/assignments/?${queryParams}`,
					{
						method: "get",
					}
				)

				const data = await response.json()

				if (response.ok) {
					setAssignments(data)
				}
			} catch (error) {
				console.error(error)
			}
		}

		fetchAssignments()
	}, [courseId])

	useEffect(() => {
		if (!courseId) return

		const fetchStudents = async () => {
			if (!courseId) return
			const data = await GetStudents(Number(courseId), true)
			if (Array.isArray(data)) {
				setStudents(data)
			} else {
				console.error("Error fetching students.", data)
			}
		}

		fetchStudents()
	}, [courseId])

	// Fetch student comparison
	const fetchSimilarities = useCallback(async () => {
		if (!studentAID || !studentBID) return
		try {
			const response = await easyFetch(
				`http://localhost:8000/api/cheating/submission-similarity-pairs/?submission_id_1__student_id=${studentAID}`,
				{
					method: "GET",
				}
			)

			const data = await response.json()

			if (response.ok) {
				console.log("Similarity results:", data.results)
				setSimilarity(data.results)
			}
		} catch (e) {
			console.error("Error fetching similarities:", e)
		}
	}, [studentAID, studentBID, asID])

	return (
		//The TextField accepts and returns a string written by the user, The dropdown returns an id correlating to a specific
		// class, and the button will handle submitting the data to the database
		<Box
			sx={(theme) => ({
				backgroundColor: theme.palette.background.paper,
				p: 2,
				borderRadius: 1,
			})}
		>
			<Stack
				spacing={2}
				sx={{ justifyContent: "center", alignItems: "center" }}
			>
				<FormControl fullWidth>
					<InputLabel>Semester</InputLabel>
					<Select
						label="Semester"
						value={semId ?? ""}
						onChange={(e) => {
							const id = Number(e.target.value)
							setSemId(Number(e.target.value))
							console.log("Select sem id: ", id)
						}}
					>
						{semesters.map((sem) => (
							<MenuItem key={sem.id} value={sem.id ?? ""}>
								{sem.name}
							</MenuItem>
						))}
					</Select>
				</FormControl>
				{/* Select Courses */}
				<FormControl fullWidth disabled={!semId}>
					<InputLabel>Courses</InputLabel>
					<Select
						label="Courses"
						value={courseId ?? ""}
						onChange={(e) => {
							const id = Number(e.target.value)
							setCourseId(Number(e.target.value))
							console.log("Course id: ", id)
						}}
					>
						{courses.map((course) => (
							<MenuItem key={course.id} value={course.id ?? ""}>
								{course.course_catalog.name}
							</MenuItem>
						))}
					</Select>
				</FormControl>
				<FormControl fullWidth disabled={!semId && !courseId}>
					<InputLabel>Assignment</InputLabel>
					<Select
						label="Assignment"
						value={asID ?? ""}
						onChange={(e) => {
							const id = Number(e.target.value)
							setasID(Number(e.target.value))
							console.log("Assignment ID: ", id)
						}}
					>
						{assignments.map((assignment) => (
							<MenuItem key={assignment.id} value={assignment.id ?? ""}>
								{assignment.title}
							</MenuItem>
						))}
					</Select>
				</FormControl>
				<FormControl fullWidth disabled={!semId && !courseId}>
					<InputLabel>Student</InputLabel>
					<Select
						label="Student"
						value={studentAID ?? ""}
						onChange={(e) => {
							const id = Number(e.target.value)
							setStudentAID(Number(e.target.value))
							console.log("Student id: ", id)
						}}
					>
						{students.map((student) => (
							<MenuItem key={student.id} value={student.id ?? ""}>
								{student.student.first_name} {student.student.last_name}
							</MenuItem>
						))}
					</Select>
				</FormControl>
				<FormControl fullWidth disabled={!semId && !courseId}>
					<InputLabel>Student</InputLabel>
					<Select
						label="Student"
						value={studentBID ?? ""}
						onChange={(e) => {
							const id = Number(e.target.value)
							setStudentBID(Number(e.target.value))
							console.log("Student id: ", id)
						}}
					>
						{students.map((student) => (
							<MenuItem key={student.id} value={student.id ?? ""}>
								{student.student.first_name} {student.student.last_name}
							</MenuItem>
						))}
					</Select>
				</FormControl>
				<Button
					variant="contained"
					onClick={fetchSimilarities}
					disabled={!asID || !studentAID || !studentBID}
				>
					Query Similarities
				</Button>
				{similarity && (
					<TableContainer sx={{ backgroundColor: grey[200] }}>
						<Table
							sx={{ width: 650, justifyContent: "center" }}
							aria-label="simple table"
						>
							<TableHead>
								<TableRow>
									<TableCell>Student Name</TableCell>
									<TableCell>Relative</TableCell>
								</TableRow>
							</TableHead>
							<TableBody>
								{similarity.map((sim: SubmissionSimilarity) => (
									<TableRow key={sim.id}>
										<TableCell>
											{sim.submission_id_1.student.first_name}{" "}
											{sim.submission_id_1.student.last_name} &amp;{" "}
											{sim.submission_id_2.student.first_name}{" "}
											{sim.submission_id_2.student.last_name}
										</TableCell>
										<TableCell>{sim.percentage}%</TableCell>
									</TableRow>
								))}
							</TableBody>
						</Table>
					</TableContainer>
				)}
			</Stack>
		</Box>
	)
}
