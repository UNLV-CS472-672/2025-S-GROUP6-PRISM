"use client"
import React, { useEffect, useState } from "react"
import {
	Typography,
	Box,
	Button,
	List,
	ListItemButton,
	ListItemText,
	Table,
	TableBody,
	TableCell,
	TableContainer,
	TableHead,
	TableRow,
	Paper,
} from "@mui/material"
import { useRouter } from "next/navigation"
import { easyFetch } from "@/utils/fetchWrapper"
import { useCourseContext } from "@/context/CourseContext"
import { GetStudents } from "@/controllers/students"
import { StudentEnrollments } from "@/types/studentTypes"
import { blueGrey, grey } from "@mui/material/colors"
/**
 * This will be used for pagination if necessary
 */
interface AssignmentResponse {
	count: number
	next: null
	previous: null
}

interface AssignmentItem {
	id: number
	assignment_number: number
	title: string
	due_date: string | null
	lock_date: string | null
	pdf_filepath: string | null
	has_base_code: boolean | null
	moss_report_directory_path: string | null
	bulk_ai_directory_path: string | null
	language: string | null
	has_policy: boolean | null
	course_catalog: number | null
	semester: number | null
}

const assignmentFields = [
	"Title",
	"Assignment Number",
	"Due Date",
	"Lock Date",
	"Details",
]

export default function Assignments() {
	const router = useRouter()

	const [assignments, setAssignments] = useState<AssignmentItem[]>([])
	const [students, setStudents] = useState<StudentEnrollments[]>([])
	const { courseInstanceId } = useCourseContext()

	useEffect(() => {
		if (!courseInstanceId) return
		const fetchAssignments = async () => {
			try {
				const queryParams = new URLSearchParams({
					course_id: String(courseInstanceId),
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

		const fetchStudents = async () => {
			if (!courseInstanceId) return
			const data = await GetStudents(Number(courseInstanceId), true)
			if (Array.isArray(data)) {
				setStudents(data)
			} else {
				console.error("Error fetching students.", data)
			}
		}

		fetchAssignments()
		fetchStudents()
	}, [courseInstanceId])

	return (
		<Box sx={{ display: "flex", gap: 2 }}>
			{/* Body Container */}
			<Box
				sx={(theme) => ({
					backgroundColor: theme.palette.background.paper,
					p: 2,
					boxShadow: 2,
					overflowX: "auto",
					width: "100%",
					height: "100%",
				})}
			>
				{/* Page buttons */}
				<Box
					sx={{
						display: "flex",
						gap: 2,
						mb: 2,
					}}
				>
					<Button variant="contained">Upload Assignment</Button>
					<Button variant="contained">Export All</Button>
				</Box>
				<Box sx={{ overflowY: "auto", height: "100%" }}>
					<TableContainer
						sx={{ backgroundColor: grey[200], p: 1, borderRadius: 1 }}
					>
						<Table sx={{}}>
							<TableHead>
								<TableRow>
									{assignmentFields.map((field, index) => (
										<TableCell key={index}>{field}</TableCell>
									))}
								</TableRow>
							</TableHead>
							<TableBody>
								{assignments.map((assignment) => (
									<TableRow key={assignment.id}>
										<TableCell>{assignment.title}</TableCell>
										<TableCell>{assignment.assignment_number}</TableCell>
										<TableCell>{assignment.due_date}</TableCell>
										<TableCell>{assignment.lock_date}</TableCell>
										<TableCell>
											<Button
												onClick={() =>
													router.push(
														`/courses/${courseInstanceId}/assignments/graphs?assignment=${assignment.id}`
													)
												}
											>
												View
											</Button>
										</TableCell>
									</TableRow>
								))}
							</TableBody>
						</Table>
					</TableContainer>
				</Box>
			</Box>
			{/* Students List */}
			<Box
				sx={(theme) => ({
					p: 2,
					minWidth: 200,
					maxWidth: 400,
					flexShrink: 0,
					flexGrow: 0,
					backgroundColor: theme.palette.background.paper,
					boxShadow: 2,
					position: "sticky",
					top: 0,
					height: "100vh",
					overflowY: "auto",
					zIndex: 10,
				})}
			>
				<Typography>Students</Typography>
				<List component="div" disablePadding>
					{students.map((student) => (
						<React.Fragment key={student.id}>
							<ListItemButton sx={{ width: "100%" }}>
								<ListItemText
									primary={`${student.student.first_name} ${student.student.last_name}`}
									sx={{ width: "100%" }}
								/>
							</ListItemButton>
						</React.Fragment>
					))}
				</List>
			</Box>
		</Box>
	)
}
