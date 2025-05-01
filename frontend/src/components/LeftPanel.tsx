"use client"

import {
	Box,
	List,
	ListItemButton,
	ListItemText,
	Divider,
	Collapse,
} from "@mui/material"
import { ExpandLess, ExpandMore } from "@mui/icons-material"
import { Semester } from "@/types/semesterTypes"
import { useCourseContext } from "@/context/CourseContext"
import { GetSemesters } from "@/controllers/semesters"
import React, { useState, useEffect } from "react"
import { useRouter } from "next/navigation"

export default function LeftPanel() {
	const router = useRouter()
	const { setSemesterId } = useCourseContext()

	const [semOpen, setSemOpen] = useState<boolean>(false)
	const [semesters, setSemesters] = useState<Semester[]>([])
	const [semId, setSemId] = useState<number | null>(null)

	const handleSemesterClick = async (semId: number) => {
		setSemesterId(semId)
		router.push(`/courses?semester=${semId}`)
	}

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

	return (
		<Box
			sx={(theme) => ({
				minWidth: 200,
				maxWidth: 250,
				minHeight: "100%",
				position: "relative",
				backgroundColor: theme.palette.primary.main,
				color: theme.palette.primary.contrastText,
				p: 2,
			})}
		>
			<List>
				<ListItemButton onClick={() => setSemOpen(!semOpen)}>
					<ListItemText primary="Semesters" />
					{semOpen ? <ExpandLess /> : <ExpandMore />}
				</ListItemButton>
				<Collapse in={semOpen} timeout="auto" unmountOnExit>
					<List component="div" disablePadding>
						{semesters.map((semester: Semester) => (
							<React.Fragment key={semester.id}>
								<ListItemButton
									onClick={() => handleSemesterClick(semester.id)}
								>
									<ListItemText>{semester.name}</ListItemText>
								</ListItemButton>
							</React.Fragment>
						))}
						{/* {dummySemesters.map((ds) => (
							<React.Fragment key={ds.id}>
								<ListItemButton>
									<ListItemText>{ds.name}</ListItemText>
								</ListItemButton>
								 
							</React.Fragment>
						))} */}
					</List>
				</Collapse>
				<ListItemButton onClick={() => router.push("/student_comparison")}>
					<ListItemText>Student Comparison</ListItemText>
				</ListItemButton>
				<ListItemButton onClick={() => router.push("/student_submissions")}>
					<ListItemText>Student Submission</ListItemText>
				</ListItemButton>
				<ListItemButton onClick={() => router.push("/assignment_creation/")}>
					<ListItemText>Assignment Creation</ListItemText>
				</ListItemButton>
				<ListItemButton onClick={() => router.push("/plagiarism_report")}>
					<ListItemText>Plagiarism Report</ListItemText>
				</ListItemButton>

				<ListItemButton onClick={() => router.push("/alerts")}>
					<ListItemText>Alerts</ListItemText>
				</ListItemButton>

				<ListItemButton onClick={() => router.push("/account")}>
					<ListItemText>Account</ListItemText>
				</ListItemButton>

				<ListItemButton onClick={() => router.push("/alerts")}>
					<ListItemText>Settings</ListItemText>
				</ListItemButton>
			</List>
		</Box>
	)
}
