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
import { dummySemesters } from "@/data/dummySemesters"
import { GetSemesters } from "@/controllers/semesters"
import React, { useState, useEffect } from "react"
import { useRouter } from "next/navigation"

export default function LeftPanel() {
	const router = useRouter()

	const [semOpen, setSemOpen] = useState<boolean>(false)
	const [semesters, setSemesters] = useState<Semester[]>([])
	const [semesterId, setSemesterId] = useState<number | null>(null)

	const handleSemesterClick = async (semesterId: number) => {
		router.push(`/courses?semester=${semesterId}`)
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
			sx={{
				width: 250,
				minWidth: 200,
				borderRadius: 1,
				border: "1px solid white",
				p: 2,
			}}
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
								<Divider />
							</React.Fragment>
						))}
						{dummySemesters.map((ds) => (
							<React.Fragment key={ds.id}>
								<ListItemButton>
									<ListItemText>{ds.name}</ListItemText>
								</ListItemButton>
								<Divider />
							</React.Fragment>
						))}
					</List>
				</Collapse>
				<Divider />
				<ListItemButton onClick={() => router.push("/student_comparison")}>
					<ListItemText>Student Comparison</ListItemText>
				</ListItemButton>
				<Divider />
				<ListItemButton onClick={() => router.push("/plagiarism_report")}>
					<ListItemText>Plagiarism Report</ListItemText>
				</ListItemButton>
				<Divider />
				<ListItemButton onClick={() => router.push("/alerts")}>
					<ListItemText>Alerts</ListItemText>
				</ListItemButton>
				<Divider />
				<ListItemButton onClick={() => router.push("/account")}>
					<ListItemText>Account</ListItemText>
				</ListItemButton>
				<Divider />
				<ListItemButton onClick={() => router.push("/alerts")}>
					<ListItemText>Settings</ListItemText>
				</ListItemButton>
				<Divider />
			</List>
		</Box>
	)
}
