"use client"
import Image from "next/image"
import styles from "./page.module.css"
import { useState, useEffect, useCallback } from "react"
import { FormControl, InputLabel, MenuItem, Select } from "@mui/material"
import Dropdown from "../components/Dropdown"
import Box from "@mui/material/Box"
import Stack from "@mui/material/Stack"
import Typography from "@mui/material/Typography"
import Table from "@mui/material/Table"
import TableBody from "@mui/material/TableBody"
import TableCell from "@mui/material/TableCell"
import TableContainer from "@mui/material/TableContainer"
import TableHead from "@mui/material/TableHead"
import TableRow from "@mui/material/TableRow"
import Paper from "@mui/material/Paper"
import { easyFetch } from "@/utils/fetchWrapper"
import { RepeatOneSharp } from "@mui/icons-material"

const list = [
	{ Label: "CS 301", id: 0 },
	{ Label: "CS 405", id: 1 },
	{ Label: "CS 477", id: 2 },
]
const cs301 = [
	{ Label: "Alice", id: 0 },
	{ Label: "Dave", id: 1 },
	{ Label: "Johnny", id: 2 },
]
const cs405 = [
	{ Label: "Marisa", id: 3 },
	{ Label: "Bob", id: 4 },
	{ Label: "Jimmy", id: 5 },
]
const cs477 = [
	{ Label: "Jack", id: 6 },
	{ Label: "Alex", id: 7 },
	{ Label: "Dizzy", id: 8 },
]
const aliceAndJohnny = [
	{ name: "Assignment 1", similarity: 21 },
	{ name: "Assignment 2", similarity: 15 },
]
const aliceAndDave = [
	{ name: "Assignment 1", similarity: 45 },
	{ name: "Assignment 2", similarity: 41 },
]
const daveAndJohnny = [
	{ name: "Assignment 1", similarity: 10 },
	{ name: "Assignment 2", similarity: 31 },
]
const marisaAndJimmy = [
	{ name: "Assignment 1", similarity: 44 },
	{ name: "Assignment 2", similarity: 41 },
	{ name: "Assignment 3", similarity: 42 },
]
const marisaAndBob = [
	{ name: "Assignment 1", similarity: 21 },
	{ name: "Assignment 2", similarity: 25 },
	{ name: "Assignment 3", similarity: 21 },
]
const bobAndJimmy = [
	{ name: "Assignment 1", similarity: 15 },
	{ name: "Assignment 2", similarity: 12 },
	{ name: "Assignment 3", similarity: 18 },
]
const jackAndAlex = [
	{ name: "Fibbonacci", similarity: 12 },
	{ name: "Dynamic Programming", similarity: 14 },
	{ name: "Sorting Algorithms", similarity: 18 },
]
const jackAndDizzy = [
	{ name: "Fibbonacci", similarity: 21 },
	{ name: "Dynamic Programming", similarity: 10 },
	{ name: "Sorting Algorithms", similarity: 15 },
]
const alexAndDizzy = [
	{ name: "Fibbonacci", similarity: 15 },
	{ name: "Dynamic Programming", similarity: 17 },
	{ name: "Sorting Algorithms", similarity: 13 },
]
//todo: Just implement the text fields and button on page
export default function StudentComparison() {
	const fetchAssignments = useCallback(async () => {
		const response = await fetch(`http://localhost:8000/api/course/classes`, {
			method: "get",
		})

		const data = await response.json()

		if (response.ok) {
			console.log("Fetched assignments:", data)
			setSelectedClass(data)
		}
	}, [])

	const fetchStudents = useCallback(async (classInstanceId: string) => {
		const response = await fetch(
			`http://localhost:8000/api/course/enrollments?class_instance${classInstanceId}`,
			{
				method: "get",
			}
		)

		const data = await response.json()

		if (response.status === 200) {
			console.log("GOOD REQUEST")
			setStudents(data)
		}
	}, [])

	//These are variables needed for work and their default values
	const [selectedClass, setSelectedClass] = useState<any[]>([])
	const [assignmentName, setAssignmentName] = useState("")
	const [student, setStudents] = useState<any[]>([])
	let studentList = [
		{ Label: "null", id: -1 },
		{ Label: "null2", id: -1 },
	]
	let isDisabled = true
	let rows = [
		{ name: "Please select", similarity: 0 },
		{ name: "Different students", similarity: 0 },
		{ name: "From same class", similarity: 0 },
	]
	let overall = 0

	//this if else HELL is meant for testing
	// i dont know if react has switch case and dont want to waste time looking for it
	// if (selectedClass == 0) {
	// 	studentList = cs301
	// 	isDisabled = false
	// } else if (selectedClass == 1) {
	// 	studentList = cs405
	// 	isDisabled = false
	// } else if (selectedClass == 2) {
	// 	studentList = cs477
	// 	isDisabled = false
	// } else {
	// 	isDisabled = true
	// }
	// if (
	// 	(studentOne == 0 && studentTwo == 1) ||
	// 	(studentOne == 1 && studentTwo == 0)
	// ) {
	// 	rows = aliceAndDave
	// 	overall = 43
	// } else if (
	// 	(studentOne == 0 && studentTwo == 2) ||
	// 	(studentOne == 2 && studentTwo == 0)
	// ) {
	// 	rows = aliceAndJohnny
	// 	overall = 18
	// } else if (
	// 	(studentOne == 2 && studentTwo == 1) ||
	// 	(studentOne == 1 && studentTwo == 2)
	// ) {
	// 	rows = daveAndJohnny
	// 	overall = 20.5
	// } else if (
	// 	(studentOne == 3 && studentTwo == 4) ||
	// 	(studentOne == 4 && studentTwo == 3)
	// ) {
	// 	rows = marisaAndBob
	// 	overall = 22.333
	// } else if (
	// 	(studentOne == 3 && studentTwo == 5) ||
	// 	(studentOne == 5 && studentTwo == 3)
	// ) {
	// 	rows = marisaAndJimmy
	// 	overall = 42.333
	// } else if (
	// 	(studentOne == 4 && studentTwo == 5) ||
	// 	(studentOne == 5 && studentTwo == 4)
	// ) {
	// 	rows = bobAndJimmy
	// 	overall = 15
	// } else if (
	// 	(studentOne == 6 && studentTwo == 7) ||
	// 	(studentOne == 7 && studentTwo == 6)
	// ) {
	// 	rows = jackAndAlex
	// 	overall = 14.666
	// } else if (
	// 	(studentOne == 6 && studentTwo == 8) ||
	// 	(studentOne == 8 && studentTwo == 6)
	// ) {
	// 	rows = jackAndDizzy
	// 	overall = 15.333
	// } else if (
	// 	(studentOne == 7 && studentTwo == 8) ||
	// 	(studentOne == 8 && studentTwo == 7)
	// ) {
	// 	rows = alexAndDizzy
	// 	overall = 15
	// }

	//These are for input into our components to be able to extract values from them
	// const handleClassSelect = (item: number | -1) => {
	// 	setSelectedClass(item)
	// }
	// const handleStudentOne = (item: number | -1) => {
	// 	setStudentOne(item)
	// }
	// const handleStudentTwo = (item: number | -1) => {
	// 	setStudentTwo(item)
	// }
	// const handleButtonClick = () => {
	// 	console.log(selectedClass)
	// 	console.log(assignmentName)
	// }
	return (
		//The TextField accepts and returns a string written by the user, The dropdown returns an id correlating to a specific
		// class, and the button will handle submitting the data to the database
		<Stack spacing={1} sx={{ justifyContent: "center", alignItems: "center" }}>
			<FormControl fullWidth>
				<InputLabel id="demo-simple-select-label">Classes</InputLabel>
				<Select
					labelId="demo-simple-select-label"
					id="demo-simple-select"
					label="Age"
					onOpen={fetchAssignments}
				>
					{selectedClass?.map((value, index) => (
						<MenuItem key={index} value={value}>
							{value["name"]}
						</MenuItem>
					))}
				</Select>
			</FormControl>
			{/* <Dropdown isDisabled={false} items={list} dropdownLabel="Classes" /> */}
			<Stack direction="row">
				<Dropdown
					isDisabled={isDisabled}
					items={studentList}
					dropdownLabel="Select Student"
				/>
				<Dropdown
					isDisabled={isDisabled}
					items={studentList}
					dropdownLabel="Select a different student"
				/>
			</Stack>
			<Typography variant="h5" gutterBottom>
				{"Overall - " + overall}
			</Typography>
			<TableContainer component={Paper}>
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
						{rows.map((row) => (
							<TableRow
								key={row.name}
								sx={{ "&:last-child td, &:last-child th": { border: 0 } }}
							>
								<TableCell component="th" scope="row">
									{row.name}
								</TableCell>
								<TableCell>{row.similarity}</TableCell>
							</TableRow>
						))}
					</TableBody>
				</Table>
			</TableContainer>
		</Stack>
	)
}
