"use client"
import react from "react"
import { useState } from "react"
import Dropdown from "@/components/Dropdown"
import FileDrop from "@/components/Filedrop"
import { File } from "@/components/Filedrop"
import TextField from "@mui/material/TextField"
import Button from "@mui/material/Button"
import Stack from "@mui/material/Stack"
import Typography from "@mui/material/Typography"

export default function AssignmentCreation() {
	// Keeps track of the files uploaded
	const [fileUploads, setFileUploads] = useState<File[]>([])

	const [selectedClass, setSelectedClass] = useState(-1)
	const [assignmentName, setAssignmentName] = useState("")
	const list = [
		{ Label: "CS 301", id: 0 },
		{ Label: "CS 405", id: 1 },
		{ Label: "CS 477", id: 2 },
	]

	const handleItemSelect = (item: number | -1) => {
		setSelectedClass(item)
	}

	const handleButtonClick = () => {
		console.log(selectedClass)
		console.log(assignmentName)
		console.log("Additional File Uploads: ")
		fileUploads.forEach((file) => console.log(file.name))
	}

	// Change Browser Tab Title to Actual Page Title
	react.useEffect(() => {
		document.title = "Assignment Creation"
	}, [])

	return (
		//The TextField accepts and returns a string written by the user, The dropdown returns an id correlating to a specific
		// class, and the button will handle submitting the data to the database
		<Stack
			spacing={2}
			sx={{ justifyContent: "center", alignItems: "center", paddingTop: 2 }}
		>
			<Typography variant="h3">Assignment Creation</Typography>
			<Typography variant="h6" gutterBottom>
				Please insert Assignment Name below.
			</Typography>
			<TextField
				required
				id="Assignment_Name"
				label="Assignment Name"
				variant="outlined"
				onChange={(event: React.ChangeEvent<HTMLInputElement>) => {
					setAssignmentName(event.target.value)
				}}
			/>
			<Typography variant="h6" gutterBottom>
				Please select which class the assignment is for below.
			</Typography>
			<Dropdown
				isDisabled={false}
				items={list}
				onSelectItem={handleItemSelect}
				dropdownLabel="Classes"
			/>
			<Typography variant="h6" gutterBottom>
				Upload any Additional Files {"("}Skeleton Code, Assignment PDF, etc.
				{")"}
			</Typography>
			<FileDrop files={fileUploads} setFileUploads={setFileUploads} />
			<Button
				variant="outlined"
				onClick={() => {
					handleButtonClick()
				}}
			>
				Create assignment
			</Button>
		</Stack>
	)
}
