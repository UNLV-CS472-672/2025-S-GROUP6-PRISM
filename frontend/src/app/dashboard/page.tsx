/* 
    Variables: Class_Name, Section_Name, Display_Name
    Buttons: Menu, Student Comparison, Alerts, n boxes with Class_Names and Section_Names
    Links: Student Comparison Tool, Alerts, Each box
    Header takes Display_Name
*/

// this is from stevens pr #39 - freatures/authentication

"use client"
import { Container, Typography, Button } from "@mui/material"
import { SignOutButton } from "@/components/AuthenticationMethod"
import { StudentComparison } from "@/student_comparison" // student comparison
import { Alerts } from "@/app/alerts" // alerts
import { useCallback, useEffect, useState } from "react"
import { useRouter } from "next/Navigation"
import type { User } from "@/types/index"
import { easyFetch } from "@/utils/fetchWrapper"

// basic button layout -> need to acces how many sections a teacher has, their name and the class name -> create that many buttons/div containers to show
// updated 3/25: added place holder buttons for now to get the styling done for future prs
// need an additional 2 buttons that are alwasy set for compare students and alerts -> respective buttons must be linked to their respective pages
// updated 3/25: buttons now link to the two respective pages using the onlick function and pushing to the route
// an alert icon appears if a section has been detected of cheating
// Updated 3/25: added a section that is set to false, this will be used to add the icon on the sections

// come back to header and replace the teacher name with the actual name assigned to the profile

// one div for the creating the sections

// the other div for the two main buttons at the top

// added a button for the menu in the header

// waiting to see how the team is going to approach linking the pages
// Updated: using stevens pr I updated the buttons to prep for the linking of the pages

// first porition of dashboard is added from stevens pr #39 - features/authentication
function Dashboard() {
	const router = useRouter()

	const [courses, setCourses] = useState<any[]>([])
	const [users, setUsers] = useState<User[]>([])
	const [loading, setLoading] = useState<Boolean>(false)
	const [asNumber, setAsNumber] = useState("")
	const [classInstance, setClassInstance] = useState("")
	const [assignments, setAssignments] = useState<any[]>([])

	const fetchUsers = useCallback(async () => {
		setLoading(true)
		try {
			const response = await easyFetch("http://localhost:8000/api/user/users", {
				method: "get",
			})

			const data = await response.json()

			if (response.ok) {
				setUsers(data)
			} else {
				console.error("Failed to fetch users.")
			}
		} catch (err) {
			console.error(err)
		} finally {
			setLoading(false)
		}
	}, [])

	useEffect(() => {
		const fetchCourses = async () => {
			try {
				const response = await fetch(
					"http://localhost:8000/api/course/sectionclassprof",
					{
						method: "get",
					}
				)

				const data = await response.json()

				if (response.ok) {
					setCourses(data)
				}
			} catch (error) {
				console.error(error)
			}
		}

		fetchCourses()
	}, [])

	const fetchAssignments = useCallback(async () => {
		try {
			const response = await fetch(
				`http://localhost:8000/api/assignment/assignments?assignment_number=${asNumber}&class_instance__name=${classInstance}`,
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
	}, [])

	return (
		<Container>
			<SignOutButton />
			<Button onClick={fetchUsers}>
				{loading ? "loading..." : "Fetch Users"}
			</Button>
			{users.length > 0 && (
				<div>
					{users.map((user, index) => (
						<ul key={index}>
							<li>{user.email}</li>
							<li>{user.first_name}</li>
							<li>{user.last_name}</li>
						</ul>
					))}
				</div>
			)}

			{/* Main banner */}
			<div>
				<div className="Banner">
					<button>menu</button>
					<h1>"Hello, teacher name"</h1>
				</div>

				{/* 2 buttons -> compare students, alerts */}
				<div className="comapreButtons">
					<button onClick={() => router.push("/student_comparison")}>
						Compare Students
					</button>
					<button onClick={() => router.push("/alerts")}>Alerts</button>
				</div>

				{/* Main Navigation */}
				<div className="sections">
					{Array.from({ length: 6 }).map((_, index) => (
						<button key={index} className="action-button">
							Action Button {index + 1}
						</button>
					))}
				</div>
			</div>
			<Container>
				{courses.map((course, index) => (
					<Container key={index}>
						<Button onClick={fetchAssignments}>{course.section_number}</Button>
					</Container>
				))}
			</Container>
		</Container>
	)
}

export default Dashboard
