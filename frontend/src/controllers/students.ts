import { easyFetch } from "@/utils/fetchWrapper"

import { Student, Response, StudentResponse } from "@/types/studentTypes"
import { APIError } from "@/types/APIError"
//need to get all students from specific course by course ID
export async function GetStudents(id: number): Promise<StudentResponse | APIError> {
	try {
		const response = await easyFetch(
			`http://localhost:8000/api/course/studentenrollments/?course_instance=${id}`,
			{ method: "GET" }
		)

		const data = await response.json()

		if (response.ok) {
			return data.results as StudentResponse
		} else {
			return {
				detail: data.detail ?? "Failed to fetch students.",
				status: response.status,
			}
		}
	} catch (e) {
		console.error(e)
		return {
			message: "Something went wrong during fetch.",
		}
	}
}
