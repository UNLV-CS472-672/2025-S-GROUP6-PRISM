import { easyFetch } from "@/utils/fetchWrapper"
import { Assignment, AssignmentResponse } from "@/types/assignmentTypes"
import { APIError } from "@/types/APIError"
//need to get all assignments from specific course by course ID
export async function GetAssignments(id: number): Promise<AssignmentResponse | APIError> {
    try {
        const response = await easyFetch(
            `http://localhost:8000/api/assignment/assignments/?course_instance=${id}`,
            { method: "GET" }
        )

        const data = await response.json()

        if (response.ok) {
            return data.results as AssignmentResponse
        } else {
            return {
                detail: data.detail ?? "Failed to fetch assignments.",
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