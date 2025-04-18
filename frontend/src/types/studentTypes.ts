// Define the interface for a single semester
export interface Student {
	id: number
	email: string
	nshe_id: number
	codegrade_id: number
	ace_id: string
    first_name: string
    last_name: string
}

// define response kek
export interface Response {
    id: number
    students: Student
    student: number
    course_instance: number
}

// Define the interface for the API response
export interface StudentResponse {
    count: number
    next: string | null
    previous: string | null
    results: Response[]
}