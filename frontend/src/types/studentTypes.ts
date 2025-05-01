export interface Student {
	id: number
	email: string
	nshe_id: number
	codegrade_id: number
	ace_id: string
    first_name: string
    last_name: string
}

export interface StudentEnrollmentsResponse {
	count: number
	next: string | null
	previous: string | null
	results: StudentEnrollments[]
}

// Define the interface for the API response
export interface StudentResponse {
    count: number
    next: string | null
    previous: string | null
    results: Response[]
}

export interface StudentEnrollments {
	id: number
	student: Student
	course_instance: number
}



