export interface Student {
	id: number
	email: string
	nshe_id: number
	codegrade_id: number
	ace_id: string
	first_name: string
	last_name: string
}

export interface StudentResponse {
	count: number
	next: string | null
	previous: string | null
	results: Student[]
}
