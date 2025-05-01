export interface AssignmentResponse {
	count: number
	next: null
	previous: null
}

export interface AssignmentItem {
	id: number
	assignment_number: number
	title: string
	due_date: string | null
	lock_date: string | null
	pdf_filepath: string | null
	has_base_code: boolean | null
	moss_report_directory_path: string | null
	bulk_ai_directory_path: string | null
	language: string | null
	has_policy: boolean | null
	course_catalog: number | null
	semester: number | null
}
