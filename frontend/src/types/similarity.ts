export type SubmissionSimilarity = {
	id: number
	submission_id_1: {
		id: number
		student: {
			id: number
			email: string
			nshe_id: number
			codegrade_id: number
			ace_id: string
			first_name: string
			last_name: string
		}
		grade: string
		created_at: string
		flagged: boolean
		file_path: string
		assignment: number
		course_instance: number
	}
	submission_id_2: {
		id: number
		student: {
			id: number
			email: string
			nshe_id: number
			codegrade_id: number
			ace_id: string
			first_name: string
			last_name: string
		}
		grade: string
		created_at: string
		flagged: boolean
		file_path: string
		assignment: number
		course_instance: number
	}
	file_name: string
	match_id: number
	percentage: number
	assignment: number
}
