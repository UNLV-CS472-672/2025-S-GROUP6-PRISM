export interface Assignment {
    id: number
    assignment_number: number
    title: string
    lock_date: string
    pdf_filepath: string
    has_base_code: boolean
    moss_report_directory_path: string
    bulk_ai_directory_path: string
    language: string
    has_policy: boolean
    course_instance: number
}

export interface AssignmentResponse {
    count: number
    next: string | null
    previous: string | null
    results: Assignment[]
}