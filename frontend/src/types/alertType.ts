export interface Alert {
	studentOne: string
	studentTwo: string
	alertType: string
}

export interface AlertResponse {
	alerts: Alert[]
}