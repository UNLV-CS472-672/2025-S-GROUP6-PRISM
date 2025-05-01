import { easyFetch } from "@/utils/fetchWrapper"
import { APIError } from "@/types/APIError"
import { Alert, AlertResponse } from "@/types/alertType"

export async function GetAlerts(
): Promise<AlertResponse | APIError> {

  try {
    // Fetch the alerts data
    const response = await easyFetch(
      `http://localhost:8000/api/cheating/submission-similarity-pairs/`,
      { method: "GET" }
    )

    const data = await response.json()

    // If Alert data was fetched, then format the data to be accepted by the alerts page
    if (response.ok) {
      let newAlerts: Alert[] = []
  
      data.results.foreach((result: any)=>{
        let newAlert: Alert = {
          studentOne: result.submissions.student,
          studentTwo: result.submissions.stduents.student,
          alertType: "Assignment"
        }

        newAlerts.concat(newAlert)
      })

      let alertResponse: AlertResponse = {alerts: newAlerts}
      
      return alertResponse
    } else {
      return {
        detail: data.detail ?? "Failed to fetch alerts.",
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