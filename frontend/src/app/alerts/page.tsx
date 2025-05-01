"use client"

import TableBody from "@mui/material/TableBody";
import Paper from "@mui/material/Paper";
import Table from "@mui/material/Table";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import { useEffect, useState } from "react";
import Box from "@mui/material/Box";
import { Alert }  from '@/types/alertType';
import { GetAlerts } from "@/controllers/alerts";
	
export default function Alerts() {
	const [alerts, setAlerts] = useState<Alert[]>([]);

	useEffect(() => {
	
		// Call the controller function to get alert data
		const fetchAlerts = async () => {
			const data = await GetAlerts()
			
			if ("alerts" in data){
				setAlerts(data.alerts)
			} else {
				console.error("Error fetching alerts: ", data)
			} 
		}
		
		fetchAlerts()
	}, [])

	return (
		<Box>
			<TableContainer component={Paper}>
				<Table sx={{ minWidth: 650 }} aria-label="alerts table">
					<TableHead>
						<TableRow>
							<TableCell>Student 1</TableCell>
							<TableCell>Student 2</TableCell>
							<TableCell>Alert Type</TableCell>
						</TableRow>
					</TableHead>
					<TableBody>
						{alerts.map((alert) => (
							<TableRow
								key={alert.studentOne + alert.studentTwo}
								sx={{ '&:last-child td, &:last-child th': { border: 0 } }}
							>
								<TableCell>{alert.studentOne}</TableCell>
								<TableCell>{alert.studentTwo}</TableCell>
								<TableCell>{alert.alertType}</TableCell>
							</TableRow>
						))}
					</TableBody>
				</Table>
			</TableContainer>
		</Box>
	);
}
