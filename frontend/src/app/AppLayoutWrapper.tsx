"use client"

import { useAuth } from "@/context/AuthContext"
import { Box, AppBar, Toolbar, Typography, Button } from "@mui/material"
import { SignOutButton } from "@/components/AuthenticationMethod"
import LeftPanel from "@/components/LeftPanel"

interface Props {
	children: React.ReactNode
}

export default function AppLayoutWrapper({ children }: Props) {
	const { user } = useAuth()
	console.log(user?.email)
	if (!user?.isLoggedIn) {
		return children
	} else {
		return (
			<Box
				sx={{
					display: "flex",
					height: "100vh",
					flexDirection: "column",
				}}
			>
				<AppBar
					position="static"
					sx={(theme) => ({
						backgroundColor: theme.palette.background.paper,
						color: theme.palette.text.primary,
						zIndex: 100,
					})}
				>
					<Toolbar>
						<Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
							PRISM
						</Typography>
						<Button variant="contained">
							<SignOutButton />
						</Button>
					</Toolbar>
				</AppBar>
				<Box
					sx={{
						display: "flex",
						height: "100%",
						overflow: "hidden",
					}}
				>
					<LeftPanel />
					<Box
						sx={{
							flexGrow: 1,
							overflow: "auto",
							p: 2,
						}}
					>
						{children}
					</Box>
				</Box>
			</Box>
		)
	}
}
