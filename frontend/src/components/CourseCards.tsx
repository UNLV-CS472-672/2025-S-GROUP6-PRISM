"use client"
import { Box, Card, CardContent, Typography, ButtonBase } from "@mui/material"
import { darken } from "@mui/material/styles"

interface Props {
	children: React.ReactNode
	onClick?: () => void
}

export default function CourseCards({ children, onClick }: Props) {
	return (
		<ButtonBase
			onClick={onClick}
			sx={(theme) => ({
				width: 300,
				minHeight: 250,
				borderRadius: "10px",
				overflow: "hidden",
				display: "block",
				textAlign: "center",
				px: 2,
				py: 3,
				boxShadow: 2,
				transition: "background-color 0.3s ease",
				backgroundColor: theme.palette.background.default,
				":hover": {
					backgroundColor: darken(theme.palette.background.paper, 0.05),
				},
			})}
		>
			{children}
		</ButtonBase>
	)
}
