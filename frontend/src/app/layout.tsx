import * as React from "react"
import { ThemeProvider } from "@mui/material/styles"
import CssBaseline from "@mui/material/CssBaseline"
import theme from "../theme"
import InitColorSchemeScript from "@mui/material/InitColorSchemeScript"
import Providers from "@/components/Providers"
import { CourseProvider } from "@/context/CourseContext"

interface Props {
	children: React.ReactNode
}

export default function RootLayout(props: { children: React.ReactNode }) {
	return (
		<html lang="en" suppressHydrationWarning>
			<body>
				<InitColorSchemeScript attribute="class" />
				<ThemeProvider theme={theme}>
					{/* CssBaseline kickstart an elegant, consistent, and simple baseline to build upon. */}
					<CssBaseline />
					<Providers>
						<CourseProvider>
							<AppLayoutWrapper>{children}</AppLayoutWrapper>
						</CourseProvider>
					</Providers>
				</ThemeProvider>
			</body>
		</html>
	)
}
