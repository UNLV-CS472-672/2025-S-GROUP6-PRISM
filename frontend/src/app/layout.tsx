import * as React from "react"
import { ThemeProvider } from "@mui/material/styles"
import AppLayoutWrapper from "./AppLayoutWrapper"
import CssBaseline from "@mui/material/CssBaseline"
import theme from "../theme"
import InitColorSchemeScript from "@mui/material/InitColorSchemeScript"
import Providers from "@/components/Providers"
import ClientLayout from "./ClientLayout"

interface Props {
	children: React.ReactNode
}

export default function RootLayout({ children }: Props) {
	return (
		<html lang="en" suppressHydrationWarning>
			<body>
				<InitColorSchemeScript attribute="class" />
				<ThemeProvider theme={theme}>
					{/* CssBaseline kickstart an elegant, consistent, and simple baseline to build upon. */}
					<CssBaseline />
						<Providers>
							<ClientLayout>
								{props.children}
							</ClientLayout>
						</Providers>
				</ThemeProvider>
			</body>
		</html>
	)
}
