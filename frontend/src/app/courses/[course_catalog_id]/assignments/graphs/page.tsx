"use client"

import {
	GetSimilarityIntervalPlot,
	GetDistributionPlot,
	GetSimilarityPlot,
} from "@/controllers/graphs"
import { Box, Typography, Tab, Tabs, Button } from "@mui/material"
import { useSearchParams } from "next/navigation"
import { useCourseContext } from "@/context/CourseContext"
import { useEffect, useState } from "react"
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch"

export default function Graphs() {
	const searchParams = useSearchParams()
	const { courseInstanceId, semesterId } = useCourseContext()
	const assignmentIdParam = searchParams.get("assignment")
	const assignmentId = assignmentIdParam ? Number(assignmentIdParam) : null

	const [similarityPlot, setSimilarityPlot] = useState<string>("")
	const [intervalPlot, setIntervalPlot] = useState<string>("")
	const [distributionPlot, setDistributionPlot] = useState<string>("")
	const [tabValue, setTabValue] = useState(0)

	useEffect(() => {
		if (!assignmentId) return

		const fetchAllPlots = async () => {
			const [simPlot, intPlot, distPlot] = await Promise.all([
				GetSimilarityPlot(assignmentId),
				GetSimilarityIntervalPlot(assignmentId),
				GetDistributionPlot(assignmentId),
			])

			setSimilarityPlot(simPlot ?? "")
			setIntervalPlot(intPlot ?? "")
			setDistributionPlot(distPlot ?? "")
		}

		fetchAllPlots()
	}, [assignmentId])

	const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
		setTabValue(newValue)
	}

	return (
		<Box
			sx={(theme) => ({
				width: "100%",
				height: "100%",
				overflow: "hidden",
				backgroundColor: theme.palette.background.paper,
				p: 2,
			})}
		>
			<Tabs
				value={tabValue}
				onChange={handleTabChange}
				sx={{ borderBottom: "1px solid grey" }}
			>
				<Tab label="Similarity Plot" />
				<Tab label="Similarity Interval Plot" />
				<Tab label="Distribution Plot" />
			</Tabs>

			{/* Content for each tab */}
			<Box sx={{ mt: 2 }}>
				{tabValue === 0 && (
					<Box
						sx={{
							width: "100%",
							height: 600,
							overflow: "hidden",
							mt: 1,
							border: "1px solid #ccc",
							borderRadius: 2,
							position: "relative",
						}}
					>
						{similarityPlot ? (
							<TransformWrapper
								initialScale={1}
								minScale={0.1}
								maxScale={5}
								limitToBounds={false}
							>
								{({ resetTransform }) => (
									<>
										<Box
											sx={{ display: "flex", justifyContent: "flex-end", p: 1 }}
										>
											<Button
												variant="outlined"
												size="small"
												onClick={() => resetTransform()}
											>
												Reset View
											</Button>
										</Box>
										<TransformComponent>
											<Box
												component="img"
												src={similarityPlot}
												alt="Similarity Plot"
												sx={{
													width: "100%",
													height: "100%",
													objectFit: "contain",
												}}
											/>
										</TransformComponent>
									</>
								)}
							</TransformWrapper>
						) : (
							<Typography variant="caption" sx={{ p: 2 }}>
								No Similarity Plot Available
							</Typography>
						)}
					</Box>
				)}

				{tabValue === 1 && (
					<Box
						sx={{
							width: "100%",
							overflow: "hidden",
							mt: 1,
							border: "1px solid #ccc",
							borderRadius: 2,
							position: "relative",
							display: "flex",
							alignItems: "center",
							justifyContent: "center",
						}}
					>
						{intervalPlot ? (
							<TransformWrapper
								initialScale={0.4}
								minScale={0.1}
								maxScale={5}
								limitToBounds={false}
							>
								{({ resetTransform }) => (
									<>
										<Box
											sx={{
												display: "flex",
												justifyContent: "flex-end",
												p: 1,
												position: "absolute",
												top: 0,
												right: 0,
												zIndex: 10,
											}}
										>
											<Button
												variant="outlined"
												size="small"
												onClick={() => resetTransform()}
											>
												Reset View
											</Button>
										</Box>
										<TransformComponent>
											<Box
												component="img"
												src={intervalPlot}
												alt="Similarity Interval Plot"
												sx={{
													width: "100%",
													height: "100%",
													objectFit: "contain",
												}}
											/>
										</TransformComponent>
									</>
								)}
							</TransformWrapper>
						) : (
							<Typography variant="caption" sx={{ p: 2 }}>
								No Similarity Interval Plot Available
							</Typography>
						)}
					</Box>
				)}

				{tabValue === 2 && (
					<Box
						sx={{
							width: "100%",
							mt: 1,
							border: "1px solid #ccc",
							borderRadius: 2,
						}}
					>
						{distributionPlot ? (
							<Box
								component="img"
								src={distributionPlot}
								alt="Distribution Plot"
								sx={{
									height: "100%",
									objectFit: "contain",
								}}
							/>
						) : (
							<Typography variant="caption" sx={{ p: 2 }}>
								No Distribution Plot Available
							</Typography>
						)}
					</Box>
				)}
			</Box>
		</Box>
	)
}
