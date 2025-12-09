import { useQueryParam } from "@/hooks/useQueryParam";
import { parseSSEChunk } from "@/lib/api";
import { API_BASE } from "@/lib/consts";
import { detectTextSchema, keySchema, nRunsSchema, nSchema } from "@/lib/states";
import { BarChart3, PlayCircle } from "lucide-react";
import { motion } from "motion/react";
import { useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, type TooltipProps } from "recharts";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { TabsContent } from "./ui/tabs";
import { Textarea } from "./ui/textarea";

type DetectRequestPayload = {
	text: string;
	n: number;
	k?: number;
	n_runs: number;
	gamma?: number;
	key: number;
};

type DetectResponsePayload = {
	p_value: number;
	length: number;
	n: number;
	k: number;
	watermarked: string;
	elapsed: number;
	error?: { status_code: number; detail: string };
};

type DetectInitEvent = {
	length: number;
	k: number;
	n: number;
	n_runs: number;
	vocab_size: number;
	gamma: number;
};

type DetectProgressEventObserved = {
	stage: "observed";
	test_result: number;
	runs_completed: number;
};

type DetectProgressEventComparing = {
	stage: "comparing";
	runs_completed: number;
	p_hat: number;
	elapsed: number;
};

type DetectProgressEvent = DetectProgressEventObserved | DetectProgressEventComparing;

type PValuePoint = {
	run: number;
	pHat: number;
	elapsed: number;
};

export default function DetectSection() {
	const [detectText, setDetectText] = useQueryParam("detectText", detectTextSchema);
	const [detectN, setDetectN] = useQueryParam("n", nSchema);

	const [detectNRuns, setDetectNRuns] = useQueryParam("nRuns", nRunsSchema);
	const [detectKey, setDetectKey] = useQueryParam("key", keySchema);

	const [isDetecting, setIsDetecting] = useState(false);
	const [detectInit, setDetectInit] = useState<DetectInitEvent | null>(null);
	const [detectProgress, setDetectProgress] = useState<PValuePoint[]>([]);
	const [detectFinal, setDetectFinal] = useState<DetectResponsePayload | null>(null);
	const [detectError, setDetectError] = useState<string | null>(null);

	const handleDetect = async () => {
		if (!detectText.trim()) return;
		setIsDetecting(true);
		setDetectInit(null);
		setDetectProgress([]);
		setDetectFinal(null);
		setDetectError(null);

		const payload: DetectRequestPayload = {
			text: detectText,
			n: detectN,
			// k: detectK,
			n_runs: detectNRuns,
			// gamma: detectGamma,
			key: detectKey,
		};

		try {
			const res = await fetch(`${API_BASE}/detect/stream`, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
					Accept: "text/event-stream",
				},
				body: JSON.stringify(payload),
			});

			if (!res.ok || !res.body) {
				throw new Error(`HTTP ${res.status}`);
			}

			const reader = res.body.getReader();
			const decoder = new TextDecoder("utf-8");
			let buffer = "";

			while (true) {
				const { value, done } = await reader.read();
				if (done) break;
				buffer += decoder.decode(value, { stream: true });

				const lastSep = buffer.lastIndexOf("\n\n");
				if (lastSep === -1) continue;

				const chunk = buffer.slice(0, lastSep + 2);
				buffer = buffer.slice(lastSep + 2);

				const events = parseSSEChunk(chunk);
				for (const { event, data } of events) {
					if (event === "init") {
						setDetectInit(data as DetectInitEvent);
					} else if (event === "progress") {
						const p = data as DetectProgressEvent;
						if (p.stage === "comparing" && typeof p.p_hat === "number") {
							setDetectProgress((prev) => [...prev, { run: p.runs_completed, pHat: p.p_hat, elapsed: p.elapsed }]);
						}
					} else if (event === "final") {
						const final = data as DetectResponsePayload;
						setDetectFinal(final);
						if (final.error) {
							setDetectError(`${final.error.status_code}: ${final.error.detail}`);
						}
					}
				}
			}
		} catch (e: any) {
			setDetectError(e?.message ?? "Streaming failed");
		} finally {
			setIsDetecting(false);
		}
	};

	const renderPValueTooltip = (props: TooltipProps<number, string>) => {
		const { active, payload, label } = props;
		if (!active || !payload || !payload.length) return null;

		// Only keep the actual p-hat series, ignore constant threshold lines
		const pEntry = payload.find((p) => p.dataKey === "pHat");
		if (!pEntry || pEntry.value == null) return null;

		return (
			<div className="rounded border border-slate-700 bg-slate-900/90 px-2 py-1 text-[11px] text-slate-100">
				<div className="text-slate-300">run {label}</div>
				<div className="font-mono">p-hat: {Number(pEntry.value).toFixed(3)}</div>
			</div>
		);
	};

	return (
		<TabsContent value="detect" className="mt-4 space-y-4">
			<Card className="border-slate-800 bg-slate-900/60 backdrop-blur">
				<CardHeader className="pb-3">
					<CardTitle className="flex items-center gap-2 text-base">
						<BarChart3 className="h-4 w-4 text-sky-400" />
						Watermark detection
					</CardTitle>
					<CardDescription>Detect if a text is likely watermarked</CardDescription>
				</CardHeader>
				<CardContent className="space-y-4">
					<div className="space-y-2">
						<Label htmlFor="detectText">Text to analyse</Label>
						<Textarea
							id="detectText"
							value={detectText}
							onChange={(e) => setDetectText(e.target.value)}
							className="min-h-[120px] resize-y border-slate-700 bg-slate-900"
							placeholder="Paste generated text here..."
						/>
					</div>

					<div className="grid gap-3 sm:grid-cols-4">
						<div className="space-y-1.5">
							<Label htmlFor="detectN">n</Label>
							<Input
								id="detectN"
								type="number"
								value={detectN}
								onChange={(e) => setDetectN(Number(e.target.value) || 0)}
								className="border-slate-700 bg-slate-900 text-sm"
							/>
						</div>
						{/* <div className="space-y-1.5">
												<Label htmlFor="detectK">k (optional, ≤ length)</Label>
												<Input
													id="detectK"
													type="number"
													value={detectK ?? ""}
													onChange={(e) => {
														const v = e.target.value;
														setDetectK(v === "" ? undefined : Number(v));
													}}
													className="border-slate-700 bg-slate-900 text-sm"
												/>
											</div> */}
						<div className="space-y-1.5">
							<Label htmlFor="detectNRuns">Null runs</Label>
							<Input
								id="detectNRuns"
								type="number"
								value={detectNRuns}
								onChange={(e) => setDetectNRuns(Number(e.target.value) || 0)}
								className="border-slate-700 bg-slate-900 text-sm"
							/>
						</div>
						{/* <div className="space-y-1.5">
												<Label htmlFor="detectGamma">Gamma</Label>
												<Input
													id="detectGamma"
													type="number"
													step="0.01"
													value={detectGamma}
													onChange={(e) => setDetectGamma(Number(e.target.value) || 0)}
													className="border-slate-700 bg-slate-900 text-sm"
												/>
											</div> */}
					</div>

					<div className="grid gap-3 sm:grid-cols-2">
						<div className="space-y-1.5">
							<Label htmlFor="detectKey">Key</Label>
							<Input
								id="detectKey"
								type="number"
								value={detectKey}
								onChange={(e) => setDetectKey(Number(e.target.value) || 0)}
								className="border-slate-700 bg-slate-900 text-sm"
							/>
						</div>
						{detectInit && (
							<div className="flex items-end justify-end gap-2 text-[11px] text-slate-400">
								<span>
									length: <span className="text-slate-200">{detectInit.length}</span>
								</span>
								<span>
									vocab: <span className="text-slate-200">{detectInit.vocab_size}</span>
								</span>
							</div>
						)}
					</div>

					<div className="flex items-center justify-between gap-3 pt-2">
						<Button size="sm" className="inline-flex items-center gap-2" onClick={handleDetect} disabled={isDetecting}>
							{isDetecting ? (
								<>
									<motion.span
										className="h-3 w-3 rounded-full bg-sky-400"
										animate={{
											scale: [1, 0.7, 1],
											opacity: [0.6, 1, 0.6],
										}}
										transition={{
											repeat: Infinity,
											duration: 0.9,
											ease: "easeInOut",
										}}
									/>
									Detecting...
								</>
							) : (
								<>
									<PlayCircle className="h-4 w-4" />
									Run detector
								</>
							)}
						</Button>

						{detectFinal && !detectFinal.error && (
							<div className="flex items-center gap-2 text-xs">
								<Badge variant="outline" className="border-slate-600 bg-slate-900 px-2 py-0.5 text-[11px]">
									p-value: <span className="ml-1 font-mono">{detectFinal.p_value.toFixed(4)}</span>
								</Badge>
								<Badge className="bg-slate-800 px-2 py-0.5 text-[11px]" variant="outline">
									status: <span className="ml-1 capitalize text-emerald-300">{detectFinal.watermarked}</span>
								</Badge>
							</div>
						)}
					</div>

					{detectError && <p className="text-xs text-red-400">{detectError}</p>}
				</CardContent>
			</Card>

			{/* p-value chart */}
			<Card className="border-slate-800 bg-slate-900/70 backdrop-blur">
				<CardHeader className="pb-2">
					<CardTitle className="flex items-center justify-between text-sm">
						<span>Running p-value</span>
					</CardTitle>
				</CardHeader>
				<CardContent>
					<div className="h-64 w-full">
						{detectProgress.length > 1 ? (
							<ResponsiveContainer width="100%" height="100%">
								<LineChart data={detectProgress}>
									<CartesianGrid strokeDasharray="3 3" />
									<XAxis
										dataKey="run"
										tick={{ fontSize: 11 }}
										label={{
											value: "runs completed",
											position: "insideBottom",
											offset: -4,
											fontSize: 11,
										}}
									/>
									<YAxis
										dataKey="pHat"
										tick={{ fontSize: 11 }}
										label={{
											value: "p-hat",
											angle: -90,
											position: "insideLeft",
											fontSize: 11,
										}}
									/>
									<Tooltip content={renderPValueTooltip} />

									<Line type="monotone" dataKey="pHat" dot={false} strokeWidth={2} />
									<Line
										type="monotone"
										dataKey={() => 0.01}
										stroke="#4ade80"
										strokeWidth={1.5}
										dot={false}
										legendType="none"
										isAnimationActive={false}
									/>

									<Line
										type="monotone"
										dataKey={() => 0.05}
										stroke="#facc15"
										strokeWidth={1.5}
										dot={false}
										legendType="none"
										isAnimationActive={false}
									/>
								</LineChart>
							</ResponsiveContainer>
						) : (
							<div className="flex h-full items-center justify-center text-xs text-slate-500">
								Run detection to see the p-value curve over null runs.
							</div>
						)}
					</div>
				</CardContent>
			</Card>
		</TabsContent>
	);
}
