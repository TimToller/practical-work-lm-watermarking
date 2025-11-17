// src/App.tsx
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { ArrowRightCircle, BarChart3, Copy, Cpu, PlayCircle, Sparkles } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import React, { useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, type TooltipProps } from "recharts";

const API_BASE = import.meta.env.VITE_API_BASE_URL?.replace(/\/+$/, "") ?? "http://localhost:8000";

type ModelInfo = {
	vocab_size: number;
	n_ctx: number;
	n_threads: number;
	n_gpu_layers: number;
	model_path: string;
};

type GenerateRequestPayload = {
	prompt: string;
	m: number;
	n: number;
	temperature: number;
	top_p: number;
	include_prompt: boolean;
	return_vanilla: boolean;
	vanilla_m?: number;
	key: number;
};

type GenerateResponsePayload = {
	text: string;
	token_ids: number[];
	used_top_n?: number | null;
	warning?: string | null;
	vanilla_text?: string | null;
	vanilla_token_ids?: number[] | null;
	elapsed: number;
	error?: { status_code: number; detail: string };
};

type GenerateTokenEvent = {
	step: number;
	text: string;
	id: number;
	prompt_offset: number;
	total_steps: number;
	vanilla_text?: string | null;
};

type DetectRequestPayload = {
	text: string;
	n: number;
	k?: number;
	n_runs: number;
	gamma: number;
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

function parseSSEChunk(chunk: string): { event: string; data: any }[] {
	// chunk is like "event: token\ndata: {...}\n\nevent: token..."
	const events: { event: string; data: any }[] = [];
	const blocks = chunk.split("\n\n").filter(Boolean);

	for (const block of blocks) {
		const lines = block.split("\n");
		let eventType = "message";
		let dataStr = "";

		for (const line of lines) {
			if (line.startsWith("event:")) {
				eventType = line.slice(6).trim();
			} else if (line.startsWith("data:")) {
				const part = line.slice(5).trim();
				dataStr += part;
			}
		}

		if (!dataStr) continue;
		try {
			const data = JSON.parse(dataStr);
			events.push({ event: eventType, data });
		} catch {
			// ignore malformed json
		}
	}

	return events;
}

const App: React.FC = () => {
	// Model info
	const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
	const [modelLoading, setModelLoading] = useState(false);
	const [modelError, setModelError] = useState<string | null>(null);

	// Generate
	const [prompt, setPrompt] = useState("How did US states get their names?");
	const [m, setM] = useState(200);
	const [n, setN] = useState(50);
	const [temperature, setTemperature] = useState(1);
	const [topP, setTopP] = useState(1);
	const [includePrompt, setIncludePrompt] = useState(false);
	const [returnVanilla, setReturnVanilla] = useState(false);
	const [key, setKey] = useState(42);
	const [isGenerating, setIsGenerating] = useState(false);
	const [generatedTokens, setGeneratedTokens] = useState<GenerateTokenEvent[]>([]);
	const [finalGenerate, setFinalGenerate] = useState<GenerateResponsePayload | null>(null);
	const [generateError, setGenerateError] = useState<string | null>(null);

	// Detect
	const [detectText, setDetectText] = useState("");
	const [detectN, setDetectN] = useState(50);
	const [detectK, setDetectK] = useState<number | undefined>(undefined);
	const [detectNRuns, setDetectNRuns] = useState(300);
	const [detectGamma, setDetectGamma] = useState(0);
	const [detectKey, setDetectKey] = useState(42);
	const [isDetecting, setIsDetecting] = useState(false);
	const [detectInit, setDetectInit] = useState<DetectInitEvent | null>(null);
	const [detectProgress, setDetectProgress] = useState<PValuePoint[]>([]);
	const [detectFinal, setDetectFinal] = useState<DetectResponsePayload | null>(null);
	const [detectError, setDetectError] = useState<string | null>(null);

	const [activeTab, setActiveTab] = useState<"generate" | "detect">("generate");
	const [copied, setCopied] = useState(false);

	// Full text that matches what we conceptually want to copy/bring-to-detect
	const fullGeneratedText = React.useMemo(() => {
		// Prefer the backend's final text if available and not an error
		if (finalGenerate?.text && !finalGenerate.error) {
			return includePrompt ? prompt + finalGenerate.text : finalGenerate.text;
		}
		const continuation = generatedTokens.map((t) => t.text).join("");
		return includePrompt ? prompt + continuation : continuation;
	}, [finalGenerate, generatedTokens, includePrompt, prompt]);

	const handleCopyGenerated = async () => {
		if (!fullGeneratedText) return;
		try {
			await navigator.clipboard.writeText(fullGeneratedText);
			setCopied(true);
			setTimeout(() => setCopied(false), 1500);
		} catch (err) {
			console.error("Failed to copy:", err);
		}
	};

	const handleBringToDetect = () => {
		if (!fullGeneratedText) return;
		setDetectText(fullGeneratedText);
		setActiveTab("detect");
	};

	// Load model info
	useEffect(() => {
		const load = async () => {
			try {
				setModelLoading(true);
				setModelError(null);
				const res = await fetch(`${API_BASE}/model`);
				if (!res.ok) throw new Error(`HTTP ${res.status}`);
				const data: ModelInfo = await res.json();
				setModelInfo(data);
			} catch (e: any) {
				setModelError(e.message ?? "Failed to load model info");
			} finally {
				setModelLoading(false);
			}
		};
		load();
	}, []);

	const handleGenerate = async () => {
		if (!prompt.trim()) return;
		setIsGenerating(true);
		setGeneratedTokens([]);
		setFinalGenerate(null);
		setGenerateError(null);

		const payload: GenerateRequestPayload = {
			prompt,
			m,
			n,
			temperature,
			top_p: topP,
			include_prompt: includePrompt,
			return_vanilla: returnVanilla,
			vanilla_m: returnVanilla ? m : undefined,
			key,
		};

		try {
			const res = await fetch(`${API_BASE}/generate/stream`, {
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
					if (event === "token") {
						const tokenEvent = data as GenerateTokenEvent;
						setGeneratedTokens((prev) => [...prev, tokenEvent]);
					} else if (event === "final") {
						const final = data as GenerateResponsePayload;
						setFinalGenerate(final);
						if (final.error) {
							setGenerateError(`${final.error.status_code}: ${final.error.detail}`);
						}
					}
				}
			}
		} catch (e: any) {
			setGenerateError(e.message ?? "Streaming failed");
		} finally {
			setIsGenerating(false);
		}
	};

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
			k: detectK,
			n_runs: detectNRuns,
			gamma: detectGamma,
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
			setDetectError(e.message ?? "Streaming failed");
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

	const joinedGeneratedText = generatedTokens.map((t) => t.text).join("");

	const effectiveText = finalGenerate?.text ?? (includePrompt ? prompt + joinedGeneratedText : joinedGeneratedText);

	return (
		<div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-slate-100">
			<div className="mx-auto flex min-h-screen max-w-6xl flex-col px-4 py-6 lg:px-6">
				{/* Header */}
				<header className="mb-6 flex items-center justify-between gap-3">
					<div className="flex items-center gap-3">
						<div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-slate-800 shadow-lg">
							<Sparkles className="h-5 w-5" />
						</div>
						<div>
							<h1 className="text-xl font-semibold tracking-tight sm:text-2xl">Watermarked LLM Playground</h1>
						</div>
					</div>
				</header>

				{/* Content */}
				<main className="grid flex-1 gap-4 md:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
					{/* Left: Generation & Detection */}
					<div className="flex flex-col gap-4">
						<Tabs
							defaultValue="generate"
							className="w-full"
							value={activeTab}
							onValueChange={(v) => setActiveTab(v as "generate" | "detect")}>
							<TabsList className="grid w-full grid-cols-2 bg-slate-900/70">
								<TabsTrigger value="generate" className="flex items-center gap-2">
									<PlayCircle className="h-4 w-4" />
									Generate
								</TabsTrigger>
								<TabsTrigger value="detect" className="flex items-center gap-2">
									<BarChart3 className="h-4 w-4" />
									Detect
								</TabsTrigger>
							</TabsList>

							{/* Generate Tab */}
							<TabsContent value="generate" className="mt-4 space-y-4">
								<Card className="border-slate-800 bg-slate-900/60 backdrop-blur">
									<CardHeader className="pb-3">
										<CardTitle className="flex items-center gap-2 text-base">
											<Sparkles className="h-4 w-4 text-amber-400" />
											Streaming generation
										</CardTitle>
									</CardHeader>
									<CardContent className="space-y-4">
										<div className="space-y-2">
											<Label htmlFor="prompt">Prompt</Label>
											<Textarea
												id="prompt"
												value={prompt}
												onChange={(e) => setPrompt(e.target.value)}
												className="min-h-[120px] resize-y border-slate-700 bg-slate-900"
												placeholder="Ask the model anything..."
											/>
										</div>

										<div className="grid gap-3 sm:grid-cols-3">
											<div className="space-y-1.5">
												<Label htmlFor="m">Max tokens (m)</Label>
												<Input
													id="m"
													type="number"
													value={m}
													onChange={(e) => setM(Number(e.target.value) || 0)}
													className="border-slate-700 bg-slate-900 text-sm"
													min={1}
												/>
											</div>
											<div className="space-y-1.5">
												<Label htmlFor="n">n (watermark dimension)</Label>
												<Input
													id="n"
													type="number"
													value={n}
													onChange={(e) => setN(Number(e.target.value) || 0)}
													className="border-slate-700 bg-slate-900 text-sm"
													min={1}
												/>
											</div>
											<div className="space-y-1.5">
												<Label htmlFor="key">Key</Label>
												<Input
													id="key"
													type="number"
													value={key}
													onChange={(e) => setKey(Number(e.target.value) || 0)}
													className="border-slate-700 bg-slate-900 text-sm"
												/>
											</div>
										</div>

										<div className="grid gap-3 sm:grid-cols-3">
											{/* <div className="space-y-1.5">
												<Label htmlFor="temperature">Temperature</Label>
												<Input
													id="temperature"
													type="number"
													step="0.01"
													value={temperature}
													onChange={(e) => setTemperature(Number(e.target.value) || 0)}
													className="border-slate-700 bg-slate-900 text-sm"
												/>
											</div>
											<div className="space-y-1.5">
												<Label htmlFor="topP">Top-p</Label>
												<Input
													id="topP"
													type="number"
													step="0.01"
													value={topP}
													onChange={(e) => setTopP(Number(e.target.value) || 0)}
													className="border-slate-700 bg-slate-900 text-sm"
												/>
											</div> */}
										</div>

										<div className="flex flex-wrap items-center justify-between gap-3 pt-2">
											<div className="flex items-center gap-3 text-xs ">
												<label className="inline-flex items-center gap-2">
													<input
														type="checkbox"
														className="h-3 w-3 rounded border-slate-600 bg-slate-900"
														checked={includePrompt}
														onChange={(e) => setIncludePrompt(e.target.checked)}
													/>
													Include prompt in final text
												</label>
											</div>
											<div className="flex flex-wrap items-center gap-3 text-xs">
												<label className="inline-flex items-center gap-2">
													<input
														type="checkbox"
														className="h-3 w-3 rounded border-slate-600 bg-slate-900"
														checked={returnVanilla}
														onChange={(e) => setReturnVanilla(e.target.checked)}
													/>
													Return vanilla baseline
												</label>
												{finalGenerate?.elapsed && (
													<span className="rounded-full bg-slate-900/70 px-3 py-1 text-[11px] text-slate-400 ring-1 ring-slate-700/60">
														elapsed: {finalGenerate.elapsed.toFixed(2)}s
													</span>
												)}
											</div>

											<Button
												size="sm"
												className="inline-flex items-center gap-2"
												onClick={handleGenerate}
												disabled={isGenerating}>
												{isGenerating ? (
													<>
														<motion.span
															className="h-3 w-3 rounded-full bg-emerald-400"
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
														Streaming...
													</>
												) : (
													<>
														<PlayCircle className="h-4 w-4" />
														Generate
													</>
												)}
											</Button>
										</div>

										{generateError && <p className="text-xs text-red-400">{generateError}</p>}

										{finalGenerate?.warning && !finalGenerate.error && (
											<p className="text-xs text-amber-300">⚠ {finalGenerate.warning}</p>
										)}
									</CardContent>
								</Card>

								{/* Generated output with animated tokens */}
								<Card className="border-slate-800 bg-slate-900/70 backdrop-blur">
									<CardHeader className="pb-2">
										<CardTitle className="flex items-center justify-between text-sm">
											<span>Generated output</span>
											<div className="flex items-center gap-2">
												<span className="text-[11px] font-normal text-slate-400">tokens: {generatedTokens.length}</span>

												<Button
													variant="outline"
													size="icon"
													className="h-7 w-7 border-slate-700 bg-slate-900"
													onClick={handleCopyGenerated}
													disabled={!fullGeneratedText}>
													<Copy className="h-3 w-3" />
												</Button>

												<Button
													variant="outline"
													size="icon"
													className="h-7 w-7 border-slate-700 bg-slate-900"
													onClick={handleBringToDetect}
													disabled={!fullGeneratedText}>
													<ArrowRightCircle className="h-3 w-3" />
												</Button>
											</div>
										</CardTitle>

										{copied && <p className="mt-1 text-[11px] text-emerald-300">Copied to clipboard ✔</p>}
									</CardHeader>
									<CardContent>
										<div className="relative min-h-[140px] rounded-xl border border-slate-800 bg-slate-950/70 p-3 text-sm leading-relaxed">
											{includePrompt || generatedTokens.length > 0 ? (
												<p className="whitespace-pre-wrap font-mono">
													{/* Static prompt part, if include_prompt is true */}
													{includePrompt && <span>{prompt}</span>}

													{/* Animated continuation tokens */}
													<AnimatePresence initial={false}>
														{generatedTokens.map((tok) => (
															<motion.span
																key={tok.step}
																className="inline"
																initial={{ opacity: 0, y: 4 }}
																animate={{ opacity: 1, y: 0 }}
																exit={{ opacity: 0 }}
																transition={{ duration: 0.12 }}>
																{tok.text}
															</motion.span>
														))}
													</AnimatePresence>
												</p>
											) : (
												<span className="text-xs text-slate-500">Generated text will appear here, token by token.</span>
											)}

											{isGenerating && (
												<motion.div
													className="pointer-events-none absolute inset-x-3 bottom-2 flex items-center gap-2 text-[11px] text-slate-400"
													initial={{ opacity: 0 }}
													animate={{ opacity: 1 }}>
													<span className="h-[3px] flex-1 overflow-hidden rounded-full bg-slate-800">
														<motion.span
															className="block h-full bg-emerald-400"
															animate={{ x: ["-100%", "0%", "100%"] }}
															transition={{ repeat: Infinity, duration: 1.2 }}
														/>
													</span>
													<span>streaming...</span>
												</motion.div>
											)}
										</div>

										{returnVanilla && finalGenerate?.vanilla_text && !finalGenerate.error && (
											<div className="mt-4 rounded-lg border border-slate-800 bg-slate-950/60 p-3 text-xs">
												<div className="mb-1 flex items-center justify-between">
													<span className="font-medium text-slate-200">Vanilla baseline</span>
												</div>
												<p className="whitespace-pre-wrap font-mono text-slate-300">{finalGenerate.vanilla_text}</p>
											</div>
										)}
									</CardContent>
								</Card>
							</TabsContent>

							{/* Detect Tab */}
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
						</Tabs>
					</div>

					{/* Right: Model info & meta */}
					<div className="flex flex-col gap-4">
						<Card className="border-slate-800 bg-slate-900/60 backdrop-blur">
							<CardHeader className="pb-3">
								<CardTitle className="flex items-center gap-2 text-base">
									<Cpu className="h-4 w-4 text-emerald-400" />
									Model info
								</CardTitle>
								<CardDescription>
									Fetched from <code className="mx-1 rounded bg-slate-800 px-1 py-0.5 text-xs text-emerald-200">/model</code> on
									load.
								</CardDescription>
							</CardHeader>
							<CardContent>
								{modelLoading && (
									<div className="flex items-center gap-2 text-xs text-slate-400">
										<motion.div
											className="h-3 w-3 rounded-full bg-emerald-400"
											animate={{
												scale: [1, 0.7, 1],
												opacity: [0.6, 1, 0.6],
											}}
											transition={{
												repeat: Infinity,
												duration: 0.9,
											}}
										/>
										Loading model details...
									</div>
								)}
								{modelError && <p className="text-xs text-red-400">{modelError}</p>}
								{modelInfo && (
									<dl className="grid grid-cols-2 gap-3 text-xs">
										<div>
											<dt className="text-slate-400">Vocab size</dt>
											<dd className="font-mono text-slate-100">{modelInfo.vocab_size}</dd>
										</div>
										<div>
											<dt className="text-slate-400">Context length</dt>
											<dd className="font-mono text-slate-100">{modelInfo.n_ctx}</dd>
										</div>
										<div>
											<dt className="text-slate-400">Threads</dt>
											<dd className="font-mono text-slate-100">{modelInfo.n_threads}</dd>
										</div>
										<div>
											<dt className="text-slate-400">GPU layers</dt>
											<dd className="font-mono text-slate-100">{modelInfo.n_gpu_layers}</dd>
										</div>
										<div className="col-span-2">
											<dt className="text-slate-400">Model path</dt>
											<dd className="truncate font-mono text-[11px] text-slate-200">{modelInfo.model_path}</dd>
										</div>
									</dl>
								)}
							</CardContent>
						</Card>
					</div>
				</main>

				<footer className="mt-6 flex items-center justify-between border-t border-slate-800 pt-3 text-[11px] text-slate-500">
					<span>
						Created by Tim Toller, paper and code based on{" "}
						<a href="https://github.com/jthickstun/watermark" target="_blank" rel="noopener noreferrer">
							https://github.com/jthickstun/watermark
						</a>
					</span>
				</footer>
			</div>
		</div>
	);
};

export default App;
