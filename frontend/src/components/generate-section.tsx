import { useQueryParam } from "@/hooks/useQueryParam";
import { parseSSEChunk } from "@/lib/api";
import { API_BASE } from "@/lib/consts";
import { getPrompt } from "@/lib/prompt";
import {
	detectTextSchema,
	includePromptSchema,
	keySchema,
	mSchema,
	nSchema,
	promptSchema,
	returnVanillaSchema,
	tabSchema,
	temperatureSchema,
} from "@/lib/states";
import { ArrowRightCircle, Copy, PlayCircle, Shuffle, Sparkles } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import React, { useState } from "react";
import { Button } from "./ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { TabsContent } from "./ui/tabs";
import { Textarea } from "./ui/textarea";

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

export default function GenerateSection() {
	const [prompt, setPrompt] = useQueryParam("prompt", promptSchema);
	const [m, setM] = useQueryParam("m", mSchema);
	const [n, setN] = useQueryParam("n", nSchema);
	const [temperature, setTemperature] = useQueryParam("temperature", temperatureSchema);
	const [includePrompt, setIncludePrompt] = useQueryParam("includePrompt", includePromptSchema);
	const [returnVanilla, setReturnVanilla] = useQueryParam("returnVanilla", returnVanillaSchema);
	const [key, setKey] = useQueryParam("key", keySchema);
	const [, setActiveTab] = useQueryParam("tab", tabSchema, { replace: true });
	const [, setDetectText] = useQueryParam("detectText", detectTextSchema);

	const [topP] = useState(1);
	const [isGenerating, setIsGenerating] = useState(false);
	const [generatedTokens, setGeneratedTokens] = useState<GenerateTokenEvent[]>([]);
	const [finalGenerate, setFinalGenerate] = useState<GenerateResponsePayload | null>(null);
	const [generateError, setGenerateError] = useState<string | null>(null);
	const [copied, setCopied] = useState(false);

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

	const shufflePrompt = () => {
		setPrompt(getPrompt());
	};

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
		} catch (e: Error) {
			setGenerateError(e.message ?? "Streaming failed");
		} finally {
			setIsGenerating(false);
		}
	};
	return (
		<TabsContent value="generate" className="mt-4 space-y-4">
			<Card className="border-slate-800 bg-slate-900/60 backdrop-blur">
				<CardHeader className="pb-3">
					<CardTitle className="flex items-center gap-2 text-base">
						<Sparkles className="h-4 w-4 text-amber-400" />
						Generate
					</CardTitle>
					<CardDescription>Generate a watermarked text</CardDescription>
				</CardHeader>
				<CardContent className="space-y-4">
					<div className="space-y-2">
						<Button variant="outline" size="icon" className="h-7 w-7 border-slate-700 bg-slate-900" onClick={shufflePrompt}>
							<Shuffle className="h-3 w-3" />
						</Button>

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
						<div className="space-y-1.5">
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
						{/* <div className="space-y-1.5">
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

						<Button size="sm" className="inline-flex items-center gap-2" onClick={handleGenerate} disabled={isGenerating}>
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

					{finalGenerate?.warning && !finalGenerate.error && <p className="text-xs text-amber-300">⚠ {finalGenerate.warning}</p>}
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
	);
}
