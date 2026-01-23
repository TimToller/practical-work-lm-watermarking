import { parseSSEChunk } from "@/lib/api";
import { API_BASE } from "@/lib/consts";
import { getPrompt } from "@/lib/prompt";
import { corruptedText, generateSettings } from "@/lib/store";
import { useAtom } from "jotai";
import { PlayCircle, Shuffle, Sparkles } from "lucide-react";
import { motion } from "motion/react";
import { useState } from "react";
import CorruptCard from "./corrupt-card";
import GenerateCard from "./generate-card";
import { Button } from "./ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { TabsContent } from "./ui/tabs";
import { Textarea } from "./ui/textarea";
import { Tooltip, TooltipContent, TooltipTrigger } from "./ui/tooltip";

export type GenerateRequestPayload = {
	prompt: string;
	m: number;
	n: number;
	temperature: number;
	top_p: number;
	return_vanilla: boolean;
	vanilla_m?: number;
	key: number;
};

export type GenerateResponsePayload = {
	text: string;
	token_ids: number[];
	used_top_n?: number | null;
	warning?: string | null;
	vanilla_text?: string | null;
	vanilla_token_ids?: number[] | null;
	elapsed: number;
	error?: { status_code: number; detail: string };
};

export type GenerateTokenEvent = {
	step: number;
	text: string;
	id: number;
	prompt_offset: number;
	total_steps: number;
	vanilla_text?: string | null;
};

export default function GenerateSection() {
	const [settings, setSettings] = useAtom(generateSettings);
	const [, setCorrupted] = useAtom(corruptedText);

	const { prompt, maxTokens: m, n, temperature, returnVanillaBaseline: returnVanilla, key, top_p: topP } = settings;

	const [isGenerating, setIsGenerating] = useState(false);
	const [generatedTokens, setGeneratedTokens] = useState<GenerateTokenEvent[]>([]);
	const [finalGenerate, setFinalGenerate] = useState<GenerateResponsePayload | null>(null);
	const [generateError, setGenerateError] = useState<string | null>(null);

	const shufflePrompt = () => {
		setSettings((prev) => ({ ...prev, prompt: getPrompt() }));
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
			return_vanilla: returnVanilla,
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
						setCorrupted(() => ({ original: final.text, corrupted: final.text }));
						if (final.error) {
							setGenerateError(`${final.error.status_code}: ${final.error.detail}`);
						}
					}
				}
			}
		} catch (e: unknown) {
			const message = e instanceof Error ? e.message : "Streaming failed";
			setGenerateError(message);
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
						<Tooltip>
							<TooltipTrigger asChild>
								<Button variant="outline" size="icon" className="h-7 w-7 border-slate-700 bg-slate-900" onClick={shufflePrompt}>
									<Shuffle className="h-3 w-3" />
								</Button>
							</TooltipTrigger>
							<TooltipContent>
								<p>Get random prompt</p>
							</TooltipContent>
						</Tooltip>

						<Label htmlFor="prompt">Prompt</Label>
						<Textarea
							id="prompt"
							value={prompt}
							onChange={(e) => setSettings((prev) => ({ ...prev, prompt: e.target.value }))}
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
								onChange={(e) => setSettings((prev) => ({ ...prev, maxTokens: Number(e.target.value) || 0 }))}
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
								onChange={(e) => setSettings((prev) => ({ ...prev, n: Number(e.target.value) || 0 }))}
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
								onChange={(e) => setSettings((prev) => ({ ...prev, key: Number(e.target.value) || 0 }))}
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
								onChange={(e) => setSettings((prev) => ({ ...prev, temperature: Number(e.target.value) || 0 }))}
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
						<div className="flex flex-wrap items-center gap-3 text-xs">
							<label className="inline-flex items-center gap-2">
								<input
									type="checkbox"
									className="h-3 w-3 rounded border-slate-600 bg-slate-900"
									checked={returnVanilla}
									onChange={(e) => setSettings((prev) => ({ ...prev, returnVanillaBaseline: e.target.checked }))}
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
			<GenerateCard finalGenerate={finalGenerate} generatedTokens={generatedTokens} isGenerating={isGenerating} prompt={prompt} />

			<CorruptCard />
		</TabsContent>
	);
}
