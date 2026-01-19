import { activeTab, detectSettings } from "@/lib/store";
import { useAtom } from "jotai";
import { ArrowRightCircle, Copy } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useMemo, useState } from "react";
import type { GenerateResponsePayload, GenerateTokenEvent } from "./generate-section";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "./ui/accordion";
import { Button } from "./ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";

interface Props {
	finalGenerate: GenerateResponsePayload | null;
	generatedTokens: GenerateTokenEvent[];
	isGenerating: boolean;
	prompt: string;
}
export default function GenerateCard({ finalGenerate, generatedTokens, isGenerating, prompt }: Props) {
	const [copied, setCopied] = useState(false);

	const [, setTab] = useAtom(activeTab);
	const [, setDetect] = useAtom(detectSettings);

	const fullGeneratedText = useMemo(() => {
		// Prefer the backend's final text if available and not an error
		if (finalGenerate?.text && !finalGenerate.error) {
			return finalGenerate.text;
		}
		const continuation = generatedTokens.map((t) => t.text).join("");
		return continuation;
	}, [finalGenerate, generatedTokens, prompt]);

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
		setDetect((prev) => ({ ...prev, text: fullGeneratedText }));
		setTab("detect");
	};
	return (
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
					{generatedTokens.length > 0 ? (
						<p className="whitespace-pre-wrap font-mono">
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

				{finalGenerate?.vanilla_text && !finalGenerate.error && (
					<Accordion type="single" collapsible className="w-full">
						<AccordionItem value="vanilla">
							<AccordionTrigger>Vanilla baseline</AccordionTrigger>
							<AccordionContent className="flex flex-col gap-4 text-balance">
								<p className="whitespace-pre-wrap font-mono text-slate-300">{finalGenerate.vanilla_text}</p>
							</AccordionContent>
						</AccordionItem>
					</Accordion>
				)}
			</CardContent>
		</Card>
	);
}
