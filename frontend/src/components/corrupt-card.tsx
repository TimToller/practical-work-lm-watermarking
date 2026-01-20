import { corruptText } from "@/lib/api";
import { activeTab, corruptedText, detectSettings } from "@/lib/store";
import { similarityPercent } from "@/lib/utils";
import { diffWordsWithSpace } from "diff";
import { useAtom } from "jotai";
import { ArrowRightCircle, Copy, Hammer, Loader2 } from "lucide-react";
import { useMemo, useState } from "react";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "./ui/accordion";
import { Button } from "./ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";

export default function CorruptCard() {
	const [{ corrupted, original }, setCorrupted] = useAtom(corruptedText);
	const [, setTab] = useAtom(activeTab);
	const [, setDetect] = useAtom(detectSettings);

	const [copied, setCopied] = useState(false);
	const [showDeletions, setShowDeletions] = useState(true);
	const [isCorrupting, setIsCorrupting] = useState(false);

	const percent = useMemo(() => similarityPercent(original ?? "", corrupted ?? ""), [original, corrupted]);

	const parts = useMemo(() => {
		if (!original || !corrupted) return [];
		return diffWordsWithSpace(original, corrupted);
	}, [original, corrupted]);

	const handleCopyGenerated = async () => {
		if (!corrupted) return;
		try {
			await navigator.clipboard.writeText(corrupted);
			setCopied(true);
			setTimeout(() => setCopied(false), 1500);
		} catch (err) {
			console.error("Failed to copy:", err);
		}
	};

	const handleBringToDetect = () => {
		if (!corrupted) return;
		setDetect((prev) => ({ ...prev, text: corrupted }));
		setTab("detect");
	};

	const handleCorrupt = async () => {
		if (!original) return;
		setIsCorrupting(true);

		const { corrupted_text } = await corruptText({
			text: original,
			similarity_goal: 0.9, // aim to corrupt 10% of the text
		});
		setIsCorrupting(false);
		setCorrupted((prev) => ({
			...prev,
			corrupted: corrupted_text,
		}));
	};

	const canShowStats = Boolean(original && corrupted);
	const canCorrupt = Boolean(corrupted);

	return (
		<Card className="border-slate-800 bg-slate-900/70 backdrop-blur">
			<CardHeader className="pb-2">
				<CardTitle className="flex items-center justify-between text-sm">
					<span>Corrupt Text</span>

					<div className="flex items-center gap-2">
						{canShowStats && <span className="text-[11px] font-normal text-slate-400">Similarity: {percent.toFixed(2)}%</span>}

						<Button
							variant="outline"
							size="icon"
							className="h-7 w-7 border-slate-700 bg-slate-900"
							onClick={handleCopyGenerated}
							disabled={!corrupted}>
							<Copy className="h-3 w-3" />
						</Button>

						<Button
							variant="outline"
							size="icon"
							className="h-7 w-7 border-slate-700 bg-slate-900"
							onClick={handleBringToDetect}
							disabled={!corrupted}>
							<ArrowRightCircle className="h-3 w-3" />
						</Button>
					</div>
				</CardTitle>

				{copied && <p className="mt-1 text-[11px] text-emerald-300">Copied to clipboard ✔</p>}
			</CardHeader>

			<CardContent>
				<div className="relative min-h-60 rounded-xl border border-slate-800 bg-slate-950/70 p-3 text-sm leading-relaxed flex">
					<textarea
						name="corrupt-text"
						className="flex-1 w-full resize-none border-0 bg-transparent font-mono text-slate-300 outline-none align-top min-h-full"
						value={corrupted ?? ""}
						onChange={(e) => setCorrupted((prev) => ({ ...prev, corrupted: e.target.value }))}
						placeholder="Corrupted text will appear here…"
					/>
				</div>

				{original && corrupted && (
					<div className="mt-3">
						<Accordion type="single" collapsible>
							<AccordionItem value="diff" className="border-slate-800">
								<AccordionTrigger className="text-slate-200 hover:no-underline">Diff preview</AccordionTrigger>
								<AccordionContent>
									<div className="mb-2 flex items-center justify-between">
										<p className="text-[11px] text-slate-400">Inline diff</p>

										<Button
											variant="outline"
											size="sm"
											className="h-7 border-slate-800 bg-slate-900 text-[11px] text-slate-300"
											onClick={() => setShowDeletions((v) => !v)}>
											{showDeletions ? "Hide deletions" : "Show deletions"}
										</Button>
									</div>

									<div className="rounded-xl border border-slate-800 bg-slate-950/40 p-3">
										<div className="whitespace-pre-wrap break-words font-mono text-sm leading-relaxed text-slate-300">
											{parts.map((p, idx) => {
												if (p.removed && !showDeletions) return null;

												const cls = p.added
													? "bg-emerald-500/15 text-emerald-200"
													: p.removed
														? "bg-rose-500/15 text-rose-200 line-through decoration-rose-300/60"
														: "text-slate-300";

												return (
													<span key={idx} className={cls}>
														{p.value}
													</span>
												);
											})}
										</div>
									</div>
								</AccordionContent>
							</AccordionItem>
						</Accordion>
					</div>
				)}

				<div className="mt-3 flex justify-end">
					<Button onClick={handleCorrupt} disabled={!canCorrupt || isCorrupting} className="gap-2">
						{isCorrupting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Hammer className="h-4 w-4" />}
						Corrupt
					</Button>
				</div>
			</CardContent>
		</Card>
	);
}
