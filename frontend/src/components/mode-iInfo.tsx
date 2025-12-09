import type { ModelInfo } from "@/lib/api";
import { getModelInfo } from "@/lib/api";
import { Cpu } from "lucide-react";
import { motion } from "motion/react";
import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";

export default function ModelInfo() {
	const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
	const [modelLoading, setModelLoading] = useState(false);
	const [modelError, setModelError] = useState<string | null>(null);

	// Load model info
	useEffect(() => {
		const load = async () => {
			try {
				setModelLoading(true);
				setModelError(null);
				const data = await getModelInfo();
				setModelInfo(data);
			} catch (e) {
				setModelError(e instanceof Error ? e.message : "Failed to load model info");
			} finally {
				setModelLoading(false);
			}
		};
		load();
	}, []);

	return (
		<div className="flex flex-col gap-4">
			<Card className="border-slate-800 bg-slate-900/60 backdrop-blur">
				<CardHeader className="pb-3">
					<CardTitle className="flex items-center gap-2 text-base">
						<Cpu className="h-4 w-4 text-emerald-400" />
						Model info
					</CardTitle>
					<CardDescription>
						Fetched from <code className="mx-1 rounded bg-slate-800 px-1 py-0.5 text-xs text-emerald-200">/model</code> on load.
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
	);
}
