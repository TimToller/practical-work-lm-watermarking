// src/App.tsx
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { BarChart3, PlayCircle } from "lucide-react";
import React from "react";
import DetectSection from "./components/detect-section";
import Footer from "./components/footer";
import GenerateSection from "./components/generate-section";
import Header from "./components/header";
import ModelInfo from "./components/mode-iInfo";
import { useQueryParam } from "./hooks/useQueryParam";
import { tabSchema } from "./lib/states";

const App: React.FC = () => {
	const [activeTab, setActiveTab] = useQueryParam("tab", tabSchema, { replace: true });
	return (
		<div className="min-h-screen bg-linear-to-br from-slate-950 via-slate-900 to-slate-950 text-slate-100">
			<div className="mx-auto flex min-h-screen max-w-6xl flex-col px-4 py-6 lg:px-6">
				<Header />
				<main className="grid flex-1 gap-4 md:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
					<div className="flex flex-col gap-4">
						<Tabs
							defaultValue="generate"
							className="w-full"
							value={activeTab?.toString()}
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
							<GenerateSection />
							<DetectSection />
						</Tabs>
					</div>
					<ModelInfo />
				</main>
				<Footer />
			</div>
		</div>
	);
};

export default App;
