import { atomWithStorage } from "jotai/utils";

export type TabKey = "generate" | "detect";

export const activeTab = atomWithStorage<TabKey>("activeTab", "generate");

export const generateSettings = atomWithStorage("generateSettings", {
	prompt: "",
	maxTokens: 200,
	n: 256,
	key: 42,
	top_p: 1,
	temperature: 0.95,
	returnVanillaBaseline: true,
});

export const generateOutput = atomWithStorage("generateOutput", {
	text: "",
	vanillaBaseline: "",
	error: "",
	isGenerating: false,
});

export const detectSettings = atomWithStorage("detectSettings", {
	text: "",
	n: 256,
	nRuns: 300,
	key: 42,
});

export const detectOutput = atomWithStorage("detectOutput", {
	isDetecting: false,
});
