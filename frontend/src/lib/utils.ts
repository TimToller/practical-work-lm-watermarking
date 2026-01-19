import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
	return twMerge(clsx(inputs));
}

function levenshtein(a: string, b: string): number {
	const m = a.length;
	const n = b.length;
	if (m === 0) return n;
	if (n === 0) return m;

	// dp row
	let prev = new Array(n + 1);
	let curr = new Array(n + 1);

	for (let j = 0; j <= n; j++) prev[j] = j;

	for (let i = 1; i <= m; i++) {
		curr[0] = i;
		const ai = a.charCodeAt(i - 1);
		for (let j = 1; j <= n; j++) {
			const cost = ai === b.charCodeAt(j - 1) ? 0 : 1;
			const del = prev[j] + 1;
			const ins = curr[j - 1] + 1;
			const sub = prev[j - 1] + cost;
			curr[j] = Math.min(del, ins, sub);
		}
		[prev, curr] = [curr, prev];
	}
	return prev[n];
}

export function similarityPercent(a: string, b: string): number {
	const maxLen = Math.max(a.length, b.length);
	if (maxLen === 0) return 100;
	const dist = levenshtein(a, b);
	const sim = 1 - dist / maxLen;
	return Math.max(0, Math.min(1, sim)) * 100;
}
