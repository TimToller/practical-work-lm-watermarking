import { API_BASE } from "./consts";

export const parseSSEChunk = (chunk: string): { event: string; data: unknown }[] => {
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
};

export type ModelInfo = {
	vocab_size: number;
	n_ctx: number;
	n_threads: number;
	n_gpu_layers: number;
	model_path: string;
};

export const getModelInfo = async () => {
	const response = await fetch(`${API_BASE}/model`);
	return response.json().then((data) => data as ModelInfo);
};

export type CorruptTextRequest = {
	text: string;
	corruption_rate: number; // between 0 and 1
};

export type CorruptTextResponse = {
	corrupted_text: string;
	corruption_rate: number;
	corruption_offset: number;
	elapsed: number;
};

export const corruptText = async (request: CorruptTextRequest): Promise<CorruptTextResponse> => {
	const response = await fetch(`${API_BASE}/corrupt`, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
		},
		body: JSON.stringify(request),
	});

	if (!response.ok) {
		throw new Error(`Error corrupting text: ${response.statusText}`);
	}

	return response.json().then((data) => data as CorruptTextResponse);
};
