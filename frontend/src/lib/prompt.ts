import prompts from "@/assets/prompts.json";

export const getPrompt = (key?: number): string => {
	if (key !== undefined && prompts[key]) {
		return prompts[key].instruction;
	}
	const randomIndex = Math.floor(Math.random() * prompts.length);
	return prompts[randomIndex].instruction;
};
