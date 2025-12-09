import z from "zod";

export const tabSchema = z.enum(["generate", "detect"]).default("generate");
export const promptSchema = z.string().min(0).default("");
export const mSchema = z.coerce.number().min(1).default(200);
export const nSchema = z.coerce.number().min(1).default(256);
export const temperatureSchema = z.coerce.number().min(0).default(1);
export const includePromptSchema = z.coerce.boolean().default(true);
export const returnVanillaSchema = z.coerce.boolean().default(true);
export const keySchema = z.coerce.number().default(42);
export const detectTextSchema = z.string().min(0).default("");
export const nRunsSchema = z.coerce.number().min(1).default(300);
