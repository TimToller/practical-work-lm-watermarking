import * as React from "react";
import type { z } from "zod";

type Options = {
	/** If true, uses history.replaceState instead of pushState. */
	replace?: boolean;
	/** If true, remove the param(s) when setting to undefined/null. */
	cleanOnUndefined?: boolean;
};

type SetStateAction<T> = T | ((prev: T) => T);

const URL_CHANGE_EVENT = "usequeryparam:change";

function encodeToSearchParams<T>(name: string, value: T, sp: URLSearchParams) {
	sp.delete(name);
	if (value == null) return;

	if (Array.isArray(value)) {
		for (const v of value) sp.append(name, String(v));
		return;
	}

	if (typeof value === "object") {
		sp.set(name, JSON.stringify(value));
		return;
	}

	sp.set(name, String(value));
}

function readRawFromSearchParams(name: string, sp: URLSearchParams) {
	const all = sp.getAll(name);
	if (all.length === 0) return undefined;
	return all.length === 1 ? all[0] : all; // string | string[]
}

function withUpdatedUrl(sp: URLSearchParams, replace: boolean) {
	const qs = sp.toString();
	const next = `${window.location.pathname}${qs ? `?${qs}` : ""}${window.location.hash ?? ""}`;
	(replace ? window.history.replaceState : window.history.pushState).call(window.history, null, "", next);
	window.dispatchEvent(new Event(URL_CHANGE_EVENT));
}

/**
 * useQueryState(name, schema, options?)
 *
 * Initial value comes from the schema:
 * - If the param exists and schema parses -> that value
 * - Else we try schema.parse(undefined) (so make your schema optional/default/catch)
 *
 * Examples:
 *   z.string().default("")
 *   z.coerce.number().catch(1)
 *   z.array(z.string()).default([])
 *   z.enum(["new","top"]).default("new")
 */
export function useQueryParam<T>(
	name: string,
	schema: z.ZodType<T>,
	options: Options = {}
): readonly [T, (next: SetStateAction<T>) => void] {
	const { replace = false, cleanOnUndefined = true } = options;

	const getDefaultFromSchema = React.useCallback((): T => {
		return schema.parse(undefined);
	}, [schema]);

	const parseFromLocation = React.useCallback((): T => {
		const sp = new URLSearchParams(window.location.search);
		const raw = readRawFromSearchParams(name, sp);

		const parsed = schema.safeParse(raw);
		if (parsed.success) return parsed.data;

		return getDefaultFromSchema();
	}, [name, schema, getDefaultFromSchema]);

	const [state, setState] = React.useState<T>(() => parseFromLocation());

	React.useEffect(() => {
		const onLocationChange = () => setState(parseFromLocation());
		window.addEventListener("popstate", onLocationChange);
		window.addEventListener(URL_CHANGE_EVENT, onLocationChange);
		return () => {
			window.removeEventListener("popstate", onLocationChange);
			window.removeEventListener(URL_CHANGE_EVENT, onLocationChange);
		};
	}, [parseFromLocation]);

	const setQueryState = React.useCallback(
		(next: SetStateAction<T>) => {
			setState((prev) => {
				const resolved = typeof next === "function" ? (next as (p: T) => T)(prev) : next;

				const sp = new URLSearchParams(window.location.search);

				if (cleanOnUndefined && (resolved as unknown) == null) {
					sp.delete(name);
				} else {
					encodeToSearchParams(name, resolved, sp);
				}

				withUpdatedUrl(sp, replace);
				return resolved;
			});
		},
		[name, replace, cleanOnUndefined]
	);

	return [state, setQueryState] as const;
}
