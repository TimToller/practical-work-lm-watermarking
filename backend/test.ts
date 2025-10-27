const tokens = [
	1, 2443, 6249, 5275, 338, 278, 6944, 310, 25508, 2472, 470, 848, 2768, 1790, 934, 470, 7047, 1584, 565, 263, 4439, 14803, 6263,
	14335, 1009, 2898, 342, 304, 10320, 284, 278, 7934, 11561, 29889, 739, 8830, 4694, 267, 278, 3050, 304, 15833, 414, 322, 871,
	599, 327, 1372, 2130, 304, 1906, 1058, 526, 7755, 304, 2125, 372, 29889, 910, 3732, 372, 29892, 297, 263, 4060, 29892, 3804,
	304, 278, 1404, 322, 470, 6316, 556, 310, 278, 848, 322, 5019, 1811, 304, 1722, 278, 934, 29889, 13, 7789, 29887, 1562, 338,
	263, 5407, 5780, 363, 429, 305, 9776, 7191, 1728, 5019, 541, 278, 10004, 322, 19870,
];

const tokens2 = [
	1, 2443, 6249, 5275, 338, 263, 11043, 1304, 304, 9563, 7035, 7191, 2629, 385, 1967, 29892, 263, 4823, 29892, 470, 916, 5745,
	393, 526, 451, 5491, 1304, 304, 27769, 11592, 2472, 29889, 2443, 6249, 5275, 508, 367, 1304, 304, 10628, 284, 278, 2114, 393,
	727, 338, 263, 7035, 2643, 297, 278, 937, 2058, 29892, 541, 372, 508, 884, 367, 1304, 304, 22649, 7035, 7191, 29889, 1670, 526,
	1784, 1422, 3519, 310, 8025, 7035, 7191, 297, 385, 1967, 29889, 3834, 310, 278, 3519, 526, 29901, 13, 1523, 2590, 1158, 313,
	15189, 2000, 1706, 949, 18272, 29897, 13, 1576, 1820, 18766, 411, 1886, 6249, 5275, 338, 23655,
];

const m = 100;
const n = 256;
const key = 42;
const seed = 1;

// const vocab_size = 32000;
const vocab_size = 32000;

function int_32(x: number): number {
	// Force to uint32
	return x >>> 0;
}

class MersenneRNG {
	private state: Uint32Array;
	private index: number;

	private readonly N = 624;
	private readonly M = 397;

	// Masks
	private readonly UPPER_MASK = 0x80000000 >>> 0; // 1 << 31 as uint32
	private readonly LOWER_MASK = 0x7fffffff >>> 0;

	// Initialization constants
	private readonly f = 1812433253 >>> 0;
	// Tempering parameters
	private readonly u = 11;
	private readonly s = 7;
	private readonly b = 0x9d2c5680 >>> 0;
	private readonly t = 15;
	private readonly c = 0xefc60000 >>> 0;
	private readonly l = 18;

	constructor(seed: number = 5489) {
		this.state = new Uint32Array(this.N);
		this.index = this.N; // force twist on first use unless initialized

		this.state[0] = int_32(seed);
		for (let i = 1; i < this.N; i++) {
			// state[i] = int_32(f * (state[i-1] ^ (state[i-1] >> 30)) + i)
			const prev = this.state[i - 1];
			const x = (prev ^ (prev >>> 30)) >>> 0;
			const mul = Math.imul(this.f, x) >>> 0; // 32-bit multiply
			this.state[i] = int_32((mul + i) >>> 0);
		}
	}

	private twist(): void {
		for (let i = 0; i < this.N; i++) {
			// x = (state[i] & UPPER_MASK) + (state[(i+1)%N] & LOWER_MASK)
			const x = (((this.state[i] & this.UPPER_MASK) >>> 0) + ((this.state[(i + 1) % this.N] & this.LOWER_MASK) >>> 0)) >>> 0;
			let xA = (x >>> 1) >>> 0;
			if ((x & 1) !== 0) {
				xA = (xA ^ 0x9908b0df) >>> 0;
			}
			this.state[i] = (this.state[(i + this.M) % this.N] ^ xA) >>> 0;
		}
		this.index = 0;
	}

	rand(): number {
		if (this.index >= this.N) {
			this.twist();
		}
		let y = this.state[this.index];

		// Tempering
		y ^= y >>> this.u;
		y ^= (y << this.s) & this.b;
		y >>>= 0;
		y ^= (y << this.t) & this.c;
		y >>>= 0;
		y ^= y >>> this.l;
		y >>>= 0;

		this.index += 1;

		// Return in [0,1), same as Python: int_32(y) * (1.0 / 2**32)
		return (y >>> 0) * (1.0 / 4294967296.0);
	}
}

// ----- Data structures & utilities -----

type Matrix = Float32Array[]; // array of rows; each row is Float32Array of length vocab_size

function zeros(rows: number, cols: number): Float64Array[] {
	const out: Float64Array[] = new Array(rows);
	for (let i = 0; i < rows; i++) {
		out[i] = new Float64Array(cols);
	}
	return out;
}

// Generate xi with the custom RNG to match Python behavior
function generateXi(nRows: number, vsize: number, rng: MersenneRNG): Matrix {
	const xi: Matrix = new Array(nRows);
	for (let i = 0; i < nRows; i++) {
		const row = new Float32Array(vsize);
		for (let j = 0; j < vsize; j++) {
			row[j] = rng.rand();
		}
		xi[i] = row;
	}
	return xi;
}

// ----- Core algorithms -----

/**
 * levenshtein variant:
 * x: token sequence (number[] of length n)
 * y: matrix of probabilities (rows = m, cols = vocab_size)
 * gamma: insertion/deletion cost (default 0.0)
 *
 * cost_sub = log(1 - y[j-1, x[i-1]])
 */
function levenshtein(x: ArrayLike<number>, y: Matrix, gamma: number = 0.0): number {
	const n = x.length;
	const m = y.length;

	const A = zeros(n + 1, m + 1);

	for (let i = 0; i <= n; i++) {
		for (let j = 0; j <= m; j++) {
			if (i === 0) {
				A[i][j] = j * gamma;
			} else if (j === 0) {
				A[i][j] = i * gamma;
			} else {
				const token = x[i - 1] | 0;
				const probRow = y[j - 1];
				const p = probRow[token];
				// safeguard for numeric stability, though rng.rand() never returns 1.0
				const clamped = p >= 1.0 ? 1.0 - Number.EPSILON : p;
				const cost_sub = Math.log(1.0 - clamped);

				// deletion
				let best = A[i - 1][j] + gamma;

				// insertion
				let tmp = A[i][j - 1] + gamma;
				if (tmp < best) best = tmp;

				// substitution
				tmp = A[i - 1][j - 1] + cost_sub;
				if (tmp < best) best = tmp;

				A[i][j] = best;
			}
		}
	}
	return A[n][m];
}

/**
 * detect:
 * tokens: token sequence to evaluate
 * n: (ignored to mirror Python signature; inferred from xi.length)
 * k: window size (length(tokens) in the original)
 * xi: matrix (n x vocab_size)
 * gamma: insertion/deletion cost
 *
 * Returns the min score over sliding windows and circular xi windows.
 */
function detect(tokens: ArrayLike<number>, _n: number, k: number, xi: Matrix, gamma: number = 0.0): number {
	const mLen = tokens.length;
	const nRows = xi.length;

	if (k !== mLen) {
		// In the Python code, k is set to len(tokens). We keep the same behavior.
	}

	// A has shape (m-(k-1), n)
	const rowsA = mLen - (k - 1);
	if (rowsA <= 0) {
		throw new Error("k must be <= tokens.length and >= 1");
	}

	let globalMin = Number.POSITIVE_INFINITY;

	// Precompute circular row indices for xi windows
	for (let i = 0; i < rowsA; i++) {
		for (let j = 0; j < nRows; j++) {
			// Build a k-row cyclic window from xi: rows (j + 0..k-1) % nRows
			// To avoid materializing the full window as a big matrix, index directly inside levenshtein:
			// But levenshtein expects a full matrix. We'll construct a lightweight view.
			const yWindow: Matrix = new Array(k);
			for (let t = 0; t < k; t++) {
				yWindow[t] = xi[(j + t) % nRows];
			}
			const subTokens = (tokens as number[]).slice(i, i + k);
			const val = levenshtein(subTokens, yWindow, gamma);
			if (val < globalMin) globalMin = val;
		}
	}

	// Equivalent to Python's `print(np.min(A))` then `return np.min(A)`
	return globalMin;
}

/**
 * calculate_detection:
 * Runs the test against xi generated by our MT, then compares to nulls built with Math.random().
 * Returns the (p_val + 1) / (n_runs + 1) estimate, matching the Python code.
 */
function calculate_detection(t: ArrayLike<number>, xi: Matrix, nRows: number, vsize: number, n_runs: number = 100): number {
	const k = t.length;
	const test_result = detect(t, nRows, k, xi);
	let p_val = 0;

	for (let run = 0; run < n_runs; run++) {
		// Null alternative uses uniform random [0,1) like numpy's np.random.rand
		const xi_alt: Matrix = new Array(nRows);
		for (let i = 0; i < nRows; i++) {
			const row = new Float32Array(vsize);
			for (let j = 0; j < vsize; j++) {
				row[j] = Math.random();
			}
			xi_alt[i] = row;
		}
		const null_result = detect(t, nRows, k, xi_alt);
		// assuming lower test values indicate presence of watermark
		if (null_result <= test_result) p_val += 1;
	}

	return (p_val + 1.0) / (n_runs + 1.0);
}

// ----- Example wiring (replace tokens/tokens2 as needed) -----

// Placeholder: supply your real token arrays (each entry must be 0..vocab_size-1).
// For a quick smoke test, you can uncomment these lines to make up tiny inputs.
// const tokens: number[] = [1, 2, 3, 4, 5];
// const tokens2: number[] = [5, 4, 3, 2, 1];

const rng = new MersenneRNG(key);

// If you want to verify MT parity against Python, you can log a few values here
// and compare to your Python run initialized with seed=42.
// Example:
// console.log([rng.rand(), rng.rand(), rng.rand(), rng.rand(), rng.rand()]);

// Generate XI with the custom RNG (matches Python behavior)
const xi = generateXi(n, vocab_size, rng);
// console.log("XI generated");

// If you have tokens/tokens2 available in scope, you can run:
const kTokens = tokens.length;
const p_val1 = calculate_detection(tokens, xi, n, vocab_size);
const p_val2 = calculate_detection(tokens2, xi, n, vocab_size);
console.log("P-VALUES: ", p_val1, p_val2);
