/**
 * BPE Tokenizer — JavaScript implementation
 *
 * Mirrors the Python tokenizer in /BPETokenizer/tokenizer.py.
 * Load tokenizer data with `loadTokenizer(url)`, then call
 * `tokenizer.encode(text)`, `tokenizer.decode(ids)`, or
 * `tokenizer.tokenize(text)` for annotated token objects.
 */

// GPT-2 pre-tokenization pattern (requires ES2018+ Unicode property escapes)
const BPE_PAT = /'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

class BPETokenizer {
  /**
   * @param {{ startId: number, vocab: Record<string, number[]>, merges: [number, number][] }} data
   */
  constructor(data) {
    this.startId = data.startId;

    // vocab: Map<id, Uint8Array>
    this.vocab = new Map();
    for (const [id, bytes] of Object.entries(data.vocab)) {
      this.vocab.set(parseInt(id, 10), new Uint8Array(bytes));
    }

    // mergeRank: Map<"idA_idB", { rank, newId }>
    this.mergeRank = new Map();
    data.merges.forEach(([idA, idB], rank) => {
      this.mergeRank.set(`${idA}_${idB}`, { rank, newId: this.startId + rank });
    });

    this._encoder = new TextEncoder();
    this._decoder = new TextDecoder("utf-8", { fatal: false });
  }

  /** BPE-encode a single pre-tokenized word given as raw bytes. */
  _encodeBytes(wordBytes) {
    // Each byte value (0-255) is its own initial token ID
    let tokens = Array.from(wordBytes);

    while (tokens.length > 1) {
      let bestRank = Infinity;
      let bestIdx = -1;
      let bestNewId = -1;

      for (let i = 0; i < tokens.length - 1; i++) {
        const entry = this.mergeRank.get(`${tokens[i]}_${tokens[i + 1]}`);
        if (entry && entry.rank < bestRank) {
          bestRank = entry.rank;
          bestIdx = i;
          bestNewId = entry.newId;
        }
      }

      if (bestIdx === -1) break;
      tokens.splice(bestIdx, 2, bestNewId);
    }

    return tokens;
  }

  /** Encode text to a list of token IDs. */
  encode(text) {
    const ids = [];
    const pat = new RegExp(BPE_PAT.source, BPE_PAT.flags);
    for (const match of text.matchAll(pat)) {
      ids.push(...this._encodeBytes(this._encoder.encode(match[0])));
    }
    return ids;
  }

  /** Decode a list of token IDs back to a string. */
  decode(ids) {
    const parts = ids.map((id) => this.vocab.get(id) ?? new Uint8Array());
    const totalLen = parts.reduce((s, p) => s + p.length, 0);
    const buf = new Uint8Array(totalLen);
    let offset = 0;
    for (const p of parts) {
      buf.set(p, offset);
      offset += p.length;
    }
    return this._decoder.decode(buf);
  }

  /**
   * Encode text and return annotated token objects.
   * @returns {{ id: number, str: string }[]}
   */
  tokenize(text) {
    return this.encode(text).map((id) => ({
      id,
      str: this._decoder.decode(this.vocab.get(id) ?? new Uint8Array()),
    }));
  }
}

/**
 * Fetch tokenizer JSON and return a ready-to-use BPETokenizer.
 * @param {string} url  Path to the exported tokenizer.json
 * @returns {Promise<BPETokenizer>}
 */
async function loadTokenizer(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load tokenizer data (HTTP ${res.status})`);
  return new BPETokenizer(await res.json());
}
