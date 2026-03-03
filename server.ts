/**
 * Pocket TTS — Deno Server
 *
 * A REST API server for neural text-to-speech with voice cloning.
 * Runs entirely on CPU using ONNX Runtime, no GPU required.
 *
 * Usage:
 *   deno run --allow-read --allow-net --allow-sys --allow-ffi --allow-env server.ts
 *   Add --port <port> after server.ts to use a specific port.
 *
 * API Endpoints:
 *   GET  /                    Server info & API reference
 *   GET  /health              Health check
 *   GET  /v1/voices           List available voices
 *   POST /v1/voices           Register a custom voice from a WAV audio upload
 *   DELETE /v1/voices/:name   Remove a registered custom voice
 *   POST /v1/audio/speech     Generate speech, streams WAV audio
 */

import * as ort from "ort";
import {
  createSentencePieceModule,
  type SpModule,
  type SpProcessor,
} from "./sentencepiece.ts";

// ─── Types ────────────────────────────────────────────────────────────────────

interface VoiceEmbedding {
  data: Float32Array;
  shape: [number, number, number]; // [1, frames, dim]
}

type StateMap = Record<string, ort.Tensor>;
type StateShapes = Record<string, { shape: number[]; dtype: string }>;

class SentencePieceProcessor {
  #sp!: SpModule;
  #proc!: SpProcessor;

  async loadModel(data: Uint8Array): Promise<void> {
    const tempName = `${crypto.randomUUID()}.model`;
    this.#sp = await createSentencePieceModule();
    this.#sp.FS.writeFile(tempName, data);
    const sv = new this.#sp.StringView(tempName);
    const asv = sv.getView();
    this.#proc = new this.#sp.SentencePieceProcessor();
    const status = this.#proc.Load(asv);
    status.delete();
    asv.delete();
    sv.delete();
    this.#sp.FS.unlink(tempName);
  }

  encodeIds(text: string): number[] {
    const sv = new this.#sp.StringView(text);
    const asv = sv.getView();
    const data = this.#proc.EncodeAsIds(asv);
    const arr: number[] = [];
    for (let i = 0; i < data.size(); i++) arr.push(data.get(i));
    data.delete();
    asv.delete();
    sv.delete();
    return arr;
  }

  decodeIds(ids: number[]): string {
    const vec = this.#sp.vecFromJSArray(ids);
    const str = this.#proc.DecodeIds(vec).slice();
    vec.delete();
    return str;
  }
}

// ─── Configuration ────────────────────────────────────────────────────────────

const SAMPLE_RATE = 24000;
const MAX_FRAMES = 500;
const CHUNK_TARGET_TOKENS = 50;
const CHUNK_GAP_SEC = 0.25;
const RESET_FLOW_STATE_EACH_CHUNK = true;
const RESET_MIMI_STATE_EACH_CHUNK = true;
const MAX_LSD = 10;
const FRAMES_AFTER_EOS = 3;
const MAX_VOICE_AUDIO_SEC = 10;

const BASE_DIR = new URL(".", import.meta.url).pathname;

// ─── Server State ─────────────────────────────────────────────────────────────

let mimiEncoderSession: ort.InferenceSession | null = null;
let textConditionerSession: ort.InferenceSession | null = null;
let flowLmMainSession: ort.InferenceSession | null = null;
let flowLmFlowSession: ort.InferenceSession | null = null;
let mimiDecoderSession: ort.InferenceSession | null = null;
let tokenizerProcessor: SentencePieceProcessor | null = null;
let predefinedVoices: Record<string, VoiceEmbedding> = {};
const customVoices = new Map<string, VoiceEmbedding>();
const voiceConditioningCache = new Map<string, StateMap>();
let stTensors: Record<number, Array<{ s: ort.Tensor; t: ort.Tensor }>> = {};
let isReady = false;

// ─── Text Preprocessing (ported from inference-worker.js) ─────────────────────

const ONES = [
  "",
  "one",
  "two",
  "three",
  "four",
  "five",
  "six",
  "seven",
  "eight",
  "nine",
  "ten",
  "eleven",
  "twelve",
  "thirteen",
  "fourteen",
  "fifteen",
  "sixteen",
  "seventeen",
  "eighteen",
  "nineteen",
];
const TENS = [
  "",
  "",
  "twenty",
  "thirty",
  "forty",
  "fifty",
  "sixty",
  "seventy",
  "eighty",
  "ninety",
];
const ORDINAL_ONES = [
  "",
  "first",
  "second",
  "third",
  "fourth",
  "fifth",
  "sixth",
  "seventh",
  "eighth",
  "ninth",
  "tenth",
  "eleventh",
  "twelfth",
  "thirteenth",
  "fourteenth",
  "fifteenth",
  "sixteenth",
  "seventeenth",
  "eighteenth",
  "nineteenth",
];
const ORDINAL_TENS = [
  "",
  "",
  "twentieth",
  "thirtieth",
  "fortieth",
  "fiftieth",
  "sixtieth",
  "seventieth",
  "eightieth",
  "ninetieth",
];

function numberToWords(
  num: number,
  options: { andword?: string; zero?: string; group?: number } = {},
): string {
  const { andword = "", zero = "zero", group = 0 } = options;
  if (num === 0) return zero;
  const convert = (n: number): string => {
    if (n < 20) return ONES[n];
    if (n < 100) {
      return TENS[Math.floor(n / 10)] + (n % 10 ? " " + ONES[n % 10] : "");
    }
    if (n < 1000) {
      const r = n % 100;
      return ONES[Math.floor(n / 100)] + " hundred" +
        (r ? (andword ? " " + andword + " " : " ") + convert(r) : "");
    }
    if (n < 1_000_000) {
      const t = Math.floor(n / 1000), r = n % 1000;
      return convert(t) + " thousand" + (r ? " " + convert(r) : "");
    }
    if (n < 1_000_000_000) {
      const m = Math.floor(n / 1_000_000), r = n % 1_000_000;
      return convert(m) + " million" + (r ? " " + convert(r) : "");
    }
    const b = Math.floor(n / 1_000_000_000), r = n % 1_000_000_000;
    return convert(b) + " billion" + (r ? " " + convert(r) : "");
  };
  if (group === 2 && num > 1000 && num < 10000) {
    const high = Math.floor(num / 100), low = num % 100;
    if (low === 0) return convert(high) + " hundred";
    else if (low < 10) {
      return convert(high) + " " + (zero === "oh" ? "oh" : zero) + " " +
        ONES[low];
    } else return convert(high) + " " + convert(low);
  }
  return convert(num);
}

function ordinalToWords(num: number): string {
  if (num < 20) return ORDINAL_ONES[num] || numberToWords(num) + "th";
  if (num < 100) {
    const tens = Math.floor(num / 10), ones = num % 10;
    if (ones === 0) return ORDINAL_TENS[tens];
    return TENS[tens] + " " + ORDINAL_ONES[ones];
  }
  const cardinal = numberToWords(num);
  if (cardinal.endsWith("y")) return cardinal.slice(0, -1) + "ieth";
  if (cardinal.endsWith("one")) return cardinal.slice(0, -3) + "first";
  if (cardinal.endsWith("two")) return cardinal.slice(0, -3) + "second";
  if (cardinal.endsWith("three")) return cardinal.slice(0, -5) + "third";
  if (cardinal.endsWith("ve")) return cardinal.slice(0, -2) + "fth";
  if (cardinal.endsWith("e")) return cardinal.slice(0, -1) + "th";
  if (cardinal.endsWith("t")) return cardinal + "h";
  return cardinal + "th";
}

const UNICODE_MAP: Record<string, string> = {
  "à": "a",
  "á": "a",
  "â": "a",
  "ã": "a",
  "ä": "a",
  "å": "a",
  "æ": "ae",
  "ç": "c",
  "è": "e",
  "é": "e",
  "ê": "e",
  "ë": "e",
  "ì": "i",
  "í": "i",
  "î": "i",
  "ï": "i",
  "ñ": "n",
  "ò": "o",
  "ó": "o",
  "ô": "o",
  "õ": "o",
  "ö": "o",
  "ø": "o",
  "ù": "u",
  "ú": "u",
  "û": "u",
  "ü": "u",
  "ý": "y",
  "ÿ": "y",
  "ß": "ss",
  "œ": "oe",
  "ð": "d",
  "þ": "th",
  "À": "A",
  "Á": "A",
  "Â": "A",
  "Ã": "A",
  "Ä": "A",
  "Å": "A",
  "Æ": "AE",
  "Ç": "C",
  "È": "E",
  "É": "E",
  "Ê": "E",
  "Ë": "E",
  "Ì": "I",
  "Í": "I",
  "Î": "I",
  "Ï": "I",
  "Ñ": "N",
  "Ò": "O",
  "Ó": "O",
  "Ô": "O",
  "Õ": "O",
  "Ö": "O",
  "Ø": "O",
  "Ù": "U",
  "Ú": "U",
  "Û": "U",
  "Ü": "U",
  "Ý": "Y",
  "\u201C": '"',
  "\u201D": '"',
  "\u2018": "'",
  "\u2019": "'",
  "\u2026": "...",
  "\u2013": "-",
  "\u2014": "-",
};

function convertToAscii(text: string): string {
  return text.split("").map((c) => UNICODE_MAP[c] || c).join("").normalize(
    "NFD",
  ).replace(/[\u0300-\u036f]/g, "");
}

const ABBREVIATIONS: [RegExp, string][] = [
  [/\bmrs\./gi, "misuss"],
  [/\bms\./gi, "miss"],
  [/\bmr\./gi, "mister"],
  [/\bdr\./gi, "doctor"],
  [/\bst\./gi, "saint"],
  [/\bco\./gi, "company"],
  [/\bjr\./gi, "junior"],
  [/\bmaj\./gi, "major"],
  [/\bgen\./gi, "general"],
  [/\bdrs\./gi, "doctors"],
  [/\brev\./gi, "reverend"],
  [/\blt\./gi, "lieutenant"],
  [/\bhon\./gi, "honorable"],
  [/\bsgt\./gi, "sergeant"],
  [/\bcapt\./gi, "captain"],
  [/\besq\./gi, "esquire"],
  [/\bltd\./gi, "limited"],
  [/\bcol\./gi, "colonel"],
  [/\bft\./gi, "fort"],
];
const CASED_ABBREVIATIONS: [RegExp, string][] = [
  [/\bTTS\b/g, "text to speech"],
  [/\bHz\b/g, "hertz"],
  [/\bkHz\b/g, "kilohertz"],
  [/\bKBs\b/g, "kilobytes"],
  [/\bKB\b/g, "kilobyte"],
  [/\bMBs\b/g, "megabytes"],
  [/\bMB\b/g, "megabyte"],
  [/\bGBs\b/g, "gigabytes"],
  [/\bGB\b/g, "gigabyte"],
  [/\bTBs\b/g, "terabytes"],
  [/\bTB\b/g, "terabyte"],
  [/\bAPIs\b/g, "a p i's"],
  [/\bAPI\b/g, "a p i"],
  [/\bCLIs\b/g, "c l i's"],
  [/\bCLI\b/g, "c l i"],
  [/\bCPUs\b/g, "c p u's"],
  [/\bCPU\b/g, "c p u"],
  [/\bGPUs\b/g, "g p u's"],
  [/\bGPU\b/g, "g p u"],
  [/\bAve\b/g, "avenue"],
  [/\betc\b/g, "etcetera"],
];

function expandAbbreviations(text: string): string {
  for (const [re, rep] of [...ABBREVIATIONS, ...CASED_ABBREVIATIONS]) {
    text = text.replace(re, rep);
  }
  return text;
}

function normalizeNumbers(text: string): string {
  text = text.replace(/#(\d)/g, (_, d) => `number ${d}`);
  text = text.replace(/(\d)([KMBT])/gi, (_, num, suffix) => {
    const map: Record<string, string> = {
      k: "thousand",
      m: "million",
      b: "billion",
      t: "trillion",
    };
    return `${num} ${map[suffix.toLowerCase()]}`;
  });
  for (let i = 0; i < 2; i++) {
    text = text.replace(/(\d)([a-z])|([a-z])(\d)/gi, (m, d1, l1, l2, d2) => {
      if (d1 && l1) return `${d1} ${l1}`;
      if (l2 && d2) return `${l2} ${d2}`;
      return m;
    });
  }
  text = text.replace(/(\d[\d,]+\d)/g, (m) => m.replace(/,/g, ""));
  text = text.replace(
    /(^|[^/])(\d\d?[/-]\d\d?[/-]\d\d(?:\d\d)?)($|[^/])/g,
    (_, pre, date, post) => pre + date.split(/[./-]/).join(" dash ") + post,
  );
  text = text.replace(/\(?\d{3}\)?[-.\s]\d{3}[-.\s]?\d{4}/g, (m) => {
    const digits = m.replace(/\D/g, "");
    return digits.length === 10
      ? `${digits.slice(0, 3).split("").join(" ")}, ${
        digits.slice(3, 6).split("").join(" ")
      }, ${digits.slice(6).split("").join(" ")}`
      : m;
  });
  text = text.replace(
    /(\d\d?):(\d\d)(?::(\d\d))?/g,
    (_, hours, minutes, seconds) => {
      const h = parseInt(hours),
        m = parseInt(minutes),
        s = seconds ? parseInt(seconds) : 0;
      if (!seconds) {
        return m === 0
          ? (h === 0 ? "0" : h > 12 ? `${hours} minutes` : `${hours} o'clock`)
          : minutes.startsWith("0")
          ? `${hours} oh ${minutes[1]}`
          : `${hours} ${minutes}`;
      }
      let res = "";
      if (h !== 0) {
        res = hours + " " +
          (m === 0
            ? "oh oh"
            : minutes.startsWith("0")
            ? `oh ${minutes[1]}`
            : minutes);
      } else if (m !== 0) {
        res = minutes + " " +
          (s === 0
            ? "oh oh"
            : seconds.startsWith("0")
            ? `oh ${seconds[1]}`
            : seconds);
      } else res = seconds;
      return res + " " +
        (s === 0 ? "" : seconds.startsWith("0") ? `oh ${seconds[1]}` : seconds);
    },
  );
  text = text.replace(
    /£([\d,]*\d+)/g,
    (_, amount) => `${amount.replace(/,/g, "")} pounds`,
  );
  text = text.replace(/\$([\d.,]*\d+)/g, (_, amount) => {
    const parts = amount.replace(/,/g, "").split(".");
    const dollars = parseInt(parts[0]) || 0,
      cents = parts[1] ? parseInt(parts[1]) : 0;
    if (dollars && cents) {
      return `${dollars} ${dollars === 1 ? "dollar" : "dollars"}, ${cents} ${
        cents === 1 ? "cent" : "cents"
      }`;
    }
    if (dollars) return `${dollars} ${dollars === 1 ? "dollar" : "dollars"}`;
    if (cents) return `${cents} ${cents === 1 ? "cent" : "cents"}`;
    return "zero dollars";
  });
  text = text.replace(
    /(\d+(?:\.\d+)+)/g,
    (m) => m.split(".").join(" point ").split("").join(" "),
  );
  text = text.replace(/(\d)\s?\*\s?(\d)/g, "$1 times $2");
  text = text.replace(/(\d)\s?\/\s?(\d)/g, "$1 over $2");
  text = text.replace(/(\d)\s?\+\s?(\d)/g, "$1 plus $2");
  text = text.replace(
    /(\d)?\s?-\s?(\d)/g,
    (_, a, b) => (a ? a : "") + " minus " + b,
  );
  text = text.replace(/(\d+)\/(\d+)/g, "$1 over $2");
  text = text.replace(
    /(\d+)(st|nd|rd|th)/gi,
    (_, num) => ordinalToWords(parseInt(num)),
  );
  text = text.replace(/\d+/g, (m) => {
    const num = parseInt(m);
    if (num > 1000 && num < 3000) {
      if (num === 2000) return "two thousand";
      if (num > 2000 && num < 2010) {
        return "two thousand " + numberToWords(num % 100);
      }
      if (num % 100 === 0) {
        return numberToWords(Math.floor(num / 100)) + " hundred";
      }
      return numberToWords(num, { zero: "oh", group: 2 });
    }
    return numberToWords(num);
  });
  return text;
}

const SPECIAL_CHARACTERS: [RegExp, string][] = [
  [/@/g, " at "],
  [/&/g, " and "],
  [/%/g, " percent "],
  [/:/g, "."],
  [/;/g, ","],
  [/\+/g, " plus "],
  [/\\/g, " backslash "],
  [/~/g, " about "],
  [/(^| )<3/g, " heart "],
  [/<=/g, " less than or equal to "],
  [/>=/g, " greater than or equal to "],
  [/</g, " less than "],
  [/>/g, " greater than "],
  [/=/g, " equals "],
  [/\//g, " slash "],
  [/_/g, " "],
];

function normalizeSpecial(text: string): string {
  text = text.replace(/https?:\/\//gi, "h t t p s colon slash slash ");
  text = text.replace(/(.) - (.)/g, "$1, $2");
  text = text.replace(/([A-Z])\.([A-Z])/gi, "$1 dot $2");
  text = text.replace(/[\(\[\{][^\)\]\}]*[\)\]\}](.)?/g, (m, after) => {
    let result = m.replace(/[\(\[\{]/g, ", ").replace(/[\)\]\}]/g, ", ");
    if (after && /[$.!?,]/.test(after)) result = result.slice(0, -2) + after;
    return result;
  });
  return text;
}

function expandSpecialCharacters(text: string): string {
  for (const [re, rep] of SPECIAL_CHARACTERS) text = text.replace(re, rep);
  return text;
}

function collapseWhitespace(text: string): string {
  return text.replace(/\s+/g, " ").replace(/ ([.\?!,])/g, "$1");
}

function dedupPunctuation(text: string): string {
  return text.replace(/\.\.\.+/g, "[ELLIPSIS]").replace(/,+/g, ",").replace(
    /[.,]*\.[.,]*/g,
    ".",
  ).replace(/[.,!]*![.,!]*/g, "!").replace(/[.,!?]*\?[.,!?]*/g, "?").replace(
    /\[ELLIPSIS\]/g,
    "...",
  );
}

function prepareText(text: string): string {
  text = text.trim();
  if (!text) return "";
  text = convertToAscii(text);
  text = normalizeNumbers(text);
  text = normalizeSpecial(text);
  text = expandAbbreviations(text);
  text = expandSpecialCharacters(text);
  text = collapseWhitespace(text);
  text = dedupPunctuation(text);
  text = text.trim();
  if (text && text[text.length - 1].match(/[a-zA-Z0-9]/)) text += ".";
  if (text && !text[0].match(/[A-Z]/)) {
    text = text[0].toUpperCase() + text.slice(1);
  }
  return text;
}

function splitTextIntoSentences(text: string): string[] {
  const matches = text.match(/[^.!?]+[.!?]+|[^.!?]+$/g);
  if (!matches) return [];
  return matches.map((s) => s.trim()).filter(Boolean);
}

function splitTokenIdsIntoChunks(
  tokenIds: number[],
  maxTokens: number,
): string[] {
  const chunks: string[] = [];
  for (let i = 0; i < tokenIds.length; i += maxTokens) {
    const chunkText = tokenizerProcessor!.decodeIds(
      tokenIds.slice(i, i + maxTokens),
    ).trim();
    if (chunkText) chunks.push(chunkText);
  }
  return chunks;
}

function splitIntoBestSentences(text: string): string[] {
  const prepared = prepareText(text);
  if (!prepared) return [];
  const sentences = splitTextIntoSentences(prepared);
  if (sentences.length === 0) return [];

  const chunks: string[] = [];
  let current = "";
  for (const sentence of sentences) {
    const sentTokens = tokenizerProcessor!.encodeIds(sentence);
    if (sentTokens.length > CHUNK_TARGET_TOKENS) {
      if (current) {
        chunks.push(current.trim());
        current = "";
      }
      for (
        const c of splitTokenIdsIntoChunks(sentTokens, CHUNK_TARGET_TOKENS)
      ) {
        if (c) chunks.push(c.trim());
      }
      continue;
    }
    if (!current) {
      current = sentence;
      continue;
    }
    const combined = `${current} ${sentence}`;
    if (tokenizerProcessor!.encodeIds(combined).length > CHUNK_TARGET_TOKENS) {
      chunks.push(current.trim());
      current = sentence;
    } else {
      current = combined;
    }
  }
  if (current) chunks.push(current.trim());
  return chunks;
}

// ─── Model State Shapes (from inference-worker.js) ────────────────────────────

const FLOW_LM_STATE_SHAPES: StateShapes = {
  state_0: { shape: [2, 1, 1000, 16, 64], dtype: "float32" },
  state_1: { shape: [0], dtype: "float32" },
  state_2: { shape: [1], dtype: "int64" },
  state_3: { shape: [2, 1, 1000, 16, 64], dtype: "float32" },
  state_4: { shape: [0], dtype: "float32" },
  state_5: { shape: [1], dtype: "int64" },
  state_6: { shape: [2, 1, 1000, 16, 64], dtype: "float32" },
  state_7: { shape: [0], dtype: "float32" },
  state_8: { shape: [1], dtype: "int64" },
  state_9: { shape: [2, 1, 1000, 16, 64], dtype: "float32" },
  state_10: { shape: [0], dtype: "float32" },
  state_11: { shape: [1], dtype: "int64" },
  state_12: { shape: [2, 1, 1000, 16, 64], dtype: "float32" },
  state_13: { shape: [0], dtype: "float32" },
  state_14: { shape: [1], dtype: "int64" },
  state_15: { shape: [2, 1, 1000, 16, 64], dtype: "float32" },
  state_16: { shape: [0], dtype: "float32" },
  state_17: { shape: [1], dtype: "int64" },
};

const MIMI_DECODER_STATE_SHAPES: StateShapes = {
  state_0: { shape: [1], dtype: "bool" },
  state_1: { shape: [1, 512, 6], dtype: "float32" },
  state_2: { shape: [1], dtype: "bool" },
  state_3: { shape: [1, 64, 2], dtype: "float32" },
  state_4: { shape: [1, 256, 6], dtype: "float32" },
  state_5: { shape: [1], dtype: "bool" },
  state_6: { shape: [1, 256, 2], dtype: "float32" },
  state_7: { shape: [1], dtype: "bool" },
  state_8: { shape: [1, 128, 0], dtype: "float32" },
  state_9: { shape: [1, 128, 5], dtype: "float32" },
  state_10: { shape: [1], dtype: "bool" },
  state_11: { shape: [1, 128, 2], dtype: "float32" },
  state_12: { shape: [1], dtype: "bool" },
  state_13: { shape: [1, 64, 0], dtype: "float32" },
  state_14: { shape: [1, 64, 4], dtype: "float32" },
  state_15: { shape: [1], dtype: "bool" },
  state_16: { shape: [1, 64, 2], dtype: "float32" },
  state_17: { shape: [1], dtype: "bool" },
  state_18: { shape: [1, 32, 0], dtype: "float32" },
  state_19: { shape: [2, 1, 8, 1000, 64], dtype: "float32" },
  state_20: { shape: [1], dtype: "int64" },
  state_21: { shape: [1], dtype: "int64" },
  state_22: { shape: [2, 1, 8, 1000, 64], dtype: "float32" },
  state_23: { shape: [1], dtype: "int64" },
  state_24: { shape: [1], dtype: "int64" },
  state_25: { shape: [1], dtype: "bool" },
  state_26: { shape: [1, 512, 16], dtype: "float32" },
  state_27: { shape: [1], dtype: "bool" },
  state_28: { shape: [1, 1, 6], dtype: "float32" },
  state_29: { shape: [1], dtype: "bool" },
  state_30: { shape: [1, 64, 2], dtype: "float32" },
  state_31: { shape: [1], dtype: "bool" },
  state_32: { shape: [1, 32, 0], dtype: "float32" },
  state_33: { shape: [1], dtype: "bool" },
  state_34: { shape: [1, 512, 2], dtype: "float32" },
  state_35: { shape: [1], dtype: "bool" },
  state_36: { shape: [1, 64, 4], dtype: "float32" },
  state_37: { shape: [1], dtype: "bool" },
  state_38: { shape: [1, 128, 2], dtype: "float32" },
  state_39: { shape: [1], dtype: "bool" },
  state_40: { shape: [1, 64, 0], dtype: "float32" },
  state_41: { shape: [1], dtype: "bool" },
  state_42: { shape: [1, 128, 5], dtype: "float32" },
  state_43: { shape: [1], dtype: "bool" },
  state_44: { shape: [1, 256, 2], dtype: "float32" },
  state_45: { shape: [1], dtype: "bool" },
  state_46: { shape: [1, 128, 0], dtype: "float32" },
  state_47: { shape: [1], dtype: "bool" },
  state_48: { shape: [1, 256, 6], dtype: "float32" },
  state_49: { shape: [2, 1, 8, 1000, 64], dtype: "float32" },
  state_50: { shape: [1], dtype: "int64" },
  state_51: { shape: [1], dtype: "int64" },
  state_52: { shape: [2, 1, 8, 1000, 64], dtype: "float32" },
  state_53: { shape: [1], dtype: "int64" },
  state_54: { shape: [1], dtype: "int64" },
  state_55: { shape: [1, 512, 16], dtype: "float32" },
};

// ─── Model Loading ────────────────────────────────────────────────────────────

async function loadModels(): Promise<void> {
  const threads = Math.min(navigator.hardwareConcurrency || 4, 8);
  const sessionOpts: ort.InferenceSession.SessionOptions = {
    executionProviders: ["cpu"],
    graphOptimizationLevel: "all",
    intraOpNumThreads: threads,
  };

  console.log(`  Loading ONNX models (${threads} threads)...`);
  [
    mimiEncoderSession,
    textConditionerSession,
    flowLmMainSession,
    flowLmFlowSession,
    mimiDecoderSession,
  ] = await Promise.all([
    ort.InferenceSession.create(
      BASE_DIR + "onnx/mimi_encoder.onnx",
      sessionOpts,
    ),
    ort.InferenceSession.create(
      BASE_DIR + "onnx/text_conditioner.onnx",
      sessionOpts,
    ),
    ort.InferenceSession.create(
      BASE_DIR + "onnx/flow_lm_main_int8.onnx",
      sessionOpts,
    ),
    ort.InferenceSession.create(
      BASE_DIR + "onnx/flow_lm_flow_int8.onnx",
      sessionOpts,
    ),
    ort.InferenceSession.create(
      BASE_DIR + "onnx/mimi_decoder_int8.onnx",
      sessionOpts,
    ),
  ]);
  console.log("  ONNX sessions created.");

  console.log("  Loading tokenizer...");
  const tokBytes = await Deno.readFile(BASE_DIR + "tokenizer.model");
  tokenizerProcessor = new SentencePieceProcessor();
  await tokenizerProcessor.loadModel(tokBytes);
  console.log("  Tokenizer ready.");

  console.log("  Loading voices...");
  const voiceBytes = await Deno.readFile(BASE_DIR + "voices.bin");
  predefinedVoices = parseVoicesBin(voiceBytes.buffer);
  console.log("  Voices:", Object.keys(predefinedVoices).join(", "));

  // Pre-condition default voice so first request is fast
  const defaultVoice = predefinedVoices["cosette"]
    ? "cosette"
    : Object.keys(predefinedVoices)[0];
  if (defaultVoice) {
    console.log(`  Pre-conditioning voice "${defaultVoice}"...`);
    await ensureVoiceConditioned(
      defaultVoice,
      predefinedVoices[defaultVoice],
      true,
    );
  }

  // Pre-allocate flow matching s/t tensors for all LSD levels
  stTensors = {};
  for (let lsd = 1; lsd <= MAX_LSD; lsd++) {
    stTensors[lsd] = [];
    for (let j = 0; j < lsd; j++) {
      const s = j / lsd, t = s + 1 / lsd;
      stTensors[lsd].push({
        s: new ort.Tensor("float32", new Float32Array([s]), [1, 1]),
        t: new ort.Tensor("float32", new Float32Array([t]), [1, 1]),
      });
    }
  }

  isReady = true;
}

// ─── Voice Utilities ──────────────────────────────────────────────────────────

function parseVoicesBin(buffer: ArrayBuffer): Record<string, VoiceEmbedding> {
  const voices: Record<string, VoiceEmbedding> = {};
  const view = new DataView(buffer);
  let offset = 0;
  const numVoices = view.getUint32(offset, true);
  offset += 4;
  for (let i = 0; i < numVoices; i++) {
    const nameBytes = new Uint8Array(buffer, offset, 32);
    const nameEnd = nameBytes.indexOf(0);
    const name = new TextDecoder().decode(
      nameBytes.subarray(0, nameEnd > 0 ? nameEnd : 32),
    ).trim();
    offset += 32;
    const numFrames = view.getUint32(offset, true);
    offset += 4;
    const embDim = view.getUint32(offset, true);
    offset += 4;
    const embSize = numFrames * embDim;
    voices[name] = {
      data: new Float32Array(buffer.slice(offset, offset + embSize * 4)),
      shape: [1, numFrames, embDim],
    };
    offset += embSize * 4;
    console.log(`    Voice "${name}": ${numFrames} frames, ${embDim} dim`);
  }
  return voices;
}

async function encodeVoiceAudio(
  audioData: Float32Array,
): Promise<VoiceEmbedding> {
  const input = new ort.Tensor("float32", audioData, [1, 1, audioData.length]);
  const out = await mimiEncoderSession!.run({ audio: input });
  const emb = out[mimiEncoderSession!.outputNames[0]];
  return {
    data: new Float32Array(emb.data as Float32Array),
    shape: emb.dims as [number, number, number],
  };
}

async function buildVoiceConditionedState(
  voiceEmb: VoiceEmbedding,
): Promise<StateMap> {
  const flowLmState = initState(flowLmMainSession!, FLOW_LM_STATE_SHAPES);
  const emptySeq = new ort.Tensor("float32", new Float32Array(0), [1, 0, 32]);
  const voiceTensor = new ort.Tensor("float32", voiceEmb.data, voiceEmb.shape);

  const result = await flowLmMainSession!.run({
    sequence: emptySeq,
    text_embeddings: voiceTensor,
    ...flowLmState,
  });

  for (let i = 2; i < flowLmMainSession!.outputNames.length; i++) {
    const name = flowLmMainSession!.outputNames[i];
    if (name.startsWith("out_state_")) {
      const idx = parseInt(name.replace("out_state_", ""));
      flowLmState[`state_${idx}`] = result[name];
    }
  }
  return flowLmState;
}

async function ensureVoiceConditioned(
  name: string,
  emb: VoiceEmbedding,
  force = false,
): Promise<void> {
  if (!force && voiceConditioningCache.has(name)) return;
  const state = await buildVoiceConditionedState(emb);
  voiceConditioningCache.set(name, state);
}

function resolveVoice(voiceName: string): VoiceEmbedding | null {
  return predefinedVoices[voiceName] ?? customVoices.get(voiceName) ?? null;
}

// ─── State Management ─────────────────────────────────────────────────────────

function initState(
  session: ort.InferenceSession,
  shapes: StateShapes,
): StateMap {
  const state: StateMap = {};
  for (const name of session.inputNames) {
    if (!name.startsWith("state_")) continue;
    const info = shapes[name];
    if (!info) {
      console.warn(`Unknown state: ${name}`);
      continue;
    }
    const { shape, dtype } = info;
    const size = shape.reduce((a, b) => a * b, 1);
    let data: Float32Array | BigInt64Array | Uint8Array;
    if (dtype === "int64") data = new BigInt64Array(size);
    else if (dtype === "bool") data = new Uint8Array(size);
    else data = new Float32Array(size);
    state[name] = new ort.Tensor(dtype, data, shape);
  }
  return state;
}

function cloneFlowState(state: StateMap): StateMap {
  return { ...state };
}

// ─── Speech Generation ────────────────────────────────────────────────────────

async function* generateSpeech(
  text: string,
  voiceName: string,
  lsd: number = MAX_LSD,
): AsyncGenerator<Float32Array> {
  const chunks = splitIntoBestSentences(text);
  if (chunks.length === 0) {
    throw new Error("No text to generate after preprocessing.");
  }

  // Resolve and condition voice
  const voiceEmb = resolveVoice(voiceName);
  if (!voiceEmb) throw new Error(`Unknown voice: "${voiceName}"`);
  await ensureVoiceConditioned(voiceName, voiceEmb);

  const baseFlowState = voiceConditioningCache.get(voiceName)!;
  let mimiState = initState(mimiDecoderSession!, MIMI_DECODER_STATE_SHAPES);
  let flowLmState = cloneFlowState(baseFlowState);

  const emptySeq = new ort.Tensor("float32", new Float32Array(0), [1, 0, 32]);
  const emptyTextEmb = new ort.Tensor("float32", new Float32Array(0), [
    1,
    0,
    1024,
  ]);

  const FIRST_CHUNK_FRAMES = 3;
  const NORMAL_CHUNK_FRAMES = 12;
  let isFirstAudioChunk = true;

  for (let chunkIdx = 0; chunkIdx < chunks.length; chunkIdx++) {
    if (RESET_FLOW_STATE_EACH_CHUNK && chunkIdx > 0) {
      flowLmState = cloneFlowState(baseFlowState);
    }
    if (RESET_MIMI_STATE_EACH_CHUNK && chunkIdx > 0) {
      mimiState = initState(mimiDecoderSession!, MIMI_DECODER_STATE_SHAPES);
    }

    const chunkText = chunks[chunkIdx];
    const tokenIds = tokenizerProcessor!.encodeIds(chunkText);
    const textInput = new ort.Tensor(
      "int64",
      BigInt64Array.from(tokenIds.map((x: number) => BigInt(x))),
      [1, tokenIds.length],
    );

    // Text conditioning
    const textCondResult = await textConditionerSession!.run({
      token_ids: textInput,
    });
    let textEmb = textCondResult[textConditionerSession!.outputNames[0]];
    if (textEmb.dims.length === 2) {
      textEmb = new ort.Tensor("float32", textEmb.data as Float32Array, [
        1,
        textEmb.dims[0],
        textEmb.dims[1],
      ]);
    }

    const condResult = await flowLmMainSession!.run({
      sequence: emptySeq,
      text_embeddings: textEmb,
      ...flowLmState,
    });
    for (let i = 2; i < flowLmMainSession!.outputNames.length; i++) {
      const name = flowLmMainSession!.outputNames[i];
      if (name.startsWith("out_state_")) {
        const idx = parseInt(name.replace("out_state_", ""));
        flowLmState[`state_${idx}`] = condResult[name];
      }
    }

    // AR generation loop
    const chunkLatents: Float32Array[] = [];
    let currentLatent = new ort.Tensor(
      "float32",
      new Float32Array(32).fill(NaN),
      [1, 1, 32],
    );
    let chunkDecodedFrames = 0;
    let eosStep: number | null = null;
    let chunkEnded = false;

    for (let step = 0; step < MAX_FRAMES; step++) {
      // Yield to event loop periodically so server stays responsive
      if (step > 0 && step % 4 === 0) {
        await new Promise((r) => setTimeout(r, 0));
      }

      const arResult = await flowLmMainSession!.run({
        sequence: currentLatent,
        text_embeddings: emptyTextEmb,
        ...flowLmState,
      });

      const conditioning = arResult["conditioning"];
      const eosLogit = (arResult["eos_logit"].data as Float32Array)[0];
      const isEos = eosLogit > -4.0;

      if (isEos && eosStep === null) eosStep = step;
      const shouldStop = eosStep !== null && step >= eosStep + FRAMES_AFTER_EOS;

      // Flow matching (Euler integration)
      const TEMP = 0.7, STD = Math.sqrt(TEMP);
      const xData = new Float32Array(32);
      for (let i = 0; i < 32; i++) {
        let u = 0, v = 0;
        while (u === 0) u = Math.random();
        while (v === 0) v = Math.random();
        xData[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) *
          STD;
      }

      const dt = 1.0 / lsd;
      for (let j = 0; j < lsd; j++) {
        const flowResult = await flowLmFlowSession!.run({
          c: conditioning,
          s: stTensors[lsd][j].s,
          t: stTensors[lsd][j].t,
          x: new ort.Tensor("float32", xData, [1, 32]),
        });
        const v = flowResult["flow_dir"].data as Float32Array;
        for (let k = 0; k < 32; k++) xData[k] += v[k] * dt;
      }

      chunkLatents.push(new Float32Array(xData));

      // Update AR state
      currentLatent = new ort.Tensor("float32", xData, [1, 1, 32]);
      for (let i = 2; i < flowLmMainSession!.outputNames.length; i++) {
        const name = flowLmMainSession!.outputNames[i];
        if (name.startsWith("out_state_")) {
          const idx = parseInt(name.replace("out_state_", ""));
          flowLmState[`state_${idx}`] = arResult[name];
        }
      }

      // Decide how many latents to decode now
      const pending = chunkLatents.length - chunkDecodedFrames;
      let decodeSize = 0;
      if (shouldStop) decodeSize = pending;
      else if (isFirstAudioChunk && pending >= FIRST_CHUNK_FRAMES) {
        decodeSize = FIRST_CHUNK_FRAMES;
      } else if (pending >= NORMAL_CHUNK_FRAMES) {
        decodeSize = NORMAL_CHUNK_FRAMES;
      }

      if (decodeSize > 0) {
        const decodeLatents = new Float32Array(decodeSize * 32);
        for (let i = 0; i < decodeSize; i++) {
          decodeLatents.set(chunkLatents[chunkDecodedFrames + i], i * 32);
        }

        const latentTensor = new ort.Tensor("float32", decodeLatents, [
          1,
          decodeSize,
          32,
        ]);
        const decodeResult = await mimiDecoderSession!.run({
          latent: latentTensor,
          ...mimiState,
        });

        const audioData = decodeResult[mimiDecoderSession!.outputNames[0]]
          .data as Float32Array;
        for (let i = 1; i < mimiDecoderSession!.outputNames.length; i++) {
          const name = mimiDecoderSession!.outputNames[i];
          mimiState[`state_${i - 1}`] = decodeResult[name];
        }

        chunkDecodedFrames += decodeSize;
        isFirstAudioChunk = false;
        yield new Float32Array(audioData);
      }

      if (shouldStop) {
        chunkEnded = true;
        break;
      }
    }

    // Insert silence gap between text chunks
    if (chunkEnded && chunkIdx < chunks.length - 1) {
      const gapSamples = Math.max(1, Math.floor(CHUNK_GAP_SEC * SAMPLE_RATE));
      yield new Float32Array(gapSamples);
    }
  }
}

// ─── WAV Encoding ─────────────────────────────────────────────────────────────

/**
 * Build a 44-byte WAV/RIFF header.
 * Pass dataSize = -1 for streaming (writes 0xFFFFFFFF as a placeholder).
 */
function buildWavHeader(sampleRate: number, dataSize: number): Uint8Array {
  const buf = new ArrayBuffer(44);
  const view = new DataView(buf);
  const enc = new TextEncoder();
  const w4 = (o: number, s: string) =>
    enc.encode(s).forEach((b, i) => view.setUint8(o + i, b));
  const sz = dataSize < 0 ? 0xffffffff : dataSize;
  const riff = dataSize < 0 ? 0xffffffff : sz + 36;

  w4(0, "RIFF");
  view.setUint32(4, riff, true);
  w4(8, "WAVE");
  w4(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byteRate = sampleRate * channels * bitsPerSample/8
  view.setUint16(32, 2, true); // blockAlign
  view.setUint16(34, 16, true); // bitsPerSample
  w4(36, "data");
  view.setUint32(40, sz, true);
  return new Uint8Array(buf);
}

function float32ToPcm16(samples: Float32Array): Uint8Array {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    out[i] = s < 0 ? Math.round(s * 0x8000) : Math.round(s * 0x7fff);
  }
  return new Uint8Array(out.buffer);
}

// ─── WAV Parsing (for voice cloning uploads) ──────────────────────────────────

interface WavData {
  samples: Float32Array;
  sampleRate: number;
  channels: number;
}

function parseWavFile(buffer: ArrayBuffer): WavData {
  const view = new DataView(buffer);

  const riff = String.fromCharCode(
    view.getUint8(0),
    view.getUint8(1),
    view.getUint8(2),
    view.getUint8(3),
  );
  if (riff !== "RIFF") throw new Error("Not a RIFF/WAV file.");

  let offset = 12;
  let audioFormat = 1, channels = 1, sampleRate = 24000, bitsPerSample = 16;
  let dataOffset = -1, dataSize = 0;

  while (offset + 8 <= view.byteLength) {
    const id = String.fromCharCode(
      view.getUint8(offset),
      view.getUint8(offset + 1),
      view.getUint8(offset + 2),
      view.getUint8(offset + 3),
    );
    const size = view.getUint32(offset + 4, true);

    if (id === "fmt ") {
      audioFormat = view.getUint16(offset + 8, true);
      channels = view.getUint16(offset + 10, true);
      sampleRate = view.getUint32(offset + 12, true);
      bitsPerSample = view.getUint16(offset + 22, true);
    } else if (id === "data") {
      dataOffset = offset + 8;
      dataSize = size;
      break;
    }
    offset += 8 + size + (size % 2); // word-align
  }

  if (dataOffset < 0) throw new Error("WAV file has no data chunk.");

  const bytesPerSample = bitsPerSample / 8;
  const totalSamples = dataSize / bytesPerSample;
  const samples = new Float32Array(totalSamples);

  if (audioFormat === 1) {
    // PCM integer
    for (let i = 0; i < totalSamples; i++) {
      if (bitsPerSample === 8) {
        samples[i] = (view.getUint8(dataOffset + i) - 128) / 128;
      } else if (bitsPerSample === 16) {
        samples[i] = view.getInt16(dataOffset + i * 2, true) / 32768;
      } else if (bitsPerSample === 24) {
        const lo = view.getUint8(dataOffset + i * 3);
        const hi = view.getInt16(dataOffset + i * 3 + 1, true);
        samples[i] = ((hi << 8) | lo) / 8388608;
      } else if (bitsPerSample === 32) {
        samples[i] = view.getInt32(dataOffset + i * 4, true) / 2147483648;
      }
    }
  } else if (audioFormat === 3) {
    // IEEE float
    for (let i = 0; i < totalSamples; i++) {
      samples[i] = view.getFloat32(dataOffset + i * 4, true);
    }
  } else {
    throw new Error(
      `Unsupported WAV audio format: ${audioFormat} (only PCM and IEEE float supported).`,
    );
  }

  return { samples, sampleRate, channels };
}

function resampleAudio(
  samples: Float32Array,
  fromRate: number,
  toRate: number,
): Float32Array {
  if (fromRate === toRate) return samples;
  const ratio = fromRate / toRate;
  const outLen = Math.floor(samples.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const src = i * ratio;
    const floor = Math.floor(src);
    const ceil = Math.min(floor + 1, samples.length - 1);
    const t = src - floor;
    out[i] = samples[floor] * (1 - t) + samples[ceil] * t;
  }
  return out;
}

function mixToMono(samples: Float32Array, channels: number): Float32Array {
  if (channels === 1) return samples;
  const frames = samples.length / channels;
  const mono = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    let sum = 0;
    for (let c = 0; c < channels; c++) sum += samples[i * channels + c];
    mono[i] = sum / channels;
  }
  return mono;
}

/**
 * Decode a WAV buffer to Float32 mono at SAMPLE_RATE (24 kHz).
 * Handles PCM16/PCM32/Float32, mono/stereo, arbitrary sample rate.
 */
function decodeAudioForVoiceCloning(buffer: ArrayBuffer): Float32Array {
  const { samples, sampleRate, channels } = parseWavFile(buffer);
  let audio = mixToMono(samples, channels);
  audio = resampleAudio(audio, sampleRate, SAMPLE_RATE);
  const maxSamples = SAMPLE_RATE * MAX_VOICE_AUDIO_SEC;
  return audio.length > maxSamples ? audio.slice(0, maxSamples) : audio;
}

// ─── HTTP Helpers ─────────────────────────────────────────────────────────────

function corsHeaders() {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  };
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body, null, 2), {
    status,
    headers: { ...corsHeaders(), "Content-Type": "application/json" },
  });
}

// ─── HTTP Handlers ─────────────────────────────────────────────────────────────

function handleInfo(): Response {
  const allVoices = [...Object.keys(predefinedVoices), ...customVoices.keys()];
  return json({
    name: "Pocket TTS Server",
    status: isReady ? "ready" : "loading",
    voices: allVoices,
    endpoints: {
      "GET  /v1/voices": "List available voices",
      "POST /v1/voices": "Register a custom voice from WAV upload",
      "DELETE /v1/voices/:name": "Remove a custom voice",
      "POST /v1/audio/speech": "Generate speech; returns streaming WAV audio",
    },
  });
}

function handleHealth(): Response {
  if (!isReady) return json({ status: "loading" }, 503);
  return json({ status: "ok" });
}

function handleListVoices(): Response {
  const builtin = Object.keys(predefinedVoices);
  const custom = [...customVoices.keys()];
  return json({ voices: [...builtin, ...custom], builtin, custom });
}

async function handleRegisterVoice(req: Request, url: URL): Promise<Response> {
  if (!isReady) return json({ error: "Models not ready yet." }, 503);

  const name = url.searchParams.get("name") || `voice_${Date.now()}`;
  const ct = req.headers.get("content-type") || "";

  let audioBuffer: ArrayBuffer;

  if (ct.includes("multipart/form-data")) {
    const form = await req.formData();
    const file = form.get("file");
    if (!file || typeof file === "string") {
      return json(
        { error: "Multipart field 'file' (WAV audio) is required." },
        400,
      );
    }
    audioBuffer = await (file as File).arrayBuffer();
  } else {
    // Treat raw body as WAV bytes
    audioBuffer = await req.arrayBuffer();
  }

  if (audioBuffer.byteLength < 44) {
    return json({
      error: "Uploaded audio is too small to be a valid WAV file.",
    }, 400);
  }

  let audio: Float32Array;
  try {
    audio = decodeAudioForVoiceCloning(audioBuffer);
  } catch (err) {
    return json({ error: `Audio decode failed: ${err}` }, 400);
  }

  const emb = await encodeVoiceAudio(audio);
  customVoices.set(name, emb);
  await ensureVoiceConditioned(name, emb, true);

  return json({ id: name, frames: emb.shape[1], status: "ready" }, 201);
}

function handleDeleteVoice(path: string): Response {
  const name = path.replace("/v1/voices/", "");
  if (predefinedVoices[name]) {
    return json({ error: "Cannot delete a built-in voice." }, 403);
  }
  if (!customVoices.has(name)) {
    return json({ error: `Voice "${name}" not found.` }, 404);
  }
  customVoices.delete(name);
  voiceConditioningCache.delete(name);
  return json({ deleted: name });
}

async function handleSpeech(req: Request, _url: URL): Promise<Response> {
  if (!isReady) return json({ error: "Models not ready yet." }, 503);

  let body: {
    input?: string;
    voice?: string;
    speed?: number;
    response_format?: string;
  };
  try {
    body = await req.json();
  } catch {
    return json({ error: "Request body must be JSON." }, 400);
  }

  const text = (body.input ?? "").trim();
  const voice = body.voice ||
    (predefinedVoices["cosette"]
      ? "cosette"
      : Object.keys(predefinedVoices)[0]);

  if (!text) return json({ error: "'input' field is required." }, 400);
  if (!resolveVoice(voice)) {
    const available = [
      ...Object.keys(predefinedVoices),
      ...customVoices.keys(),
    ];
    return json({
      error: `Unknown voice "${voice}". Available: ${available.join(", ")}`,
    }, 400);
  }

  console.log(
    `[speech] voice="${voice}" text="${text.slice(0, 60)}${
      text.length > 60 ? "…" : ""
    }"`,
  );

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      // WAV header with streaming placeholder size (0xFFFFFFFF)
      controller.enqueue(buildWavHeader(SAMPLE_RATE, -1));
      try {
        for await (const chunk of generateSpeech(text, voice)) {
          controller.enqueue(float32ToPcm16(chunk));
        }
      } catch (err) {
        console.error("[speech] generation error:", err);
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      ...corsHeaders(),
      "Content-Type": "audio/wav",
      "Transfer-Encoding": "chunked",
      "X-Voice": voice,
      "Cache-Control": "no-cache",
    },
  });
}

// ─── Router ───────────────────────────────────────────────────────────────────

async function router(req: Request): Promise<Response> {
  if (req.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: corsHeaders() });
  }
  const { pathname } = new URL(req.url);
  const m = req.method;

  try {
    if (m === "GET" && pathname === "/") return handleInfo();
    if (m === "GET" && pathname === "/health") return handleHealth();
    if (m === "GET" && pathname === "/v1/voices") return handleListVoices();
    if (m === "POST" && pathname === "/v1/voices") {
      return await handleRegisterVoice(req, new URL(req.url));
    }
    if (m === "DELETE" && pathname.startsWith("/v1/voices/")) {
      return handleDeleteVoice(pathname);
    }
    if (m === "POST" && pathname === "/v1/audio/speech") {
      return await handleSpeech(req, new URL(req.url));
    }
    return json({ error: "Not found." }, 404);
  } catch (err) {
    console.error("[router] unhandled error:", err);
    return json({ error: String(err) }, 500);
  }
}

// ─── Entry Point ──────────────────────────────────────────────────────────────

console.log("Pocket TTS — Deno Server");
console.log("Loading models...");
await loadModels();
console.log("Models ready.\n");

const port_flag = Deno.args.findIndex(val => val === "--port")
const port_val = port_flag > -1 && Deno.args.length > port_flag + 1 ? Deno.args[port_flag+1] : ""
const port = port_val && Number.isInteger(+port_val) ? +port_val : 0

const server = Deno.serve({
  port,
  onListen: ({ port }) => {
    console.log(`Server listening on http://localhost:${port}`);
    console.log(`  Voices: ${Object.keys(predefinedVoices).join(", ")}`);
    console.log(`  Try: curl -s http://localhost:${port}/v1/audio/speech \\`);
    console.log(`         -H 'Content-Type: application/json' \\`);
    console.log(
      `         -d '{"input":"Hello world!","voice":"cosette"}' > speech.wav`,
    );
  },
}, router);

await server.finished;
