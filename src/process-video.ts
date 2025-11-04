import 'dotenv/config';
import { execa } from 'execa';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { ZodError, z } from 'zod';

const {
  WHISPER_API_URL,
  WHISPER_API_LANGUAGE = '',
  WHISPER_API_TASK = 'transcribe',
  WHISPER_API_WORD_TS = 'false',

  // Fallback (only if you built whisper locally)
  WHISPER_BIN,
  WHISPER_MODEL,
  WHISPER_THREADS = '4',
  WHISPER_BEAM_SIZE = '5',

  LLM_PROVIDER: RAW_LLM_PROVIDER = 'ollama',
  LLM_MODEL: LEGACY_LLM_MODEL,

  OLLAMA_API_BASE,
  OLLAMA_API_KEY = 'ollama',
  OLLAMA_MODEL,

  API_BASE: RAW_OPENAI_BASE,
  OPENAI_API_KEY: RAW_OPENAI_KEY = '',
  OPENAI_MODEL,

  OUT_DIR = './dist',
} = process.env as Record<string, string>;

const stripTrailingSlash = (url: string) => url.replace(/\/$/, '');
const ensureV1 = (base: string) => {
  const trimmed = stripTrailingSlash(base);
  return trimmed.endsWith('/v1') ? trimmed : `${trimmed}/v1`;
};

const legacyOllamaDefault = 'http://localhost:11434/v1';
const normalizedProvider = RAW_LLM_PROVIDER.toLowerCase() === 'openai' ? 'openai' : 'ollama';
const ollamaBaseCandidate = OLLAMA_API_BASE
  ?? (RAW_OPENAI_BASE && RAW_OPENAI_BASE !== 'https://api.openai.com/v1' ? RAW_OPENAI_BASE : undefined)
  ?? legacyOllamaDefault;
const openAiBaseCandidate = RAW_OPENAI_BASE && RAW_OPENAI_BASE !== legacyOllamaDefault
  ? RAW_OPENAI_BASE
  : undefined;
const ollamaBase = ensureV1(ollamaBaseCandidate);
const openAiBase = ensureV1(openAiBaseCandidate ?? 'https://api.openai.com/v1');
const llmBase = normalizedProvider === 'openai' ? openAiBase : ollamaBase;
const LLM_ENDPOINT = `${llmBase}/chat/completions`;

const LLM_MODEL = LEGACY_LLM_MODEL
  ?? (normalizedProvider === 'openai'
    ? (OPENAI_MODEL ?? 'gpt-4o-mini')
    : (OLLAMA_MODEL ?? 'llama3.1:8b-instruct'));
const LLM_API_KEY = normalizedProvider === 'openai'
  ? RAW_OPENAI_KEY
  : (OLLAMA_API_KEY || 'ollama');

if (normalizedProvider === 'openai' && !LLM_API_KEY) {
  throw new Error('Set OPENAI_API_KEY when LLM_PROVIDER=openai');
}

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ensureDir = (p: string) => fs.mkdir(p, { recursive: true });
const exists = async (p: string) => { try { await fs.stat(p); return true; } catch { return false; } };
const baseNameNoExt = (p: string) => path.basename(p, path.extname(p));

const ChaptersSchema = z.array(z.object({ start: z.string(), title: z.string().min(3) }));
const YouTubeSchema = z.object({
  title: z.string().min(10).max(70),
  description: z.string().min(50),
  tags: z.array(z.string()).min(5).max(25),
  chapters: ChaptersSchema.optional(),
});
const SocialSchema = z.object({
  x: z.object({ main: z.string().max(280), thread: z.array(z.string().max(280)).max(3).optional() }),
  bluesky: z.object({ post: z.string().max(300) }),
  linkedin: z.object({ post: z.string().min(50), hashtags: z.array(z.string()).max(10).optional() }),
  reddit: z.object({ title: z.string().min(10).max(180), body: z.string().min(50) }),
});
const BlogSchema = z.object({
  title: z.string().min(10).max(120),
  content: z.string().min(200),
});
const OutputSchema = z.object({ youtube: YouTubeSchema, socials: SocialSchema, blog: BlogSchema });
export type Output = z.infer<typeof OutputSchema>;

// OpenAI-compatible LLM client (Ollama by default, OpenAI optional)
async function chatJSON<T>(messages: any[]): Promise<T> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (LLM_API_KEY) headers.Authorization = `Bearer ${LLM_API_KEY}`;
  console.log('LLM request to', LLM_ENDPOINT, 'with model', LLM_MODEL);
  const res = await fetch(LLM_ENDPOINT, {
    method: 'POST',
    headers,
    body: JSON.stringify({ model: LLM_MODEL, messages, }),
  });
  if (!res.ok) throw new Error(`LLM error ${res.status}: ${await res.text()}`);
  const data = await res.json();
  const content = data.choices?.[0]?.message?.content as string | undefined;
  if (!content) throw new Error('Empty LLM response');
  const match = content.match(/```json\n([\s\S]*?)\n```/m) || content.match(/\{[\s\S]*\}$/m);
  const jsonText = match ? (match[1] ?? match[0]) : content;

  try {
    try { return JSON.parse(jsonText) as T; }
    catch { return JSON.parse(jsonText.replace(/,\s*([}\]])/g, '$1')) as T; }
  } catch (err) {
    return jsonText as unknown as T;
  }
}

function makeSystemPrompt() {
  return `
  You are a YouTube strategist, social copywriter, and technical blog writer. 
  The videos are coding related tutorials or explainers.

  IMPORTANT: Do not make up information about the video that is not in the transcript. Only use what is provided. Decipher the content as accurately as possible from
  the transcript before making assumptions about the content. The transcript contains all the information you need to complete the task.

  When creating social texts try to make them engaging and clickable, using relevant hashtags where appropriate, also try to make it
  catchy to attract viewers. The copy should contain exactly what is in the transcript summarized in an engaging way, it should not
  contain any made up or irrelevant information. Try to make it a bit more expansive than 1-2 sentences to give more context and use
  emojis, bullet lists to engage the audience.

  I want you to come up with a title for the video based on the transcript and make it catchy and engaging. (clickbait style is ok)
  Then, generate a YouTube description that summarizes the video content, includes relevant keywords, and has a call to action.
  Also, create a list of 5-25 relevant tags for the video. The tags should be single words or short phrases without the '#' symbol.
  
  Create YouTube chapters for the content if possible, with timestamps in MM:SS format, the chapters should cover the main sections of the video and you
  should aim for 3-10 chapters depending on the length of the video. The name of the chapter should be descriptive of what the transcript section covers.

  The final output should be in JSON format with the following structure:
  {
    "youtube": {
      // max 70 characters
      "title": "string",
      "description": "string",
      // min 5, max 25
      "tags": ["string", "..."], 
      "chapters": [{"start": "MM:SS", "title": "string"}, "..."]
    },

  Additionally, create social media copy for the following platforms:
  - X (Twitter): A main tweet (<= 280 chars) and up to 3 follow-up tweets for a thread.
  - Bluesky: A post (<= 300 chars) summarizing the video.
  - LinkedIn: A post (>= 50 chars) with relevant hashtags (up to 10).
  - Reddit: A title (10-180 chars) and body (>= 50 chars) suitable for a relevant subreddit.

  Additionally, create a blog post based on the transcript:
  - Title: A compelling blog post title (10-120 chars)
  - Content: A well-structured blog post in Markdown format (minimum 200 chars) that:
    * Expands on the concepts from the transcript
    * Includes proper headings (##, ###)
    * Contains code examples deduced from the transcript (use proper markdown code blocks with language syntax highlighting)
    * Has clear explanations of the code
    * Follows a logical structure (introduction, main content sections, conclusion)
    * Uses bullet points and formatting for readability
    * If code is mentioned in the transcript, recreate it accurately in the blog post

  Return ONLY JSON with main keys "youtube", "socials", and "blog".
  Ensure the JSON is properly formatted and all strings are properly escaped.
  Example output:
  {
    "youtube": {
      "title": "Your Catchy Video Title",
      "description": "A well-crafted description of the video content...",
      "tags": ["tag1", "tag2", "..."],
      "chapters": [{"start": "00:00", "title": "Introduction"}, {"start": "02:15", "title": "Main Topic"}, "..."]
    },
    "socials": {
      "x": {"main": "This will blow your mind! ...", "thread": ["Follow-up tweet 1...", "..."]},
      "bluesky": {"post": "Bluesky post content..."},
      "linkedin": {"post": "LinkedIn post content...", "hashtags": ["hashtag1", "..."]},
      "reddit": {"title": "Reddit post title...", "body": "Reddit post body..."}
    },
    "blog": {
      "title": "Your Blog Post Title",
      "content": "# Your Blog Post Title\\n\\n## Introduction\\n\\nContent here...\\n\\n## Main Section\\n\\nMore content...\\n\\n\`\`\`javascript\\nconst example = 'code';\\n\`\`\`\\n\\n## Conclusion\\n\\nFinal thoughts..."
    }
  }
  Do not add any explanations outside the JSON. Adhere to the structure strictly and more is preferred to less.

  `;
}
function makeUserPrompt(videoTitle: string, transcript: string) {
  return ` \nTranscript of the video:\n${transcript}`;
}

// ---- Audio extraction
async function extractWav(input: string, outDir: string) {
  const wav = path.join(outDir, 'audio_16k.wav');
  await execa('ffmpeg', ['-y', '-i', input, '-ac', '1', '-ar', '16000', wav], { stdio: 'inherit' });
  return wav;
}

// ---- Transcription via container (preferred)
async function transcribeViaApi(wavPath: string, outDir: string) {
  if (!WHISPER_API_URL) throw new Error('WHISPER_API_URL not set');
  const existingSrcText = await fs.readFile(path.join(outDir, 'transcript.txt')).catch(() => null);
  const existingTranscript = await fs.readFile(path.join(outDir, 'original_transcript.txt')).catch(() => null);
  if (existingSrcText && existingTranscript) {
    console.log('Using existing transcript files.');
    return { srtText: existingTranscript.toString('utf8'), transcriptText: existingSrcText.toString('utf8') };
  }
  const buf = await fs.readFile(wavPath);
  const form = new FormData();
  form.append('file', new Blob([buf]), path.basename(wavPath));
  if (WHISPER_API_LANGUAGE) form.append('language', WHISPER_API_LANGUAGE);
  form.append('task', WHISPER_API_TASK);
  if (WHISPER_API_WORD_TS === 'true') form.append('word_ts', 'true');
  const res = await fetch(`${WHISPER_API_URL.replace(/\/$/, '')}/transcribe`, {
    method: 'POST',
    body: form
  });
  if (!res.ok) throw new Error(`Whisper API ${res.status}: ${await res.text()}`);
  const j = await res.json();
  const srtText: string = j.srt || (j.segments?.map((s: any) => `${s.text}`).join('\n') ?? '');
  const transcriptText: string = j.text || '';
  await fs.writeFile(path.join(outDir, 'transcript.txt'), srtText, 'utf8');
  await fs.writeFile(path.join(outDir, "original_transcript.txt"), srtText, 'utf8');
  return { srtText, transcriptText };
}



async function transcribeSmart(wavPath: string, outDir: string) {
  return transcribeViaApi(wavPath, outDir);

}

async function generateOutputs({ titleGuess, transcript }: { titleGuess: string; transcript: string; }) {
  //const MAX = 12000;
  const shortTx = transcript;
  const raw = await chatJSON<any>([
    { role: 'system', content: makeSystemPrompt() },
    { role: 'user', content: makeUserPrompt(titleGuess, shortTx) },
  ]);
  console.log('Raw LLM output:', raw);
  return OutputSchema.parse(raw);
}

async function writeOutputs(baseOut: string, out: Output, srt: string, transcript: string) {
  await ensureDir(baseOut);
  await fs.writeFile(path.join(baseOut, 'youtube.json'), JSON.stringify(out.youtube, null, 2), 'utf8');
  await fs.writeFile(path.join(baseOut, 'socials.json'), JSON.stringify(out.socials, null, 2), 'utf8');
  await fs.writeFile(path.join(baseOut, 'blog.json'), JSON.stringify(out.blog, null, 2), 'utf8');
  await fs.writeFile(path.join(baseOut, 'transcript.srt'), srt, 'utf8');
  await fs.writeFile(path.join(baseOut, 'transcript.txt'), transcript, 'utf8');

  const ytMd = `# YouTube\n\n**Title**\n\n${out.youtube.title}\n\n**Description**\n\n${out.youtube.description}\n\n**Tags**\n\n${out.youtube.tags.map(t => `#${t}`).join(' ')}\n\n`
    + (out.youtube.chapters ? `**Chapters**\n\n${out.youtube.chapters.map(c => `- ${c.start} — ${c.title}`).join('\n')}\n` : '');
  await fs.writeFile(path.join(baseOut, 'youtube.md'), ytMd, 'utf8');

  const socialsMd = `# Social Copy\n\n## X (Tweet)\n${out.socials.x.main}\n`
    + (out.socials.x.thread?.length ? `\n**Thread**\n${out.socials.x.thread.map((t, i) => `${i + 1}. ${t}`).join('\n')}` : '')
    + `\n\n## Bluesky\n${out.socials.bluesky.post}\n\n`
    + `## LinkedIn\n${out.socials.linkedin.post}\n\n`
    + (out.socials.linkedin.hashtags?.length ? out.socials.linkedin.hashtags.map(h => `#${h}`).join(' ') + '\n\n' : '')
    + `## Reddit\n**Title:** ${out.socials.reddit.title}\n\n${out.socials.reddit.body}\n`;
  await fs.writeFile(path.join(baseOut, 'socials.md'), socialsMd, 'utf8');

  // Write blog post as markdown
  await fs.writeFile(path.join(baseOut, 'blog.md'), out.blog.content, 'utf8');
}

async function extractAudioAndProcess(mp4Path: string, transcriptOnly = false) {
  const abs = path.resolve(mp4Path);
  const base = baseNameNoExt(abs);
  const outBase = path.resolve(OUT_DIR, base);
  await ensureDir(outBase);

  const totalSteps = transcriptOnly ? 3 : 4;
  console.log(`[1/${totalSteps}] Extracting audio → ${outBase}`);
  const wav = await extractWav(abs, outBase);

  console.log(`[2/${totalSteps}] Transcribing (HTTP if WHISPER_API_URL, else local binary)`);
  const { srtText, transcriptText } = await transcribeSmart(wav, outBase);

  if (transcriptOnly) {
    console.log(`[3/${totalSteps}] Writing transcript files`);
    await ensureDir(outBase);
    await fs.writeFile(path.join(outBase, 'transcript.srt'), srtText, 'utf8');
    await fs.writeFile(path.join(outBase, 'transcript.txt'), transcriptText, 'utf8');

    console.log(`\nDone. Open ${outBase} for:`);
    console.log('- transcript.srt / transcript.txt');
    return;
  }
  console.log('[3/4] Generating metadata & social copy with LLM');

  const outputs = await generateOutputs({ titleGuess: base, transcript: transcriptText });

  console.log('[4/4] Writing files');
  await writeOutputs(outBase, outputs, srtText, transcriptText);

  console.log(`\nDone. Open ${outBase} for:`);
  console.log('- youtube.md (title/description/tags/chapters)');
  console.log('- socials.md (X, Bluesky, LinkedIn, Reddit)');
  console.log('- blog.md (blog post with code examples)');
  console.log('- transcript.srt / transcript.txt');
}

// CLI
const args = process.argv.slice(2);
if (args[0] === '--help' || args.length === 0) {
  console.log(`\nUsage:\n  pnpm one <path/to/video.mp4> [--transcriptOnly]\n  pnpm watch ./inbox [--transcriptOnly]\n\nOptions:\n  --watch <dir>       Watch a folder and process new .mp4 files\n  --transcriptOnly    Only generate transcript, skip LLM metadata generation\n`);
  process.exit(0);
}

const transcriptOnly = args.includes('--transcriptOnly');
const filteredArgs = args.filter(arg => arg !== '--transcriptOnly');

if (filteredArgs[0] === '--watch') {
  const dir = path.resolve(filteredArgs[1] || './inbox');
  console.log(`Watching ${dir} for new .mp4 files...${transcriptOnly ? ' (transcript only mode)' : ''}`);
  await ensureDir(dir);
  const seen = new Set<string>();
  setInterval(async () => {
    const files = await fs.readdir(dir);
    for (const f of files) if (f.endsWith('.mp4')) {
      const full = path.join(dir, f);
      if (seen.has(full)) continue;
      seen.add(full);
      extractAudioAndProcess(full, transcriptOnly).catch(err => { console.error('Error:', err); seen.delete(full); });
    }
  }, 2000);
} else {
  const file = filteredArgs[0];
  if (!file) { console.error('Provide a path to an .mp4'); process.exit(1); }
  if (!(await exists(file))) { console.error('File does not exist:', file); process.exit(1); }
  await extractAudioAndProcess(file, transcriptOnly);
}
