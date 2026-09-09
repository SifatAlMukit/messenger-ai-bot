const axios = require('axios');

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/models';
const API_TIMEOUT = 60000;
const MAX_RETRIES = 3;
const RETRY_DELAY = 2000;

// ─── Unified System Prompt ───
// openrouter.js এও একই prompt ব্যবহার হয়
const SYSTEM_INSTRUCTION = `You are a helpful AI assistant chatting via Facebook Messenger.

Formatting rules — Messenger cannot render markdown, so:
1. Never use **bold**, *italic*, or # headers
2. Use plain text with line breaks for readability
3. Use • for bullet points, 1. 2. 3. for numbered lists
4. Use [brackets] for emphasis if needed
5. For code: use triple backtick blocks with a language name
6. For math: use Unicode directly (×, ÷, ±, π, √, ∞, x², etc.)
7. Reply in the same language the user writes in`;

// ═══════════════════════════════════════
// OpenAI-style history → Gemini format
// ═══════════════════════════════════════
// - "assistant" → "model"
// - system prompt আলাদা (system_instruction)
// - contents এ user/model পর্যায়ক্রমে আসতে হবে
// - ছবি → inline_data (raw base64, prefix ছাড়া)

function buildContents(chatHistory, userText, base64Images) {
    const contents = [];

    // Chat history
    for (const msg of chatHistory) {
        contents.push({
            role: msg.role === 'assistant' ? 'model' : 'user',
            parts: [{ text: msg.content }]
        });
    }

    // Current user message parts
    const userParts = [];

    if (userText) {
        userParts.push({ text: userText });
    }

    // সব pending images যোগ করা
    if (base64Images && base64Images.length > 0) {
        for (const base64Image of base64Images) {
            const match = base64Image.match(/^data:([^;]+);base64,(.+)$/s);
            if (match) {
                userParts.push({
                    inline_data: {
                        mime_type: match[1],
                        data: match[2]  // raw base64, prefix ছাড়া
                    }
                });
            }
        }
    }

    if (userParts.length === 0) {
        userParts.push({ text: 'হ্যালো!' });
    }

    // Gemini requires strictly alternating user/model roles
    // শেষ item "user" হলে একটা dummy "model" দিতে হবে
    const lastItem = contents[contents.length - 1];
    if (lastItem && lastItem.role === 'user') {
        contents.push({ role: 'model', parts: [{ text: '.' }] });
    }

    contents.push({ role: 'user', parts: userParts });

    // প্রথম মেসেজ "model" হলে Gemini error দেয়
    if (contents.length > 0 && contents[0].role === 'model') {
        contents.unshift({ role: 'user', parts: [{ text: '.' }] });
    }

    return contents;
}

// ═══════════════════════════════════════
// Gemini API Call (retry সহ)
// ═══════════════════════════════════════

async function getResponse(chatHistory, userText, base64Images, model, retries = MAX_RETRIES) {
    if (!GEMINI_API_KEY) {
        throw new Error('GEMINI_API_KEY is not set');
    }

    // Single string বা array — দুটোই handle করা
    const imagesArray = Array.isArray(base64Images)
        ? base64Images
        : (base64Images ? [base64Images] : []);

    const contents = buildContents(chatHistory, userText, imagesArray);

    for (let attempt = 1; attempt <= retries; attempt++) {
        try {
            console.log(`🔄 [Gemini] Attempt ${attempt}/${retries} — ${model} | Images: ${imagesArray.length}`);

            const response = await axios.post(
                `${BASE_URL}/${model}:generateContent?key=${GEMINI_API_KEY}`,
                {
                    system_instruction: {
                        parts: [{ text: SYSTEM_INSTRUCTION }]
                    },
                    contents,
                    generationConfig: {
                        temperature: 0.7,
                        maxOutputTokens: 8192,
                        topP: 0.95
                    }
                },
                {
                    headers: { 'Content-Type': 'application/json' },
                    timeout: API_TIMEOUT
                }
            );

            const candidate = response.data?.candidates?.[0];

            if (!candidate?.content?.parts) {
                const reason = candidate?.finishReason || 'UNKNOWN';
                throw new Error(`No content. Finish reason: ${reason}`);
            }

            // Gemini 2.5+ thinking parts ফিল্টার করা
            const textParts = candidate.content.parts
                .filter(p => !p.thought)
                .map(p => p.text)
                .filter(Boolean);

            const text = textParts.join('\n').trim();

            if (!text) {
                throw new Error('Empty text after filtering');
            }

            console.log(`✅ [Gemini] Success on attempt ${attempt}`);
            return text;

        } catch (error) {
            const status = error.response?.status;
            const errMsg = error.response?.data?.error?.message || error.message;
            console.error(`❌ [Gemini] Attempt ${attempt}: [${status || 'N/A'}] ${errMsg}`);

            // Auth error → retry করে লাভ নেই
            if (status === 403 || status === 401) {
                throw error;
            }

            if (attempt < retries) {
                const waitTime = RETRY_DELAY * attempt;
                console.log(`⏳ Waiting ${waitTime / 1000}s...`);
                await sleep(waitTime);
            } else {
                throw error;
            }
        }
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

module.exports = { getResponse, SYSTEM_INSTRUCTION };
