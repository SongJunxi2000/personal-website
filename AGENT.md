# Agent Instructions — Personal Writing Website

This file instructs Claude Code on how to assist with content on this site.
Read this file at the start of every session involving content changes.

---

## Site Overview

- **Framework:** Astro (static), located in `astro-src/`
- **Content directory:** `astro-src/src/content/` — three subfolders: `language/`, `stillness/`, `apokalyi/`
- **Languages supported:** English (`en`) and Chinese (`zh`)
- **Language toggle:** CSS-driven via `data-lang` attribute on `<html>`; preference stored in `localStorage`

---

## Post File Format

Every post is a Markdown file with the following frontmatter structure:

```markdown
---
title: "English title"
title_zh: "中文标题"
date: "YYYY-MM-DD"
excerpt: "One-line English excerpt."
excerpt_zh: "一行中文摘要。"
project: language | stillness | apokalyi
readingTime: <number>          # estimated minutes
auto_translated: true | false  # true = ZH was machine-translated, pending review
---

<div data-lang="en">

English body content here (Markdown works inside the div).

</div>

<div data-lang="zh">

<!-- AUTO-TRANSLATED — PENDING REVIEW -->

Chinese body content here.

</div>
```

**Important:** Both `<div data-lang="en">` and `<div data-lang="zh">` sections must be present in every published post. If one language is missing, the page will show blank content for that language.

---

## Trigger: New Post Added

When the user says they have added a new post (or drops a new `.md` file into a content folder), do the following **in order**:

### Step 1 — Identify the post
- Find the new file in `astro-src/src/content/`
- Read the entire file including frontmatter

### Step 2 — Proofread the original
- Check for: grammar, clarity, consistency of voice, broken Markdown syntax, missing frontmatter fields, concise
- Report issues inline with suggested corrections
- Ask the user to confirm fixes before applying

### Step 3 — Determine the source language
- If the post body is inside `<div data-lang="en">` → source is English, translate to Chinese
- If the post body is inside `<div data-lang="zh">` → source is Chinese, translate to English
- If no `data-lang` divs exist yet, ask the user which language the post was written in

### Step 4 — Translate
- Translate the full body content into the other language
- Also translate: `title`, `excerpt` (into `title_zh` / `excerpt_zh` or vice versa)
- Preserve all Markdown formatting: headings, italics, bold, blockquotes, horizontal rules
- Preserve structural HTML (`<div data-lang>` wrappers, any other HTML elements)
- Do NOT translate proper nouns, technical terms, or quoted text unless clearly appropriate
- For Chinese → English: use natural, literary English that matches the voice of the site
- For English → Chinese: use fluent, contemporary Chinese prose — avoid overly literal phrasing

### Step 5 — Write the translation back to the file
- Add the translated content into the appropriate `<div data-lang>` section in the same file
- Add `<!-- AUTO-TRANSLATED — PENDING REVIEW -->` as the first line inside the translated `<div>`
- Set `auto_translated: true` in frontmatter
- Do NOT remove or modify the source language content

### Step 6 — Flag for review
After writing, tell the user:
- Which file was updated
- A summary of any proofreading changes made
- The translation is marked as **auto-translated** and needs their review before publishing
- Remind them: once they have reviewed and approved the translation, they should set `auto_translated: false` in the frontmatter

---

## Trigger: Post Modified

When the user says they have changed something in an existing post, do the following:

### Step 1 — Read the updated file
- Identify what changed (frontmatter, source language body, or translated body)

### Step 2 — If the source language content changed
- Re-translate only the changed sections (do not retranslate unchanged passages)
- If the change is small (a sentence or phrase), update only that part in the translation
- If the change is structural (added/removed section, rewritten paragraph), retranslate the affected section
- Re-add `<!-- AUTO-TRANSLATED — PENDING REVIEW -->` at the top of the translated section
- Set `auto_translated: true` in frontmatter

### Step 3 — If only the translated content changed
- The user is reviewing or correcting a translation — do not overwrite their work
- If `auto_translated` is still `true`, remind them to set it to `false` once satisfied

### Step 4 — Flag for review
Same as Step 6 above.

---

## Trigger: User Asks for Full Review

When the user asks to review a post (proofreading, translation quality, or both):

1. Read the file
2. Evaluate the source language content: grammar, voice, clarity, structure
3. Evaluate the translation: accuracy, naturalness, preservation of tone
4. Present findings as a numbered list with suggested changes
5. Apply changes only after user confirmation

---

## Voice & Style Guidelines

These apply to both proofreading and translation.

**English voice:**
- First person, reflective, unhurried
- Short paragraphs (3–5 sentences)
- Literary but accessible — not academic
- Avoid: corporate language, hedging, filler phrases ("In conclusion…", "It is important to note…")

**Chinese voice (中文风格):**
- 第一人称，沉思，不急不躁
- 段落简短，语言流畅自然
- 书面语与现代口语之间的平衡——不要文绉绉，也不要随意
- 避免：过于直译的欧化句式，冗余的连接词

---

## File Naming Convention

- Slugs are the filename without `.md`
- Use lowercase, hyphenated English slugs even for Chinese-language posts
- Example: `on-sitting.md` (not `关于坐禅.md`)

---

## What NOT to do

- Do not modify any files in `astro-src/src/layouts/`, `astro-src/src/styles/`, or `astro-src/src/pages/` unless the user explicitly asks
- Do not auto-publish or run `npm run build` unless asked
- Do not change `auto_translated: false` back to `true`
- Do not remove the `<!-- AUTO-TRANSLATED — PENDING REVIEW -->` comment — the user removes it when satisfied
