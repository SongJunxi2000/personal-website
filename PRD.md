# Product Requirements Document: Personal Writing Website

## Vision & Design Philosophy

This is a personal website centered on writing, ideas, and contemplation. The design draws from **Zen Buddhism, wabi-sabi aesthetics, and the pursuit of inner peace** — valuing imperfection, simplicity, and the beauty of empty space.

**Design north star:** The site should feel like entering a quiet room. Every element earns its place through necessity, not decoration. The visitor should feel their breathing slow down.

### Inspiration References

- **wabi.ai** — Restrained typography, generous whitespace, soft transitions, content emerging gently from negative space. The page breathes.
- **off.site** — Poetic use of text as visual element, deliberate pacing, bold but minimal typography, dark palette option with high emotional impact.

### Core Design Principles

1. **Ma (間) — Negative space is content.** Whitespace is not empty — it is the most important design element. Margins and padding should be generous to the point of feeling luxurious.
2. **Wabi-sabi — Beauty in imperfection.** Avoid overly polished, corporate aesthetics. Favor organic textures, slightly asymmetric layouts, and natural warmth.
3. **应无所住 — When the heart dies, the Way is born.** Subtle, non-repeating micro-interactions that reward slow, attentive browsing. Nothing flashy.
4. **Kanso (簡素) — Simplicity.** Strip away everything that doesn't serve the content. If in doubt, remove it.

---

## Site Architecture

### Pages (9 total)

| Page | Route | Purpose |
|------|-------|--------|
| Home | `/` | Landing page — a single poetic statement or rotating quote, minimal navigation entry point |
| Writing | `/writing` | Unified index — all posts across all projects, sorted by recency |
| Language | `/language` | Project index for NLP & natural language processing writing |
| Language Post | `/language/[slug]` | Individual post within the Language project |
| Stillness | `/stillness` | Project index for Buddhism & meditation writing |
| Stillness Post | `/stillness/[slug]` | Individual post within the Stillness project |
| ApokalYi | `/apokalyi` | Project index for I-Ching & divination writing |
| ApokalYi Post | `/apokalyi/[slug]` | Individual post within the ApokalYi project |
| About | `/about` | Personal bio, philosophy, portrait or abstract visual |
| Connect | `/connect` | Contact information, social links, and an optional message form |

### Projects Overview

The site's writing is organized into three distinct projects, each with its own identity and accent color while sharing the same global design system.

| Project | Route | Topic | Accent Color |
|---------|-------|-------|--------------|
| **Language** | `/language` | Natural language processing, linguistics, machine learning | Cool slate — `#6B7FA3` |
| **Stillness** | `/stillness` | Buddhism, meditation, contemplative practice | Soft sage — `#7A9E87` |
| **ApokalYi** | `/apokalyi` | I-Ching, divination, oracular thought | Warm amber — `#A08B5B` |

Each project inherits all global tokens but overrides `--accent` and `--accent-hover` with its own color. Everything else — typography, spacing, layout, motion — remains identical.

---

## Page-by-Page Specifications

### 1. Home (`/`)

**Purpose:** First impression. Set the emotional tone. Make the visitor pause.

**Layout:**
- Full viewport height on load (100vh), content vertically and horizontally centered
- A single short phrase, quote, or personal mantra displayed in large, elegant serif typography
- Below the fold or after a gentle scroll: a minimal navigation row or subtle prompt to explore further
- No hero image. No banner. Just text and space.

**Interactions:**
- Text fades in softly on load (opacity 0 → 1 over ~1.5s, no dramatic easing)
- Gentle parallax or subtle vertical shift on scroll (2–5px movement, not aggressive)
- Navigation items appear on scroll or hover — they do not compete with the opening statement

**Content example (placeholder):**
> "Between stimulus and response there is a space. In that space is our freedom."

**Design notes:**
- The home page should feel like opening a book to a blank page with a single inscription
- Consider a very faint background texture — rice paper, linen, or subtle noise — to add warmth without distraction

---

### 2. Writing Index (`/writing`)

**Purpose:** A unified view of all writing across all three projects. The reader sees the full body of work in one place.

**Layout:**
- Page title: "Writing" or a personal label
- Post list displayed as a clean vertical stack, reverse chronological
- Each post entry shows: **title** (primary), **project label** (small, colored with the project's accent — "Language", "Stillness", or "ApokalYi"), **date** (secondary, muted), and an optional one-line excerpt
- No thumbnails, no tags cloud, no sidebar. Content only.
- Generous vertical spacing between entries (at least 48–64px)

**Interactions:**
- Entries subtly shift or change opacity on hover (opacity 0.6 → 1.0, or a gentle horizontal translate of 4–8px)
- The project label takes on its accent color on hover, reinforcing the visual identity of each project
- Staggered fade-in on page load — entries appear one by one with 80–120ms delay between each
- Optional: year dividers for posts spanning multiple years (just "2025" as a small, muted label)

**Filtering:**
- Three project filter tabs at the top: "All", "Language", "Stillness", "ApokalYi" — styled as minimal text toggles, not buttons. Active tab shows the project's accent color.
- Filter switches without page reload (client-side)
- Optional: a minimal search input, styled as just a thin underline

**Design notes:**
- The project label is the only color on this page — it should feel like a quiet signal, not a tag
- Think of this page as a shared table of contents for three books that live together on the same shelf

---

### 3. Project Index Pages (`/language`, `/stillness`, `/apokalyi`)

**Purpose:** Introduce a specific project and list all its posts. This is the dedicated home for each body of work.

**Layout:**
- Top section: project name in large serif type, followed by 2–4 sentences of description/introduction — what this project is about, what draws you to this topic
- Below: the post list, identical in structure to the Writing index but without the project label column (all posts belong to this project)
- The page carries the project's accent color throughout: links, hover states, any subtle decorative elements

**Interactions:**
- Same staggered fade-in as the Writing index
- Hover states use the project-specific accent color

**Design notes — per project:**
- **Language (`/language`):** The intro might reference the strangeness and beauty of how meaning is encoded in text. The cool slate accent should feel intellectual, precise, calm.
- **Stillness (`/stillness`):** The intro should read like the opening lines of a dharma talk — unhurried, inviting. The sage accent should feel like moss, breath, morning light.
- **ApokalYi (`/apokalyi`):** The intro can lean into the oracular — mysterious, ancient, associative. The amber accent should feel like candlelight, old ink, the turning of hexagrams.

---

### 4. Post Page (`/language/[slug]`, `/stillness/[slug]`, `/apokalyi/[slug]`)

**Purpose:** The core reading experience. This page should be world-class for long-form reading. Each post knows which project it belongs to and renders with that project's accent color.

**Project context:**
- A small breadcrumb or project label at the top (e.g., "← Stillness") links back to the project index — muted, unobtrusive
- The reading progress bar uses the project's accent color (very faintly — it should barely be noticeable)
- No other visual difference from the global post layout

**Layout:**
- Narrow content column: **max-width 680px**, centered on the page
- Post title at the top, large serif font
- Date and optional reading time below the title, small and muted
- Body text: optimized for readability — 18–20px font size, 1.6–1.8 line height, dark gray (not pure black) on warm off-white
- Generous paragraph spacing (1.5em–2em between paragraphs)
- Pull quotes styled distinctly: larger font, indented or offset, with a subtle vertical rule or extra spacing
- No sidebar, no related posts, no share buttons, no comments section

**Typography for body text:**
- Serif font for body (Georgia, Libre Baskerville, or Newsreader from Google Fonts)
- Headings within the post can use the same serif at a larger weight or a contrasting sans-serif
- Emphasis (italic) and strong (bold) should be used sparingly — the reader should feel the calm of uniform text

**Interactions:**
- Smooth scroll behavior globally
- A minimal, thin progress bar at the very top of the viewport (1–2px height) showing reading progress — muted color, barely noticeable
- Back-to-top: a small, subtle arrow that appears after scrolling 50% down — fades in, positioned bottom-right, very understated

**Design notes:**
- Study the reading experience on sites like Medium (column width), Craig Mod's essays (typography and pacing), and iA Writer (clarity)
- The goal is that after 30 seconds of reading, the reader forgets they're on a website
- Footnotes, if used, should be inline-expandable (click to reveal) rather than jumping to the bottom

---

### 5. About (`/about`)

**Purpose:** Personal introduction. Who is the person behind the writing?

**Layout:**
- Two possible approaches (choose during build):
  - **Option A — Vertical flow:** A portrait or abstract image at the top (full-width or centered, with rounded corners or soft edges), followed by 2–4 paragraphs of bio text in the same narrow column as blog posts
  - **Option B — Side-by-side:** On desktop, image on one side (40% width) and text on the other (60%), collapsing to vertical on mobile
- Bio text should be personal and warm — first person, conversational
- Below the bio: a short "currently" section (what you're reading, thinking about, working on) — this can be updated casually and adds life
- Optional: a small list of values or interests, displayed horizontally with generous spacing (not as a bulleted list)

**Interactions:**
- Minimal. This is a still, grounded page.
- Image can have a very subtle Ken Burns effect (slow, barely perceptible zoom over 20+ seconds)

**Design notes:**
- The About page is the human core of the site. It should feel intimate, like a letter.
- If no photo is available, consider an abstract visual — a brushstroke, ink wash, or botanical illustration that aligns with the Zen aesthetic

---

### 6. Connect (`/connect`)

**Purpose:** Let people reach out. Provide social/professional links.

**Layout:**
- A short, warm invitation to connect (1–2 sentences)
- Email address displayed as plain text (no mailto disguises or captchas — trust the visitor)
- Social links displayed as text labels (not icon-only) with subtle hover effects: "Twitter", "GitHub", "LinkedIn", etc.
- Links arranged vertically with generous spacing, or horizontally if few (3 or fewer)
- Optional: a minimal contact form with just two fields — name and message. No subject line, no dropdown, no captcha. Style it as a calm invitation, not a corporate form.

**Interactions:**
- Social links: subtle underline animation on hover (underline slides in from left)
- Form submission: a quiet "thank you" message replaces the form. No confetti, no redirect.

**Design notes:**
- This page should feel like the back cover of a book — a gentle closing gesture
- Keep it sparse. Five links maximum.

---

## Global Design System

### Color Palette

**Light mode (default):**

| Token | Value | Usage |
|-------|-------|-------|
| `--bg` | `#F5F1EB` | Page background — warm off-white, like aged paper |
| `--bg-subtle` | `#EDE8E0` | Card backgrounds, subtle sections |
| `--text-primary` | `#2C2C2C` | Body text — warm dark gray, never pure black |
| `--text-secondary` | `#8A8478` | Dates, captions, muted labels |
| `--text-tertiary` | `#B5AFA6` | Placeholder text, disabled states |
| `--accent` | `#7C6F5B` | Links, interactive elements — muted earth tone |
| `--accent-hover` | `#5C5245` | Hover state for accent |
| `--border` | `#DDD7CE` | Subtle dividers, input borders |
| `--reading-progress` | `#C4B9A8` | Progress bar, barely visible |

**Project accent overrides (applied via a data attribute, e.g., `data-project="stillness"` on `<body>`):**

| Project | `--accent` | `--accent-hover` | Character |
|---------|------------|------------------|----------|
| Language | `#6B7FA3` | `#4E6285` | Cool slate — intellectual, precise |
| Stillness | `#7A9E87` | `#5C8069` | Soft sage — breath, morning, moss |
| ApokalYi | `#A08B5B` | `#7D6C41` | Warm amber — old ink, candlelight |

These override only `--accent` and `--accent-hover`. All other tokens remain global.

**Dark mode (optional, respect `prefers-color-scheme`):**

| Token | Value | Usage |
|-------|-------|-------|
| `--bg` | `#1A1918` | Deep warm black |
| `--bg-subtle` | `#242220` | Elevated surfaces |
| `--text-primary` | `#E8E2D9` | Body text |
| `--text-secondary` | `#8A8478` | Muted text (same tone works) |
| `--accent` | `#B5A48A` | Lighter earth tone for dark bg |
| `--border` | `#3A3634` | Dark dividers |

### Typography

| Element | Font | Weight | Size | Line Height |
|---------|------|--------|------|-------------|
| Display (home quote) | Serif (e.g., `Cormorant Garamond`) | 300 (Light) | 48–64px desktop / 32–40px mobile | 1.2 |
| Page titles | Serif (same family) | 400 (Regular) | 36–42px desktop / 28–32px mobile | 1.2 |
| Post titles (in list) | Serif (same family) | 400 | 24–28px | 1.3 |
| Body text | Serif (e.g., `Libre Baskerville`, `Newsreader`) | 400 | 18–20px | 1.7 |
| Captions / dates | Sans-serif (e.g., `Inter`, `DM Sans`) | 400 | 13–14px | 1.4 |
| Navigation | Sans-serif | 400 | 14–16px | 1.0 |

**Font pairing recommendation:**
- Headings + Display: **Cormorant Garamond** (Google Fonts, free) — elegant, high contrast, literary feel
- Body: **Libre Baskerville** or **Source Serif Pro** — highly readable at body sizes, warm
- UI / Nav / Dates: **Inter** or **DM Sans** — clean, neutral, doesn't compete

### Spacing Scale

Use a base-8 scale consistently:

| Token | Value |
|-------|-------|
| `--space-xs` | 4px |
| `--space-sm` | 8px |
| `--space-md` | 16px |
| `--space-lg` | 32px |
| `--space-xl` | 64px |
| `--space-2xl` | 128px |
| `--space-3xl` | 192px |

**Key spacing rules:**
- Section padding (top/bottom): `--space-2xl` (128px) minimum
- Content max-width: 680px for reading, 960px for layout pages
- Side margins on mobile: `--space-lg` (32px) minimum — do not let content touch screen edges

### Navigation

- **Style:** Minimal, text-only. No hamburger menu on desktop. On mobile, a simple slide-in or full-screen overlay.
- **Position:** Top of page, not fixed/sticky. The navigation should scroll away — it is not more important than the content.
- **Items:** Home (or ensō mark), Writing, Language, Stillness, ApokalYi, About, Connect
- **Visual treatment:** Small sans-serif, uppercase or small-caps, letter-spacing 1–2px, muted color until hovered
- **Active state:** Subtle underline or slightly bolder weight. When inside a project (e.g., on `/stillness` or `/stillness/[slug]`), that project's nav item shows its accent color — a quiet signal of where you are.
- **Grouping (optional):** On desktop, "Language · Stillness · ApokalYi" can be grouped together with a slightly smaller type size or subtle visual separation from the top-level nav items
- **Optional:** A small circle or dot (ensō-inspired) as the "home" link instead of text

### Footer

- Nearly invisible. A single line at the bottom of each page.
- Content: Just the year and your name, or a short phrase. Example: `© 2026` or `应无所住`
- Muted color (`--text-tertiary`), small type, generous top margin

---

## Animation & Interaction Guidelines

### Philosophy
Animations should be **felt, not seen**. They exist to create a sense of calm transitions, not to impress. If a visitor consciously notices an animation, it's too much.

### Specifications

| Element | Animation | Duration | Easing |
|---------|-----------|----------|--------|
| Page transitions | Fade in (opacity 0→1) | 600–800ms | `ease-out` |
| Content stagger | Elements fade in sequentially | 80–120ms delay per item | `ease-out` |
| Hover (links) | Underline slides in from left | 300ms | `ease-in-out` |
| Hover (post items) | Opacity shift + subtle translate-x | 300ms | `ease-out` |
| Scroll reveal | Elements fade + translate-y (20px→0) | 500ms | `ease-out` |
| Reading progress bar | Width tracks scroll % | Real-time (no transition needed) | — |
| Back-to-top button | Fade in after 50% scroll | 400ms | `ease-in-out` |

### What NOT to animate
- No parallax backgrounds
- No scroll-jacking (never hijack native scroll)
- No animated gradients or color shifts
- No bouncing or elastic easing
- No loading spinners — if content loads fast, show nothing; if slow, a simple fade-in is enough

---

## Responsive Design

### Breakpoints

| Name | Width | Notes |
|------|-------|-------|
| Mobile | < 640px | Single column, generous side padding |
| Tablet | 640–1024px | Same as mobile layout with slightly wider content |
| Desktop | > 1024px | Full layout, centered content column |

### Key responsive behaviors
- Content column width adjusts smoothly (no jarring breakpoint jumps)
- Typography scales down proportionally on mobile (use `clamp()` for fluid sizing)
- Navigation collapses to a simple icon/text toggle on mobile
- Images scale to 100% width on mobile with maintained aspect ratio
- Spacing reduces proportionally but never feels cramped — mobile should still feel spacious

---

## Technical Recommendations

### Stack

| Layer | Technology | Rationale |
|-------|------------|----------|
| Framework | **Astro** (static output) | Ships zero JS by default; purpose-built for content sites; perfect fit for performance targets |
| Styling | **Vanilla CSS with custom properties** | Maps directly to the design token system; no build complexity; full control |
| Content | **Markdown files** in a `/content/[project]/` directory | Write posts in `.md` with frontmatter (`title`, `date`, `excerpt`, `project`) — no CMS needed |
| Fonts | **Google Fonts** (self-hosted for performance) | Cormorant Garamond + Libre Baskerville + Inter |
| Hosting | **Vercel** | Free tier, fast global CDN, automatic deploys from Git |
| Backend | **None** | Fully static. No database, no server, no CMS. |
| Domain | Custom domain | Essential for personal branding |

**Contact form:** If added, use a third-party service (e.g., Formspree or Resend) — no backend code lives in this repo.

### Performance targets
- First Contentful Paint: < 1.0s
- Total page weight: < 500KB (fonts will be the heaviest asset)
- No JavaScript required for core reading experience (progressive enhancement only)
- Lighthouse score: 95+ across all categories

### Accessibility
- Semantic HTML throughout (`<article>`, `<nav>`, `<main>`, `<footer>`)
- Proper heading hierarchy (one `<h1>` per page)
- Color contrast ratios: minimum 4.5:1 for body text, 3:1 for large text
- All interactive elements keyboard-accessible with visible focus states
- `prefers-reduced-motion` respected — all animations disabled when set
- `prefers-color-scheme` respected for dark mode

---

## Content Strategy (guidance for the site owner)

### Writing voice
- First person, reflective, unhurried
- Short paragraphs (3–5 sentences max)
- Use whitespace in writing the way the design uses it visually — let ideas breathe

### Post cadence
- Quality over frequency. One thoughtful essay per month is better than weekly noise.
- No "sorry I haven't posted" posts. The site exists outside of urgency.

### Recommended first posts
- An introductory post about why this site exists
- A reflection on a topic you care deeply about
- A short, poetic observation (to show the range of the site)

---

## Build Phases

### Phase 1 — Foundation
- Set up Astro project with static output mode
- Implement global layout: typography, colors, spacing, navigation, footer
- Build the Home page
- Build the About page with placeholder content
- Deploy to Vercel

### Phase 2 — Writing System
- Set up Markdown content pipeline with `project` field in frontmatter (`language` | `stillness` | `apokalyi`)
- Build the unified Writing index page with project labels and filter tabs
- Build the three project index pages (`/language`, `/stillness`, `/apokalyi`) with intro copy
- Build the post template (shared layout, project-aware accent color via `data-project`)
- Add 2–3 placeholder posts per project to verify the reading experience
- Implement reading progress bar with project accent color

### Phase 3 — Polish
- Build the Connect page
- Add all animations and micro-interactions
- Implement dark mode
- Responsive testing and refinement
- Accessibility audit
- Performance optimization (font subsetting, image optimization)

### Phase 4 — Launch
- Replace placeholder content with real content
- Set up custom domain
- Final cross-browser testing
- Share with a few trusted people for feedback
- Go live

---

## Success Criteria

The website is successful when:

1. A first-time visitor feels a sense of calm within 3 seconds of landing
2. Someone can read a full essay without any UI element distracting them
3. The site loads in under 1 second on a decent connection
4. You (the owner) feel proud sharing the URL with anyone
5. The design feels timeless — not trendy — a year from now
