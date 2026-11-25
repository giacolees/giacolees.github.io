# giacolees — research journal

A statically generated Astro research journal, deployed to GitHub Pages.

## Run locally

Use npm 9.6.5 or later. A project-local Node.js 22 runtime is installed with dependencies so the npm scripts use a compatible version. Astro requires Node.js 22.23.4 or later when run directly.

```sh
npm install
npm run dev
```

Open the local URL printed by Astro (normally http://localhost:4321).

```sh
npm run check
npm run build
npm run preview
```

The production website is generated in `dist/`. The custom error page is `dist/404.html`; a future static host should use that file for missing URLs.

## Make it yours

Edit `src/site.ts` for the site name, homepage heading, and default description. The header logo and favicon are in `public/icons/`. Edit `src/pages/about.astro` for the biography. Adjust the palette and spacing in `src/styles/global.css`.

Add a Markdown file to `src/content/posts/`:

```yaml
---
title: A new question
date: 2026-02-01
description: A short summary of the note.
section: computation # computation, learning, or internet
tags: [research, tools] # optional
sample: false # optional; defaults to false
---
```

Write the article below the metadata. The filename determines its URL: `a-new-question.md` becomes `/posts/a-new-question/`. Use lowercase, hyphenated filenames; nested folders create nested URLs. Posts appear newest first. Use ISO dates (`YYYY-MM-DD`) to keep dates unambiguous.

Markdown supports headings, lists, links, blockquotes, fenced code with a language, and footnotes (`[^name]` with a corresponding `[^name]: Citation` definition). Page titles and descriptions are generated from metadata. Start article sections at `##` because the page supplies the main heading.

## Design and motion

Site assets live in `public/`: icons, article images, and the scripts and data for interactive demos. Decorative window controls are noninteractive and hidden from assistive technology.

Homepage entries reveal once using IntersectionObserver (350ms fade and 8px rise). Content remains visible without JavaScript. Reduced motion disables the reveal and transitions. Articles have a thin red progress line calculated against the article body using requestAnimationFrame; it reaches the end when the body’s bottom reaches the viewport bottom. Code blocks scroll independently and can receive keyboard focus when they overflow.

The site uses system fonts and local assets, with no remote font requests. No preview annotation strip is included.

## Topics, tags, and revisions

Every article has a required `section` field: `computation`, `learning`, or `internet`. The homepage links to an archive for each section. Edit titles and descriptions in `src/topics.ts`; to add a topic, add its identifier to `sectionIds` and its title/description to `sections`, then assign posts to it. Section pages are generated automatically.

Tags are optional, cross-topic labels. Use lowercase, hyphenated tags (for example `machine-learning`). Click a tag on an article or use `/tags/` to browse its archive. These links work without JavaScript.

Add an optional `updated: 2026-02-10` date to revised articles. It must be on or after `date`. Publication order and RSS publication dates stay based on the original date.

Articles link to the previous (older) and next (newer) entry across the full notebook. About content is in `src/pages/about.astro`; add your real biography and verified contact/profile links there.

## RSS

The feed is available at `/rss.xml` and linked in the footer and page metadata. The default production origin is `https://giacolees.github.io`. Override it for another host:

```sh
SITE_URL=https://your-domain.example npm run build
```

The feed includes all posts, original publication dates, descriptions, topic categories, and tags. Sample entries are identified in their descriptions.

Topic cards lift 2px with a crisp shadow on hover and keyboard focus (150ms). Links gain a pale purple highlight (120ms). Supporting browsers crossfade same-origin page navigations over 150ms using native CSS cross-document view transitions; other browsers navigate normally. Full page navigation preserves article-progress and post-reveal initialization. Reduced motion removes the lift and all added transitions while retaining static link/focus feedback. These effects work without JavaScript.

## GitHub Pages

The Pages workflow builds Astro with Node 22.23.5 and deploys `dist/` on pushes to `main`. For local development, run `nvm use`, `npm ci`, and `npm run dev`. The former Hugo site is backed up outside this repository.
