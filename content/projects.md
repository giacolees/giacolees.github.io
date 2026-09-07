---
title: "Open Source Projects"
draft: false
---

A selection of public projects, grouped by topic. Everything lives on [GitHub](https://github.com/giacolees?tab=repositories).

## LLMs & NLP

<div style="display:flex;flex-direction:column;gap:1rem;margin-top:0.5rem;">

<div style="border:1px solid var(--accent);padding:1rem;">
<a href="https://github.com/giacolees/BPE-tokenizer" target="_blank" rel="noopener" style="font-weight:bold;color:var(--accent);text-decoration:none;">BPE-tokenizer ↗</a>
<p style="margin:0.4rem 0 0;font-size:0.9rem;opacity:0.8;">A from-scratch Byte-Pair Encoding tokenizer with four progressively optimized training algorithms — from naive O(V×M) to 85× faster with an inverted index and heap. Companion code for the <a href="/posts/tokenizers/">Tokenizers are easy!</a> post.</p>
<div style="margin-top:0.5rem;font-size:0.75rem;opacity:0.6;">Python · Tokenization · Algorithms</div>
</div>

<div style="border:1px solid var(--accent);padding:1rem;">
<a href="https://github.com/giacolees/local-llms-recipes" target="_blank" rel="noopener" style="font-weight:bold;color:var(--accent);text-decoration:none;">local-llms-recipes ↗</a>
<p style="margin:0.4rem 0 0;font-size:0.9rem;opacity:0.8;">LLM inference configurations and launchers for different GPU hardware profiles (4×A100, 2×A6000) across vLLM and llama.cpp. Companion code for the <a href="/posts/hardware-aware-programming-for-dummies/">Hardware-Aware Programming</a> post.</p>
<div style="margin-top:0.5rem;font-size:0.75rem;opacity:0.6;">Shell · vLLM · llama.cpp · CUDA GPUs</div>
</div>

<div style="border:1px solid var(--accent);padding:1rem;">
<a href="https://github.com/giacolees/multimodal-document-understanding" target="_blank" rel="noopener" style="font-weight:bold;color:var(--accent);text-decoration:none;">multimodal-document-understanding ↗</a>
<p style="margin:0.4rem 0 0;font-size:0.9rem;opacity:0.8;">Benchmark for testing Vision LLMs on unanswerable question detection over document images.</p>
<div style="margin-top:0.5rem;font-size:0.75rem;opacity:0.6;">Python · Vision LLMs · Benchmarking</div>
</div>

</div>

## Obsidian Plugins

Local-first plugins — your notes never leave the device.

<div style="display:flex;flex-direction:column;gap:1rem;margin-top:0.5rem;">

<div style="border:1px solid var(--accent);padding:1rem;">
<a href="https://github.com/giacolees/obsidian-math-convert" target="_blank" rel="noopener" style="font-weight:bold;color:var(--accent);text-decoration:none;">obsidian-math-convert ↗</a>
<p style="margin:0.4rem 0 0;font-size:0.9rem;opacity:0.8;">Obsidian plugin that converts photos or screenshots of equations to LaTeX locally — no cloud, no subscription, runs fully offline via WebAssembly.</p>
<div style="margin-top:0.5rem;font-size:0.75rem;opacity:0.6;">JavaScript · Obsidian · WebAssembly</div>
</div>

<div style="border:1px solid var(--accent);padding:1rem;">
<a href="https://github.com/giacolees/obsidian-local-voiceover" target="_blank" rel="noopener" style="font-weight:bold;color:var(--accent);text-decoration:none;">obsidian-local-voiceover ↗</a>
<p style="margin:0.4rem 0 0;font-size:0.9rem;opacity:0.8;">Private, local text-to-speech for Obsidian. Speak selected English text on-device — no API key or cloud uploads.</p>
<div style="margin-top:0.5rem;font-size:0.75rem;opacity:0.6;">TypeScript · Obsidian · On-device TTS</div>
</div>

</div>

## Perception & Research

<div style="display:flex;flex-direction:column;gap:1rem;margin-top:0.5rem;">

<div style="border:1px solid var(--accent);padding:1rem;">
<a href="https://github.com/giacolees/fault-detection-in-autonomous-vehicle-perception-systems" target="_blank" rel="noopener" style="font-weight:bold;color:var(--accent);text-decoration:none;">fault-detection-in-autonomous-vehicle-perception-systems ↗</a>
<p style="margin:0.4rem 0 0;font-size:0.9rem;opacity:0.8;">Research for the Master's thesis focused on advancing anomaly and fault detection techniques for autonomous vehicle perception systems.</p>
<div style="margin-top:0.5rem;font-size:0.75rem;opacity:0.6;">Python · Anomaly Detection · Autonomous Driving</div>
</div>

</div>

## Dev Environment & Agent Tooling

Dotfiles, terminal setup, and AI-agent workflow tooling.

<div style="display:flex;flex-direction:column;gap:1rem;margin-top:0.5rem;">

<div style="border:1px solid var(--accent);padding:1rem;">
<a href="https://github.com/giacolees/dotfiles" target="_blank" rel="noopener" style="font-weight:bold;color:var(--accent);text-decoration:none;">dotfiles ↗</a>
<p style="margin:0.4rem 0 0;font-size:0.9rem;opacity:0.8;">Cross-platform dotfiles managed with chezmoi, applying cleanly on both macOS and Ubuntu.</p>
<div style="margin-top:0.5rem;font-size:0.75rem;opacity:0.6;">Shell · chezmoi · macOS · Ubuntu</div>
</div>

<div style="border:1px solid var(--accent);padding:1rem;">
<a href="https://github.com/giacolees/herdr-openlogi" target="_blank" rel="noopener" style="font-weight:bold;color:var(--accent);text-decoration:none;">herdr-openlogi ↗</a>
<p style="margin:0.4rem 0 0;font-size:0.9rem;opacity:0.8;">Logitech mouse → herdr via OpenLogi binding overlay.</p>
<div style="margin-top:0.5rem;font-size:0.75rem;opacity:0.6;">Shell · herdr · Ghostty · macOS</div>
</div>

<div style="border:1px solid var(--accent);padding:1rem;">
<a href="https://github.com/giacolees/plan-herdr-subagents" target="_blank" rel="noopener" style="font-weight:bold;color:var(--accent);text-decoration:none;">plan-herdr-subagents ↗</a>
<p style="margin:0.4rem 0 0;font-size:0.9rem;opacity:0.8;">Single pi package for the entire /plan grill-and-plan workflow — grilling + domain-modeling → pointer plan → sequential workers → reviewer on herdr subagents.</p>
<div style="margin-top:0.5rem;font-size:0.75rem;opacity:0.6;">TypeScript · pi · Subagents · Planning</div>
</div>

</div>

## Side Projects

<div style="display:flex;flex-direction:column;gap:1rem;margin-top:0.5rem;">

<div style="border:1px solid var(--accent);padding:1rem;">
<a href="https://github.com/giacolees/openFanta-draft" target="_blank" rel="noopener" style="font-weight:bold;color:var(--accent);text-decoration:none;">openFanta-draft ↗</a>
<p style="margin:0.4rem 0 0;font-size:0.9rem;opacity:0.8;">Tools for the Fantacalcio (Italian fantasy football) 2026/27 auction — listone import, TIX/FIX player indexes, all managed with uv.</p>
<div style="margin-top:0.5rem;font-size:0.75rem;opacity:0.6;">Python · uv · Data Analysis</div>
</div>

</div>
