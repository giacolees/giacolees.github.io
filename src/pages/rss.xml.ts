import type { APIRoute } from 'astro';
import { getCollection } from 'astro:content';
import { site } from '../site';
import { sections } from '../topics';
const escape = (value: string) => value.replace(/[<>&"']/g, char => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;', "'": '&apos;' })[char]!);
export const GET: APIRoute = async ({ site: base }) => {
  if (!base) throw new Error('Set site in astro.config.mjs to generate RSS URLs.');
  const posts = (await getCollection('posts')).sort((a,b) => b.data.date.valueOf() - a.data.date.valueOf());
  const items = posts.map(post => {
    const url = new URL(`/posts/${post.id}/`, base).href;
    return `<item><title>${escape(post.data.title)}</title><link>${escape(url)}</link><guid isPermaLink="true">${escape(url)}</guid><description>${escape(post.data.description + (post.data.sample ? ' (Sample post)' : ''))}</description><pubDate>${post.data.date.toUTCString()}</pubDate><category>${escape(sections[post.data.section].title)}</category>${(post.data.tags ?? []).map(tag => `<category>${escape(tag)}</category>`).join('')}</item>`;
  }).join('');
  return new Response(`<?xml version="1.0" encoding="UTF-8"?><rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom"><channel><title>${escape(site.name)} — research journal</title><link>${escape(base.href)}</link><description>${escape(site.description)}</description><language>en</language><atom:link href="${escape(new URL('/rss.xml', base).href)}" rel="self" type="application/rss+xml"/>${items}</channel></rss>`, { headers: { 'Content-Type': 'application/rss+xml; charset=utf-8' } });
};
